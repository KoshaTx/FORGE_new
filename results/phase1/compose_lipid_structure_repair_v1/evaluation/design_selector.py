"""Scoped development selector: qualified design correction before frozen support objectives.

This module neither constructs chemistry nor converts TRAIN support into chemical
validity. Unknown candidates remain in the pool. Source-exact assessment and
reference diagnostics are supplied by separately pinned evaluators.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class ComponentIdentity:
    role: str
    smiles: str
    global_train_novel: bool
    role_train_novel: bool


@dataclass(frozen=True)
class SelectionCandidate:
    request: int
    ordinal: int
    smiles: str | None
    connected: bool
    exact: bool
    local_features_observed: bool
    product_train_novel: bool
    components: tuple[ComponentIdentity, ...] = ()

    def __post_init__(self):
        if self.request < 0 or self.ordinal < 0:
            raise ValueError("Request and candidate ordinal must be nonnegative")
        if self.connected and not self.smiles:
            raise ValueError("A connected candidate requires a product")
        if self.exact and (not self.connected or not self.components):
            raise ValueError("Source-exact candidates need a connected product and components")
        if self.local_features_observed and not self.exact:
            raise ValueError("The selection support flag is defined only for exact candidates")
        roles = [part.role for part in self.components]
        if len(roles) != len(set(roles)) or any(
            not p.role or not p.smiles for p in self.components
        ):
            raise ValueError("Components need unique nonempty roles and identities")
        if not self.exact and self.components:
            raise ValueError("Only source-exact candidates may contribute component diversity")


def first_exact(candidates: Sequence[SelectionCandidate]) -> SelectionCandidate:
    if not candidates or len({c.request for c in candidates}) != 1:
        raise ValueError("One nonempty candidate pool per request is required")
    ordered = sorted(candidates, key=lambda c: c.ordinal)
    if len({c.ordinal for c in ordered}) != len(ordered):
        raise ValueError("Candidate ordinals must be unique per request")
    return next((c for c in ordered if c.exact), ordered[0])


def _counts(selection):
    groups = defaultdict(Counter)
    novelty = Counter()
    for candidate in selection:
        if candidate.connected:
            groups["product"][candidate.smiles] += 1
            novelty["product"] += candidate.product_train_novel
        for part in candidate.components:
            key = "component:" + part.role
            groups[key][part.smiles] += 1
            novelty[key + ":global"] += part.global_train_novel
            novelty[key + ":role"] += part.role_train_novel
    return dict(groups), novelty


def _multiplicity_power_product(counts):
    """For fixed N, smaller product(count**count) is greater Shannon entropy."""
    return math.prod(value**value for value in counts.values())


def check_noninferiority(baseline, selected):
    """No numerical tolerance can waive a diversity or novelty floor."""
    if len(selected) != len(baseline) or len({c.request for c in selected}) != len(selected):
        raise ValueError("Selection must contain exactly one candidate per request")
    before_by_request = {c.request: c for c in baseline}
    if set(before_by_request) != {c.request for c in selected}:
        raise ValueError("Selection changed the requested population")
    if any(c.exact != before_by_request[c.request].exact for c in selected):
        raise ValueError("Selection changed source-exact request coverage")
    before, before_novel = _counts(baseline)
    after, after_novel = _counts(selected)
    if set(before) != set(after):
        raise ValueError("Selection changed the diversity role space")
    for group, counts in before.items():
        new = after[group]
        if sum(counts.values()) != sum(new.values()):
            raise ValueError("Selection changed a diversity denominator")
        if len(new) < len(counts):
            raise ValueError("Selection reduced a unique-identity count")
        if sum(v * v for v in new.values()) > sum(v * v for v in counts.values()):
            raise ValueError("Selection reduced Simpson effective count")
        if _multiplicity_power_product(new) > _multiplicity_power_product(counts):
            raise ValueError("Selection reduced Shannon effective count")
    if any(after_novel[key] < count for key, count in before_novel.items()):
        raise ValueError("Selection reduced TRAIN novelty")


def select_design_supported(
    pools: Sequence[Sequence[SelectionCandidate]],
    *,
    baseline: Sequence[SelectionCandidate],
    design_pass: frozenset[tuple[int, int]],
    node_limit: int = 10000,
    time_limit: float = 5.0,
) -> tuple[list[SelectionCandidate], dict[str, Any]]:
    """Select one family's pools while preserving the saved current exact statuses.

    Lexicographic objectives maximize qualified design successes, then distinct
    supported products, supported requests, and deterministic ordinal cost.
    Unproved results retain the supplied current baseline with explicit status.
    """
    if (
        type(node_limit) is not int
        or node_limit < 1
        or not math.isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError("Finite positive solver bounds are required")
    pools = sorted(pools, key=lambda pool: first_exact(pool).request)
    baseline = sorted(baseline, key=lambda c: c.request)
    if len(baseline) != len(pools) or any(
        c not in pool for c, pool in zip(baseline, pools, strict=True)
    ):
        raise ValueError("Saved current baseline must belong to every augmented pool")
    if len({candidate.request for candidate in baseline}) != len(baseline):
        raise ValueError("Request identifiers must be unique")
    candidates = []
    excluded = []
    for pool, first in zip(pools, baseline, strict=True):
        roles = {part.role for part in first.components}
        for candidate in sorted(pool, key=lambda c: c.ordinal):
            if first.exact and candidate.exact and {p.role for p in candidate.components} == roles:
                candidates.append(candidate)
            elif not first.exact and candidate == first:
                candidates.append(candidate)
            else:
                excluded.append(
                    {
                        "request": candidate.request,
                        "ordinal": candidate.ordinal,
                        "reason": "not_exact_or_different_registered_role_set",
                    }
                )
    candidate_keys = {(c.request, c.ordinal) for pool in pools for c in pool}
    if not design_pass <= candidate_keys:
        raise ValueError("Design pass set contains an unknown candidate")
    if any((c.request, c.ordinal) in design_pass and not c.exact for pool in pools for c in pool):
        raise ValueError("Qualified design success must retain exact source L1")
    if all(sum(c.request == first.request for c in candidates) == 1 for first in baseline):
        return baseline, {"status": "no_alternative", "excluded": excluded, "stages": []}
    costs, terms, lower, upper = [], [], [], []

    def variable():
        costs.append(0.0)
        return len(costs) - 1

    def constraint(values, low=-np.inf, high=np.inf):
        terms.append(values)
        lower.append(low)
        upper.append(high)

    requests = defaultdict(list)
    groups = defaultdict(lambda: defaultdict(list))
    novelty = defaultdict(list)
    novel_groups = defaultdict(lambda: defaultdict(list))
    supported_products = defaultdict(list)
    for candidate in candidates:
        i = variable()
        requests[candidate.request].append(i)
        if candidate.connected:
            groups["product"][candidate.smiles].append(i)
            if candidate.product_train_novel:
                novelty["product"].append(i)
                novel_groups["product"][candidate.smiles].append(i)
        for part in candidate.components:
            key = "component:" + part.role
            groups[key][part.smiles].append(i)
            if part.global_train_novel:
                novelty[key + ":global"].append(i)
                novel_groups[key + ":global"][part.smiles].append(i)
            if part.role_train_novel:
                novelty[key + ":role"].append(i)
                novel_groups[key + ":role"][part.smiles].append(i)
        if candidate.local_features_observed:
            supported_products[candidate.smiles].append(i)
    for indices in requests.values():
        constraint(dict.fromkeys(indices, 1), low=1, high=1)
    original_counts, original_novelty = _counts(baseline)
    for group, identities in sorted(groups.items()):
        unique, squared, entropy = {}, {}, {}
        for _, indices in sorted(identities.items()):
            maximum = len({candidates[i].request for i in indices})
            prefix = [variable() for _ in range(maximum)]
            constraint({**dict.fromkeys(indices, 1), **dict.fromkeys(prefix, -1)}, low=0, high=0)
            for left, right in zip(prefix, prefix[1:]):
                constraint({left: 1, right: -1}, low=0)
            unique[prefix[0]] = 1
            for count, index in enumerate(prefix, 1):
                squared[index] = 2 * count - 1
                entropy[index] = count * math.log(count) - (
                    (count - 1) * math.log(count - 1) if count > 1 else 0.0
                )
        original = original_counts[group]
        constraint(unique, low=len(original))
        constraint(squared, high=sum(count**2 for count in original.values()))
        constraint(entropy, high=sum(count * math.log(count) for count in original.values()))
    for key, count in original_novelty.items():
        constraint(dict.fromkeys(novelty[key], 1), low=count)
    baseline_novel_unique = _novel_unique(baseline)
    for key, identities in sorted(novel_groups.items()):
        presence = []
        for _, indices in sorted(identities.items()):
            y = variable()
            presence.append(y)
            constraint({**dict.fromkeys(indices, 1), y: -1}, low=0)
            constraint({**dict.fromkeys(indices, 1), y: -len(indices)}, high=0)
        constraint(dict.fromkeys(presence, 1), low=len(baseline_novel_unique[key]))
    supported_presence = []
    for indices in supported_products.values():
        y = variable()
        supported_presence.append(y)
        constraint({**dict.fromkeys(indices, 1), y: -1}, low=0)
        constraint({**dict.fromkeys(indices, 1), y: -len(indices)}, high=0)
    objectives = [
        (
            "design_clean_requests",
            {i: -1 for i, c in enumerate(candidates) if (c.request, c.ordinal) in design_pass},
        ),
        ("supported_distinct", dict.fromkeys(supported_presence, -1)),
        (
            "supported_requests",
            {i: -1 for i, c in enumerate(candidates) if c.local_features_observed},
        ),
        ("ordinal", {i: c.ordinal for i, c in enumerate(candidates)}),
    ]
    stages = []
    chosen = baseline
    for name, objective in objectives:
        rr, cc, values = [], [], []
        for row, coefficients in enumerate(terms):
            for column, value in coefficients.items():
                if value:
                    rr.append(row)
                    cc.append(column)
                    values.append(value)
        matrix = coo_matrix((values, (rr, cc)), shape=(len(terms), len(costs))).tocsc()
        vector = np.zeros(len(costs))
        for index, value in objective.items():
            vector[index] = value
        result = milp(
            vector,
            integrality=np.ones(len(costs)),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(matrix, lower, upper),
            options={"node_limit": node_limit, "time_limit": time_limit, "mip_rel_gap": 0.0},
        )
        stages.append({"objective": name, "status": int(result.status), "message": result.message})
        if result.status != 0 or result.x is None:
            return baseline, {
                "status": "censored_retained_current_baseline",
                "stages": stages,
                "excluded": excluded,
            }
        if not np.isfinite(result.x).all() or not np.allclose(
            result.x, np.rint(result.x), atol=1e-6, rtol=0
        ):
            return baseline, {
                "status": "nonintegral_retained_current_baseline",
                "stages": stages,
                "excluded": excluded,
            }
        rounded = np.rint(result.x).astype(np.int64)
        chosen = [candidates[i] for i in range(len(candidates)) if rounded[i]]
        try:
            check_strict_noninferiority(baseline, chosen)
        except ValueError as error:
            return baseline, {
                "status": "failed_independent_check_retained_current_baseline",
                "reason": str(error),
                "stages": stages,
                "excluded": excluded,
            }
        optimum = sum(value * int(rounded[index]) for index, value in objective.items())
        constraint(objective, low=optimum, high=optimum)
        stages[-1]["objective_value"] = optimum
    before_supported = {c.smiles for c in baseline if c.local_features_observed}
    after_supported = {c.smiles for c in chosen if c.local_features_observed}
    before_design = sum((c.request, c.ordinal) in design_pass for c in baseline)
    after_design = sum((c.request, c.ordinal) in design_pass for c in chosen)
    if after_design < before_design:
        raise ValueError("Primary qualified-design objective decreased")
    if after_design == before_design and len(after_supported) < len(before_supported):
        raise ValueError("Support objective decreased at unchanged primary design value")
    return sorted(chosen, key=lambda c: c.request), {
        "status": "optimal_independently_verified",
        "stages": stages,
        "excluded": excluded,
        "requests": len(baseline),
        "eligible_candidates": len(candidates),
        "design_clean_before": before_design,
        "design_clean_after": after_design,
        "supported_distinct_before": len(before_supported),
        "supported_distinct_after": len(after_supported),
    }


def _novel_unique(selection):
    groups = defaultdict(set)
    for candidate in selection:
        if candidate.connected and candidate.product_train_novel:
            groups["product"].add(candidate.smiles)
        for part in candidate.components:
            key = "component:" + part.role
            if part.global_train_novel:
                groups[key + ":global"].add(part.smiles)
            if part.role_train_novel:
                groups[key + ":role"].add(part.smiles)
    return groups


def check_strict_noninferiority(baseline, selected):
    check_noninferiority(baseline, selected)
    before, after = _novel_unique(baseline), _novel_unique(selected)
    if any(len(after[key]) < len(values) for key, values in before.items()):
        raise ValueError("Selection reduced distinct TRAIN-novel identities")
