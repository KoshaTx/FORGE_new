"""Deterministic TRAIN-only component splits for a bounded graph-score diagnostic."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.chemistry.smiles import canonical_constitution
from forge.core.io import iter_csv

ROLES = ("amine_head", "oxoester_aldehyde_body_tail")
GROUPS = ("fit", "amine_disjoint", "aldehyde_disjoint", "repeated_component")


class UgiCandidateGraphCoverageError(ValueError):
    """The fixed TRAIN coverage or split contract is inconsistent."""


def _canonical(value: Any, memo: dict[str, str]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UgiCandidateGraphCoverageError("missing component structure")
    if value not in memo:
        memo[value] = str(canonical_constitution(value, error=UgiCandidateGraphCoverageError))
    return memo[value]


def load_train_assignments(path: Path) -> dict[str, dict[str, Any]]:
    """Mask before accessing any identity or structure; canonicalize TRAIN components once."""
    output: dict[str, dict[str, Any]] = {}
    memo: dict[str, str] = {}
    for row in iter_csv(path):
        if row["primary_product_fold"] != "train":
            continue
        product_id = row["product_id"]
        if not product_id or product_id in output:
            raise UgiCandidateGraphCoverageError(
                f"missing or duplicate TRAIN product: {product_id}"
            )
        output[product_id] = {
            "product_id": product_id,
            "primary_product_fold": "train",
            "component_smiles": {role: _canonical(row[f"{role}_smiles"], memo) for role in ROLES},
        }
    if not output:
        raise UgiCandidateGraphCoverageError("no TRAIN assignment rows")
    return output


def _bucket(value: str, domain: str, bucket_count: int) -> int:
    return int(hashlib.sha256((domain + value).encode()).hexdigest(), 16) % bucket_count


def _measure(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "source_weight_sum": math.fsum(row["source_weight"] for row in rows),
        "original_source_probability_mass": math.fsum(row["source_probability"] for row in rows),
        "components_by_role": {
            role: len({row["component_smiles"][role] for row in rows}) for role in ROLES
        },
    }


def build_selection(
    coverage_rows: Sequence[Mapping[str, Any]],
    train_assignments: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep the old applicability set, hash both components, then source-weight draws.

    The caller authenticates source files and verifies complete cache coverage and weights.
    Old coverage partition fields are retained as historical metadata; only the new
    ``partition_groups`` and ``component_withheld`` fields define this diagnostic's split.
    No component structures outside explicit TRAIN assignment records are interpreted.
    """
    partition = config["partition"]
    bucket_count = partition["bucket_count"]
    evaluation_bucket = partition["evaluation_bucket"]
    if (
        type(bucket_count) is not int
        or bucket_count < 2
        or (type(evaluation_bucket) is not int or not 0 <= evaluation_bucket < bucket_count)
    ):
        raise UgiCandidateGraphCoverageError("invalid component/product bucket policy")
    if any(
        not isinstance(partition[key], str) or not partition[key]
        for key in ("component_domain", "product_domain")
    ):
        raise UgiCandidateGraphCoverageError("empty partition domain")
    if set(config["draws"]) != set(GROUPS) or any(
        type(config["draws"][group]) is not int or config["draws"][group] <= 0 for group in GROUPS
    ):
        raise UgiCandidateGraphCoverageError("all four fixed draw counts must be positive integers")
    seed = config["seeds"]["draws"]
    if type(seed) is not int or seed < 0:
        raise UgiCandidateGraphCoverageError("draw seed must be a nonnegative integer")
    ids, indices = set(), set()
    memo: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    populations: dict[str, list[dict[str, Any]]] = {name: [] for name in GROUPS}
    for original in coverage_rows:
        product_id = original["product_id"]
        cache_index = original["cache_index"]
        if not isinstance(product_id, str) or not product_id or product_id in ids:
            raise UgiCandidateGraphCoverageError(
                f"missing or duplicate coverage product: {product_id}"
            )
        if type(cache_index) is not int or cache_index < 0 or cache_index in indices:
            raise UgiCandidateGraphCoverageError(f"invalid or duplicate cache index: {cache_index}")
        ids.add(product_id)
        indices.add(cache_index)
        if type(original["applicable"]) is not bool:
            raise UgiCandidateGraphCoverageError(f"malformed applicability: {product_id}")
        for field in ("source_weight", "source_probability"):
            value = original[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (not math.isfinite(value) or value <= 0)
            ):
                raise UgiCandidateGraphCoverageError(f"invalid {field}: {product_id}")
        assignment = train_assignments.get(product_id)
        if assignment is None or assignment["primary_product_fold"] != "train":
            raise UgiCandidateGraphCoverageError(f"missing explicit TRAIN assignment: {product_id}")
        if assignment["product_id"] != product_id:
            raise UgiCandidateGraphCoverageError(
                f"TRAIN assignment identity mismatch: {product_id}"
            )
        components = {
            role: _canonical(assignment["component_smiles"][role], memo) for role in ROLES
        }
        if (
            "amine_smiles" in original
            and _canonical(original["amine_smiles"], memo) != components[ROLES[0]]
        ):
            raise UgiCandidateGraphCoverageError(
                f"coverage/assignment amine identity mismatch: {product_id}"
            )
        withheld = {
            role: _bucket(f"{role}:{components[role]}", partition["component_domain"], bucket_count)
            == evaluation_bucket
            for role in ROLES
        }
        product_bucket = _bucket(product_id, partition["product_domain"], bucket_count)
        memberships = []
        if withheld[ROLES[0]]:
            memberships.append("amine_disjoint")
        if withheld[ROLES[1]]:
            memberships.append("aldehyde_disjoint")
        if not any(withheld.values()):
            memberships.append(
                "repeated_component" if product_bucket == evaluation_bucket else "fit"
            )
        row = {
            **original,
            "component_smiles": components,
            "component_withheld": withheld,
            "candidate_graph_product_bucket": product_bucket,
            "partition_groups": memberships,
        }
        rows.append(row)
        if original["applicable"]:
            for group in memberships:
                populations[group].append(row)
    if not rows or any(not population for population in populations.values()):
        raise UgiCandidateGraphCoverageError(
            "fixed subdivision has an empty population; no reseed allowed"
        )
    rows.sort(key=lambda row: row["cache_index"])
    for population in populations.values():
        population.sort(key=lambda row: row["cache_index"])
    fit_components = {
        role: {row["component_smiles"][role] for row in populations["fit"]} for role in ROLES
    }
    for role, group in zip(ROLES, ("amine_disjoint", "aldehyde_disjoint"), strict=True):
        if fit_components[role] & {row["component_smiles"][role] for row in populations[group]}:
            raise UgiCandidateGraphCoverageError(f"component leaked from fit into {group}")
    rng = np.random.default_rng(seed)
    groups = {}
    for name in GROUPS:
        population = populations[name]
        weights = np.asarray([row["source_weight"] for row in population], dtype=np.float64)
        total = float(weights.sum())
        if not math.isfinite(total) or total <= 0:
            raise UgiCandidateGraphCoverageError(f"invalid source weight total: {name}")
        chosen = rng.choice(
            len(population), size=config["draws"][name], replace=True, p=weights / total
        )
        selected = [
            {
                **population[int(index)],
                "draw_index": draw,
                "selection_group": name,
                "candidate_graph_draw_probability": float(weights[int(index)] / total),
            }
            for draw, index in enumerate(chosen)
        ]
        groups[name] = {
            "population": _measure(population),
            "sampled_unique_products": len({row["product_id"] for row in selected}),
            "sampled_components_by_role": {
                role: len({row["component_smiles"][role] for row in selected}) for role in ROLES
            },
            "draws": selected,
        }
    sampled_fit = {
        role: {row["component_smiles"][role] for row in groups["fit"]["draws"]} for role in ROLES
    }
    for group in groups.values():
        group["draws_with_component_seen_in_actual_fit_draws"] = {
            role: sum(row["component_smiles"][role] in sampled_fit[role] for row in group["draws"])
            for role in ROLES
        }
    amine_ids = {row["product_id"] for row in populations["amine_disjoint"]}
    aldehyde_ids = {row["product_id"] for row in populations["aldehyde_disjoint"]}
    overlap_ids = amine_ids & aldehyde_ids
    overlap = [row for row in rows if row["product_id"] in overlap_ids]
    excluded = [row for row in rows if not row["applicable"]]
    return {
        "groups": groups,
        "coverage_rows": rows,
        "coverage": {
            "schema_version": "forge.ugi_candidate_graph_coverage.v1",
            "partition": dict(partition),
            "draw_seed": seed,
            "group_order": list(GROUPS),
            "population_order": "ascending_cache_index",
            "all_train": _measure(rows),
            "applicable": _measure([row for row in rows if row["applicable"]]),
            "excluded": _measure(excluded),
            "exclusion_reasons": dict(
                sorted(Counter(reason for row in excluded for reason in row["reasons"]).items())
            ),
            "component_disjoint_evaluation_overlap": {
                **_measure(overlap),
                "product_ids": sorted(overlap_ids),
                "sampled_product_ids_in_both_evaluation_groups": sorted(
                    {row["product_id"] for row in groups["amine_disjoint"]["draws"]}
                    & {row["product_id"] for row in groups["aldehyde_disjoint"]["draws"]}
                ),
            },
            "nonclaims": [
                "Prior applicability is preserved and does not certify target graph support or full terminal producibility.",
                "The two component-disjoint evaluation populations overlap; both remain TRAIN data for the original backbone.",
                "Source weights define replacement draw probabilities and must not be applied a second time to sampled exposures.",
            ],
        },
    }
