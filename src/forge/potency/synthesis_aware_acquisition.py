"""Choose an experimental batch under a synthesis budget.

The bottleneck in this project is not generation and it is not routing. 92.1% of samples clear
terminal admission, and 97.1% of prediction-supported designs resolve to purchasable material.
The bottleneck is that the activity model is weak and the experiments that would improve it
are expensive. So the interesting decision is not "which molecules score highest" but "which
batch of makeable molecules will both find something and teach us how to rank the rest".

This module implements that as a budgeted greedy selection over three quantities:

  value        a conservative lower bound on activity, so nothing obviously dead is bought
  information  reduction in posterior variance over a declared target population
  cost         documented synthetic burden, in preparations or steps

Two guards are deliberate. An activity floor is applied before anything else, so "exploration"
can never mean synthesising a molecule we expect to fail; and the information term is measured
over an explicit target set, so it rewards learning about the population we care about rather
than learning about outliers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from forge.potency.reaction_factorized_surrogate import ReactionFactorizedSurrogate


class SynthesisAwareAcquisitionError(RuntimeError):
    """Raised when a batch cannot be selected under the declared constraints."""


@dataclass(frozen=True)
class Candidate:
    """One feasible design: its structures, its conservative score, and what it costs."""

    identifier: str
    product: str
    amine: str
    aldehyde: str
    isocyanide: str
    cost: float = 1.0

    def as_record(self) -> dict[str, str]:
        return {
            "product": self.product,
            "amine": self.amine,
            "aldehyde": self.aldehyde,
            "isocyanide": self.isocyanide,
        }


@dataclass(frozen=True)
class AcquisitionPolicy:
    """Everything that must be fixed before candidates are seen.

    ``value_weight`` trades conservative value against information. 0 is pure experimental
    design, 1 is pure exploitation and reproduces score ranking. ``activity_floor`` is applied
    as a hard filter on the conservative bound before any trade-off is computed, expressed as a
    quantile of the feasible pool so it does not depend on the scale of the endpoint.
    """

    batch_size: int
    budget: float
    value_weight: float = 0.5
    activity_floor_quantile: float = 0.5
    cost_normalised: bool = True

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise SynthesisAwareAcquisitionError("batch_size must be positive")
        if self.budget <= 0:
            raise SynthesisAwareAcquisitionError("budget must be positive")
        if not 0.0 <= self.value_weight <= 1.0:
            raise SynthesisAwareAcquisitionError("value_weight must lie in [0, 1]")
        if not 0.0 <= self.activity_floor_quantile < 1.0:
            raise SynthesisAwareAcquisitionError("activity_floor_quantile must lie in [0, 1)")


@dataclass
class AcquisitionResult:
    selected: list[Candidate] = field(default_factory=list)
    total_cost: float = 0.0
    variance_before: float = 0.0
    variance_after: float = 0.0
    eligible_after_floor: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def variance_reduction(self) -> float:
        return self.variance_before - self.variance_after

    def summary(self) -> dict[str, Any]:
        return {
            "selected": [c.identifier for c in self.selected],
            "batch_size": len(self.selected),
            "total_cost": self.total_cost,
            "eligible_after_floor": self.eligible_after_floor,
            "variance_before": self.variance_before,
            "variance_after": self.variance_after,
            "variance_reduction": self.variance_reduction,
        }


def apply_activity_floor(
    surrogate: ReactionFactorizedSurrogate,
    candidates: Sequence[Candidate],
    quantile: float,
) -> tuple[list[Candidate], float]:
    """Drop candidates whose conservative bound falls below a quantile of the pool.

    This lives outside the selection rules on purpose. The floor is a property of the
    experiment, not of any one acquisition method, so the benchmark applies it once and hands
    the identical filtered pool to every arm. Applying it inside a single selector would
    quietly confine that selector to the top of the predicted distribution while its baselines
    roamed the whole pool, which is a comparison about the floor rather than about the rule.
    """
    if not candidates:
        raise SynthesisAwareAcquisitionError("no candidates supplied to the activity floor")
    if not 0.0 <= quantile < 1.0:
        raise SynthesisAwareAcquisitionError("activity floor quantile must lie in [0, 1)")
    bounds = surrogate.lower_bound([c.as_record() for c in candidates])
    floor = float(np.quantile(bounds, quantile))
    kept = [c for c, b in zip(candidates, bounds, strict=True) if b >= floor]
    if not kept:
        raise SynthesisAwareAcquisitionError("activity floor removed every candidate")
    return kept, floor


def _normalise(values: np.ndarray) -> np.ndarray:
    """Map to [0, 1] so value and information are commensurable before weighting."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def select_batch(
    surrogate: ReactionFactorizedSurrogate,
    candidates: Sequence[Candidate],
    target: Sequence[dict[str, str]],
    policy: AcquisitionPolicy,
) -> AcquisitionResult:
    """Greedy budgeted selection maximising weighted value plus information per unit cost.

    Greedy is used rather than an exact combinatorial solve because the information term is
    computed against a live posterior that changes as the batch grows, and because a batch of
    a dozen from a pool of thousands does not warrant more. Every marginal gain is recomputed
    against the posterior conditioned on what has already been chosen, so the batch is
    selected jointly rather than as the top-k of a fixed score.
    """
    policy.validate()
    if not candidates:
        raise SynthesisAwareAcquisitionError("no candidates supplied")
    if not target:
        raise SynthesisAwareAcquisitionError(
            "target population is empty; information would be undefined"
        )

    # The floor is applied by the caller through apply_activity_floor, so that every arm in a
    # comparison receives the same pool. Passing a non-zero quantile here filters again, which
    # is correct for standalone use and must be left at zero inside a benchmark.
    pool, _ = (
        apply_activity_floor(surrogate, candidates, policy.activity_floor_quantile)
        if policy.activity_floor_quantile > 0.0
        else (list(candidates), float("-inf"))
    )
    pool_value = _normalise(surrogate.lower_bound([c.as_record() for c in pool]))

    result = AcquisitionResult(eligible_after_floor=len(pool))

    # The target Gram is fixed for the round; the covariance is carried forward by rank-one
    # updates rather than refitted, which is what keeps a thousand-candidate pool tractable.
    gram = surrogate.target_gram(target)
    _, covariance = surrogate._require_fit()
    result.variance_before = surrogate.variance_over_gram(gram, covariance)

    pool_records = [c.as_record() for c in pool]
    chosen: list[Candidate] = []
    remaining = set(range(len(pool)))
    running_cost = 0.0

    while len(chosen) < policy.batch_size and remaining:
        affordable = sorted(i for i in remaining if running_cost + pool[i].cost <= policy.budget)
        if not affordable:
            break

        gains = surrogate.variance_reduction_scores(
            [pool_records[i] for i in affordable], gram, covariance
        )
        combined = policy.value_weight * pool_value[affordable] + (
            1.0 - policy.value_weight
        ) * _normalise(gains)
        if policy.cost_normalised:
            combined = combined / np.array([max(pool[i].cost, 1e-9) for i in affordable])

        slot = int(np.argmax(combined))
        pick = affordable[slot]
        chosen.append(pool[pick])
        running_cost += pool[pick].cost
        remaining.discard(pick)
        covariance = surrogate.rank_one_update(covariance, pool_records[pick])
        result.trace.append(
            {
                "step": len(chosen),
                "identifier": pool[pick].identifier,
                "marginal_variance_reduction": float(gains[slot]),
                "cumulative_cost": running_cost,
            }
        )

    if not chosen:
        raise SynthesisAwareAcquisitionError("budget admitted no candidate")

    result.selected = chosen
    result.total_cost = running_cost
    result.variance_after = surrogate.variance_over_gram(gram, covariance)
    return result


# --------------------------------------------------------------------- baselines
#
# Every baseline receives the same feasible pool, the same budget and the same batch size, so
# the comparison isolates the selection rule. They are here rather than in the benchmark script
# because a baseline that lives beside the method is harder to quietly disadvantage.


def _budgeted(
    candidates: Sequence[Candidate], order: Sequence[int], policy: AcquisitionPolicy
) -> list[Candidate]:
    chosen: list[Candidate] = []
    cost = 0.0
    for index in order:
        candidate = candidates[index]
        if len(chosen) >= policy.batch_size:
            break
        if cost + candidate.cost > policy.budget:
            continue
        chosen.append(candidate)
        cost += candidate.cost
    return chosen


def select_random(
    candidates: Sequence[Candidate], policy: AcquisitionPolicy, rng: np.random.Generator
) -> list[Candidate]:
    order = rng.permutation(len(candidates))
    return _budgeted(candidates, order.tolist(), policy)


def select_top_mean(
    surrogate: ReactionFactorizedSurrogate,
    candidates: Sequence[Candidate],
    policy: AcquisitionPolicy,
) -> list[Candidate]:
    mu, _ = surrogate.predict([c.as_record() for c in candidates])
    return _budgeted(candidates, np.argsort(-mu).tolist(), policy)


def select_top_lower_bound(
    surrogate: ReactionFactorizedSurrogate,
    candidates: Sequence[Candidate],
    policy: AcquisitionPolicy,
) -> list[Candidate]:
    bounds = surrogate.lower_bound([c.as_record() for c in candidates])
    return _budgeted(candidates, np.argsort(-bounds).tolist(), policy)


def select_uncertainty(
    surrogate: ReactionFactorizedSurrogate,
    candidates: Sequence[Candidate],
    policy: AcquisitionPolicy,
) -> list[Candidate]:
    _, sd = surrogate.predict([c.as_record() for c in candidates])
    return _budgeted(candidates, np.argsort(-sd).tolist(), policy)


def select_component_diversity(
    candidates: Sequence[Candidate], policy: AcquisitionPolicy, rng: np.random.Generator
) -> list[Candidate]:
    """Round-robin over unseen precursor identities, then fall back to random.

    This is the strong non-model baseline: it is what a thoughtful chemist does with the
    component table and no surrogate at all, and the method has to beat it to be worth having.
    """
    order = rng.permutation(len(candidates)).tolist()
    chosen: list[Candidate] = []
    cost = 0.0
    seen: dict[str, set[str]] = {"amine": set(), "aldehyde": set(), "isocyanide": set()}
    for _ in range(len(order)):
        best, best_new = None, -1
        for index in order:
            candidate = candidates[index]
            if candidate in chosen or cost + candidate.cost > policy.budget:
                continue
            new = sum(
                1
                for role, value in (
                    ("amine", candidate.amine),
                    ("aldehyde", candidate.aldehyde),
                    ("isocyanide", candidate.isocyanide),
                )
                if value not in seen[role]
            )
            if new > best_new:
                best, best_new = candidate, new
        if best is None or len(chosen) >= policy.batch_size:
            break
        chosen.append(best)
        cost += best.cost
        seen["amine"].add(best.amine)
        seen["aldehyde"].add(best.aldehyde)
        seen["isocyanide"].add(best.isocyanide)
    return chosen


def select_mixture(
    surrogate: ReactionFactorizedSurrogate,
    candidates: Sequence[Candidate],
    policy: AcquisitionPolicy,
    rng: np.random.Generator,
    *,
    exploit_fraction: float,
) -> list[Candidate]:
    """Allocate batch positions between top-mean exploitation and component diversity.

    This is the control the reaction-factorized rule actually has to beat. That rule currently
    sits between two simple extremes - top-mean finds hits, component diversity teaches the
    ranker - and a reviewer is entitled to ask whether it is an elaborate way of interpolating
    between them. Comparing against the endpoints does not answer that; comparing against the
    line joining them does.

    Positions are allocated deterministically, exploitation first, then diversity fills the
    remainder from what is left. Duplicates cannot arise because the diversity pass never sees
    an already-chosen candidate.
    """
    if not 0.0 <= exploit_fraction <= 1.0:
        raise SynthesisAwareAcquisitionError("exploit_fraction must lie in [0, 1]")
    n_exploit = int(round(exploit_fraction * policy.batch_size))

    chosen: list[Candidate] = []
    cost = 0.0
    if n_exploit:
        exploit_policy = AcquisitionPolicy(
            batch_size=n_exploit,
            budget=policy.budget,
            value_weight=policy.value_weight,
            activity_floor_quantile=0.0,
            cost_normalised=policy.cost_normalised,
        )
        chosen = select_top_mean(surrogate, candidates, exploit_policy)
        cost = sum(c.cost for c in chosen)

    taken = {c.identifier for c in chosen}
    remainder = [c for c in candidates if c.identifier not in taken]
    n_diverse = policy.batch_size - len(chosen)
    if n_diverse > 0 and remainder:
        diverse_policy = AcquisitionPolicy(
            batch_size=n_diverse,
            budget=max(policy.budget - cost, 0.0),
            value_weight=policy.value_weight,
            activity_floor_quantile=0.0,
            cost_normalised=policy.cost_normalised,
        )
        chosen.extend(select_component_diversity(remainder, diverse_policy, rng))
    return chosen


def select_uncertainty_mixture(
    surrogate: ReactionFactorizedSurrogate,
    candidates: Sequence[Candidate],
    policy: AcquisitionPolicy,
    *,
    exploit_fraction: float,
) -> list[Candidate]:
    """The same interpolation, with uncertainty sampling as the learning endpoint."""
    if not 0.0 <= exploit_fraction <= 1.0:
        raise SynthesisAwareAcquisitionError("exploit_fraction must lie in [0, 1]")
    n_exploit = int(round(exploit_fraction * policy.batch_size))

    chosen: list[Candidate] = []
    cost = 0.0
    if n_exploit:
        exploit_policy = AcquisitionPolicy(
            batch_size=n_exploit,
            budget=policy.budget,
            value_weight=policy.value_weight,
            activity_floor_quantile=0.0,
            cost_normalised=policy.cost_normalised,
        )
        chosen = select_top_mean(surrogate, candidates, exploit_policy)
        cost = sum(c.cost for c in chosen)

    taken = {c.identifier for c in chosen}
    remainder = [c for c in candidates if c.identifier not in taken]
    n_learn = policy.batch_size - len(chosen)
    if n_learn > 0 and remainder:
        learn_policy = AcquisitionPolicy(
            batch_size=n_learn,
            budget=max(policy.budget - cost, 0.0),
            value_weight=policy.value_weight,
            activity_floor_quantile=0.0,
            cost_normalised=policy.cost_normalised,
        )
        chosen.extend(select_uncertainty(surrogate, remainder, learn_policy))
    return chosen


SELECTORS: dict[str, Callable[..., Any]] = {
    "random": select_random,
    "top_mean": select_top_mean,
    "top_lower_bound": select_top_lower_bound,
    "uncertainty": select_uncertainty,
    "component_diversity": select_component_diversity,
    "reaction_factorized": select_batch,
    "mixture": select_mixture,
    "uncertainty_mixture": select_uncertainty_mixture,
}
