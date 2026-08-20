"""Tests for budgeted synthesis-aware batch acquisition.

The properties worth defending are the ones a reviewer would attack: that the budget and the
activity floor are real constraints rather than decoration, that the batch is chosen jointly
rather than as a top-k, and that the policy degenerates to score ranking when told to.
"""

from __future__ import annotations

import numpy as np
import pytest

from forge.potency.oracle.reaction_factorized_surrogate import ReactionFactorizedSurrogate
from forge.potency.ranking.synthesis_aware_acquisition import (
    AcquisitionPolicy,
    Candidate,
    SynthesisAwareAcquisitionError,
    select_batch,
    select_component_diversity,
    select_random,
    select_top_lower_bound,
)

TAILS = ["CCCC", "CCCCC", "CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC"]
HEADS = ["NCCN(C)C", "NCCCN(C)C", "NCCN1CCCC1"]


def make_candidates(n: int = 12) -> list[Candidate]:
    out = []
    for index in range(n):
        tail = TAILS[index % len(TAILS)]
        head = HEADS[index % len(HEADS)]
        out.append(
            Candidate(
                identifier=f"X{index:02d}",
                product=f"{tail}NC(=O)C(CCCOC(=O){tail})N{head[1:]}",
                amine=head,
                aldehyde=f"{tail}C(=O)OCCCC=O",
                isocyanide=f"[C-]#[N+]{tail}",
                cost=1.0 + (index % 3),
            )
        )
    return out


def fitted_surrogate(candidates: list[Candidate]) -> ReactionFactorizedSurrogate:
    records = [c.as_record() for c in candidates[:6]]
    values = [float(i) for i in range(len(records))]
    return ReactionFactorizedSurrogate(n_bits=64).fit(records, values)


def test_selects_a_batch_within_size_and_budget():
    candidates = make_candidates()
    surrogate = fitted_surrogate(candidates)
    policy = AcquisitionPolicy(batch_size=3, budget=6.0, activity_floor_quantile=0.0)
    result = select_batch(surrogate, candidates, [c.as_record() for c in candidates], policy)
    assert len(result.selected) <= 3
    assert result.total_cost <= 6.0
    assert len({c.identifier for c in result.selected}) == len(result.selected)


def test_budget_binds_before_batch_size():
    candidates = make_candidates()
    surrogate = fitted_surrogate(candidates)
    cheapest = min(c.cost for c in candidates)
    policy = AcquisitionPolicy(batch_size=10, budget=cheapest, activity_floor_quantile=0.0)
    result = select_batch(surrogate, candidates, [c.as_record() for c in candidates], policy)
    assert len(result.selected) == 1
    assert result.total_cost <= cheapest


def test_activity_floor_removes_the_weak_half():
    candidates = make_candidates()
    surrogate = fitted_surrogate(candidates)
    policy = AcquisitionPolicy(batch_size=2, budget=99.0, activity_floor_quantile=0.5)
    result = select_batch(surrogate, candidates, [c.as_record() for c in candidates], policy)
    assert result.eligible_after_floor <= len(candidates)
    bounds = surrogate.lower_bound([c.as_record() for c in candidates])
    floor = float(np.quantile(bounds, 0.5))
    for chosen in result.selected:
        assert surrogate.lower_bound([chosen.as_record()])[0] >= floor


def test_variance_never_increases_and_is_reported_consistently():
    candidates = make_candidates()
    surrogate = fitted_surrogate(candidates)
    policy = AcquisitionPolicy(batch_size=4, budget=99.0, activity_floor_quantile=0.0)
    target = [c.as_record() for c in candidates]
    result = select_batch(surrogate, candidates, target, policy)
    assert result.variance_after <= result.variance_before + 1e-9
    assert result.variance_reduction >= -1e-9
    # the reported endpoint must equal a direct recomputation on the chosen batch
    direct = surrogate.variance_reduction([c.as_record() for c in result.selected], target)
    assert result.variance_reduction == pytest.approx(direct, rel=1e-6, abs=1e-9)


def test_pure_value_weight_reproduces_score_ranking():
    """value_weight = 1 must degenerate to the exploitative baseline, or the knob is a lie."""
    candidates = [c for c in make_candidates() if c.cost == 1.0]
    surrogate = fitted_surrogate(make_candidates())
    policy = AcquisitionPolicy(
        batch_size=2,
        budget=99.0,
        value_weight=1.0,
        activity_floor_quantile=0.0,
        cost_normalised=False,
    )
    chosen = select_batch(surrogate, candidates, [c.as_record() for c in candidates], policy)
    ranked = select_top_lower_bound(surrogate, candidates, policy)
    assert [c.identifier for c in chosen.selected] == [c.identifier for c in ranked]


def test_pure_information_weight_differs_from_score_ranking():
    candidates = make_candidates()
    surrogate = fitted_surrogate(candidates)
    policy = AcquisitionPolicy(
        batch_size=4,
        budget=99.0,
        value_weight=0.0,
        activity_floor_quantile=0.0,
        cost_normalised=False,
    )
    informative = select_batch(surrogate, candidates, [c.as_record() for c in candidates], policy)
    exploitative = select_top_lower_bound(surrogate, candidates, policy)
    assert [c.identifier for c in informative.selected] != [c.identifier for c in exploitative]


def test_batch_is_chosen_jointly_not_as_top_k():
    """Marginal gains must fall as the batch grows; a top-k rule would not show that."""
    candidates = make_candidates(16)
    surrogate = fitted_surrogate(candidates)
    policy = AcquisitionPolicy(
        batch_size=5,
        budget=99.0,
        value_weight=0.0,
        activity_floor_quantile=0.0,
        cost_normalised=False,
    )
    result = select_batch(surrogate, candidates, [c.as_record() for c in candidates], policy)
    gains = [step["marginal_variance_reduction"] for step in result.trace]
    assert gains[0] >= gains[-1]
    assert len(result.trace) == len(result.selected)


def test_cost_normalisation_prefers_cheaper_designs():
    candidates = make_candidates()
    surrogate = fitted_surrogate(candidates)
    common = dict(batch_size=3, budget=99.0, value_weight=0.5, activity_floor_quantile=0.0)
    cheap = select_batch(
        surrogate,
        candidates,
        [c.as_record() for c in candidates],
        AcquisitionPolicy(**common, cost_normalised=True),
    )
    plain = select_batch(
        surrogate,
        candidates,
        [c.as_record() for c in candidates],
        AcquisitionPolicy(**common, cost_normalised=False),
    )
    assert cheap.total_cost <= plain.total_cost


def test_component_diversity_baseline_covers_more_roles_than_random():
    candidates = make_candidates(18)
    policy = AcquisitionPolicy(batch_size=6, budget=99.0)
    rng = np.random.default_rng(0)
    diverse = select_component_diversity(candidates, policy, rng)
    drawn = select_random(candidates, policy, np.random.default_rng(0))
    assert len({c.aldehyde for c in diverse}) >= len({c.aldehyde for c in drawn})


def test_baselines_respect_the_same_budget():
    candidates = make_candidates()
    policy = AcquisitionPolicy(batch_size=10, budget=4.0)
    for chosen in (
        select_random(candidates, policy, np.random.default_rng(1)),
        select_component_diversity(candidates, policy, np.random.default_rng(1)),
    ):
        assert sum(c.cost for c in chosen) <= 4.0


def test_rejects_empty_inputs_and_bad_policies():
    candidates = make_candidates()
    surrogate = fitted_surrogate(candidates)
    policy = AcquisitionPolicy(batch_size=2, budget=9.0)
    with pytest.raises(SynthesisAwareAcquisitionError, match="no candidates"):
        select_batch(surrogate, [], [candidates[0].as_record()], policy)
    with pytest.raises(SynthesisAwareAcquisitionError, match="target population is empty"):
        select_batch(surrogate, candidates, [], policy)
    with pytest.raises(SynthesisAwareAcquisitionError, match="batch_size"):
        AcquisitionPolicy(batch_size=0, budget=1.0).validate()
    with pytest.raises(SynthesisAwareAcquisitionError, match="budget"):
        AcquisitionPolicy(batch_size=1, budget=0.0).validate()
    with pytest.raises(SynthesisAwareAcquisitionError, match="value_weight"):
        AcquisitionPolicy(batch_size=1, budget=1.0, value_weight=1.5).validate()


def test_budget_smaller_than_every_candidate_is_an_error_not_an_empty_batch():
    candidates = make_candidates()
    surrogate = fitted_surrogate(candidates)
    policy = AcquisitionPolicy(batch_size=3, budget=0.5, activity_floor_quantile=0.0)
    with pytest.raises(SynthesisAwareAcquisitionError, match="budget admitted no candidate"):
        select_batch(surrogate, candidates, [c.as_record() for c in candidates], policy)


def test_activity_floor_is_separable_from_the_selection_rule():
    """The floor must be appliable once, outside any selector, so all arms share a pool.

    Applying it inside a single selector confines that arm to the top of the predicted
    distribution while its baselines sample freely. On a learning metric that is a handicap,
    not an advantage, and it turns the comparison into one about the floor.
    """
    from forge.potency.ranking.synthesis_aware_acquisition import apply_activity_floor

    candidates = make_candidates(20)
    surrogate = fitted_surrogate(candidates)
    kept, floor = apply_activity_floor(surrogate, candidates, 0.5)
    assert 0 < len(kept) < len(candidates)
    bounds = surrogate.lower_bound([c.as_record() for c in kept])
    assert np.all(bounds >= floor)

    # a zero-quantile policy must not filter again, so a pre-floored pool passes through intact
    policy = AcquisitionPolicy(
        batch_size=3, budget=99.0, activity_floor_quantile=0.0, cost_normalised=False
    )
    result = select_batch(surrogate, kept, [c.as_record() for c in kept], policy)
    assert result.eligible_after_floor == len(kept)
    assert all(c in kept for c in result.selected)


def test_floor_rejects_an_empty_pool_and_a_bad_quantile():
    from forge.potency.ranking.synthesis_aware_acquisition import apply_activity_floor

    candidates = make_candidates()
    surrogate = fitted_surrogate(candidates)
    with pytest.raises(SynthesisAwareAcquisitionError, match="no candidates"):
        apply_activity_floor(surrogate, [], 0.5)
    with pytest.raises(SynthesisAwareAcquisitionError, match="quantile"):
        apply_activity_floor(surrogate, candidates, 1.0)
