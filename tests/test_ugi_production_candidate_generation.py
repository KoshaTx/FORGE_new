from __future__ import annotations

import pytest

from experiments.phase1.product_l1.sampling.ugi_production_candidate_generation import (
    TERMINAL_LEDGER_REQUIRED_FIELDS,
    ProductionCandidateDesign,
    UgiProductionCandidateGenerationError,
    production_draw_seed,
)
from experiments.phase1.synthesis_guidance.schedule.ugi_production_candidate_schedule import ARM_IDS


def test_production_design_has_matched_two_arm_budget() -> None:
    design = ProductionCandidateDesign(
        draws_per_arm=16384,
        shard_draws=128,
        particle_seed_base=11,
        terminal_seed_base=12,
        device="cpu",
        checkpoint=6,
    )
    assert design.shard_count == 128
    assert design.terminal_attempts == 32768
    assert design.terminal_attempts == design.draws_per_arm * len(ARM_IDS)


def test_production_seed_is_reproducible_and_purpose_separated() -> None:
    seed = production_draw_seed(11, purpose="particle", draw_index=7)
    assert seed == production_draw_seed(11, purpose="particle", draw_index=7)
    assert seed != production_draw_seed(11, purpose="terminal", draw_index=7)
    assert seed != production_draw_seed(11, purpose="particle", draw_index=8)


def test_invalid_production_sharding_fails_closed() -> None:
    design = ProductionCandidateDesign(
        draws_per_arm=16384,
        shard_draws=127,
        particle_seed_base=11,
        terminal_seed_base=12,
        device="cpu",
        checkpoint=6,
    )
    with pytest.raises(UgiProductionCandidateGenerationError):
        _ = design.shard_count


def test_terminal_ledger_retains_probability_and_provenance_fields() -> None:
    assert {
        "arm_id",
        "draw_index",
        "program_sha256",
        "program",
        "native_terminal",
        "broad_prior_probability",
        "support_proposal_probability",
        "particle_seed",
        "terminal_seed",
    }.issubset(TERMINAL_LEDGER_REQUIRED_FIELDS)
