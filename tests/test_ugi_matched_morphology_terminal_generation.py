from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.phase1.product_l1.sampling.ugi_matched_morphology_terminal_generation import (
    TERMINAL_LEDGER_REQUIRED_FIELDS,
    MatchedTerminalDesign,
    UgiMatchedMorphologyTerminalGenerationError,
    _validate_schedule_records,
    load_matched_terminal_contract,
    matched_draw_seed,
)
from experiments.phase1.synthesis_guidance.schedule.ugi_matched_morphology_allocation_schedule import (
    ARM_IDS,
)

REPO = Path(__file__).resolve().parents[1]


def test_matched_terminal_design_has_equal_budgets_and_48_shards() -> None:
    design = MatchedTerminalDesign(
        draws_per_arm=3072,
        shard_draws=64,
        particle_seed_base=11,
        terminal_seed_base=12,
        device="cpu",
        checkpoint=6,
    )
    assert design.shard_count == 48
    assert design.terminal_attempts == 3072 * len(ARM_IDS)


def test_draw_seed_is_common_across_arms_but_purpose_separated() -> None:
    common = matched_draw_seed(11, purpose="particle", draw_index=7)
    assert common == matched_draw_seed(11, purpose="particle", draw_index=7)
    assert common != matched_draw_seed(11, purpose="particle", draw_index=8)
    assert common != matched_draw_seed(11, purpose="terminal", draw_index=7)


def test_invalid_shard_partition_fails_closed() -> None:
    design = MatchedTerminalDesign(
        draws_per_arm=3072,
        shard_draws=100,
        particle_seed_base=11,
        terminal_seed_base=12,
        device="cpu",
        checkpoint=6,
    )
    with pytest.raises(UgiMatchedMorphologyTerminalGenerationError):
        _ = design.shard_count


def test_terminal_ledger_contract_retains_scoring_and_provenance_fields() -> None:
    assert {
        "arm_id",
        "draw_index",
        "program_sha256",
        "program",
        "native_terminal",
        "broad_prior_probability",
        "support_proposal_probability",
        "potency_proposal_probability",
        "particle_seed",
        "terminal_seed",
    }.issubset(TERMINAL_LEDGER_REQUIRED_FIELDS)


def test_frozen_matched_terminal_contract_has_full_three_arm_budget() -> None:
    contract = load_matched_terminal_contract(
        REPO,
        REPO / "configs/model/phase1_ugi_matched_morphology_terminal_generation_v1.json",
    )
    assert len(contract.schedule) == 3072
    assert contract.design.terminal_attempts == 9216
    assert [int(row["draw_index"]) for row in contract.schedule] == list(range(3072))


def test_schedule_validation_rejects_a_tampered_program_hash() -> None:
    schedule = json.loads(
        (
            REPO / "results/phase1/ugi_matched_morphology_allocation_schedule_v1/schedule.json"
        ).read_text()
    )
    records = copy.deepcopy(schedule["records"][:1])
    records[0]["arms"]["broad_prior"]["program_sha256"] = "0" * 64
    with pytest.raises(UgiMatchedMorphologyTerminalGenerationError):
        _validate_schedule_records(records, draws=1)
