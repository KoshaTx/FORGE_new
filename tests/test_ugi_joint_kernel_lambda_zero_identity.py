from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.phase1.synthesis_guidance.guidance.ugi_joint_kernel_lambda_zero_identity import (
    LEDGER_FIELDS,
    RecordingGuidanceLane,
    load_execution_contract,
    run_seed_qualification,
)
from experiments.phase1.synthesis_guidance.schedule.ugi_nonzero_guidance_runner import (
    GuidanceStateReceipt,
    GuidanceTerminalCompletionReceipt,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_joint_kernel_lambda_zero_identity_v1.json"


def _sha256_payload(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class _FakeState:
    step: int
    particle_count: int


class _FakeLane:
    def initialize(
        self,
        programs: tuple[bytes, ...],
        *,
        seed: int,
        particle_seeds: tuple[int, ...],
        device: str,
    ) -> GuidanceStateReceipt:
        assert seed >= 0
        assert device == "cpu"
        assert len(programs) == len(particle_seeds) == 64
        return GuidanceStateReceipt(
            state=_FakeState(step=0, particle_count=len(programs)),
            product_transition_calls=0,
            consumed_particle_seed_manifest_sha256=_sha256_payload(particle_seeds),
        )

    def advance(self, state: _FakeState, *, target_step: int) -> GuidanceStateReceipt:
        return GuidanceStateReceipt(
            state=_FakeState(step=target_step, particle_count=state.particle_count),
            product_transition_calls=state.particle_count * (target_step - state.step),
        )

    def snapshot(self, state: _FakeState) -> GuidanceStateReceipt:
        return GuidanceStateReceipt(state=state, product_transition_calls=0)

    def complete_terminal(
        self,
        state: _FakeState,
        *,
        particle_index: int,
        seed: int,
        checkpoint_index: int,
    ) -> GuidanceTerminalCompletionReceipt:
        assert 0 <= particle_index < state.particle_count
        assert seed >= 0
        assert checkpoint_index == state.step
        return GuidanceTerminalCompletionReceipt(
            terminal=None,
            product_transition_calls=8 - state.step,
            error_detail="synthetic_terminal_failure",
        )

    def apply_ancestry(
        self,
        state: _FakeState,
        ancestors: tuple[int, ...],
    ) -> GuidanceStateReceipt:
        assert ancestors == tuple(range(state.particle_count))
        return GuidanceStateReceipt(state=state, product_transition_calls=0)


def test_contract_authenticates_schedule_and_structural_gate() -> None:
    contract = load_execution_contract(REPO, CONFIG)
    assert tuple(contract.schedule.by_seed()) == tuple(range(20260821, 20260829))
    assert len(contract.schedule_metadata) == 128
    assert contract.applicability.thresholds_sha256 == (
        "f43efb355cca547eb067f3f4fb93199151e18f0716f55304805f6a8752da67d7"
    )
    assert contract.applicability.supported_patterns == {
        ("amine",): "amine_only",
        ("aldehyde", "isocyanide"): "aldehyde_isocyanide_pair",
    }


def test_exact_measured_candidate_is_structurally_neutral() -> None:
    contract = load_execution_contract(REPO, CONFIG)
    with gzip.open(REPO / "results/m0_07/agile_oracle_curated.csv.gz", "rt", newline="") as handle:
        row = next(csv.DictReader(handle))
    result = contract.applicability.classify_candidate(
        {
            "label": row["label"],
            "product_smiles": row["model_smiles"],
            "amine_smiles": row["A_smiles"],
            "aldehyde_smiles": row["B_smiles"],
            "isocyanide_smiles": row["C_smiles"],
        }
    )
    assert result["exact_measured_combination"] is True
    assert result["action"] == "neutral"
    assert result["reason"] == "exact_measured_combination"


def test_fake_lane_runs_existing_identity_gate_and_captures_one_arm() -> None:
    result, ledger = run_seed_qualification(
        REPO,
        CONFIG,
        20260821,
        lane=_FakeLane(),
    )
    assert result["lambda_zero_identity"]["bitwise_identity_passed"] is True
    assert result["lambda_zero_identity"]["historical_hard_reference_passed"] is None
    assert result["lambda_zero_identity"]["product_transition_calls_per_arm"] == 1280
    assert result["lambda_zero_identity"]["terminal_completions_per_arm"] == 256
    census = result["terminal_applicability_census"]
    assert census["scheduled_terminal_attempts"] == 256
    assert census["terminal_present"] == 0
    assert census["actions"] == {"abstain": 256}
    assert census["reasons"] == {"terminal_completion_failed": 256}
    with gzip.GzipFile(fileobj=__import__("io").BytesIO(ledger), mode="rb") as handle:
        rows = list(csv.DictReader(__import__("io").StringIO(handle.read().decode())))
    assert len(rows) == 256
    assert set(rows[0]) == set(LEDGER_FIELDS)
    assert {int(row["checkpoint"]) for row in rows} == {2, 4, 6, 8}


def test_recording_lane_rejects_a_third_arm() -> None:
    lane = RecordingGuidanceLane(_FakeLane())
    programs = tuple(f"program-{index}".encode() for index in range(64))
    seeds = tuple(range(64))
    lane.initialize(programs, seed=7, particle_seeds=seeds, device="cpu")
    lane.initialize(programs, seed=7, particle_seeds=seeds, device="cpu")
    try:
        lane.initialize(programs, seed=7, particle_seeds=seeds, device="cpu")
    except RuntimeError as error:
        assert "more than two arms" in str(error)
    else:  # pragma: no cover
        raise AssertionError("third arm initialization unexpectedly succeeded")
