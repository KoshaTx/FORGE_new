from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.archive.producers.phase1_qualify_ugi_grouped_zero_guidance_identity import (
    _atomic_write_once,
)
from experiments.phase1.synthesis_guidance.guidance.ugi_grouped_zero_guidance_identity import (
    UgiGroupedZeroGuidanceIdentityError,
    _assert_completion_identity,
    _expanded_programs,
    _productive_final_invocation_seed,
    _validate_blocked_runner_plan,
    _validate_restartable_v2,
)
from experiments.phase1.synthesis_guidance.schedule.ugi_nonzero_guidance_runner import (
    GuidanceTerminalCompletionReceipt,
    load_grouped_smc_schedule_qualification,
)

REPO = Path(__file__).resolve().parents[1]


def _json(path: str) -> dict:
    return json.loads((REPO / path).read_text())


def test_current_restartable_v2_preserves_all_old_batch_partitions() -> None:
    value = _json("results/phase1/ugi_restartable_sampler_equivalence_v2/result.json")
    _validate_restartable_v2(value)

    malformed = copy.deepcopy(value)
    malformed["comparisons"] = malformed["comparisons"][:2]
    with pytest.raises(
        UgiGroupedZeroGuidanceIdentityError,
        match="old batch partitions",
    ):
        _validate_restartable_v2(malformed)


def test_current_runner_plan_remains_fail_closed_for_nonzero_execution() -> None:
    value = _json("results/phase1/ugi_nonzero_guidance_runner_v1/plan.json")
    _validate_blocked_runner_plan(value)

    authorized = copy.deepcopy(value)
    authorized["authorization"]["nonzero_guidance_authorized"] = True
    with pytest.raises(
        UgiGroupedZeroGuidanceIdentityError,
        match="fail-closed",
    ):
        _validate_blocked_runner_plan(authorized)


def test_frozen_first_assignment_expands_to_sixteen_groups_of_four() -> None:
    schedule = load_grouped_smc_schedule_qualification(
        REPO / "results/phase1/ugi_grouped_smc_schedule_qualification_v1/result.json"
    )
    assignment = schedule.by_seed()[20260821]
    expanded = _expanded_programs(assignment.programs)

    assert len(assignment.programs) == 16
    assert len(expanded) == 64
    assert all(
        expanded[program_index * 4 : (program_index + 1) * 4]
        == (assignment.programs[program_index],) * 4
        for program_index in range(16)
    )
    assert _productive_final_invocation_seed(assignment.stochastic_particle_seeds[0]) == (
        assignment.stochastic_particle_seeds[0] + 1
    )
    assert _productive_final_invocation_seed(assignment.stochastic_particle_seeds[-1]) == (
        assignment.stochastic_particle_seeds[-1] + 1
    )


def test_completion_identity_handles_equal_failures_and_rejects_drift() -> None:
    left = GuidanceTerminalCompletionReceipt(
        terminal=None,
        product_transition_calls=6,
        error_detail="synthetic completion failure",
    )
    right = GuidanceTerminalCompletionReceipt(
        terminal=None,
        product_transition_calls=6,
        error_detail="synthetic completion failure",
    )
    record = _assert_completion_identity(
        left,
        right,
        expected_transition_calls=6,
        label="test",
    )
    assert record["terminal_present"] is False

    drifted = GuidanceTerminalCompletionReceipt(
        terminal=None,
        product_transition_calls=6,
        error_detail="different failure",
    )
    with pytest.raises(
        UgiGroupedZeroGuidanceIdentityError,
        match="completion errors differ",
    ):
        _assert_completion_identity(
            left,
            drifted,
            expected_transition_calls=6,
            label="test",
        )


def test_atomic_writer_is_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    _atomic_write_once(output, b"first\n")
    assert output.read_bytes() == b"first\n"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _atomic_write_once(output, b"second\n")
    assert output.read_bytes() == b"first\n"
