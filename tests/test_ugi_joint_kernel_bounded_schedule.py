from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from forge.design.audit.ugi_joint_program_support_audit import build_joint_program_kernel
from forge.design.flow.ugi_morphology_program import UgiMorphologyProgram
from forge.design.flow.ugi_restartable_terminal_support_adapter import (
    canonical_morphology_program_bytes,
)
from forge.design.schedule.ugi_joint_kernel_bounded_schedule import (
    _program_feasible,
    build_complete_program_pool,
    select_bounded_schedule,
)
from forge.design.schedule.ugi_nonzero_guidance_runner import (
    load_grouped_smc_schedule_qualification,
)


def _program(nodes: tuple[int, int, int]) -> UgiMorphologyProgram:
    return UgiMorphologyProgram(
        node_counts=nodes,
        junction_budgets=(1, 0, 0),
        cycle_ranks=(0, 0, 0),
        attachment_counts=(1, 1, 1),
    )


def _bounds() -> dict[str, int]:
    return {
        "maximum_children": 3,
        "maximum_component_atoms": 32,
        "maximum_total_atoms": 80,
        "maximum_junction_budget": 7,
        "maximum_cycle_rank": 2,
        "maximum_attachment_count": 2,
    }


def test_pool_accepts_only_complete_exact_or_local_frozen_tuples() -> None:
    anchors = tuple(_program((8, 10 + 4 * index, 12)) for index in range(4))
    measured = tuple(program for program in anchors for _ in range(4))
    kernel = build_joint_program_kernel(measured)
    local = _program((8, 15, 12))
    outside = _program((20, 30, 24))

    pool = build_complete_program_pool(
        kernel,
        anchors=anchors,
        frozen_draw=(local, outside),
        bounds=_bounds(),
    )
    payloads = {candidate.payload for candidate in pool}

    assert canonical_morphology_program_bytes(local) in payloads
    assert canonical_morphology_program_bytes(outside) not in payloads
    assert all(
        candidate.neighborhood in {"exact_anchor", "local_smoothed_neighborhood"}
        for candidate in pool
    )
    assert all(_program_feasible(candidate.program, _bounds()) for candidate in pool)


def test_weighted_schedule_is_deterministic_disjoint_and_unstratified() -> None:
    anchors = tuple(_program((4 + index % 16, 10 + index // 16, 10)) for index in range(160))
    kernel = build_joint_program_kernel(anchors)
    pool = build_complete_program_pool(
        kernel,
        anchors=anchors,
        frozen_draw=(),
        bounds=_bounds(),
    )
    first = select_bounded_schedule(pool)
    second = select_bounded_schedule(pool)

    assert first == second
    hashes = [
        record["morphology_program_sha256"] for schedule in first for record in schedule["programs"]
    ]
    assert len(first) == 8
    assert len(hashes) == 128
    assert len(set(hashes)) == 128
    assert all(len(schedule["particles"]) == 64 for schedule in first)


def test_frozen_result_loads_through_existing_grouped_schedule_loader() -> None:
    path = Path("results/phase1/ugi_joint_kernel_bounded_schedule_v1/result.json")
    if not path.is_file():
        pytest.skip("bounded schedule artifact has not been materialized")
    result = json.loads(path.read_text())
    qualification = load_grouped_smc_schedule_qualification(path)

    assert len(qualification.assignments) == 8
    assert (
        result["bounded_joint_program_contract"]["schedule_builder_mutation_or_role_composition"]
        is False
    )
    assert result["bounded_joint_program_contract"]["program_overlap_across_seeds"] == 0
    assert (
        result["bounded_joint_program_contract"]["biological_labels_or_oracle_outputs_read"]
        is False
    )
    assert result["scope"]["nonzero_guidance"] is False
    assert result["scope"]["oracle_calls"] == 0


def test_selected_schedule_materializes_for_the_real_loader(tmp_path: Path) -> None:
    anchors = tuple(_program((4 + index % 16, 10 + index // 16, 10)) for index in range(160))
    kernel = build_joint_program_kernel(anchors)
    pool = build_complete_program_pool(
        kernel,
        anchors=anchors,
        frozen_draw=(),
        bounds=_bounds(),
    )
    schedules = list(select_bounded_schedule(pool))
    receipt = {
        "schema_version": "phase1_ugi_grouped_smc_schedule_qualification.v1",
        "status": "grouped_smc_schedule_qualified_nonexecuting",
        "design": {
            "programs_per_seed": 16,
            "particles_per_program": 4,
            "particles_per_seed": 64,
            "ancestry_group_key": ["seed", "program_index"],
            "within_program_ancestry_has_four_choices": True,
            "cross_program_ancestry_forbidden": True,
        },
        "integration_contract": {
            "sampler_receives_each_program_four_times": True,
            "ancestry_selection_invoked_separately_per_program_group": True,
            "particle_index_must_not_be_reinterpreted_as_program_index": True,
            "sixty_four_unique_programs_forbidden": True,
        },
        "seed_schedules": schedules,
    }
    receipt["result_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    qualification = load_grouped_smc_schedule_qualification(path)

    assert len(qualification.assignments) == 8
    assert all(len(assignment.programs) == 16 for assignment in qualification.assignments)
    assert (
        len(
            {program for assignment in qualification.assignments for program in assignment.programs}
        )
        == 128
    )
    assert np.isfinite(float(anchors[0].node_count))
