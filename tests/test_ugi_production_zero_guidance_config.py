from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from forge.design.schedule.ugi_matched_budget_orchestration import _schedule_sha256
from forge.design.guidance.ugi_production_zero_guidance_config import (
    UgiProductionZeroGuidanceConfigError,
    load_production_zero_guidance_rehearsal_plan,
)
from forge.design.flow.ugi_restartable_terminal_support_adapter import (
    canonical_morphology_program_bytes,
)
from forge.route.engine.planner import PlannerBudgetLimits
from forge.route.engine.planner_cache import PlannerCacheContext
from forge.route.terminals.terminal_assessment import (
    required_three_role_route_reservation,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_production_zero_guidance_rehearsal_v1.json"
SOURCE_INPUTS_SHA256 = "adfc6767d44d9aa56cfb46bba2253e485ec35f5dace0af0f5a11f9cd72bcd3c6"
GENERATOR_SHA256 = "90f5f0bd3e41e2884f6588d9875db0b7bf34e5c5ea7fdc1f9d8565fba8c0c532"
CLOSURE_SHA256 = "a97507ac6a9eeba41d0cc351666cffdd21069d13bc78c0db671ab3a30c710b5d"


def _context(*, maximum_verifier_calls: int = 2) -> PlannerCacheContext:
    return PlannerCacheContext(
        planner_id="test-production-planner",
        planner_sha256="1" * 64,
        search_policy_sha256="2" * 64,
        value_policy_sha256="3" * 64,
        l1_reaction_sha256="4" * 64,
        upstream_reaction_registry_sha256="5" * 64,
        variant_registry_sha256="6" * 64,
        verifier_sha256="7" * 64,
        l3_snapshot_sha256=SOURCE_INPUTS_SHA256,
        l3_region="US",
        l3_accessed_at_utc="2026-08-03T03:04:10Z",
        l3_expires_at_utc="2026-08-09T05:16:00Z",
        software_versions=(("rdkit", "test"),),
        identity_policy="constitutional",
        stereochemistry_policy="excluded",
        budget_limits=PlannerBudgetLimits(
            maximum_depth=4,
            maximum_logical_planner_calls=1,
            maximum_expansions=3,
            maximum_product_candidates=4,
            maximum_verifier_calls=maximum_verifier_calls,
            maximum_elapsed_milliseconds=1000,
        ),
    )


def _load(*, context: PlannerCacheContext | None = None):
    return load_production_zero_guidance_rehearsal_plan(
        REPO,
        CONFIG,
        planner_context=context or _context(),
    )


def _write_changed_config(tmp_path: Path, mutate) -> Path:
    value = json.loads(CONFIG.read_text())
    mutate(value)
    path = tmp_path / "changed-config.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def test_real_frozen_config_builds_only_the_typed_deterministic_schedule() -> None:
    first = _load()
    second = _load()

    assert first.to_dict() == second.to_dict()
    assert first.plan_sha256 == second.plan_sha256
    assert len(first.programs) == len(first.schedule) == 12
    assert first.seed == 20260802
    assert first.sample_steps == first.terminal_checkpoint_index == 8
    assert first.identity.generator_checkpoint_sha256 == GENERATOR_SHA256
    assert first.identity.closure_checkpoint_sha256 == CLOSURE_SHA256
    assert first.identity.production_generator_manifest_sha256 == (
        "c27352c11a2e210fd76a6e4ae510bbbb89e51c92c28514fd9b4693f82a50e68a"
    )
    assert first.identity.restartable_equivalence_receipt_sha256 == (
        "5a2b9139141abc7eb6fe6bf6b1851b6a52dcfd2f45756a1ee1958bab3090d778"
    )
    assert first.identity.terminal_decoder_id == "ugi_joint_terminal_completion:argmax:v1"
    assert first.schedule_sha256 == (
        "5a17e4f830a7be33aa75f6e89930d35075ae33d9862472ed049c02f27948a04c"
    )
    assert first.schedule_sha256 == _schedule_sha256(first.schedule)
    assert first.config_sha256 == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert [artifact.name for artifact in first.artifacts] == sorted(
        {
            "closure_checkpoint",
            "generator_checkpoint",
            "production_generator_manifest",
            "program_draw",
            "restartable_equivalence_config",
            "restartable_equivalence_result",
        }
    )
    for index, (program, entry) in enumerate(zip(first.programs, first.schedule, strict=True)):
        assert entry.unit_id == f"production-zero-guidance-{index:04d}"
        assert entry.program_index == entry.particle_index == index
        assert entry.checkpoint_index == 8
        assert entry.rollout_index == 0
        assert entry.productive_generation_calls == 1
        assert entry.morphology_program == canonical_morphology_program_bytes(program)
        assert entry.generator_checkpoint_sha256 == GENERATOR_SHA256
        assert entry.closure_checkpoint_sha256 == CLOSURE_SHA256

    output = first.to_dict()
    assert output["execution"] == {
        "guidance_strength": 0,
        "program_count": 12,
        "sample_steps": 8,
        "seed": 20260802,
        "terminal_checkpoint_index": 8,
        "production_generator_executed": False,
        "production_route_source_executed": False,
    }
    assert output["scope_guards"] == {
        "production_generator_executed": False,
        "production_route_source_executed": False,
        "nonzero_synthesis_guidance": False,
        "synthesis_scalar_defined": False,
        "biological_guidance": False,
        "candidate_selection": False,
        "private_holdout_accessed": False,
    }


def test_terminal_decoder_identity_matches_selected_generator_lane_exactly() -> None:
    from forge.design.sampling.ugi_selected_restartable_generator import TERMINAL_DECODER_ID

    plan = _load()
    assert TERMINAL_DECODER_ID == "ugi_joint_terminal_completion:argmax:v1"
    assert plan.identity.terminal_decoder_id == TERMINAL_DECODER_ID


def test_route_reservation_is_derived_from_typed_context_and_scaled_exactly() -> None:
    low = _load(context=_context(maximum_verifier_calls=2))
    high_context = _context(maximum_verifier_calls=5)
    high = _load(context=high_context)

    assert low.per_unit_route_reservation == required_three_role_route_reservation(
        _context(maximum_verifier_calls=2).budget_limits
    )
    assert high.per_unit_route_reservation == required_three_role_route_reservation(
        high_context.budget_limits
    )
    assert low.per_unit_route_reservation.logical_planner_calls == 3
    assert low.per_unit_route_reservation.logical_verifier_calls == 7
    assert high.per_unit_route_reservation.logical_verifier_calls == 16
    assert high.budget_limits.route.logical_planner_calls == 36
    assert high.budget_limits.route.physical_planner_calls == 36
    assert high.budget_limits.route.logical_verifier_calls == 192
    assert high.budget_limits.route.physical_verifier_calls == 192
    assert all(
        entry.route_reservation == high.per_unit_route_reservation for entry in high.schedule
    )


@pytest.mark.parametrize(
    "artifact_name",
    (
        "closure_checkpoint",
        "generator_checkpoint",
        "production_generator_manifest",
        "program_draw",
        "restartable_equivalence_config",
        "restartable_equivalence_result",
    ),
)
def test_every_on_disk_artifact_pin_fails_closed(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    config_path = _write_changed_config(
        tmp_path,
        lambda value: value["artifacts"][artifact_name].update({"sha256": "0" * 64}),
    )
    with pytest.raises(
        UgiProductionZeroGuidanceConfigError,
        match=f"artifact {artifact_name} SHA-256 mismatch",
    ):
        load_production_zero_guidance_rehearsal_plan(
            REPO,
            config_path,
            planner_context=_context(),
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("guidance_strength", 0.1),
        ("program_count", 11),
        ("sample_steps", 7),
        ("seed", 1),
        ("terminal_checkpoint_index", 7),
        ("generator_implementation_sha256", "0" * 64),
        ("terminal_decoder_id", "valence_constrained_argmax.v1"),
    ),
)
def test_execution_identity_drift_fails_closed(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    config_path = _write_changed_config(
        tmp_path,
        lambda value: value["execution"].update({field: replacement}),
    )
    with pytest.raises(UgiProductionZeroGuidanceConfigError):
        load_production_zero_guidance_rehearsal_plan(
            REPO,
            config_path,
            planner_context=_context(),
        )


def test_any_expansive_scope_guard_fails_closed(tmp_path: Path) -> None:
    config_path = _write_changed_config(
        tmp_path,
        lambda value: value["scope_guards"].update({"biological_guidance": True}),
    )
    with pytest.raises(UgiProductionZeroGuidanceConfigError, match="scope guards changed"):
        load_production_zero_guidance_rehearsal_plan(
            REPO,
            config_path,
            planner_context=_context(),
        )


def test_planner_context_must_match_exact_l3_source_and_window() -> None:
    with pytest.raises(UgiProductionZeroGuidanceConfigError, match="L3 identity"):
        _load(context=replace(_context(), l3_snapshot_sha256="f" * 64))
    with pytest.raises(UgiProductionZeroGuidanceConfigError, match="L3 identity"):
        _load(
            context=replace(
                _context(),
                l3_accessed_at_utc="2026-08-03T03:04:11Z",
            )
        )


def test_assessment_at_expiry_is_rejected_by_accessed_inclusive_expires_exclusive_rule(
    tmp_path: Path,
) -> None:
    config_path = _write_changed_config(
        tmp_path,
        lambda value: value["assessment"].update({"assessment_as_of_utc": "2026-08-09T05:16:00Z"}),
    )
    with pytest.raises(UgiProductionZeroGuidanceConfigError, match="not current"):
        load_production_zero_guidance_rehearsal_plan(
            REPO,
            config_path,
            planner_context=_context(),
        )


def test_per_component_planner_context_must_own_one_logical_call() -> None:
    limits = replace(_context().budget_limits, maximum_logical_planner_calls=2)
    context = replace(_context(), budget_limits=limits)
    with pytest.raises(UgiProductionZeroGuidanceConfigError, match="exactly one logical"):
        _load(context=context)


def test_duplicate_json_keys_fail_closed_before_any_artifact_is_used(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":"x","schema_version":"y"}\n')
    with pytest.raises(UgiProductionZeroGuidanceConfigError, match="duplicate JSON key"):
        load_production_zero_guidance_rehearsal_plan(
            REPO,
            path,
            planner_context=_context(),
        )
