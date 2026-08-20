from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from forge.design.schedule.ugi_matched_budget_orchestration import (
    DiagnosticRouteAssessment,
    LockedMatchedTerminal,
    MatchedArm,
    MatchedBudgetLimits,
    MatchedDisposition,
    MatchedGenerationRequest,
    MatchedScheduleEntry,
    RouteComputeUsage,
    UgiMatchedBudgetError,
    run_diagnostic_matched_budget_arms,
)

GENERATOR_CHECKPOINT_SHA256 = "a" * 64
CLOSURE_CHECKPOINT_SHA256 = "b" * 64
CACHE_SNAPSHOT_SHA256 = "c" * 64


def _schedule(count: int = 3) -> tuple[MatchedScheduleEntry, ...]:
    return tuple(
        MatchedScheduleEntry(
            unit_id=f"unit-{index}",
            morphology_program=f"morphology-program-{index}".encode(),
            program_index=index,
            particle_index=index + 10,
            checkpoint_index=2,
            generator_checkpoint_sha256=GENERATOR_CHECKPOINT_SHA256,
            closure_checkpoint_sha256=CLOSURE_CHECKPOINT_SHA256,
            rollout_index=0,
            productive_generation_calls=1,
            route_reservation=RouteComputeUsage(1, 1, 1, 1),
        )
        for index in range(count)
    )


def _terminal(
    request: MatchedGenerationRequest,
    *,
    locked: bool = True,
    arm_dependent_bytes: bool = False,
    morphology_program_sha256: str | None = None,
    generator_checkpoint_sha256: str | None = None,
    closure_checkpoint_sha256: str | None = None,
) -> LockedMatchedTerminal:
    arm_suffix = f":{request.arm.value}" if arm_dependent_bytes else ""
    terminal_bytes = (
        f"terminal:{request.entry.unit_id}:{request.productive_seed}{arm_suffix}".encode()
    )
    return LockedMatchedTerminal(
        unit_id=request.entry.unit_id,
        morphology_program_sha256=(
            morphology_program_sha256 or request.entry.morphology_program_sha256
        ),
        checkpoint_index=request.entry.checkpoint_index,
        generator_checkpoint_sha256=(
            generator_checkpoint_sha256 or request.entry.generator_checkpoint_sha256
        ),
        closure_checkpoint_sha256=(
            closure_checkpoint_sha256 or request.entry.closure_checkpoint_sha256
        ),
        terminal_id=f"{request.arm.value}:{request.entry.unit_id}",
        terminal_locked=locked,
        terminal_valid=True,
        exact_l1=True,
        terminal_bytes=terminal_bytes,
        generation_trace_bytes=(
            f"trace:{request.entry.unit_id}:{request.productive_seed}".encode()
        ),
    )


def test_matched_contract_uses_shared_productive_rng_and_common_censoring() -> None:
    events: list[str] = []
    generation_seeds: dict[tuple[str, str], int] = {}
    route_seeds: dict[tuple[str, str], int] = {}
    post_hoc_manifests: list[str] = []

    def generate(request: MatchedGenerationRequest) -> LockedMatchedTerminal:
        events.append(f"generate:{request.arm.value}:{request.entry.unit_id}")
        generation_seeds[(request.arm.value, request.entry.unit_id)] = request.productive_seed
        return _terminal(request)

    def assess(terminal, context):
        events.append(f"assess:{context.arm.value}:{terminal.unit_id}")
        route_seeds[(context.arm.value, terminal.unit_id)] = context.route_seed
        if context.arm is MatchedArm.POST_HOC:
            assert context.post_hoc_lock_manifest_sha256
            post_hoc_manifests.append(context.post_hoc_lock_manifest_sha256)
            usage = RouteComputeUsage(1, 1, 1, 1)
        else:
            assert context.post_hoc_lock_manifest_sha256 is None
            usage = RouteComputeUsage(1, 1, 1, 1)
        return DiagnosticRouteAssessment(
            value_policy_id="fake-structured-policy-v1",
            diagnostic_fields=(
                ("closure_state", "fake-complete"),
                ("uncertainty_state", "fake-low"),
            ),
            usage=usage,
        )

    limits = MatchedBudgetLimits(
        productive_generation_calls=2,
        terminal_completions=2,
        final_candidates=2,
        route=RouteComputeUsage(2, 2, 2, 2),
    )
    result = run_diagnostic_matched_budget_arms(
        _schedule(),
        budget_limits=limits,
        base_seed=20260803,
        zero_guidance=True,
        cache_snapshot_sha256=CACHE_SNAPSHOT_SHA256,
        generate_terminal=generate,
        assess_terminal=assess,
    )

    assert result.zero_guidance_bitwise_equivalent is True
    assert result.cache_snapshot_sha256 == CACHE_SNAPSHOT_SHA256
    assert result.guided_cache_clone_id != result.post_hoc_cache_clone_id
    assert result.shared_censored_unit_ids == ("unit-2",)
    assert [record.disposition for record in result.guided_records] == [
        MatchedDisposition.ASSESSED,
        MatchedDisposition.ASSESSED,
        MatchedDisposition.SHARED_BUDGET_CENSORED,
    ]
    assert [record.disposition for record in result.post_hoc_records] == [
        MatchedDisposition.ASSESSED,
        MatchedDisposition.ASSESSED,
        MatchedDisposition.SHARED_BUDGET_CENSORED,
    ]
    assert result.guided_ledger.productive_generation_calls == 2
    assert result.post_hoc_ledger.productive_generation_calls == 2
    assert result.guided_ledger.terminal_completions == 2
    assert result.post_hoc_ledger.terminal_completions == 2
    assert result.guided_ledger.final_candidates == 2
    assert result.post_hoc_ledger.final_candidates == 2
    assert result.guided_ledger.route == RouteComputeUsage(2, 2, 2, 2)
    assert result.post_hoc_ledger.route == RouteComputeUsage(2, 2, 2, 2)
    assert result.guided_ledger.planner_cache_hits == 0
    assert result.post_hoc_ledger.planner_cache_hits == 0
    assert result.post_hoc_ledger.verifier_cache_hits == 0

    for unit_id in ("unit-0", "unit-1"):
        assert (
            generation_seeds[(MatchedArm.GUIDED.value, unit_id)]
            == generation_seeds[(MatchedArm.POST_HOC.value, unit_id)]
        )
        assert (
            route_seeds[(MatchedArm.GUIDED.value, unit_id)]
            != route_seeds[(MatchedArm.POST_HOC.value, unit_id)]
        )
    assert (MatchedArm.GUIDED.value, "unit-2") not in generation_seeds
    assert (MatchedArm.POST_HOC.value, "unit-2") not in generation_seeds
    assert len(set(post_hoc_manifests)) == 1
    first_post_hoc_assessment = events.index("assess:post_hoc:unit-0")
    assert events.index("generate:post_hoc:unit-1") < first_post_hoc_assessment

    repeated = run_diagnostic_matched_budget_arms(
        _schedule(),
        budget_limits=limits,
        base_seed=20260803,
        zero_guidance=True,
        cache_snapshot_sha256=CACHE_SNAPSHOT_SHA256,
        generate_terminal=lambda request: _terminal(request),
        assess_terminal=lambda terminal, context: DiagnosticRouteAssessment(
            value_policy_id="fake-structured-policy-v1",
            diagnostic_fields=(
                ("closure_state", "fake-complete"),
                ("uncertainty_state", "fake-low"),
            ),
            usage=RouteComputeUsage(1, 1, 1, 1),
        ),
    )
    assert repeated == result


def test_zero_guidance_fails_closed_on_productive_byte_divergence() -> None:
    with pytest.raises(UgiMatchedBudgetError, match="zero-guidance"):
        run_diagnostic_matched_budget_arms(
            _schedule(1),
            budget_limits=MatchedBudgetLimits(1, 1, 1, RouteComputeUsage(1, 1, 1, 1)),
            base_seed=5,
            zero_guidance=True,
            cache_snapshot_sha256=CACHE_SNAPSHOT_SHA256,
            generate_terminal=lambda request: _terminal(request, arm_dependent_bytes=True),
            assess_terminal=lambda terminal, context: DiagnosticRouteAssessment(
                value_policy_id="fake-v1",
                diagnostic_fields=(("state", "fake"),),
                usage=RouteComputeUsage(1, 1, 1, 1),
            ),
        )


def test_post_hoc_terminal_must_be_locked_before_post_hoc_assessment() -> None:
    assessed_arms: list[MatchedArm] = []

    def generate(request: MatchedGenerationRequest) -> LockedMatchedTerminal:
        return _terminal(
            request,
            locked=request.arm is not MatchedArm.POST_HOC,
        )

    def assess(terminal, context):
        assessed_arms.append(context.arm)
        return DiagnosticRouteAssessment(
            value_policy_id="fake-v1",
            diagnostic_fields=(("state", "fake"),),
            usage=RouteComputeUsage(1, 1, 1, 1),
        )

    with pytest.raises(UgiMatchedBudgetError, match="sealed terminal"):
        run_diagnostic_matched_budget_arms(
            _schedule(1),
            budget_limits=MatchedBudgetLimits(1, 1, 1, RouteComputeUsage(1, 1, 1, 1)),
            base_seed=7,
            zero_guidance=False,
            cache_snapshot_sha256=CACHE_SNAPSHOT_SHA256,
            generate_terminal=generate,
            assess_terminal=assess,
        )

    assert assessed_arms == [MatchedArm.GUIDED]


def test_cache_clones_do_not_inherit_guided_warm_state() -> None:
    cache_by_clone: dict[str, set[str]] = {}
    contexts: dict[MatchedArm, tuple[str, str]] = {}

    def assess(terminal, context):
        contexts[context.arm] = (
            context.cache_snapshot_sha256,
            context.cache_clone_id,
        )
        cache = cache_by_clone.setdefault(context.cache_clone_id, set())
        cache_hit = "shared-fake-route-key" in cache
        cache.add("shared-fake-route-key")
        return DiagnosticRouteAssessment(
            value_policy_id="fake-v1",
            diagnostic_fields=(("state", "fake"),),
            usage=RouteComputeUsage(1, 0 if cache_hit else 1, 1, 1),
        )

    result = run_diagnostic_matched_budget_arms(
        _schedule(2),
        budget_limits=MatchedBudgetLimits(2, 2, 2, RouteComputeUsage(2, 2, 2, 2)),
        base_seed=9,
        zero_guidance=True,
        cache_snapshot_sha256=CACHE_SNAPSHOT_SHA256,
        generate_terminal=lambda request: _terminal(request),
        assess_terminal=assess,
    )

    assert contexts[MatchedArm.GUIDED][0] == CACHE_SNAPSHOT_SHA256
    assert contexts[MatchedArm.POST_HOC][0] == CACHE_SNAPSHOT_SHA256
    assert contexts[MatchedArm.GUIDED][1] != contexts[MatchedArm.POST_HOC][1]
    assert result.guided_ledger.route == RouteComputeUsage(2, 1, 2, 2)
    assert result.post_hoc_ledger.route == RouteComputeUsage(2, 1, 2, 2)
    assert result.guided_ledger.planner_cache_hits == 1
    assert result.post_hoc_ledger.planner_cache_hits == 1
    assert result.guided_records[0].cache_clone_id == result.guided_cache_clone_id
    assert result.post_hoc_records[0].cache_clone_id == result.post_hoc_cache_clone_id


def test_arm_cannot_change_morphology_program_or_checkpoint_contract() -> None:
    def generate(request: MatchedGenerationRequest) -> LockedMatchedTerminal:
        return _terminal(
            request,
            morphology_program_sha256=(
                "0" * 64
                if request.arm is MatchedArm.POST_HOC
                else request.entry.morphology_program_sha256
            ),
        )

    def wrong_checkpoint(request: MatchedGenerationRequest) -> LockedMatchedTerminal:
        return _terminal(
            request,
            closure_checkpoint_sha256=(
                "d" * 64
                if request.arm is MatchedArm.POST_HOC
                else request.entry.closure_checkpoint_sha256
            ),
        )

    with pytest.raises(UgiMatchedBudgetError, match="closure checkpoint hash mismatch"):
        run_diagnostic_matched_budget_arms(
            _schedule(1),
            budget_limits=MatchedBudgetLimits(1, 1, 1, RouteComputeUsage(1, 1, 1, 1)),
            base_seed=12,
            zero_guidance=False,
            cache_snapshot_sha256=CACHE_SNAPSHOT_SHA256,
            generate_terminal=wrong_checkpoint,
            assess_terminal=lambda terminal, context: DiagnosticRouteAssessment(
                value_policy_id="fake-v1",
                diagnostic_fields=(("state", "fake"),),
                usage=RouteComputeUsage(1, 1, 1, 1),
            ),
        )

    with pytest.raises(UgiMatchedBudgetError, match="morphology program changed"):
        run_diagnostic_matched_budget_arms(
            _schedule(1),
            budget_limits=MatchedBudgetLimits(1, 1, 1, RouteComputeUsage(1, 1, 1, 1)),
            base_seed=11,
            zero_guidance=False,
            cache_snapshot_sha256=CACHE_SNAPSHOT_SHA256,
            generate_terminal=generate,
            assess_terminal=lambda terminal, context: DiagnosticRouteAssessment(
                value_policy_id="fake-v1",
                diagnostic_fields=(("state", "fake"),),
                usage=RouteComputeUsage(1, 1, 1, 1),
            ),
        )


def test_route_assessment_cannot_exceed_frozen_unit_reservation() -> None:
    with pytest.raises(UgiMatchedBudgetError, match="frozen reservation"):
        run_diagnostic_matched_budget_arms(
            _schedule(1),
            budget_limits=MatchedBudgetLimits(1, 1, 1, RouteComputeUsage(2, 2, 2, 2)),
            base_seed=13,
            zero_guidance=False,
            cache_snapshot_sha256=CACHE_SNAPSHOT_SHA256,
            generate_terminal=lambda request: _terminal(request),
            assess_terminal=lambda terminal, context: DiagnosticRouteAssessment(
                value_policy_id="fake-v1",
                diagnostic_fields=(("state", "fake"),),
                usage=RouteComputeUsage(2, 2, 1, 1),
            ),
        )


def test_invalid_usage_and_duplicate_schedule_coordinates_fail_closed() -> None:
    with pytest.raises(UgiMatchedBudgetError, match="cannot exceed"):
        RouteComputeUsage(1, 2, 0, 0)

    duplicated = (_schedule(1)[0], _schedule(1)[0])
    with pytest.raises(UgiMatchedBudgetError, match="unit IDs must be unique"):
        run_diagnostic_matched_budget_arms(
            duplicated,
            budget_limits=MatchedBudgetLimits(2, 2, 2, RouteComputeUsage(2, 2, 2, 2)),
            base_seed=17,
            zero_guidance=False,
            cache_snapshot_sha256=CACHE_SNAPSHOT_SHA256,
            generate_terminal=lambda request: _terminal(request),
            assess_terminal=lambda terminal, context: DiagnosticRouteAssessment(
                value_policy_id="fake-v1",
                diagnostic_fields=(("state", "fake"),),
                usage=RouteComputeUsage(1, 1, 1, 1),
            ),
        )


def test_frozen_matched_budget_audit_is_deterministic_and_provenance_pinned() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts/phase1_audit_ugi_matched_budget_orchestration.py"
    config = repo / "configs/model/phase1_ugi_matched_budget_orchestration_v1.json"
    build_audit = runpy.run_path(script)["build_audit"]

    first = build_audit(repo, config)
    repeated = build_audit(repo, config)

    assert first == repeated
    assert first["decision"] == "diagnostic_matched_budget_orchestration_contract_qualified"
    assert first["matched_contract"]["zero_guidance_bitwise_equivalent"] is True
    assert first["event_contract"]["all_post_hoc_terminals_locked_before_route_assessment"]
    assert not first["event_contract"]["warm_cache_inheritance"]
    assert not first["production_synthesis_guidance"]
    assert not first["biological_guidance"]
    assert not first["candidate_selection_performed"]
    assert not first["scalar_synthesis_value_defined"]
    assert not first["real_route_planner_executed"]
    assert not first["sealed_holdout_accessed"]
    assert set(first["inputs"]) == {
        "audit_script",
        "config",
        "contract_source",
        "contract_tests",
        "fixed_budget_rollout_result",
        "restartable_sampler_result",
    }
    assert all(not record["path"].startswith("/") for record in first["inputs"].values())
