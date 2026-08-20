from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from forge.design.schedule.ugi_matched_budget_orchestration import (
    LockedMatchedTerminal,
    MatchedArm,
    MatchedAssessmentContext,
    RouteComputeUsage,
)
from forge.design.schedule.ugi_matched_planner_cache_binding import (
    preflight_lazy_matched_planner_cache_binding,
)
from forge.design.schedule.ugi_nonzero_guidance_runner import (
    FrozenSeedProgramAssignment,
    GuidanceAssessmentContext,
    GuidanceCacheFinalization,
    GuidanceCacheIsolationContract,
    GuidanceComputeBudget,
    GuidanceRouteEvaluation,
    GuidanceSchedule,
    GuidanceStateReceipt,
    GuidanceTerminalCompletionReceipt,
    NonzeroGuidanceExecutionReview,
    ParticleGroupDesign,
    UgiNonzeroGuidanceRunnerError,
    build_nonzero_guidance_runner_plan,
    development_guidance_run_to_dict,
    finalize_lazy_matched_cache_binding,
    load_grouped_smc_schedule_qualification,
    run_development_matched_guidance,
)
from forge.route.engine.planner import PlannerBudgetLimits
from forge.route.engine.planner_cache import (
    FilePlannerCache,
    PlannerCacheContext,
    PlannerCacheError,
)

REPO = Path(__file__).resolve().parents[1]
FAKE_SHA = "a" * 64
POLICY_ID = "binary-exact-dossier-route-completion-utility-v1"


def _assessment_receipt(
    terminal: LockedMatchedTerminal,
    context: GuidanceAssessmentContext,
) -> str:
    return hashlib.sha256(
        f"assessment:{terminal.terminal_id}:{context.route_seed}".encode()
    ).hexdigest()


def _dossier_receipt(
    terminal: LockedMatchedTerminal,
    utility: float | None,
) -> str | None:
    if utility != 1.0:
        return None
    return hashlib.sha256(f"dossier:{terminal.terminal_id}".encode()).hexdigest()


def _cache_contract() -> GuidanceCacheIsolationContract:
    return GuidanceCacheIsolationContract(
        base_snapshot_sha256="c" * 64,
        planner_context_sha256="d" * 64,
        cache_preflight_sha256="e" * 64,
        guided_clone_id="fake-guided-clone",
        post_hoc_clone_id="fake-post-hoc-clone",
        isolated_overlay_roots=True,
        production_adapter_qualified=False,
    )


def _cache_finalizer():
    contract = _cache_contract()

    def finalize() -> GuidanceCacheFinalization:
        return GuidanceCacheFinalization(
            audit_sha256=hashlib.sha256(b"fake-cache-audit").hexdigest(),
            base_snapshot_sha256=contract.base_snapshot_sha256,
            planner_context_sha256=contract.planner_context_sha256,
            cache_preflight_sha256=contract.cache_preflight_sha256,
            guided_clone_id=contract.guided_clone_id,
            post_hoc_clone_id=contract.post_hoc_clone_id,
            base_unchanged=True,
            overlay_roots_isolated=True,
            overlay_start_states_identical=True,
        )

    return finalize


def _planner_context() -> PlannerCacheContext:
    limits = PlannerBudgetLimits(
        maximum_depth=3,
        maximum_logical_planner_calls=3,
        maximum_expansions=16,
        maximum_product_candidates=16,
        maximum_verifier_calls=13,
        maximum_elapsed_milliseconds=10_000,
    )
    return PlannerCacheContext(
        planner_id="nonzero-runner-test",
        planner_sha256=hashlib.sha256(b"planner").hexdigest(),
        search_policy_sha256=hashlib.sha256(b"search").hexdigest(),
        value_policy_sha256=hashlib.sha256(POLICY_ID.encode()).hexdigest(),
        l1_reaction_sha256=hashlib.sha256(b"l1").hexdigest(),
        upstream_reaction_registry_sha256=hashlib.sha256(b"upstream").hexdigest(),
        variant_registry_sha256=hashlib.sha256(b"variants").hexdigest(),
        verifier_sha256=hashlib.sha256(b"verifier").hexdigest(),
        l3_snapshot_sha256=hashlib.sha256(b"l3").hexdigest(),
        l3_region="US",
        l3_accessed_at_utc="2026-08-01T00:00:00Z",
        l3_expires_at_utc="2026-08-31T00:00:00Z",
        software_versions=(("rdkit", "test"),),
        identity_policy="canonical_constitutional_smiles",
        stereochemistry_policy="phase1_stereo_free",
        budget_limits=limits,
    )


def _actual_cache_binding(tmp_path: Path):
    binding = preflight_lazy_matched_planner_cache_binding(
        FilePlannerCache(tmp_path / "base"),
        FilePlannerCache(tmp_path / "guided"),
        FilePlannerCache(tmp_path / "post_hoc"),
        _planner_context(),
        assessment_at_utc="2026-08-03T00:00:00Z",
    )
    contract = GuidanceCacheIsolationContract(
        base_snapshot_sha256=binding.preflight.base.snapshot_sha256,
        planner_context_sha256=binding.preflight.context_sha256,
        cache_preflight_sha256=binding.preflight.preflight_sha256,
        guided_clone_id="runner-guided-cache-clone",
        post_hoc_clone_id="runner-post-hoc-cache-clone",
        isolated_overlay_roots=True,
        production_adapter_qualified=False,
    )
    return binding, contract


def _assignment(seed: int) -> FrozenSeedProgramAssignment:
    programs = (b"program-a", b"program-b")
    return FrozenSeedProgramAssignment(
        seed=seed,
        schedule_sha256=hashlib.sha256(f"schedule:{seed}".encode()).hexdigest(),
        programs=programs,
        program_sha256s=tuple(hashlib.sha256(value).hexdigest() for value in programs),
        stochastic_particle_seeds=tuple(seed * 100 + index for index in range(8)),
    )


@dataclass(frozen=True)
class FakeState:
    programs: tuple[bytes, ...]
    particle_tokens: tuple[int, ...]
    step: int
    device: str


class FakeRestartableLane:
    def __init__(self) -> None:
        self.events: list[tuple[str, int, int]] = []

    def initialize(
        self,
        programs: tuple[bytes, ...],
        *,
        seed: int,
        particle_seeds: tuple[int, ...],
        device: str,
    ) -> GuidanceStateReceipt:
        assert len(particle_seeds) == len(programs)
        self.events.append(("initialize", seed, len(programs)))
        return GuidanceStateReceipt(
            state=FakeState(
                programs=programs,
                particle_tokens=tuple(range(len(programs))),
                step=0,
                device=device,
            ),
            product_transition_calls=0,
            consumed_particle_seed_manifest_sha256=hashlib.sha256(
                json.dumps(
                    particle_seeds,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        )

    def advance(self, state: FakeState, *, target_step: int) -> GuidanceStateReceipt:
        assert state.step <= target_step
        self.events.append(("advance", state.step, target_step))
        return GuidanceStateReceipt(
            state=FakeState(
                programs=state.programs,
                particle_tokens=state.particle_tokens,
                step=target_step,
                device=state.device,
            ),
            product_transition_calls=len(state.programs) * (target_step - state.step),
        )

    def snapshot(self, state: FakeState) -> GuidanceStateReceipt:
        self.events.append(("snapshot", state.step, len(state.programs)))
        return GuidanceStateReceipt(state=state, product_transition_calls=0)

    def complete_terminal(
        self,
        state: FakeState,
        *,
        particle_index: int,
        seed: int,
        checkpoint_index: int,
    ) -> GuidanceTerminalCompletionReceipt:
        token = state.particle_tokens[particle_index]
        self.events.append(("complete", checkpoint_index, token))
        # Productive completion remains stochastic after ancestry, so copied
        # ancestors can still yield constitutionally distinct final identities.
        identity = f"C{token + 1}N-final-{particle_index}"
        payload = {
            "canonical_identity": identity,
            "utility": float(token % 4 == 0),
            "token": token,
            "checkpoint": checkpoint_index,
        }
        terminal_bytes = json.dumps(
            {
                "canonical_identity": identity,
                "checkpoint": checkpoint_index,
                "seed": seed,
                "token": token,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        trace_bytes = f"trace:{state.step}:{token}:{seed}".encode()
        return GuidanceTerminalCompletionReceipt(
            terminal=LockedMatchedTerminal(
                unit_id=f"unit-{checkpoint_index}-{particle_index}",
                morphology_program_sha256=hashlib.sha256(
                    state.programs[particle_index]
                ).hexdigest(),
                checkpoint_index=checkpoint_index,
                generator_checkpoint_sha256=FAKE_SHA,
                closure_checkpoint_sha256="b" * 64,
                terminal_id=f"terminal-{checkpoint_index}-{token}-{seed}",
                terminal_locked=True,
                terminal_valid=True,
                exact_l1=True,
                terminal_bytes=terminal_bytes,
                generation_trace_bytes=trace_bytes,
                payload=payload,
            ),
            product_transition_calls=8 - state.step,
        )

    def apply_ancestry(
        self,
        state: FakeState,
        ancestors: tuple[int, ...],
    ) -> GuidanceStateReceipt:
        assert len(ancestors) == len(state.programs)
        for destination, source in enumerate(ancestors):
            if state.programs[destination] != state.programs[source]:
                raise UgiNonzeroGuidanceRunnerError("fake lane rejected cross-program ancestry")
        self.events.append(("ancestry", state.step, sum(ancestors)))
        return GuidanceStateReceipt(
            state=FakeState(
                programs=state.programs,
                particle_tokens=tuple(state.particle_tokens[index] for index in ancestors),
                step=state.step,
                device=state.device,
            ),
            product_transition_calls=0,
        )


def _design() -> ParticleGroupDesign:
    return ParticleGroupDesign(morphology_program_count=2, particles_per_program=4)


def _schedule() -> GuidanceSchedule:
    return GuidanceSchedule(
        sample_steps=8,
        checkpoints=(2, 4, 6),
        checkpoint_betas=(0.25, 0.5, 0.75),
        rollouts_per_particle_checkpoint=1,
        final_selection_count=4,
    )


def _budget() -> GuidanceComputeBudget:
    particle_count = _design().particle_count
    terminal_completions = particle_count * 4
    return GuidanceComputeBudget(
        product_transition_calls=(
            particle_count * 8 + particle_count * ((8 - 2) + (8 - 4) + (8 - 6))
        ),
        terminal_completions=terminal_completions,
        logical_planner_calls=terminal_completions * 3,
        logical_verifier_calls=terminal_completions * 13,
        final_candidates=4,
    )


def _evaluator_factory(events: list[tuple[str, int, int]]):
    def evaluate(
        terminal: LockedMatchedTerminal,
        context: GuidanceAssessmentContext,
    ) -> GuidanceRouteEvaluation:
        remaining = context.reservation
        events.append(
            (
                f"assess:{context.treatment_arm}_{context.assessment_phase}",
                context.checkpoint,
                terminal.payload["token"],
            )
        )
        utility = terminal.payload["utility"]
        return GuidanceRouteEvaluation(
            route_completion_utility=utility,
            value_policy_id=POLICY_ID,
            usage=RouteComputeUsage(
                logical_planner_calls=min(3, remaining.logical_planner_calls),
                physical_planner_calls=min(2, remaining.physical_planner_calls),
                logical_verifier_calls=min(13, remaining.logical_verifier_calls),
                physical_verifier_calls=min(11, remaining.physical_verifier_calls),
            ),
            assessment_receipt_sha256=_assessment_receipt(terminal, context),
            route_dossier_sha256=_dossier_receipt(terminal, utility),
        )

    return evaluate


def _bound_evaluator(binding, contexts: list[GuidanceAssessmentContext]):
    def evaluate(
        terminal: LockedMatchedTerminal,
        context: GuidanceAssessmentContext,
    ) -> GuidanceRouteEvaluation:
        contexts.append(context)
        binding.bind(
            MatchedAssessmentContext(
                arm=MatchedArm(context.treatment_arm),
                route_seed=context.route_seed,
                remaining_budget=context.reservation,
                unit_reservation=context.reservation,
                cache_snapshot_sha256=context.base_snapshot_sha256,
                cache_clone_id=context.cache_clone_id,
                post_hoc_lock_manifest_sha256=(context.post_hoc_productive_lock_sha256),
            )
        )
        utility = terminal.payload["utility"]
        return GuidanceRouteEvaluation(
            route_completion_utility=utility,
            value_policy_id=POLICY_ID,
            usage=RouteComputeUsage(
                logical_planner_calls=3,
                physical_planner_calls=0,
                logical_verifier_calls=13,
                physical_verifier_calls=0,
            ),
            assessment_receipt_sha256=_assessment_receipt(terminal, context),
            route_dossier_sha256=_dossier_receipt(terminal, utility),
        )

    return evaluate


def _canonical_identity(terminal_bytes: bytes) -> str:
    return json.loads(terminal_bytes)["canonical_identity"]


def _run(lane: FakeRestartableLane, *, seed: int = 17):
    result = run_development_matched_guidance(
        _assignment(seed),
        lane=lane,
        evaluator=_evaluator_factory(lane.events),
        canonical_identity=_canonical_identity,
        design=_design(),
        schedule=_schedule(),
        budget=_budget(),
        guidance_strength=8.0,
        device="diagnostic-device",
        expected_value_policy_id=POLICY_ID,
        cache_contract=_cache_contract(),
        cache_finalizer=_cache_finalizer(),
    )
    return result, lane.events


def test_grouped_nonzero_guidance_resamples_only_within_programs() -> None:
    result, _ = _run(FakeRestartableLane())

    assert len(result.guided.checkpoint_groups) == 6
    assert any(record.resampled for record in result.guided.checkpoint_groups)
    assert any(
        record.local_ancestors != tuple(range(4)) for record in result.guided.checkpoint_groups
    )
    for record in result.guided.checkpoint_groups:
        allowed = set(record.global_particle_indices)
        assert set(record.global_ancestors) <= allowed
        assert (
            tuple(record.global_particle_indices[index] for index in record.local_ancestors)
            == record.global_ancestors
        )


def test_genealogical_tempered_values_follow_selected_ancestors() -> None:
    result, _ = _run(FakeRestartableLane())
    first = result.guided.checkpoint_groups[0]
    second = result.guided.checkpoint_groups[2]

    assert first.checkpoint == 2
    assert second.checkpoint == 4
    assert second.tempered_before == first.carried_tempered_after
    assert first.carried_tempered_after == tuple(
        first.tempered_targets[index] for index in first.local_ancestors
    )


def test_post_hoc_shadow_assessments_are_delayed_and_nonselecting() -> None:
    lane = FakeRestartableLane()
    result, events = _run(lane)

    final_completion_indices = [
        index for index, event in enumerate(events) if event[0] == "complete" and event[1] == 8
    ]
    post_hoc_shadow_indices = [
        index
        for index, event in enumerate(events)
        if event[0] == "assess:post_hoc_checkpoint_shadow"
    ]
    assert len(final_completion_indices) == 16
    assert post_hoc_shadow_indices
    assert max(final_completion_indices) < min(post_hoc_shadow_indices)
    assert result.post_hoc.shadow_assessment_started_after_productive_lock is True
    assert result.post_hoc.shadow_assessments_selecting is False
    assert result.post_hoc.productive_pool_size == 8
    assert len(result.post_hoc.selected) == 4
    assert result.guided.realized_terminal_completions == 32
    assert result.post_hoc.realized_terminal_completions == 32
    assert result.guided.realized_product_transition_calls == 160
    assert result.guided.realized_logical_planner_calls == 96
    assert result.guided.realized_physical_planner_calls == 64
    assert result.guided.planner_cache_hits == 32
    assert result.guided.realized_verifier_calls == 416
    assert result.guided.realized_physical_verifier_calls == 352
    assert result.guided.verifier_cache_hits == 64
    assert all(record.shadow_batch_manifest_sha256 for record in result.guided.checkpoint_groups)
    assert all(
        len(record.evaluation_receipt_sha256s) == 4 for record in result.guided.checkpoint_groups
    )


def test_final_selection_is_deduplicated_and_keyed_deterministically() -> None:
    first, _ = _run(FakeRestartableLane(), seed=41)
    repeated, _ = _run(FakeRestartableLane(), seed=41)

    assert first.guided.selected == repeated.guided.selected
    assert first.post_hoc.selected == repeated.post_hoc.selected
    assert all(candidate.support_bonus for candidate in first.guided.selected[:2])
    assert len({candidate.canonical_identity for candidate in first.guided.selected}) == 4


def test_zero_guidance_keeps_productive_arms_bitwise_paired() -> None:
    lane = FakeRestartableLane()
    events: list[tuple[str, int, int]] = []
    result = run_development_matched_guidance(
        _assignment(53),
        lane=lane,
        evaluator=_evaluator_factory(events),
        canonical_identity=_canonical_identity,
        design=_design(),
        schedule=_schedule(),
        budget=_budget(),
        guidance_strength=0.0,
        device="cpu",
        expected_value_policy_id=POLICY_ID,
        cache_contract=_cache_contract(),
        cache_finalizer=_cache_finalizer(),
    )

    assert all(not record.resampled for record in result.guided.checkpoint_groups)
    assert result.guided.productive_terminal_ids == result.post_hoc.productive_terminal_ids
    assert (
        result.guided.productive_canonical_identities
        == result.post_hoc.productive_canonical_identities
    )
    assert result.guided.productive_lock_manifest_sha256 == (
        result.post_hoc.productive_lock_manifest_sha256
    )


def test_censored_checkpoint_is_controller_neutral_without_utility_imputation() -> None:
    lane = FakeRestartableLane()

    def evaluator(
        terminal: LockedMatchedTerminal,
        context: GuidanceAssessmentContext,
    ) -> GuidanceRouteEvaluation:
        utility = terminal.payload["utility"]
        return GuidanceRouteEvaluation(
            route_completion_utility=utility,
            value_policy_id=POLICY_ID,
            usage=RouteComputeUsage(
                logical_planner_calls=1,
                physical_planner_calls=1,
                logical_verifier_calls=1,
                physical_verifier_calls=1,
            ),
            assessment_receipt_sha256=_assessment_receipt(terminal, context),
            route_dossier_sha256=_dossier_receipt(terminal, utility),
        )

    original = lane.complete_terminal

    def invalid_at_first_checkpoint(*args, **kwargs):
        completion = original(*args, **kwargs)
        terminal = completion.terminal
        assert terminal is not None
        if kwargs["checkpoint_index"] == 2 and kwargs["particle_index"] == 0:
            return replace(
                completion,
                terminal=replace(terminal, terminal_valid=False, exact_l1=False),
            )
        return completion

    lane.complete_terminal = invalid_at_first_checkpoint  # type: ignore[method-assign]
    result = run_development_matched_guidance(
        _assignment(71),
        lane=lane,
        evaluator=evaluator,
        canonical_identity=_canonical_identity,
        design=_design(),
        schedule=_schedule(),
        budget=_budget(),
        guidance_strength=8.0,
        device="cpu",
        expected_value_policy_id=POLICY_ID,
        cache_contract=_cache_contract(),
        cache_finalizer=_cache_finalizer(),
    )

    record = result.guided.checkpoint_groups[0]
    assert record.route_completion_utilities[0] is None
    assert record.incremental_log_weights[0] == 0.0
    assert record.tempered_targets[0] == record.tempered_before[0]


def test_planner_censored_exact_l1_rollout_retains_none_and_neutral_increment() -> None:
    lane = FakeRestartableLane()

    def evaluator(
        terminal: LockedMatchedTerminal,
        context: GuidanceAssessmentContext,
    ) -> GuidanceRouteEvaluation:
        censored = context.checkpoint == 2 and terminal.payload["token"] == 0
        utility = None if censored else terminal.payload["utility"]
        return GuidanceRouteEvaluation(
            route_completion_utility=utility,
            value_policy_id=POLICY_ID,
            usage=RouteComputeUsage(
                logical_planner_calls=1,
                physical_planner_calls=1,
                logical_verifier_calls=1,
                physical_verifier_calls=1,
            ),
            assessment_receipt_sha256=_assessment_receipt(terminal, context),
            route_dossier_sha256=_dossier_receipt(terminal, utility),
        )

    result = run_development_matched_guidance(
        _assignment(73),
        lane=lane,
        evaluator=evaluator,
        canonical_identity=_canonical_identity,
        design=_design(),
        schedule=_schedule(),
        budget=_budget(),
        guidance_strength=8.0,
        device="cpu",
        expected_value_policy_id=POLICY_ID,
        cache_contract=_cache_contract(),
        cache_finalizer=_cache_finalizer(),
    )

    record = result.guided.checkpoint_groups[0]
    assert record.dispositions[0] == "assessed"
    assert record.route_completion_utilities[0] is None
    assert record.incremental_log_weights[0] == 0.0
    assert record.value_policy_ids[0] == POLICY_ID
    assert record.evaluation_receipt_sha256s[0] is not None
    assert record.censored_assessment_count == 1


def test_invalid_productive_final_is_excluded_before_canonicalization() -> None:
    lane = FakeRestartableLane()
    original = lane.complete_terminal

    def invalid_final(*args, **kwargs):
        completion = original(*args, **kwargs)
        terminal = completion.terminal
        assert terminal is not None
        if kwargs["checkpoint_index"] == 8 and kwargs["particle_index"] == 0:
            return replace(
                completion,
                terminal=replace(terminal, terminal_valid=False, exact_l1=False),
            )
        return completion

    lane.complete_terminal = invalid_final  # type: ignore[method-assign]

    def valid_only_identity(terminal_bytes: bytes) -> str:
        return _canonical_identity(terminal_bytes)

    result = run_development_matched_guidance(
        _assignment(79),
        lane=lane,
        evaluator=_evaluator_factory(lane.events),
        canonical_identity=valid_only_identity,
        design=_design(),
        schedule=_schedule(),
        budget=_budget(),
        guidance_strength=0.0,
        device="cpu",
        expected_value_policy_id=POLICY_ID,
        cache_contract=_cache_contract(),
        cache_finalizer=_cache_finalizer(),
    )

    assert result.guided.invalid_terminal_count >= 1
    assert result.guided.productive_canonical_identities[0] is None
    assert len(result.guided.selected) == 4


def test_censored_final_selection_never_serializes_none_as_zero() -> None:
    lane = FakeRestartableLane()

    def evaluator(
        terminal: LockedMatchedTerminal,
        context: GuidanceAssessmentContext,
    ) -> GuidanceRouteEvaluation:
        utility = None if context.checkpoint == 8 else 0.0
        return GuidanceRouteEvaluation(
            route_completion_utility=utility,
            value_policy_id=POLICY_ID,
            usage=RouteComputeUsage(
                logical_planner_calls=1,
                physical_planner_calls=1,
                logical_verifier_calls=1,
                physical_verifier_calls=1,
            ),
            assessment_receipt_sha256=_assessment_receipt(terminal, context),
            route_dossier_sha256=_dossier_receipt(terminal, utility),
        )

    result = run_development_matched_guidance(
        _assignment(83),
        lane=lane,
        evaluator=evaluator,
        canonical_identity=_canonical_identity,
        design=_design(),
        schedule=_schedule(),
        budget=_budget(),
        guidance_strength=0.0,
        device="cpu",
        expected_value_policy_id=POLICY_ID,
        cache_contract=_cache_contract(),
        cache_finalizer=_cache_finalizer(),
    )

    assert all(candidate.route_completion_utility is None for candidate in result.guided.selected)
    assert all(candidate.censored for candidate in result.guided.selected)
    assert all(not candidate.support_bonus for candidate in result.guided.selected)


def test_qualified_schedule_loader_pins_all_seed_program_assignments() -> None:
    qualification = load_grouped_smc_schedule_qualification(
        REPO / "results/phase1/ugi_grouped_smc_schedule_qualification_v1/result.json"
    )

    assert [value.seed for value in qualification.assignments] == list(range(20260821, 20260829))
    assert (
        len({program for value in qualification.assignments for program in value.programs}) == 128
    )
    assert (
        len(
            {
                seed
                for value in qualification.assignments
                for seed in value.stochastic_particle_seeds
            }
        )
        == 512
    )
    assert all(
        value.qualification_file_sha256 == qualification.file_sha256
        for value in qualification.assignments
    )
    assert all(
        value.qualification_result_sha256 == qualification.result_sha256
        for value in qualification.assignments
    )


def test_qualified_schedule_loader_rejects_rehashed_program_tampering(
    tmp_path: Path,
) -> None:
    source = REPO / "results/phase1/ugi_grouped_smc_schedule_qualification_v1/result.json"
    content = json.loads(source.read_text())
    schedule = content["seed_schedules"][0]
    schedule["programs"][0]["program"]["node_counts"][0] += 1
    schedule_content = {key: schedule[key] for key in ("seed", "programs", "particles")}
    schedule["schedule_sha256"] = hashlib.sha256(
        json.dumps(
            schedule_content,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    result_content = {key: value for key, value in content.items() if key != "result_sha256"}
    content["result_sha256"] = hashlib.sha256(
        json.dumps(
            result_content,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(content, sort_keys=True, separators=(",", ":")) + "\n")

    with pytest.raises(UgiNonzeroGuidanceRunnerError, match="program hash changed"):
        load_grouped_smc_schedule_qualification(tampered)


def test_particle_seed_manifest_must_be_functionally_consumed() -> None:
    class IgnoringParticleSeedsLane(FakeRestartableLane):
        def initialize(self, *args, **kwargs):
            receipt = super().initialize(*args, **kwargs)
            return replace(
                receipt,
                consumed_particle_seed_manifest_sha256="f" * 64,
            )

    with pytest.raises(
        UgiNonzeroGuidanceRunnerError,
        match="particle-seed manifest",
    ):
        _run(IgnoringParticleSeedsLane())


def test_serialized_result_keeps_censor_null_and_row_level_receipts() -> None:
    lane = FakeRestartableLane()

    def evaluator(
        terminal: LockedMatchedTerminal,
        context: GuidanceAssessmentContext,
    ) -> GuidanceRouteEvaluation:
        utility = None if context.checkpoint == 8 else terminal.payload["utility"]
        return GuidanceRouteEvaluation(
            route_completion_utility=utility,
            value_policy_id=POLICY_ID,
            usage=RouteComputeUsage(
                logical_planner_calls=1,
                physical_planner_calls=1,
                logical_verifier_calls=1,
                physical_verifier_calls=1,
            ),
            assessment_receipt_sha256=_assessment_receipt(terminal, context),
            route_dossier_sha256=_dossier_receipt(terminal, utility),
        )

    result = run_development_matched_guidance(
        _assignment(87),
        lane=lane,
        evaluator=evaluator,
        canonical_identity=_canonical_identity,
        design=_design(),
        schedule=_schedule(),
        budget=_budget(),
        guidance_strength=0.0,
        device="cpu",
        expected_value_policy_id=POLICY_ID,
        cache_contract=_cache_contract(),
        cache_finalizer=_cache_finalizer(),
    )
    serialized = development_guidance_run_to_dict(result)

    assert serialized["scope"]["success_probability"] is None
    assert serialized["run"]["guided"]["selected"][0]["route_completion_utility"] is None
    assert serialized["run"]["guided"]["productive_admissions"]
    assert serialized["run"]["guided"]["productive_assessments"]
    assert all(
        record["assessment_receipt_sha256"]
        for record in serialized["run"]["guided"]["productive_assessments"]
        if record["disposition"] == "canonical_representative_assessed"
    )
    round_trip = json.loads(json.dumps(serialized, sort_keys=True, separators=(",", ":")))
    assert round_trip["run"]["guided"]["selected"][0]["route_completion_utility"] is None


def test_real_cache_binding_finalizes_and_route_seeds_are_paired(tmp_path: Path) -> None:
    binding, contract = _actual_cache_binding(tmp_path)
    contexts: list[GuidanceAssessmentContext] = []
    result = run_development_matched_guidance(
        _assignment(89),
        lane=FakeRestartableLane(),
        evaluator=_bound_evaluator(binding, contexts),
        canonical_identity=_canonical_identity,
        design=_design(),
        schedule=_schedule(),
        budget=_budget(),
        guidance_strength=0.0,
        device="cpu",
        expected_value_policy_id=POLICY_ID,
        cache_contract=contract,
        cache_finalizer=lambda: finalize_lazy_matched_cache_binding(binding, contract),
    )

    assert result.cache_finalization.base_unchanged is True
    assert result.cache_finalization.overlay_roots_isolated is True
    assert result.guided.cache_clone_id != result.post_hoc.cache_clone_id
    guided = sorted(
        (value.assessment_phase, value.checkpoint, value.route_seed)
        for value in contexts
        if value.treatment_arm == "guided"
    )
    post_hoc = sorted(
        (value.assessment_phase, value.checkpoint, value.route_seed)
        for value in contexts
        if value.treatment_arm == "post_hoc"
    )
    assert guided == post_hoc
    assert len(result.guided.productive_assessments) == 8
    assert all(value.evaluation_receipt_sha256 for value in result.guided.productive_assessments)


def test_real_cache_binding_rejects_base_mutation_at_finalization(tmp_path: Path) -> None:
    binding, contract = _actual_cache_binding(tmp_path)

    def mutate_then_finalize() -> GuidanceCacheFinalization:
        rogue = tmp_path / "base" / "rogue.json"
        rogue.parent.mkdir(parents=True, exist_ok=True)
        rogue.write_text("{}\n")
        return finalize_lazy_matched_cache_binding(binding, contract)

    with pytest.raises(PlannerCacheError):
        run_development_matched_guidance(
            _assignment(97),
            lane=FakeRestartableLane(),
            evaluator=_bound_evaluator(binding, []),
            canonical_identity=_canonical_identity,
            design=_design(),
            schedule=_schedule(),
            budget=_budget(),
            guidance_strength=0.0,
            device="cpu",
            expected_value_policy_id=POLICY_ID,
            cache_contract=contract,
            cache_finalizer=mutate_then_finalize,
        )


def test_real_cache_binding_rejects_guided_overlay_mutation_after_post_hoc_bind(
    tmp_path: Path,
) -> None:
    binding, contract = _actual_cache_binding(tmp_path)

    def mutate_then_finalize() -> GuidanceCacheFinalization:
        rogue = tmp_path / "guided" / "rogue.json"
        rogue.parent.mkdir(parents=True, exist_ok=True)
        rogue.write_text("{}\n")
        return finalize_lazy_matched_cache_binding(binding, contract)

    with pytest.raises(PlannerCacheError):
        run_development_matched_guidance(
            _assignment(101),
            lane=FakeRestartableLane(),
            evaluator=_bound_evaluator(binding, []),
            canonical_identity=_canonical_identity,
            design=_design(),
            schedule=_schedule(),
            budget=_budget(),
            guidance_strength=0.0,
            device="cpu",
            expected_value_policy_id=POLICY_ID,
            cache_contract=contract,
            cache_finalizer=mutate_then_finalize,
        )


def test_route_blind_dedup_precedes_assessment_and_ignores_payload_mutation() -> None:
    class DuplicateLane(FakeRestartableLane):
        def complete_terminal(self, *args, **kwargs):
            completion = super().complete_terminal(*args, **kwargs)
            terminal = completion.terminal
            assert terminal is not None
            if kwargs["checkpoint_index"] == 8 and kwargs["particle_index"] in {0, 1}:
                content = json.loads(terminal.terminal_bytes)
                content["canonical_identity"] = "shared-route-blind-identity"
                payload = dict(terminal.payload)
                payload["canonical_identity"] = f"payload-alias-{kwargs['particle_index']}"
                terminal = replace(
                    terminal,
                    terminal_bytes=json.dumps(
                        content,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode(),
                    payload=payload,
                )
                completion = replace(completion, terminal=terminal)
            return completion

    assessed_finals: list[str] = []

    def evaluator(
        terminal: LockedMatchedTerminal,
        context: GuidanceAssessmentContext,
    ) -> GuidanceRouteEvaluation:
        if context.assessment_phase == "productive_final":
            assessed_finals.append(f"{context.treatment_arm}:{terminal.terminal_id}")
            terminal.payload["canonical_identity"] = "mutated-after-admission"
        utility = terminal.payload["utility"]
        return GuidanceRouteEvaluation(
            route_completion_utility=utility,
            value_policy_id=POLICY_ID,
            usage=RouteComputeUsage(
                logical_planner_calls=1,
                physical_planner_calls=1,
                logical_verifier_calls=1,
                physical_verifier_calls=1,
            ),
            assessment_receipt_sha256=_assessment_receipt(terminal, context),
            route_dossier_sha256=_dossier_receipt(terminal, utility),
        )

    result = run_development_matched_guidance(
        _assignment(103),
        lane=DuplicateLane(),
        evaluator=evaluator,
        canonical_identity=_canonical_identity,
        design=_design(),
        schedule=_schedule(),
        budget=_budget(),
        guidance_strength=0.0,
        device="cpu",
        expected_value_policy_id=POLICY_ID,
        cache_contract=_cache_contract(),
        cache_finalizer=_cache_finalizer(),
    )

    shared = [
        value
        for value in result.guided.productive_admissions
        if value.canonical_identity == "shared-route-blind-identity"
    ]
    assert len(shared) == 2
    assert {value.disposition for value in shared} == {
        "canonical_representative",
        "duplicate_not_assessed",
    }
    representative_id = next(
        value.terminal_id for value in shared if value.disposition == "canonical_representative"
    )
    assert sum(value.endswith(representative_id) for value in assessed_finals) == 2
    duplicate_assessment = next(
        value
        for value in result.guided.productive_assessments
        if value.disposition == "duplicate_shared_assessment"
    )
    assert duplicate_assessment.representative_terminal_id == representative_id
    assert "mutated-after-admission" not in result.guided.productive_canonical_identities


def test_execution_review_fails_closed_until_every_gate_passes() -> None:
    review = NonzeroGuidanceExecutionReview(
        receipt_sha256="c" * 64,
        route_completion_utility_frozen=True,
        current_l3_snapshot_frozen=False,
        cumulative_source_runtime_qualified=False,
        production_zero_guidance_passed=True,
        particle_group_amendment_frozen=True,
        nonzero_guidance_authorized=False,
        biological_guidance_authorized=False,
        candidate_selection_authorized=False,
        sealed_holdout_access_authorized=False,
    )

    with pytest.raises(UgiNonzeroGuidanceRunnerError, match="nonzero guidance is blocked"):
        review.require_nonzero_execution_authorized()


def test_exact_64_particle_budget_is_reproduced() -> None:
    design = ParticleGroupDesign(morphology_program_count=16, particles_per_program=4)
    schedule = GuidanceSchedule(
        sample_steps=8,
        checkpoints=(2, 4, 6),
        checkpoint_betas=(0.25, 0.5, 0.75),
        rollouts_per_particle_checkpoint=1,
        final_selection_count=32,
    )
    budget = GuidanceComputeBudget(
        product_transition_calls=1280,
        terminal_completions=256,
        logical_planner_calls=768,
        logical_verifier_calls=3328,
        final_candidates=32,
    )

    budget.validate_exact_schedule(design, schedule)
    assert budget.route_budget_per_terminal(design, schedule) == (3, 13)


def test_cross_program_ancestry_is_rejected_by_lane() -> None:
    lane = FakeRestartableLane()
    design = _design()
    state = lane.initialize(
        design.expanded_programs((b"program-a", b"program-b")),
        seed=3,
        particle_seeds=tuple(range(8)),
        device="cpu",
    )

    with pytest.raises(UgiNonzeroGuidanceRunnerError, match="cross-program"):
        lane.apply_ancestry(state.state, (4, 1, 2, 3, 0, 5, 6, 7))


def test_blocked_runner_plan_rederives_all_frozen_assignments() -> None:
    plan = build_nonzero_guidance_runner_plan(
        REPO,
        REPO / "configs/model/phase1_ugi_nonzero_guidance_runner_v1.json",
    )

    assert plan["status"] == ("device_agnostic_runner_prepared_nonzero_execution_blocked")
    assert plan["authorization"]["nonzero_guidance_authorized"] is False
    assert plan["scope"]["production_nonzero_execution"] is False
    assert plan["scope"]["prospective_candidate_lock"] is False
    assignments = plan["grouped_smc_schedule"]["assignments"]
    assert len(assignments) == 8
    assert all(len(value["program_sha256s"]) == 16 for value in assignments)
    assert all(value["particle_seed_manifest_sha256"] for value in assignments)


@pytest.mark.parametrize(
    ("field", "mutate", "match"),
    [
        (
            "schedule",
            lambda value: value.__setitem__("synthesis_checkpoints", [1, 4, 6]),
            "schedule differs",
        ),
        (
            "post_hoc_contract",
            lambda value: value.__setitem__(
                "shadow_assessments_influence_endpoint_selection", True
            ),
            "post-hoc shadow contract changed",
        ),
        (
            "final_selection_contract",
            lambda value: value.__setitem__(
                "representative_choice_may_not_use_route_outcome", False
            ),
            "route-blind final-selection contract changed",
        ),
    ],
)
def test_blocked_runner_plan_rejects_contract_drift(
    tmp_path: Path,
    field: str,
    mutate,
    match: str,
) -> None:
    source = REPO / "configs/model/phase1_ugi_nonzero_guidance_runner_v1.json"
    config = json.loads(source.read_text())
    mutate(config[field])
    changed = tmp_path / "changed-runner-config.json"
    changed.write_text(json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n")

    with pytest.raises(UgiNonzeroGuidanceRunnerError, match=match):
        build_nonzero_guidance_runner_plan(REPO, changed)
