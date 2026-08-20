from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from forge.design.schedule.ugi_matched_budget_orchestration import (
    DiagnosticRouteAssessment,
    LockedMatchedTerminal,
    MatchedArm,
    MatchedAssessmentContext,
    MatchedBudgetLimits,
    MatchedGenerationRequest,
    MatchedScheduleEntry,
    RouteComputeUsage,
    run_diagnostic_matched_budget_arms,
)
from forge.design.schedule.ugi_matched_planner_cache_binding import (
    MatchedPlannerCacheBindingPreflight,
    preflight_lazy_matched_planner_cache_binding,
)
from forge.route.engine.planner import (
    AvailabilityState,
    EvidenceRecord,
    EvidenceTier,
    ForwardVerificationState,
    KnowledgeDisposition,
    KnowledgeResult,
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RecursiveRouteAssessor,
    RouteTarget,
)
from forge.route.engine.planner_cache import (
    CachedRoutePlanner,
    FilePlannerCache,
    PlannerCacheContext,
    PlannerCacheError,
    PlannerCacheKey,
)

ASSESSMENT_AT_UTC = "2026-08-15T00:00:00Z"


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _limits() -> PlannerBudgetLimits:
    return PlannerBudgetLimits(
        maximum_depth=3,
        maximum_logical_planner_calls=1,
        maximum_expansions=4,
        maximum_product_candidates=4,
        maximum_verifier_calls=4,
        maximum_elapsed_milliseconds=1_000,
    )


def _context(**overrides: object) -> PlannerCacheContext:
    values: dict[str, object] = {
        "planner_id": "synthetic-exact-evidence-v1",
        "planner_sha256": _hash("planner"),
        "search_policy_sha256": _hash("search"),
        "value_policy_sha256": _hash("structured-diagnostic"),
        "l1_reaction_sha256": _hash("l1"),
        "upstream_reaction_registry_sha256": _hash("upstream"),
        "variant_registry_sha256": _hash("variants"),
        "verifier_sha256": _hash("verifier"),
        "l3_snapshot_sha256": _hash("l3"),
        "l3_region": "US",
        "l3_accessed_at_utc": "2026-08-01T00:00:00Z",
        "l3_expires_at_utc": "2026-08-31T00:00:00Z",
        "software_versions": (("rdkit", "test"), ("route_schema", "v1")),
        "identity_policy": "canonical_constitutional_smiles",
        "stereochemistry_policy": "phase1_stereo_free",
        "budget_limits": _limits(),
    }
    values.update(overrides)
    return PlannerCacheContext(**values)  # type: ignore[arg-type]


def _terminal_result(role: str = "amine_head") -> KnowledgeResult:
    return KnowledgeResult(
        disposition=KnowledgeDisposition.TERMINAL,
        evidence=(
            EvidenceRecord(
                evidence_id=f"terminal-{role}",
                tier=EvidenceTier.ACCEPTED_TERMINAL,
                source_sha256=_hash(f"source-{role}"),
                source_locator=f"synthetic:{role}",
                exact_substrate=True,
                forward_verification=ForwardVerificationState.NOT_APPLICABLE,
                availability=AvailabilityState.CURRENT_CLOSED,
            ),
        ),
        detail="synthetic accepted terminal",
    )


@dataclass
class FixedSource:
    calls: int = 0

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        self.calls += 1
        return _terminal_result(target.role)


def _assessment(context: PlannerCacheContext, smiles: str = "CCO"):
    return RecursiveRouteAssessor(FixedSource()).assess(
        RouteTarget("amine_head", smiles),
        PlannerBudgetLedger(context.budget_limits),
    )


def _put(cache: FilePlannerCache, context: PlannerCacheContext, smiles: str = "CCO") -> None:
    assessment = _assessment(context, smiles)
    key = PlannerCacheKey.build(
        assessment.target,
        context,
        PlannerBudgetLedger(context.budget_limits),
    )
    cache.put(key, assessment)


def _binding(tmp_path: Path, *, context: PlannerCacheContext | None = None):
    frozen_context = context or _context()
    return preflight_lazy_matched_planner_cache_binding(
        FilePlannerCache(tmp_path / "base"),
        FilePlannerCache(tmp_path / "guided"),
        FilePlannerCache(tmp_path / "post_hoc"),
        frozen_context,
        assessment_at_utc=ASSESSMENT_AT_UTC,
    )


def _assessment_context(
    binding,
    *,
    arm: MatchedArm,
    clone_id: str,
    snapshot_sha256: str | None = None,
) -> MatchedAssessmentContext:
    return MatchedAssessmentContext(
        arm=arm,
        route_seed=17,
        remaining_budget=RouteComputeUsage(1, 1, 0, 0),
        unit_reservation=RouteComputeUsage(1, 1, 0, 0),
        cache_snapshot_sha256=snapshot_sha256 or binding.preflight.base.snapshot_sha256,
        cache_clone_id=clone_id,
        post_hoc_lock_manifest_sha256=(
            _hash("post-hoc-lock") if arm is MatchedArm.POST_HOC else None
        ),
    )


def _schedule(count: int = 2) -> tuple[MatchedScheduleEntry, ...]:
    return tuple(
        MatchedScheduleEntry(
            unit_id=f"unit-{index}",
            morphology_program=f"program-{index}".encode(),
            program_index=index,
            particle_index=index,
            checkpoint_index=2,
            generator_checkpoint_sha256=_hash("generator"),
            closure_checkpoint_sha256=_hash("closure"),
            rollout_index=0,
            productive_generation_calls=1,
            route_reservation=RouteComputeUsage(1, 1, 0, 0),
        )
        for index in range(count)
    )


def _terminal(request: MatchedGenerationRequest) -> LockedMatchedTerminal:
    payload = f"terminal:{request.entry.unit_id}:{request.productive_seed}".encode()
    return LockedMatchedTerminal(
        unit_id=request.entry.unit_id,
        morphology_program_sha256=request.entry.morphology_program_sha256,
        checkpoint_index=request.entry.checkpoint_index,
        generator_checkpoint_sha256=request.entry.generator_checkpoint_sha256,
        closure_checkpoint_sha256=request.entry.closure_checkpoint_sha256,
        terminal_id=f"{request.arm.value}:{request.entry.unit_id}",
        terminal_locked=True,
        terminal_valid=True,
        exact_l1=True,
        terminal_bytes=payload,
        generation_trace_bytes=f"trace:{request.entry.unit_id}".encode(),
    )


def test_preflight_is_root_independent_empty_and_roundtrips(tmp_path: Path) -> None:
    first = _binding(tmp_path / "first")
    second = _binding(tmp_path / "second")

    assert first.preflight == second.preflight
    assert first.preflight.guided_before.entry_count == 0
    assert first.preflight.post_hoc_before.entry_count == 0
    assert first.preflight.preflight_sha256 == second.preflight.preflight_sha256
    assert (
        MatchedPlannerCacheBindingPreflight.from_dict(first.preflight.to_dict()) == first.preflight
    )


def test_lazy_binding_composes_with_runner_and_cached_planner(tmp_path: Path) -> None:
    context = _context()
    binding = _binding(tmp_path, context=context)
    planners: dict[MatchedArm, CachedRoutePlanner] = {}
    sources: dict[MatchedArm, FixedSource] = {}

    def assess(terminal, assessment_context):
        cache = binding.bind(assessment_context)
        if assessment_context.arm not in planners:
            source = FixedSource()
            sources[assessment_context.arm] = source
            planners[assessment_context.arm] = CachedRoutePlanner(
                RecursiveRouteAssessor(source),
                cache,
                context,
            )
        ledger = PlannerBudgetLedger(context.budget_limits)
        route_assessment = planners[assessment_context.arm].assess(
            RouteTarget("amine_head", "CCO"),
            ledger,
        )
        return DiagnosticRouteAssessment(
            value_policy_id="synthetic-structured-diagnostic-v1",
            diagnostic_fields=(("outcome", route_assessment.outcome.value),),
            usage=RouteComputeUsage(
                logical_planner_calls=ledger.logical_planner_calls,
                physical_planner_calls=ledger.physical_cache_misses,
                logical_verifier_calls=ledger.verifier_calls,
                physical_verifier_calls=ledger.verifier_calls,
            ),
        )

    run = run_diagnostic_matched_budget_arms(
        _schedule(),
        budget_limits=MatchedBudgetLimits(
            productive_generation_calls=2,
            terminal_completions=2,
            final_candidates=2,
            route=RouteComputeUsage(2, 2, 0, 0),
        ),
        base_seed=20260803,
        zero_guidance=True,
        cache_snapshot_sha256=binding.preflight.base.snapshot_sha256,
        generate_terminal=_terminal,
        assess_terminal=assess,
    )
    audit = binding.finalize()

    assert run.zero_guidance_bitwise_equivalent is True
    assert binding.guided_bound and binding.post_hoc_bound
    assert run.guided_cache_clone_id == audit.guided_clone_id
    assert run.post_hoc_cache_clone_id == audit.post_hoc_clone_id
    assert run.guided_ledger.route == run.post_hoc_ledger.route == RouteComputeUsage(2, 1, 0, 0)
    assert sources[MatchedArm.GUIDED].calls == sources[MatchedArm.POST_HOC].calls == 1
    assert audit.base_before == audit.base_after
    assert audit.guided_before == audit.post_hoc_before
    assert audit.guided_after == audit.post_hoc_after
    assert audit.guided_after.entry_count == 1
    with pytest.raises(PlannerCacheError, match="already finalized"):
        binding.bind(
            _assessment_context(
                binding,
                arm=MatchedArm.POST_HOC,
                clone_id=run.post_hoc_cache_clone_id,
            )
        )


def test_nonempty_immutable_base_is_shared_without_arm_overlay_writes(tmp_path: Path) -> None:
    context = _context()
    base = FilePlannerCache(tmp_path / "base")
    _put(base, context)
    binding = preflight_lazy_matched_planner_cache_binding(
        base,
        FilePlannerCache(tmp_path / "guided"),
        FilePlannerCache(tmp_path / "post_hoc"),
        context,
        assessment_at_utc=ASSESSMENT_AT_UTC,
    )
    sources: list[FixedSource] = []
    for arm, clone_id in (
        (MatchedArm.GUIDED, "guided-clone"),
        (MatchedArm.POST_HOC, "post-hoc-clone"),
    ):
        cache = binding.bind(_assessment_context(binding, arm=arm, clone_id=clone_id))
        source = FixedSource()
        sources.append(source)
        planner = CachedRoutePlanner(RecursiveRouteAssessor(source), cache, context)
        ledger = PlannerBudgetLedger(context.budget_limits)
        planner.assess(RouteTarget("amine_head", "CCO"), ledger)
        assert ledger.physical_cache_hits == 1
        assert ledger.physical_cache_misses == 0

    audit = binding.finalize()
    assert [source.calls for source in sources] == [0, 0]
    assert audit.base_before.entry_count == audit.base_after.entry_count == 1
    assert audit.guided_after.entry_count == audit.post_hoc_after.entry_count == 0


def test_preflight_rejects_nonempty_or_overlapping_overlay_roots(tmp_path: Path) -> None:
    context = _context()
    guided = FilePlannerCache(tmp_path / "guided")
    post_hoc = FilePlannerCache(tmp_path / "post_hoc")
    _put(guided, context)
    _put(post_hoc, context)

    with pytest.raises(PlannerCacheError, match="must be empty"):
        preflight_lazy_matched_planner_cache_binding(
            FilePlannerCache(tmp_path / "base"),
            guided,
            post_hoc,
            context,
            assessment_at_utc=ASSESSMENT_AT_UTC,
        )

    with pytest.raises(PlannerCacheError, match="must not overlap"):
        preflight_lazy_matched_planner_cache_binding(
            FilePlannerCache(tmp_path / "overlap-base"),
            FilePlannerCache(tmp_path / "overlap-base" / "guided"),
            FilePlannerCache(tmp_path / "other"),
            context,
            assessment_at_utc=ASSESSMENT_AT_UTC,
        )


def test_mutation_between_preflight_and_each_lazy_bind_fails_closed(tmp_path: Path) -> None:
    context = _context()
    guided_root = FilePlannerCache(tmp_path / "guided")
    binding = preflight_lazy_matched_planner_cache_binding(
        FilePlannerCache(tmp_path / "base"),
        guided_root,
        FilePlannerCache(tmp_path / "post_hoc"),
        context,
        assessment_at_utc=ASSESSMENT_AT_UTC,
    )
    _put(guided_root, context)
    with pytest.raises(PlannerCacheError, match="guided overlay root changed"):
        binding.bind(_assessment_context(binding, arm=MatchedArm.GUIDED, clone_id="guided-clone"))

    post_hoc_root = FilePlannerCache(tmp_path / "second" / "post_hoc")
    binding = preflight_lazy_matched_planner_cache_binding(
        FilePlannerCache(tmp_path / "second" / "base"),
        FilePlannerCache(tmp_path / "second" / "guided"),
        post_hoc_root,
        context,
        assessment_at_utc=ASSESSMENT_AT_UTC,
    )
    binding.bind(_assessment_context(binding, arm=MatchedArm.GUIDED, clone_id="guided-clone"))
    _put(post_hoc_root, context)
    with pytest.raises(PlannerCacheError, match="post_hoc overlay root changed"):
        binding.bind(
            _assessment_context(binding, arm=MatchedArm.POST_HOC, clone_id="post-hoc-clone")
        )


def test_base_mutation_and_wrong_snapshot_fail_before_cache_binding(tmp_path: Path) -> None:
    context = _context()
    base = FilePlannerCache(tmp_path / "base")
    binding = preflight_lazy_matched_planner_cache_binding(
        base,
        FilePlannerCache(tmp_path / "guided"),
        FilePlannerCache(tmp_path / "post_hoc"),
        context,
        assessment_at_utc=ASSESSMENT_AT_UTC,
    )

    with pytest.raises(PlannerCacheError, match="runner cache snapshot differs"):
        binding.bind(
            _assessment_context(
                binding,
                arm=MatchedArm.GUIDED,
                clone_id="guided-clone",
                snapshot_sha256=_hash("wrong-snapshot"),
            )
        )
    assert not binding.guided_bound

    _put(base, context)
    with pytest.raises(PlannerCacheError, match="base cache changed"):
        binding.bind(_assessment_context(binding, arm=MatchedArm.GUIDED, clone_id="guided-clone"))
    assert not binding.guided_bound


def test_cache_root_retargeting_after_preflight_fails_closed(tmp_path: Path) -> None:
    context = _context()
    guided = FilePlannerCache(tmp_path / "guided")
    binding = preflight_lazy_matched_planner_cache_binding(
        FilePlannerCache(tmp_path / "base"),
        guided,
        FilePlannerCache(tmp_path / "post_hoc"),
        context,
        assessment_at_utc=ASSESSMENT_AT_UTC,
    )
    guided.root = tmp_path / "replacement-guided"

    with pytest.raises(PlannerCacheError, match="guided cache root changed"):
        binding.bind(_assessment_context(binding, arm=MatchedArm.GUIDED, clone_id="guided-clone"))
    assert not binding.guided_bound


def test_arm_order_and_clone_id_reuse_fail_closed(tmp_path: Path) -> None:
    binding = _binding(tmp_path)
    with pytest.raises(PlannerCacheError, match="before the guided"):
        binding.bind(
            _assessment_context(binding, arm=MatchedArm.POST_HOC, clone_id="post-hoc-clone")
        )

    guided = _assessment_context(binding, arm=MatchedArm.GUIDED, clone_id="guided-clone")
    assert binding.bind(guided).clone_id == "guided-clone"
    assert binding.bind(guided).clone_id == "guided-clone"
    with pytest.raises(PlannerCacheError, match="changed its cache clone ID"):
        binding.bind(replace(guided, cache_clone_id="different-guided-clone"))
    with pytest.raises(PlannerCacheError, match="reused one cache clone ID"):
        binding.bind(_assessment_context(binding, arm=MatchedArm.POST_HOC, clone_id="guided-clone"))

    post_hoc = _assessment_context(
        binding,
        arm=MatchedArm.POST_HOC,
        clone_id="post-hoc-clone",
    )
    assert binding.bind(post_hoc).clone_id == "post-hoc-clone"
    with pytest.raises(PlannerCacheError, match="post-hoc runner callback changed"):
        binding.bind(replace(post_hoc, cache_clone_id="different-post-hoc-clone"))
    with pytest.raises(PlannerCacheError, match="guided cache callback occurred"):
        binding.bind(guided)


def test_post_hoc_phase_freezes_guided_overlay_and_finalize_requires_both(tmp_path: Path) -> None:
    context = _context()
    guided_file = FilePlannerCache(tmp_path / "guided")
    binding = preflight_lazy_matched_planner_cache_binding(
        FilePlannerCache(tmp_path / "base"),
        guided_file,
        FilePlannerCache(tmp_path / "post_hoc"),
        context,
        assessment_at_utc=ASSESSMENT_AT_UTC,
    )
    binding.bind(_assessment_context(binding, arm=MatchedArm.GUIDED, clone_id="guided-clone"))
    with pytest.raises(PlannerCacheError, match="both matched cache arms"):
        binding.finalize()

    binding.bind(_assessment_context(binding, arm=MatchedArm.POST_HOC, clone_id="post-hoc-clone"))
    _put(guided_file, context)
    with pytest.raises(PlannerCacheError, match="guided overlay changed"):
        binding.finalize()


def test_expired_l3_context_fails_during_preflight(tmp_path: Path) -> None:
    expired = _context(l3_expires_at_utc="2026-08-15T00:00:00Z")
    with pytest.raises(PlannerCacheError, match="expired"):
        _binding(tmp_path, context=expired)
