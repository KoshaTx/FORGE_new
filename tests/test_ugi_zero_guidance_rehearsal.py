from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from forge.product.ugi_held_component_gate import load_ugi_reaction_contract
from forge.product.ugi_matched_budget_orchestration import (
    LockedMatchedTerminal,
    MatchedArm,
    MatchedBudgetLimits,
    MatchedGenerationRequest,
    MatchedScheduleEntry,
    RouteComputeUsage,
)
from forge.product.ugi_zero_guidance_rehearsal import (
    HASH_PINNED_MATCHED_RUNNER_SHA256,
    QualifiedRoutePlannerFactoryAdapter,
    RestartableGeneratorClosureAdapter,
    RestartableGeneratorClosureIdentity,
    UgiZeroGuidanceRehearsalError,
    ZeroGuidanceRehearsalContract,
    current_terminal_route_assessment_source_sha256,
    run_zero_guidance_route_rehearsal,
)
from forge.route.planner import (
    AvailabilityState,
    EvidenceRecord,
    EvidenceTier,
    ForwardVerificationState,
    KnowledgeDisposition,
    KnowledgeResult,
    PlannerBudgetLimits,
    RecursiveRouteAssessor,
    RouteTarget,
)
from forge.route.planner_cache import (
    CachedRoutePlanner,
    FilePlannerCache,
    PlannerCacheContext,
)
from forge.route.planner_cache_snapshot import (
    build_file_planner_cache_snapshot_manifest,
    planner_cache_context_sha256,
)
from forge.route.terminal_assessment import (
    DEFAULT_IDENTITY_POLICY,
    DEFAULT_STEREOCHEMISTRY_POLICY,
    QualifiedUgiL1Reverifier,
    ValidatedUgiTerminalPayload,
    required_three_role_route_reservation,
)

REPO = Path(__file__).resolve().parents[1]
ASSESSMENT_AT_UTC = "2026-08-15T00:00:00Z"
COMPONENTS = {
    "amine_head": "CN",
    "oxoester_aldehyde_body_tail": "CC=O",
    "isocyanide_tail": "[C-]#[N+]C",
}
PRODUCT = "CNC(=O)C(C)NC"


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _planner_limits() -> PlannerBudgetLimits:
    return PlannerBudgetLimits(
        maximum_depth=3,
        maximum_logical_planner_calls=1,
        maximum_expansions=4,
        maximum_product_candidates=4,
        maximum_verifier_calls=4,
        maximum_elapsed_milliseconds=1_000,
    )


def _context() -> PlannerCacheContext:
    return PlannerCacheContext(
        planner_id="synthetic-qualified-route-source-v1",
        planner_sha256=_hash("planner"),
        search_policy_sha256=_hash("search"),
        value_policy_sha256=_hash("structured-nonscalar-value"),
        l1_reaction_sha256=_hash("l1-reaction"),
        upstream_reaction_registry_sha256=_hash("upstream"),
        variant_registry_sha256=_hash("variants"),
        verifier_sha256=_hash("verifier"),
        l3_snapshot_sha256=_hash("l3"),
        l3_region="US",
        l3_accessed_at_utc="2026-08-01T00:00:00Z",
        l3_expires_at_utc="2026-08-31T00:00:00Z",
        software_versions=(("rdkit", "test"), ("route_schema", "v1")),
        identity_policy=DEFAULT_IDENTITY_POLICY,
        stereochemistry_policy=DEFAULT_STEREOCHEMISTRY_POLICY,
        budget_limits=_planner_limits(),
    )


def _reaction():
    return load_ugi_reaction_contract(REPO / "data/vendor/qualified_reactions_v1.json")


def _payload() -> ValidatedUgiTerminalPayload:
    return ValidatedUgiTerminalPayload.from_recovered_components(
        product_smiles=PRODUCT,
        components_by_role=COMPONENTS,
        l1_reaction=_reaction(),
        l1_reaction_sha256=_hash("l1-reaction"),
        component_recovery_contract_sha256=_hash("component-recovery"),
    )


def _identity() -> RestartableGeneratorClosureIdentity:
    return RestartableGeneratorClosureIdentity(
        generator_checkpoint_sha256=_hash("selected-generator"),
        closure_checkpoint_sha256=_hash("selected-closure"),
        production_generator_manifest_sha256=_hash("production-manifest"),
        restartable_equivalence_receipt_sha256=_hash("restartable-equivalence"),
        generator_implementation_sha256=_hash("generator-implementation"),
        terminal_decoder_id="bond-stochastic-v1",
    )


def _schedule(
    context: PlannerCacheContext,
    *,
    count: int = 2,
    identity: RestartableGeneratorClosureIdentity | None = None,
) -> tuple[MatchedScheduleEntry, ...]:
    selected = identity or _identity()
    reservation = required_three_role_route_reservation(context.budget_limits)
    return tuple(
        MatchedScheduleEntry(
            unit_id=f"unit-{index}",
            morphology_program=f"sealed-program-{index}".encode(),
            program_index=index,
            particle_index=index,
            checkpoint_index=2,
            generator_checkpoint_sha256=selected.generator_checkpoint_sha256,
            closure_checkpoint_sha256=selected.closure_checkpoint_sha256,
            rollout_index=0,
            productive_generation_calls=1,
            route_reservation=reservation,
        )
        for index in range(count)
    )


def _budget(schedule: tuple[MatchedScheduleEntry, ...]) -> MatchedBudgetLimits:
    route = RouteComputeUsage()
    for entry in schedule:
        route = route.plus(entry.route_reservation)
    return MatchedBudgetLimits(
        productive_generation_calls=len(schedule),
        terminal_completions=len(schedule),
        final_candidates=len(schedule),
        route=route,
    )


def _contract(
    context: PlannerCacheContext,
    base: FilePlannerCache,
    schedule: tuple[MatchedScheduleEntry, ...],
    *,
    identity: RestartableGeneratorClosureIdentity | None = None,
    **overrides: object,
) -> ZeroGuidanceRehearsalContract:
    selected = identity or _identity()
    manifest = build_file_planner_cache_snapshot_manifest(
        base,
        context,
        assessment_at_utc=ASSESSMENT_AT_UTC,
    )
    values: dict[str, object] = {
        "base_seed": 20260803,
        "guidance_strength": 0.0,
        "assessment_at_utc": ASSESSMENT_AT_UTC,
        "budget_limits": _budget(schedule),
        "generator_identity": selected,
        "planner_context_sha256": planner_cache_context_sha256(context),
        "expected_cache_snapshot_sha256": manifest.snapshot_sha256,
        "terminal_route_assessment_source_sha256": (
            current_terminal_route_assessment_source_sha256()
        ),
        "route_qualification_sha256": _hash("synthetic-route-qualification"),
        "matched_runner_source_sha256": HASH_PINNED_MATCHED_RUNNER_SHA256,
    }
    values.update(overrides)
    return ZeroGuidanceRehearsalContract(**values)  # type: ignore[arg-type]


@dataclass
class FixedQualifiedSource:
    outcome: KnowledgeDisposition = KnowledgeDisposition.TERMINAL
    calls: int = 0

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        self.calls += 1
        if self.outcome is KnowledgeDisposition.MISSING_KNOWLEDGE:
            return KnowledgeResult(
                disposition=self.outcome,
                evidence=(),
                detail="synthetic missing route knowledge",
            )
        return KnowledgeResult(
            disposition=KnowledgeDisposition.TERMINAL,
            evidence=(
                EvidenceRecord(
                    evidence_id=f"terminal-{target.role}",
                    tier=EvidenceTier.ACCEPTED_TERMINAL,
                    source_sha256=_hash(f"source-{target.role}"),
                    source_locator=f"synthetic:{target.role}",
                    exact_substrate=True,
                    forward_verification=ForwardVerificationState.NOT_APPLICABLE,
                    availability=AvailabilityState.CURRENT_CLOSED,
                ),
            ),
            detail="synthetic current terminal material",
        )


@dataclass
class RecordingGenerator:
    payload: ValidatedUgiTerminalPayload
    calls: list[MatchedGenerationRequest]
    roots: tuple[Path, Path]

    def __call__(self, request: MatchedGenerationRequest) -> LockedMatchedTerminal:
        if not self.calls:
            assert all(not any(root.rglob("*.json")) for root in self.roots)
        self.calls.append(request)
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
            terminal_bytes=self.payload.canonical_bytes,
            generation_trace_bytes=(
                f"restartable-trace:{request.entry.unit_id}:{request.productive_seed}\n".encode()
            ),
            payload=self.payload,
        )


def _run_inputs(tmp_path: Path):
    context = _context()
    base = FilePlannerCache(tmp_path / "base")
    guided = FilePlannerCache(tmp_path / "guided")
    post_hoc = FilePlannerCache(tmp_path / "post_hoc")
    schedule = _schedule(context)
    contract = _contract(context, base, schedule)
    generated = RecordingGenerator(_payload(), [], (guided.root, post_hoc.root))
    adapter = RestartableGeneratorClosureAdapter(_identity(), generated)
    sources: dict[MatchedArm, list[FixedQualifiedSource]] = {
        MatchedArm.GUIDED: [],
        MatchedArm.POST_HOC: [],
    }

    def planner_factory(cache, planner_context, assessment_context, terminal):
        assert terminal.unit_id in {entry.unit_id for entry in schedule}
        source = FixedQualifiedSource()
        sources[assessment_context.arm].append(source)
        return CachedRoutePlanner(
            RecursiveRouteAssessor(source),
            cache,
            planner_context,
        )

    return (
        context,
        base,
        guided,
        post_hoc,
        schedule,
        contract,
        generated,
        adapter,
        sources,
        QualifiedRoutePlannerFactoryAdapter(
            _hash("synthetic-route-qualification"),
            planner_factory,
        ),
    )


def test_real_zero_guidance_rehearsal_composes_all_typed_seams(tmp_path: Path) -> None:
    (
        context,
        base,
        guided,
        post_hoc,
        schedule,
        contract,
        generated,
        adapter,
        sources,
        planner_factory,
    ) = _run_inputs(tmp_path)

    result = run_zero_guidance_route_rehearsal(
        schedule,
        contract=contract,
        generator=adapter,
        planner_context=context,
        base_cache=base,
        guided_overlay=guided,
        post_hoc_overlay=post_hoc,
        l1_reverifier=QualifiedUgiL1Reverifier(
            _reaction(),
            _hash("l1-reaction"),
        ),
        planner_factory=planner_factory,
    )

    assert len(generated.calls) == 2 * len(schedule)
    assert [request.productive_seed for request in generated.calls[::2]] == [
        request.productive_seed for request in generated.calls[1::2]
    ]
    assert result.matched_run.zero_guidance_bitwise_equivalent is True
    assert result.matched_run.guided_ledger == result.matched_run.post_hoc_ledger
    assert len(result.guided_route_receipts) == len(schedule)
    assert len(result.post_hoc_route_receipts) == len(schedule)
    assert all(item.chemistry_projection_identical is True for item in result.chemistry_projections)
    assert result.cache_audit.base_before == result.cache_audit.base_after
    assert result.cache_audit.guided_before == result.cache_audit.post_hoc_before
    assert result.cache_audit.guided_after == result.cache_audit.post_hoc_after
    assert result.cache_audit.guided_after.entry_count == 3
    assert [sum(source.calls for source in arm) for arm in sources.values()] == [3, 3]
    serialized = result.to_dict()
    assert serialized["chemistry_projection_identity_proven"] is True
    assert serialized["scope"] == {
        "guidance_strength": 0,
        "synthesis_scalar_defined": False,
        "nonzero_synthesis_guidance": False,
        "biological_guidance": False,
        "candidate_selection": False,
        "private_holdout_accessed": False,
    }
    assert json.loads(result.canonical_bytes)["result_sha256"] == result.result_sha256


@pytest.mark.parametrize("guidance_strength", [0.01, -0.1, float("nan"), True])
def test_nonzero_or_malformed_guidance_is_rejected(guidance_strength: object) -> None:
    with pytest.raises(UgiZeroGuidanceRehearsalError, match="guidance_strength=0 only"):
        ZeroGuidanceRehearsalContract(
            base_seed=1,
            guidance_strength=guidance_strength,  # type: ignore[arg-type]
            assessment_at_utc=ASSESSMENT_AT_UTC,
            budget_limits=MatchedBudgetLimits(1, 1, 1, RouteComputeUsage()),
            generator_identity=_identity(),
            planner_context_sha256=_hash("context"),
            expected_cache_snapshot_sha256=_hash("cache"),
            terminal_route_assessment_source_sha256=_hash("assessment"),
            route_qualification_sha256=_hash("qualification"),
        )


@pytest.mark.parametrize(
    "drift",
    [
        "adapter",
        "generator_checkpoint",
        "closure_checkpoint",
        "planner_context",
        "cache",
        "route_qualification",
        "terminal_assessment",
    ],
)
def test_hash_drift_fails_before_productive_generation(tmp_path: Path, drift: str) -> None:
    (
        context,
        base,
        guided,
        post_hoc,
        schedule,
        contract,
        generated,
        adapter,
        _,
        planner_factory,
    ) = _run_inputs(tmp_path)
    if drift == "adapter":
        adapter = RestartableGeneratorClosureAdapter(
            replace(_identity(), terminal_decoder_id="different-decoder"),
            generated,
        )
    elif drift == "generator_checkpoint":
        schedule = (replace(schedule[0], generator_checkpoint_sha256=_hash("wrong")), *schedule[1:])
    elif drift == "closure_checkpoint":
        schedule = (replace(schedule[0], closure_checkpoint_sha256=_hash("wrong")), *schedule[1:])
    elif drift == "planner_context":
        contract = replace(contract, planner_context_sha256=_hash("wrong"))
    elif drift == "cache":
        contract = replace(contract, expected_cache_snapshot_sha256=_hash("wrong"))
    elif drift == "route_qualification":
        contract = replace(contract, route_qualification_sha256=_hash("wrong"))
    else:
        contract = replace(
            contract,
            terminal_route_assessment_source_sha256=_hash("wrong"),
        )

    with pytest.raises(UgiZeroGuidanceRehearsalError):
        run_zero_guidance_route_rehearsal(
            schedule,
            contract=contract,
            generator=adapter,
            planner_context=context,
            base_cache=base,
            guided_overlay=guided,
            post_hoc_overlay=post_hoc,
            l1_reverifier=QualifiedUgiL1Reverifier(_reaction(), _hash("l1-reaction")),
            planner_factory=planner_factory,
        )
    assert not generated.calls


def test_exact_uncensored_budget_drift_fails_before_generation(tmp_path: Path) -> None:
    (
        context,
        base,
        guided,
        post_hoc,
        schedule,
        contract,
        generated,
        adapter,
        _,
        planner_factory,
    ) = _run_inputs(tmp_path)
    drifted = replace(
        contract,
        budget_limits=replace(
            contract.budget_limits,
            productive_generation_calls=contract.budget_limits.productive_generation_calls + 1,
        ),
    )

    with pytest.raises(UgiZeroGuidanceRehearsalError, match="exact uncensored"):
        run_zero_guidance_route_rehearsal(
            schedule,
            contract=drifted,
            generator=adapter,
            planner_context=context,
            base_cache=base,
            guided_overlay=guided,
            post_hoc_overlay=post_hoc,
            l1_reverifier=QualifiedUgiL1Reverifier(_reaction(), _hash("l1-reaction")),
            planner_factory=planner_factory,
        )
    assert not generated.calls


def test_terminal_or_trace_divergence_fails_closed(tmp_path: Path) -> None:
    (
        context,
        base,
        guided,
        post_hoc,
        schedule,
        contract,
        _,
        _,
        _,
        planner_factory,
    ) = _run_inputs(tmp_path)
    payload = _payload()

    def divergent(request: MatchedGenerationRequest) -> LockedMatchedTerminal:
        trace = b"same-trace\n"
        if request.arm is MatchedArm.POST_HOC:
            trace = b"different-post-hoc-trace\n"
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
            terminal_bytes=payload.canonical_bytes,
            generation_trace_bytes=trace,
            payload=payload,
        )

    with pytest.raises(RuntimeError, match="zero-guidance terminal/trace bytes diverged"):
        run_zero_guidance_route_rehearsal(
            schedule,
            contract=contract,
            generator=RestartableGeneratorClosureAdapter(_identity(), divergent),
            planner_context=context,
            base_cache=base,
            guided_overlay=guided,
            post_hoc_overlay=post_hoc,
            l1_reverifier=QualifiedUgiL1Reverifier(_reaction(), _hash("l1-reaction")),
            planner_factory=planner_factory,
        )


def test_arm_specific_route_qualification_cannot_masquerade_as_parity(
    tmp_path: Path,
) -> None:
    (
        context,
        base,
        guided,
        post_hoc,
        schedule,
        contract,
        _,
        adapter,
        _,
        _,
    ) = _run_inputs(tmp_path)

    def asymmetric_factory(cache, planner_context, assessment_context, terminal):
        assert terminal.unit_id in {entry.unit_id for entry in schedule}
        outcome = (
            KnowledgeDisposition.TERMINAL
            if assessment_context.arm is MatchedArm.GUIDED
            else KnowledgeDisposition.MISSING_KNOWLEDGE
        )
        return CachedRoutePlanner(
            RecursiveRouteAssessor(FixedQualifiedSource(outcome)),
            cache,
            planner_context,
        )

    with pytest.raises(UgiZeroGuidanceRehearsalError, match="structured route values differ"):
        run_zero_guidance_route_rehearsal(
            schedule,
            contract=contract,
            generator=adapter,
            planner_context=context,
            base_cache=base,
            guided_overlay=guided,
            post_hoc_overlay=post_hoc,
            l1_reverifier=QualifiedUgiL1Reverifier(_reaction(), _hash("l1-reaction")),
            planner_factory=QualifiedRoutePlannerFactoryAdapter(
                _hash("synthetic-route-qualification"),
                asymmetric_factory,
            ),
        )
