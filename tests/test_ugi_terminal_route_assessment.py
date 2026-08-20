from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from rdkit import Chem

from forge.corpus.ugi_held_component_gate import load_ugi_reaction_contract
from forge.model.ugi_adapter_features import CORE_POSITION_TO_INDEX, ORIGIN_TO_INDEX
from forge.synthesis.engine.planner import (
    AssessmentOutcome,
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
    SynthesisAssessment,
    budget_exhausted_assessment,
)
from forge.synthesis.engine.planner_cache import (
    CachedRoutePlanner,
    FilePlannerCache,
    PlannerCacheContext,
)
from forge.synthesis.matched import (
    LockedMatchedTerminal,
    MatchedArm,
    MatchedAssessmentContext,
    RouteComputeUsage,
)
from forge.synthesis.terminals.terminal_assessment import (
    DEFAULT_IDENTITY_POLICY,
    DEFAULT_STEREOCHEMISTRY_POLICY,
    ExactL1ForwardVerification,
    QualifiedUgiL1Reverifier,
    UgiRoleComponent,
    UgiTerminalRouteAssessmentError,
    UgiTerminalRouteAssessmentReceipt,
    ValidatedUgiTerminalPayload,
    assess_locked_ugi_terminal_routes,
    required_three_role_route_reservation,
)

REPO = Path(__file__).resolve().parents[1]
ROLE_NAMES = (
    "amine_head",
    "oxoester_aldehyde_body_tail",
    "isocyanide_tail",
)
COMPONENTS = {
    "amine_head": "CN",
    "oxoester_aldehyde_body_tail": "CC=O",
    "isocyanide_tail": "[C-]#[N+]C",
}
PRODUCT = "CNC(=O)C(C)NC"
ASSESSMENT_AT = "2026-08-15T00:00:00Z"


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _reaction():
    return load_ugi_reaction_contract(REPO / "data/vendor/qualified_reactions_v1.json")


def _limits(**overrides: int) -> PlannerBudgetLimits:
    values = {
        "maximum_depth": 3,
        "maximum_logical_planner_calls": 1,
        "maximum_expansions": 4,
        "maximum_product_candidates": 4,
        "maximum_verifier_calls": 4,
        "maximum_elapsed_milliseconds": 1_000,
    }
    values.update(overrides)
    return PlannerBudgetLimits(**values)


def _payload() -> ValidatedUgiTerminalPayload:
    return ValidatedUgiTerminalPayload.from_recovered_components(
        product_smiles=PRODUCT,
        components_by_role=COMPONENTS,
        l1_reaction=_reaction(),
        l1_reaction_sha256=_hash("l1-reaction"),
        component_recovery_contract_sha256=_hash("component-recovery"),
    )


def _reverifier() -> QualifiedUgiL1Reverifier:
    return QualifiedUgiL1Reverifier(
        reaction_contract=_reaction(),
        l1_reaction_sha256=_hash("l1-reaction"),
    )


def _context(
    *,
    limits: PlannerBudgetLimits | None = None,
    l1_reaction_sha256: str | None = None,
) -> PlannerCacheContext:
    return PlannerCacheContext(
        planner_id="exact-evidence-lookup-v1",
        planner_sha256=_hash("planner"),
        search_policy_sha256=_hash("search"),
        value_policy_sha256=_hash("structured-value"),
        l1_reaction_sha256=l1_reaction_sha256 or _hash("l1-reaction"),
        upstream_reaction_registry_sha256=_hash("upstream-reactions"),
        variant_registry_sha256=_hash("variants"),
        verifier_sha256=_hash("verifier"),
        l3_snapshot_sha256=_hash("l3-snapshot"),
        l3_region="US",
        l3_accessed_at_utc="2026-08-01T00:00:00Z",
        l3_expires_at_utc="2026-08-31T00:00:00Z",
        software_versions=(("rdkit", "test"), ("route_schema", "v1")),
        identity_policy=DEFAULT_IDENTITY_POLICY,
        stereochemistry_policy=DEFAULT_STEREOCHEMISTRY_POLICY,
        budget_limits=limits or _limits(),
    )


def _terminal(
    payload: ValidatedUgiTerminalPayload,
    **overrides: object,
) -> LockedMatchedTerminal:
    values: dict[str, object] = {
        "unit_id": "unit-7",
        "morphology_program_sha256": _hash("morphology"),
        "checkpoint_index": 7,
        "generator_checkpoint_sha256": _hash("generator"),
        "closure_checkpoint_sha256": _hash("closure"),
        "terminal_id": "terminal-7",
        "terminal_locked": True,
        "terminal_valid": True,
        "exact_l1": True,
        "terminal_bytes": payload.canonical_bytes,
        "generation_trace_bytes": b"sealed-generation-trace-v1\n",
        "payload": payload,
    }
    values.update(overrides)
    return LockedMatchedTerminal(**values)  # type: ignore[arg-type]


def _assessment_context(
    context: PlannerCacheContext,
    *,
    arm: MatchedArm = MatchedArm.GUIDED,
    unit_reservation: RouteComputeUsage | None = None,
    remaining_budget: RouteComputeUsage | None = None,
    post_hoc_lock_manifest_sha256: str | None = None,
) -> MatchedAssessmentContext:
    required = required_three_role_route_reservation(context.budget_limits)
    return MatchedAssessmentContext(
        arm=arm,
        route_seed=20260803,
        remaining_budget=remaining_budget or required,
        unit_reservation=unit_reservation or required,
        cache_snapshot_sha256=_hash("cache-snapshot"),
        cache_clone_id=f"{arm.value}-clone-0",
        post_hoc_lock_manifest_sha256=post_hoc_lock_manifest_sha256,
    )


def _terminal_evidence(role: str) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"terminal-{role}",
        tier=EvidenceTier.ACCEPTED_TERMINAL,
        source_sha256=_hash(f"source-{role}"),
        source_locator=f"test:{role}",
        exact_substrate=True,
        forward_verification=ForwardVerificationState.NOT_APPLICABLE,
        availability=AvailabilityState.CURRENT_CLOSED,
    )


@dataclass
class TerminalSource:
    calls: list[RouteTarget]

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        self.calls.append(target)
        evidence = _terminal_evidence(target.role)
        return KnowledgeResult(
            disposition=KnowledgeDisposition.TERMINAL,
            evidence=(evidence,),
            detail="accepted current terminal material",
        )


@dataclass
class RecordingPlanner:
    delegate: object
    targets: list[RouteTarget]
    ledgers: list[PlannerBudgetLedger]

    def assess(
        self,
        target: RouteTarget,
        budget: PlannerBudgetLedger,
    ) -> SynthesisAssessment:
        self.targets.append(target)
        self.ledgers.append(budget)
        assert budget.logical_planner_calls == 0
        assert budget.physical_cache_hits == 0
        assert budget.physical_cache_misses == 0
        return self.delegate.assess(target, budget)  # type: ignore[union-attr]


class OneRoleBudgetExhaustionPlanner:
    def __init__(self) -> None:
        self.targets: list[RouteTarget] = []
        self.ledgers: list[PlannerBudgetLedger] = []
        self.source = TerminalSource([])

    def assess(
        self,
        target: RouteTarget,
        budget: PlannerBudgetLedger,
    ) -> SynthesisAssessment:
        self.targets.append(target)
        self.ledgers.append(budget)
        assert budget.consume_logical_planner_call()
        budget.record_cache_miss()
        if target.role == "amine_head":
            assert not budget.consume_expansion()
            return budget_exhausted_assessment(target, "maximum_expansions")
        return RecursiveRouteAssessor(self.source).assess_uncached(target, budget)


def _cached_recording_planner(
    tmp_path: Path,
    context: PlannerCacheContext,
) -> tuple[RecordingPlanner, TerminalSource]:
    source = TerminalSource([])
    cached = CachedRoutePlanner(
        RecursiveRouteAssessor(source),
        FilePlannerCache(tmp_path / "cache"),
        context,
    )
    return RecordingPlanner(cached, [], []), source


def test_validated_payload_has_exact_order_and_canonical_bytes() -> None:
    payload = _payload()

    assert tuple(component.role for component in payload.components) == ROLE_NAMES
    assert payload.by_role() == {
        role: payload.components[index] for index, role in enumerate(ROLE_NAMES)
    }
    assert payload == ValidatedUgiTerminalPayload.from_bytes(payload.canonical_bytes)
    assert payload.sha256 == hashlib.sha256(payload.canonical_bytes).hexdigest()

    repeated = _payload()
    assert repeated.canonical_bytes == payload.canonical_bytes
    assert repeated.sha256 == payload.sha256


def test_payload_recovers_exact_components_from_product_semantics() -> None:
    product = Chem.MolFromSmiles(
        "[CH3:10][CH2:11][NH:4][C:3](=[O:5])[CH:2]([CH3:20])[NH:1][CH2:30][CH3:31]"
    )
    assert product is not None
    origin_by_map = {
        1: "amine_head",
        30: "amine_head",
        31: "amine_head",
        2: "oxoester_aldehyde_body_tail",
        20: "oxoester_aldehyde_body_tail",
        3: "isocyanide_tail",
        4: "isocyanide_tail",
        10: "isocyanide_tail",
        11: "isocyanide_tail",
        5: "assembly_introduced",
    }
    origins: list[int] = []
    positions: list[int] = []
    for atom in product.GetAtoms():
        map_number = atom.GetAtomMapNum()
        origins.append(ORIGIN_TO_INDEX[origin_by_map[map_number]])
        if map_number in {1, 2, 3, 4}:
            position = f"map_{map_number}"
        elif map_number == 5:
            position = "template_introduced_0"
        else:
            position = "not_core"
        positions.append(CORE_POSITION_TO_INDEX[position])
        atom.SetAtomMapNum(0)

    payload = ValidatedUgiTerminalPayload.from_product_semantics(
        product=product,
        origin_states=origins,
        core_position_states=positions,
        l1_reaction=_reaction(),
        l1_reaction_sha256=_hash("l1-reaction"),
        component_recovery_contract_sha256=_hash("component-recovery"),
    )

    assert {role: component.canonical_smiles for role, component in payload.by_role().items()} == {
        "amine_head": "CCN",
        "oxoester_aldehyde_body_tail": "CC=O",
        "isocyanide_tail": "[C-]#[N+]CC",
    }
    assert payload.l1_forward_verification.exact_product_reconstructed


def test_payload_rejects_role_reordering_and_wrong_product() -> None:
    payload = _payload()
    reordered = payload.to_dict()
    reordered["components"] = list(reversed(reordered["components"]))
    with pytest.raises(UgiTerminalRouteAssessmentError, match="frozen order"):
        ValidatedUgiTerminalPayload.from_dict(reordered)

    with pytest.raises(UgiTerminalRouteAssessmentError, match="do not exactly reconstruct"):
        ValidatedUgiTerminalPayload.from_recovered_components(
            product_smiles="CC",
            components_by_role=COMPONENTS,
            l1_reaction=_reaction(),
            l1_reaction_sha256=_hash("l1-reaction"),
            component_recovery_contract_sha256=_hash("component-recovery"),
        )


def test_route_seam_reverifies_forged_exact_l1_payload_before_planner_calls(
    tmp_path: Path,
) -> None:
    valid = _payload()
    forged = ValidatedUgiTerminalPayload(
        product_smiles="CC",
        components=tuple(
            UgiRoleComponent(
                role=component.role,
                canonical_smiles=component.canonical_smiles,
            )
            for component in valid.components
        ),
        l1_forward_verification=ExactL1ForwardVerification(
            exact_product_reconstructed=True,
            maximum_outcomes=(valid.l1_forward_verification.maximum_outcomes),
            maximum_outcomes_saturated=(valid.l1_forward_verification.maximum_outcomes_saturated),
            outcome_count=valid.l1_forward_verification.outcome_count,
        ),
        l1_reaction_sha256=valid.l1_reaction_sha256,
        component_recovery_contract_sha256=(valid.component_recovery_contract_sha256),
    )
    forged = ValidatedUgiTerminalPayload.from_bytes(forged.canonical_bytes)
    context = _context()
    planner, _ = _cached_recording_planner(tmp_path, context)

    with pytest.raises(UgiTerminalRouteAssessmentError, match="did not reconstruct"):
        assess_locked_ugi_terminal_routes(
            _terminal(forged),
            l1_reverifier=_reverifier(),
            planner=planner,
            planner_context=context,
            assessment_context=_assessment_context(context),
            assessment_at_utc=ASSESSMENT_AT,
        )
    assert not planner.targets


def test_route_seam_assesses_exactly_three_roles_with_isolated_ledgers(
    tmp_path: Path,
) -> None:
    payload = _payload()
    context = _context()
    planner, source = _cached_recording_planner(tmp_path, context)

    receipt = assess_locked_ugi_terminal_routes(
        _terminal(payload),
        l1_reverifier=_reverifier(),
        planner=planner,
        planner_context=context,
        assessment_context=_assessment_context(context),
        assessment_at_utc=ASSESSMENT_AT,
    )

    assert tuple(target.role for target in planner.targets) == ROLE_NAMES
    assert tuple(target.role for target in source.calls) == ROLE_NAMES
    assert len({id(ledger) for ledger in planner.ledgers}) == 3
    assert all(ledger.logical_planner_calls == 1 for ledger in planner.ledgers)
    assert all(ledger.physical_cache_misses == 1 for ledger in planner.ledgers)
    assert receipt.product_value.route_complete
    assert receipt.realized_route_usage == RouteComputeUsage(3, 3, 1, 1)
    assert all(not item.target.product_context_smiles for item in receipt.role_assessments)
    assert all(item.cache_key_payload_json for item in receipt.role_assessments)
    assert receipt == UgiTerminalRouteAssessmentReceipt.from_bytes(receipt.canonical_bytes)
    assert receipt.to_dict()["scalar_value"] is None
    assert receipt.to_dict()["success_probability"] is None
    assert receipt.to_dict()["candidate_selected"] is False
    assert receipt.to_dict()["biological_guidance"] is False

    cached_receipt = assess_locked_ugi_terminal_routes(
        _terminal(payload),
        l1_reverifier=_reverifier(),
        planner=planner,
        planner_context=context,
        assessment_context=_assessment_context(context),
        assessment_at_utc=ASSESSMENT_AT,
    )
    assert cached_receipt.realized_route_usage == RouteComputeUsage(3, 0, 1, 1)
    assert all(
        item.budget_after.physical_cache_hits == 1 for item in cached_receipt.role_assessments
    )


@pytest.mark.parametrize(
    "terminal_overrides",
    [
        {"terminal_locked": False},
        {"terminal_valid": False, "exact_l1": False},
        {"exact_l1": False},
        {"terminal_bytes": b"different-sealed-terminal\n"},
    ],
)
def test_terminal_wrapper_mismatch_fails_before_any_route_call(
    tmp_path: Path,
    terminal_overrides: dict[str, object],
) -> None:
    payload = _payload()
    context = _context()
    planner, _ = _cached_recording_planner(tmp_path, context)

    with pytest.raises(UgiTerminalRouteAssessmentError):
        assess_locked_ugi_terminal_routes(
            _terminal(payload, **terminal_overrides),
            l1_reverifier=_reverifier(),
            planner=planner,
            planner_context=context,
            assessment_context=_assessment_context(context),
            assessment_at_utc=ASSESSMENT_AT,
        )
    assert not planner.targets


def test_l1_context_mismatch_fails_before_any_route_call(tmp_path: Path) -> None:
    payload = _payload()
    context = _context(l1_reaction_sha256=_hash("other-l1"))
    planner, _ = _cached_recording_planner(tmp_path, context)

    with pytest.raises(UgiTerminalRouteAssessmentError, match="different L1"):
        assess_locked_ugi_terminal_routes(
            _terminal(payload),
            l1_reverifier=_reverifier(),
            planner=planner,
            planner_context=context,
            assessment_context=_assessment_context(context),
            assessment_at_utc=ASSESSMENT_AT,
        )
    assert not planner.targets


def test_expired_l3_context_fails_before_any_route_call(tmp_path: Path) -> None:
    payload = _payload()
    context = replace(
        _context(),
        l3_expires_at_utc="2026-08-10T00:00:00Z",
    )
    planner, _ = _cached_recording_planner(tmp_path, context)

    with pytest.raises(UgiTerminalRouteAssessmentError, match="not current"):
        assess_locked_ugi_terminal_routes(
            _terminal(payload),
            l1_reverifier=_reverifier(),
            planner=planner,
            planner_context=context,
            assessment_context=_assessment_context(context),
            assessment_at_utc=ASSESSMENT_AT,
        )
    assert not planner.targets


def test_malformed_arm_fails_before_any_route_call(tmp_path: Path) -> None:
    payload = _payload()
    context = _context()
    planner, _ = _cached_recording_planner(tmp_path, context)
    malformed = replace(_assessment_context(context), arm="guided")  # type: ignore[arg-type]

    with pytest.raises(UgiTerminalRouteAssessmentError, match="arm is unsupported"):
        assess_locked_ugi_terminal_routes(
            _terminal(payload),
            l1_reverifier=_reverifier(),
            planner=planner,
            planner_context=context,
            assessment_context=malformed,
            assessment_at_utc=ASSESSMENT_AT,
        )
    assert not planner.targets


@pytest.mark.parametrize("budget_field", ["unit_reservation", "remaining_budget"])
def test_insufficient_outer_reservation_fails_atomically(
    tmp_path: Path,
    budget_field: str,
) -> None:
    payload = _payload()
    context = _context()
    planner, _ = _cached_recording_planner(tmp_path, context)
    insufficient = RouteComputeUsage(2, 2, 0, 0)
    kwargs = {budget_field: insufficient}

    with pytest.raises(UgiTerminalRouteAssessmentError, match="cannot"):
        assess_locked_ugi_terminal_routes(
            _terminal(payload),
            l1_reverifier=_reverifier(),
            planner=planner,
            planner_context=context,
            assessment_context=_assessment_context(context, **kwargs),
            assessment_at_utc=ASSESSMENT_AT,
        )
    assert not planner.targets


@pytest.mark.parametrize("budget_field", ["unit_reservation", "remaining_budget"])
def test_missing_product_l1_verifier_reservation_fails_before_any_chemistry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget_field: str,
) -> None:
    payload = _payload()
    context = _context()
    planner, _ = _cached_recording_planner(tmp_path, context)
    reverification_calls: list[str] = []

    def record_reverification(*args: object, **kwargs: object) -> object:
        reverification_calls.append("called")
        return payload.l1_forward_verification

    monkeypatch.setattr(
        QualifiedUgiL1Reverifier,
        "require_exact",
        record_reverification,
    )
    per_role_verifier_ceiling = len(ROLE_NAMES) * context.budget_limits.maximum_verifier_calls
    missing_product_l1 = RouteComputeUsage(
        logical_planner_calls=len(ROLE_NAMES),
        physical_planner_calls=len(ROLE_NAMES),
        logical_verifier_calls=per_role_verifier_ceiling,
        physical_verifier_calls=per_role_verifier_ceiling,
    )

    with pytest.raises(UgiTerminalRouteAssessmentError, match="cannot"):
        assess_locked_ugi_terminal_routes(
            _terminal(payload),
            l1_reverifier=_reverifier(),
            planner=planner,
            planner_context=context,
            assessment_context=_assessment_context(
                context,
                **{budget_field: missing_product_l1},
            ),
            assessment_at_utc=ASSESSMENT_AT,
        )
    assert not reverification_calls
    assert not planner.targets


def test_one_role_budget_exhaustion_does_not_starve_later_roles() -> None:
    payload = _payload()
    context = _context(limits=_limits(maximum_expansions=0))
    planner = OneRoleBudgetExhaustionPlanner()

    receipt = assess_locked_ugi_terminal_routes(
        _terminal(payload),
        l1_reverifier=_reverifier(),
        planner=planner,
        planner_context=context,
        assessment_context=_assessment_context(context),
        assessment_at_utc=ASSESSMENT_AT,
    )

    assert tuple(target.role for target in planner.targets) == ROLE_NAMES
    assert len({id(ledger) for ledger in planner.ledgers}) == 3
    outcomes = {item.role: item.assessment.outcome for item in receipt.role_assessments}
    assert outcomes == {
        "amine_head": AssessmentOutcome.BUDGET_EXHAUSTED,
        "oxoester_aldehyde_body_tail": AssessmentOutcome.COMPLETE,
        "isocyanide_tail": AssessmentOutcome.COMPLETE,
    }
    assert not receipt.product_value.route_complete
    assert planner.ledgers[1].exhaustion_events == []
    assert planner.ledgers[2].exhaustion_events == []


def test_post_hoc_requires_frozen_lock_manifest_before_routes(tmp_path: Path) -> None:
    payload = _payload()
    context = _context()
    planner, _ = _cached_recording_planner(tmp_path, context)

    with pytest.raises(UgiTerminalRouteAssessmentError, match="post_hoc_lock_manifest"):
        assess_locked_ugi_terminal_routes(
            _terminal(payload),
            l1_reverifier=_reverifier(),
            planner=planner,
            planner_context=context,
            assessment_context=_assessment_context(context, arm=MatchedArm.POST_HOC),
            assessment_at_utc=ASSESSMENT_AT,
        )
    assert not planner.targets


def test_receipt_rejects_noncanonical_or_corrupted_bytes(tmp_path: Path) -> None:
    payload = _payload()
    context = _context()
    planner, _ = _cached_recording_planner(tmp_path, context)
    receipt = assess_locked_ugi_terminal_routes(
        _terminal(payload),
        l1_reverifier=_reverifier(),
        planner=planner,
        planner_context=context,
        assessment_context=_assessment_context(context),
        assessment_at_utc=ASSESSMENT_AT,
    )

    with pytest.raises(UgiTerminalRouteAssessmentError, match="not canonical"):
        UgiTerminalRouteAssessmentReceipt.from_bytes(receipt.canonical_bytes + b" ")

    corrupted = json.loads(receipt.canonical_bytes)
    corrupted["role_assessments"][0]["assessment"]["outcome"] = "missing_knowledge"
    raw = (json.dumps(corrupted, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with pytest.raises((UgiTerminalRouteAssessmentError, ValueError)):
        UgiTerminalRouteAssessmentReceipt.from_bytes(raw)


def test_guided_context_rejects_post_hoc_manifest_before_routes(tmp_path: Path) -> None:
    payload = _payload()
    context = _context()
    planner, _ = _cached_recording_planner(tmp_path, context)

    with pytest.raises(UgiTerminalRouteAssessmentError, match="guided"):
        assess_locked_ugi_terminal_routes(
            _terminal(payload),
            l1_reverifier=_reverifier(),
            planner=planner,
            planner_context=context,
            assessment_context=_assessment_context(
                context,
                post_hoc_lock_manifest_sha256=_hash("manifest"),
            ),
            assessment_at_utc=ASSESSMENT_AT,
        )
    assert not planner.targets
