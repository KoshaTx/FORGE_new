from __future__ import annotations

from collections import Counter

import pytest

from forge.route.engine.planner import (
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
    RouteStepProposal,
    RouteTarget,
)
from forge.route.assessment.ugi3_support_boundary import (
    AuthenticatedInternalRoleRegistry,
    MolecularSupportState,
    RoutePriorityLane,
    SupportBoundaryNormalizedUgi3Source,
    TargetQualification,
    Ugi3SupportBoundaryError,
    qualification_key,
    route_priority_lane,
)

EMPTY_INTERNAL_ROLES = AuthenticatedInternalRoleRegistry(frozenset())

LIMITS = PlannerBudgetLimits(
    maximum_depth=3,
    maximum_logical_planner_calls=1,
    maximum_expansions=3,
    maximum_product_candidates=3,
    maximum_verifier_calls=1,
    maximum_elapsed_milliseconds=0,
)


def _qualification(**overrides: object) -> TargetQualification:
    values: dict[str, object] = {
        "exact_l1_eligible": True,
        "supported_ugi_role": True,
        "role_handle_qualified": True,
        "molecular_support_state": MolecularSupportState.WITHIN_DECLARED_SUPPORT,
    }
    values.update(overrides)
    return TargetQualification(**values)  # type: ignore[arg-type]


def _evidence(tier: EvidenceTier = EvidenceTier.PROVENANCE_ONLY) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"support-boundary-{tier.value}",
        tier=tier,
        source_sha256="a" * 64,
        source_locator=f"frozen://support-boundary/{tier.value}",
        exact_substrate=True,
        forward_verification=ForwardVerificationState.NOT_RUN,
        availability=AvailabilityState.UNASSESSED,
    )


class _StaticSource:
    def __init__(self, result: KnowledgeResult):
        self.result = result
        self.calls: list[RouteTarget] = []

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        self.calls.append(target)
        return self.result


def _assess(
    target: RouteTarget,
    result: KnowledgeResult,
    qualification: TargetQualification,
):
    wrapped = SupportBoundaryNormalizedUgi3Source(
        _StaticSource(result),
        {qualification_key(target): qualification},
        authenticated_internal_roles=EMPTY_INTERNAL_ROLES,
    )
    return RecursiveRouteAssessor(wrapped).assess(
        target,
        PlannerBudgetLedger(LIMITS),
    )


@pytest.mark.parametrize(
    "delegate_detail",
    [
        "target is outside the frozen exact-evidence adapter index",
        "no admitted upstream route exists in the declared search support",
        "identity is absent from a frozen route-value registry",
    ],
)
def test_cross_path_registry_misses_normalize_to_missing(delegate_detail: str) -> None:
    target = RouteTarget(role="amine_head", canonical_smiles="CCCNCCN")
    evidence = _evidence()
    assessment = _assess(
        target,
        KnowledgeResult(
            disposition=KnowledgeDisposition.OUTSIDE_SUPPORT,
            evidence=(evidence,),
            detail=delegate_detail,
        ),
        _qualification(),
    )

    assert assessment.outcome is AssessmentOutcome.MISSING_KNOWLEDGE
    assert assessment.route_tree.evidence == (evidence,)
    assert assessment.route_tree.evidence[0].tier is EvidenceTier.PROVENANCE_ONLY
    assert delegate_detail in assessment.route_tree.detail


def test_quantitative_qualified_absent_census_is_all_missing() -> None:
    targets = tuple(
        RouteTarget(role=role, canonical_smiles=f"C{'C' * index}N")
        for role in (
            "amine_head",
            "oxoester_aldehyde_body_tail",
            "isocyanide_tail",
        )
        for index in range(1, 5)
    )
    delegate = _StaticSource(
        KnowledgeResult(
            disposition=KnowledgeDisposition.OUTSIDE_SUPPORT,
            evidence=(_evidence(),),
            detail="registry index miss",
        )
    )
    wrapped = SupportBoundaryNormalizedUgi3Source(
        delegate,
        {qualification_key(target): _qualification() for target in targets},
        authenticated_internal_roles=EMPTY_INTERNAL_ROLES,
    )
    outcomes = Counter(
        RecursiveRouteAssessor(wrapped).assess(target, PlannerBudgetLedger(LIMITS)).outcome
        for target in targets
    )

    assert outcomes == {AssessmentOutcome.MISSING_KNOWLEDGE: 12}


def test_explicit_machine_readable_exclusion_preserves_outside() -> None:
    target = RouteTarget(role="isocyanide_tail", canonical_smiles="[C-]#[N+]C1CCCC1")
    qualification = _qualification(
        role_handle_qualified=False,
        molecular_support_state=MolecularSupportState.OUTSIDE_DECLARED_SUPPORT,
        declared_exclusion_code="ugi_handle_policy_failed",
        declared_exclusion_policy_locator="policy://ugi3/handles/v1#isocyanide",
    )
    assessment = _assess(
        target,
        KnowledgeResult(
            disposition=KnowledgeDisposition.OUTSIDE_SUPPORT,
            evidence=(_evidence(),),
            detail="explicit handle exclusion",
        ),
        qualification,
    )

    assert assessment.outcome is AssessmentOutcome.OUTSIDE_SUPPORT
    assert route_priority_lane(assessment.outcome, qualification) is (
        RoutePriorityLane.DECLARED_SUPPORT_REVIEW
    )


def test_outside_without_required_exclusion_metadata_is_invalid() -> None:
    target = RouteTarget(role="isocyanide_tail", canonical_smiles="[C-]#[N+]CC")
    assessment = _assess(
        target,
        KnowledgeResult(
            disposition=KnowledgeDisposition.OUTSIDE_SUPPORT,
            evidence=(),
            detail="unsupported without policy evidence",
        ),
        _qualification(role_handle_qualified=False),
    )

    assert assessment.outcome is AssessmentOutcome.INVALID_INPUT


def test_unassessed_molecular_support_is_invalid_not_missing() -> None:
    target = RouteTarget(role="amine_head", canonical_smiles="CCNCCN")
    assessment = _assess(
        target,
        KnowledgeResult(
            disposition=KnowledgeDisposition.OUTSIDE_SUPPORT,
            evidence=(_evidence(),),
            detail="index miss before molecular-support qualification",
        ),
        _qualification(
            molecular_support_state=MolecularSupportState.UNASSESSED,
        ),
    )

    assert assessment.outcome is AssessmentOutcome.INVALID_INPUT
    assert "unassessed" in assessment.route_tree.detail


@pytest.mark.parametrize(
    "qualification",
    [
        _qualification(exact_l1_eligible=False),
        _qualification(supported_ugi_role=False),
    ],
)
def test_malformed_root_qualification_is_invalid(
    qualification: TargetQualification,
) -> None:
    target = RouteTarget(role="amine_head", canonical_smiles="CCN")
    assessment = _assess(
        target,
        KnowledgeResult(
            disposition=KnowledgeDisposition.OUTSIDE_SUPPORT,
            evidence=(),
            detail="index miss",
        ),
        qualification,
    )

    assert assessment.outcome is AssessmentOutcome.INVALID_INPUT


def test_unsupported_role_is_invalid_without_calling_delegate() -> None:
    target = RouteTarget(role="not_a_ugi_role", canonical_smiles="CCN")
    qualification = _qualification()
    assessment = _assess(
        target,
        KnowledgeResult(
            disposition=KnowledgeDisposition.OUTSIDE_SUPPORT,
            evidence=(),
            detail="must not decide unsupported role",
        ),
        qualification,
    )

    assert assessment.outcome is AssessmentOutcome.INVALID_INPUT


def test_authenticated_internal_roles_pass_through_by_exact_membership() -> None:
    roles = frozenset(
        {
            "ugi3_upstream_material:component-001",
            "ugi3_retrieved_upstream_material:component-001",
            "ugi3_targeted_upstream:component-001:route-001",
            "exact_c18_terminal_acid",
            "exact_c18_internal_alkynol",
            "exact_c18_terminal_alkynol",
            "terminal_protected_propargyl_alcohol",
            "terminal_alkyl_bromide",
            "protected_internal_alkynol",
            "internal_alkynol",
            "terminal_alkynol",
        }
    )
    result = KnowledgeResult(
        disposition=KnowledgeDisposition.MISSING_KNOWLEDGE,
        evidence=(_evidence(),),
        detail="authenticated internal route node",
    )
    delegate = _StaticSource(result)
    wrapped = SupportBoundaryNormalizedUgi3Source(
        delegate,
        {},
        authenticated_internal_roles=AuthenticatedInternalRoleRegistry(roles),
    )

    for role in sorted(roles):
        assert wrapped.lookup(RouteTarget(role=role, canonical_smiles="CCO")) is result

    assert tuple(target.role for target in delegate.calls) == tuple(sorted(roles))


@pytest.mark.parametrize(
    "lookalike_role",
    [
        "ugi3_upstream_material:component-002",
        "ugi3_retrieved_upstream_material:component-001:extra",
        "ugi3_targeted_upstream:component-001:route-001:extra",
        "exact_c18_terminal_acid_extra",
        "terminal_alkynol_extra",
    ],
)
def test_unknown_internal_role_lookalikes_are_rejected_without_delegate_call(
    lookalike_role: str,
) -> None:
    delegate = _StaticSource(
        KnowledgeResult(
            disposition=KnowledgeDisposition.TERMINAL,
            evidence=(_evidence(EvidenceTier.ACCEPTED_TERMINAL),),
            detail="must not be reached",
        )
    )
    wrapped = SupportBoundaryNormalizedUgi3Source(
        delegate,
        {},
        authenticated_internal_roles=AuthenticatedInternalRoleRegistry(
            frozenset(
                {
                    "ugi3_upstream_material:component-001",
                    "ugi3_retrieved_upstream_material:component-001",
                    "ugi3_targeted_upstream:component-001:route-001",
                    "exact_c18_terminal_acid",
                    "terminal_alkynol",
                }
            )
        ),
    )

    decision = wrapped.lookup(RouteTarget(role=lookalike_role, canonical_smiles="CCO"))

    assert decision.disposition is KnowledgeDisposition.INVALID_INPUT
    assert delegate.calls == []


def test_authenticated_internal_role_registry_is_immutable_and_exact() -> None:
    with pytest.raises(Ugi3SupportBoundaryError, match="frozenset"):
        AuthenticatedInternalRoleRegistry(  # type: ignore[arg-type]
            {"ugi3_upstream_material:component-001"}
        )
    with pytest.raises(Ugi3SupportBoundaryError, match="exact machine-readable"):
        AuthenticatedInternalRoleRegistry(frozenset({"ugi3_upstream_material:*"}))
    with pytest.raises(Ugi3SupportBoundaryError, match="cannot bypass"):
        AuthenticatedInternalRoleRegistry(frozenset({"amine_head"}))

    registry = AuthenticatedInternalRoleRegistry(
        frozenset({"ugi3_upstream_material:component-001"})
    )
    with pytest.raises(AttributeError):
        registry.roles.add("ugi3_upstream_material:component-002")  # type: ignore[attr-defined]


def test_internal_registry_does_not_remove_root_qualification_requirement() -> None:
    delegate = _StaticSource(
        KnowledgeResult(
            disposition=KnowledgeDisposition.TERMINAL,
            evidence=(_evidence(EvidenceTier.ACCEPTED_TERMINAL),),
            detail="must not be reached",
        )
    )
    wrapped = SupportBoundaryNormalizedUgi3Source(
        delegate,
        {},
        authenticated_internal_roles=AuthenticatedInternalRoleRegistry(
            frozenset({"ugi3_upstream_material:component-001"})
        ),
    )

    decision = wrapped.lookup(RouteTarget(role="amine_head", canonical_smiles="CCN"))

    assert decision.disposition is KnowledgeDisposition.INVALID_INPUT
    assert delegate.calls == []


@pytest.mark.parametrize(
    "disposition",
    [
        KnowledgeDisposition.INCOMPATIBLE,
        KnowledgeDisposition.MISSING_KNOWLEDGE,
        KnowledgeDisposition.EXECUTION_ERROR,
    ],
)
def test_non_support_blockers_are_preserved(disposition: KnowledgeDisposition) -> None:
    target = RouteTarget(role="amine_head", canonical_smiles="CCN")
    result = KnowledgeResult(
        disposition=disposition,
        evidence=(_evidence(EvidenceTier.FAMILY_PROJECTED),),
        detail=f"typed {disposition.value}",
    )
    wrapped = SupportBoundaryNormalizedUgi3Source(
        _StaticSource(result),
        {qualification_key(target): _qualification()},
        authenticated_internal_roles=EMPTY_INTERNAL_ROLES,
    )

    assert wrapped.lookup(target) is result


class _ExactOverlayLikeSource:
    def __init__(self, root: RouteTarget, leaf: RouteTarget):
        exact = EvidenceRecord(
            evidence_id="support-boundary-exact-source",
            tier=EvidenceTier.EXACT_SOURCE,
            source_sha256="c" * 64,
            source_locator="frozen://support-boundary/exact-source",
            exact_substrate=True,
            forward_verification=(ForwardVerificationState.VERIFIED_EXACT_PRODUCT_UNIQUE),
            availability=AvailabilityState.UNASSESSED,
        )
        terminal = EvidenceRecord(
            evidence_id="support-boundary-terminal",
            tier=EvidenceTier.ACCEPTED_TERMINAL,
            source_sha256="b" * 64,
            source_locator="frozen://support-boundary/terminal",
            exact_substrate=True,
            forward_verification=ForwardVerificationState.NOT_APPLICABLE,
            availability=AvailabilityState.CURRENT_CLOSED,
        )
        proposal = RouteStepProposal(
            reaction_id="exact_overlay_test",
            reactants=(leaf,),
            evidence=(exact,),
            forward_product_count=1,
            verifier_calls_required=0,
            product_candidates_considered=1,
        )
        self._results = {
            root.identity: KnowledgeResult(
                disposition=KnowledgeDisposition.EXPAND,
                evidence=(exact,),
                proposal=proposal,
                detail="exact identity overlay",
            ),
            leaf.identity: KnowledgeResult(
                disposition=KnowledgeDisposition.TERMINAL,
                evidence=(terminal,),
                detail="current exact terminal",
            ),
        }

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        return self._results[target.identity]


def test_exact_overlay_completion_and_budget_outcome_are_preserved() -> None:
    root = RouteTarget(role="oxoester_aldehyde_body_tail", canonical_smiles="CCCC=O")
    leaf = RouteTarget(
        role="ugi3_upstream_material:test-component",
        canonical_smiles="CCCCO",
    )
    wrapped = SupportBoundaryNormalizedUgi3Source(
        _ExactOverlayLikeSource(root, leaf),
        {qualification_key(root): _qualification()},
        authenticated_internal_roles=AuthenticatedInternalRoleRegistry(frozenset({leaf.role})),
    )
    complete = RecursiveRouteAssessor(wrapped).assess(
        root,
        PlannerBudgetLedger(LIMITS),
    )
    depth_zero = PlannerBudgetLimits(
        maximum_depth=0,
        maximum_logical_planner_calls=1,
        maximum_expansions=1,
        maximum_product_candidates=1,
        maximum_verifier_calls=0,
        maximum_elapsed_milliseconds=0,
    )
    budgeted = RecursiveRouteAssessor(wrapped).assess(
        root,
        PlannerBudgetLedger(depth_zero),
    )

    assert complete.outcome is AssessmentOutcome.COMPLETE
    assert budgeted.outcome is AssessmentOutcome.BUDGET_EXHAUSTED


def test_route_priority_semantics_require_explicit_exclusion() -> None:
    qualified = _qualification()
    assert (
        route_priority_lane(
            AssessmentOutcome.MISSING_KNOWLEDGE,
            qualified,
        )
        is RoutePriorityLane.TARGETED_EVIDENCE
    )
    with pytest.raises(
        Ugi3SupportBoundaryError,
        match="explicit exclusion",
    ):
        route_priority_lane(AssessmentOutcome.OUTSIDE_SUPPORT, qualified)


def test_exclusion_fields_are_all_or_none_and_machine_readable() -> None:
    with pytest.raises(Ugi3SupportBoundaryError, match="supplied together"):
        _qualification(declared_exclusion_code="support_limit")
    with pytest.raises(Ugi3SupportBoundaryError, match="machine-readable"):
        _qualification(
            declared_exclusion_code="Not machine readable",
            declared_exclusion_policy_locator="policy://ugi3/support",
        )
    with pytest.raises(Ugi3SupportBoundaryError, match="contradicts"):
        _qualification(
            declared_exclusion_code="declared_policy_exclusion",
            declared_exclusion_policy_locator="policy://ugi3/support#excluded",
        )


def test_no_evidence_tier_can_be_promoted_by_normalization() -> None:
    target = RouteTarget(role="amine_head", canonical_smiles="CCNCCN")
    for tier in (EvidenceTier.PROVENANCE_ONLY, EvidenceTier.FAMILY_PROJECTED):
        evidence = _evidence(tier)
        wrapped = SupportBoundaryNormalizedUgi3Source(
            _StaticSource(
                KnowledgeResult(
                    disposition=KnowledgeDisposition.OUTSIDE_SUPPORT,
                    evidence=(evidence,),
                    detail="index miss with nonclosing evidence",
                )
            ),
            {qualification_key(target): _qualification()},
            authenticated_internal_roles=EMPTY_INTERNAL_ROLES,
        )
        decision = wrapped.lookup(target)

        assert decision.disposition is KnowledgeDisposition.MISSING_KNOWLEDGE
        assert decision.evidence == (evidence,)
        assert decision.evidence[0].tier is tier
        assert decision.proposal is None
