from __future__ import annotations

from dataclasses import dataclass, field, replace

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
from forge.value.synthesis.synthesis import (
    BurdenEstimate,
    ComponentSynthesisValue,
    Dominance,
    EvidenceSupport,
    ForwardConsistency,
    ProductSynthesisValue,
    SynthesisValueError,
    compare_component_synthesis_values,
    compare_product_synthesis_values,
    component_synthesis_value_from_assessment,
)


@dataclass
class MappingKnowledgeSource:
    records: dict[str, KnowledgeResult]
    calls: list[str] = field(default_factory=list)

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        self.calls.append(target.canonical_smiles)
        return self.records[target.canonical_smiles]


def _target(smiles: str, role: str = "aldehyde_tail") -> RouteTarget:
    return RouteTarget(role=role, canonical_smiles=smiles)


def _evidence(
    evidence_id: str,
    *,
    tier: EvidenceTier,
    availability: AvailabilityState = AvailabilityState.UNASSESSED,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        tier=tier,
        source_sha256=evidence_id[0] * 64,
        source_locator=f"source:{evidence_id}",
        exact_substrate=tier is not EvidenceTier.FAMILY_PROJECTED,
        forward_verification=(
            ForwardVerificationState.NOT_APPLICABLE
            if tier is EvidenceTier.ACCEPTED_TERMINAL
            else ForwardVerificationState.VERIFIED_EXACT_PRODUCT_UNIQUE
        ),
        availability=availability,
    )


def _terminal(smiles: str, evidence_id: str) -> tuple[str, KnowledgeResult]:
    evidence = _evidence(
        evidence_id,
        tier=EvidenceTier.ACCEPTED_TERMINAL,
        availability=AvailabilityState.CURRENT_CLOSED,
    )
    return smiles, KnowledgeResult(
        disposition=KnowledgeDisposition.TERMINAL,
        evidence=(evidence,),
        detail="current accepted terminal",
    )


def _expansion(
    product: str,
    reactants: tuple[str, ...],
    evidence_id: str,
) -> tuple[str, KnowledgeResult]:
    evidence = _evidence(evidence_id, tier=EvidenceTier.EXACT_SOURCE)
    proposal = RouteStepProposal(
        reaction_id=f"reaction:{evidence_id}",
        reactants=tuple(_target(smiles) for smiles in reactants),
        evidence=(evidence,),
        forward_product_count=1,
        verifier_calls_required=1,
        product_candidates_considered=1,
    )
    return product, KnowledgeResult(
        disposition=KnowledgeDisposition.EXPAND,
        evidence=(evidence,),
        proposal=proposal,
        detail="exact-source expansion",
    )


def _limits() -> PlannerBudgetLimits:
    return PlannerBudgetLimits(
        maximum_depth=4,
        maximum_logical_planner_calls=4,
        maximum_expansions=8,
        maximum_product_candidates=8,
        maximum_verifier_calls=8,
        maximum_elapsed_milliseconds=10_000,
    )


def _assess(records: dict[str, KnowledgeResult], smiles: str):
    return RecursiveRouteAssessor(MappingKnowledgeSource(records)).assess(
        _target(smiles),
        PlannerBudgetLedger(_limits()),
    )


def _blocked_value(outcome: AssessmentOutcome) -> ComponentSynthesisValue:
    disposition = {
        AssessmentOutcome.MISSING_KNOWLEDGE: KnowledgeDisposition.MISSING_KNOWLEDGE,
        AssessmentOutcome.OUTSIDE_SUPPORT: KnowledgeDisposition.OUTSIDE_SUPPORT,
        AssessmentOutcome.INCOMPATIBLE: KnowledgeDisposition.INCOMPATIBLE,
    }[outcome]
    assessment = _assess(
        {
            "CC": KnowledgeResult(
                disposition=disposition,
                evidence=(),
                detail=f"explicit {outcome.value}",
            )
        },
        "CC",
    )
    return component_synthesis_value_from_assessment(assessment)


def test_exact_recursive_assessment_becomes_structured_value_and_roundtrips() -> None:
    assessment = _assess(
        dict(
            [
                _expansion("CCN=C", ("CCN", "CO"), "a"),
                _terminal("CCN", "b"),
                _expansion("CO", ("C",), "c"),
                _terminal("C", "d"),
            ]
        ),
        "CCN=C",
    )
    value = component_synthesis_value_from_assessment(assessment)

    assert value.route_complete
    assert value.forward_consistency is ForwardConsistency.EXACT_UNIQUE
    assert value.evidence_support is EvidenceSupport.EXACT_IDENTITY
    assert value.route_step_count == 2
    assert value.maximum_route_depth == 2
    assert value.leaf_count == 2
    assert value.current_terminal_leaf_count == 2
    assert value.protection_burden == BurdenEstimate.unknown()
    assert value.purification_burden == BurdenEstimate.unknown()
    assert ComponentSynthesisValue.from_dict(value.to_dict()) == value


def test_direct_current_terminal_is_closed_without_fabricating_route_evidence() -> None:
    assessment = _assess(dict([_terminal("CCCC", "a")]), "CCCC")
    value = component_synthesis_value_from_assessment(assessment)

    assert value.route_complete
    assert value.route_step_count == 0
    assert value.maximum_route_depth == 0
    assert value.forward_consistency is ForwardConsistency.NOT_APPLICABLE
    assert value.evidence_support is EvidenceSupport.NOT_APPLICABLE


def test_distinct_noncomplete_blocker_taxonomies_are_incomparable() -> None:
    missing = _blocked_value(AssessmentOutcome.MISSING_KNOWLEDGE)
    outside = _blocked_value(AssessmentOutcome.OUTSIDE_SUPPORT)
    incompatible = _blocked_value(AssessmentOutcome.INCOMPATIBLE)

    assert compare_component_synthesis_values(missing, outside) is Dominance.INCOMPARABLE
    assert compare_component_synthesis_values(outside, incompatible) is Dominance.INCOMPARABLE
    assert compare_component_synthesis_values(incompatible, missing) is Dominance.INCOMPARABLE


def test_more_burden_or_weaker_evidence_never_improves_a_value() -> None:
    baseline = replace(
        _blocked_value(AssessmentOutcome.MISSING_KNOWLEDGE),
        evidence_support=EvidenceSupport.EXACT_IDENTITY,
        protection_burden=BurdenEstimate.known(1),
        purification_burden=BurdenEstimate.known(1),
    )
    more_burden = replace(
        baseline,
        protection_burden=BurdenEstimate.known(2),
        purification_burden=BurdenEstimate.known(3),
    )
    weaker_evidence = replace(
        baseline,
        evidence_support=EvidenceSupport.FAMILY_PROJECTED,
    )

    assert compare_component_synthesis_values(more_burden, baseline) is Dominance.WORSE
    assert compare_component_synthesis_values(weaker_evidence, baseline) is Dominance.WORSE
    assert compare_component_synthesis_values(baseline, more_burden) is Dominance.BETTER
    assert compare_component_synthesis_values(baseline, weaker_evidence) is Dominance.BETTER


def test_unknown_burden_is_not_silently_treated_as_zero() -> None:
    baseline = replace(
        _blocked_value(AssessmentOutcome.MISSING_KNOWLEDGE),
        protection_burden=BurdenEstimate.known(1),
    )
    unknown = replace(baseline, protection_burden=BurdenEstimate.unknown())

    assert compare_component_synthesis_values(unknown, baseline) is Dominance.INCOMPARABLE
    assert compare_component_synthesis_values(baseline, unknown) is Dominance.INCOMPARABLE


def test_product_value_preserves_roles_and_forbids_scalar_success_claims() -> None:
    component = component_synthesis_value_from_assessment(
        _assess(dict([_terminal("CCCC", "a")]), "CCCC")
    )
    product = ProductSynthesisValue(
        product_smiles="CCNC(=O)CC",
        l1_forward_consistent=True,
        components=(("amine", component), ("aldehyde", component), ("isocyanide", component)),
    )
    serialized = product.to_dict()

    assert product.route_complete
    assert serialized["scalar_value"] is None
    assert serialized["success_probability"] is None
    assert ProductSynthesisValue.from_dict(serialized) == product
    assert compare_product_synthesis_values(product, product) is Dominance.EQUAL

    serialized["success_probability"] = 0.9
    with pytest.raises(SynthesisValueError, match="scalar probabilities"):
        ProductSynthesisValue.from_dict(serialized)


def test_product_comparison_is_rolewise_and_pareto_safe() -> None:
    exact = replace(
        _blocked_value(AssessmentOutcome.MISSING_KNOWLEDGE),
        evidence_support=EvidenceSupport.EXACT_IDENTITY,
    )
    weak = replace(exact, evidence_support=EvidenceSupport.FAMILY_PROJECTED)
    left = ProductSynthesisValue(
        product_smiles="product:left",
        l1_forward_consistent=True,
        components=(("amine", exact), ("aldehyde", weak)),
    )
    right = ProductSynthesisValue(
        product_smiles="product:right",
        l1_forward_consistent=True,
        components=(("amine", weak), ("aldehyde", exact)),
    )

    assert compare_product_synthesis_values(left, right) is Dominance.INCOMPARABLE


def test_product_l1_failure_prevents_closure_and_ranks_below_exact_l1() -> None:
    component = component_synthesis_value_from_assessment(
        _assess(dict([_terminal("CCCC", "a")]), "CCCC")
    )
    exact = ProductSynthesisValue(
        product_smiles="CCNC(=O)CC",
        l1_forward_consistent=True,
        components=(("amine", component), ("aldehyde", component), ("isocyanide", component)),
    )
    failed = replace(exact, l1_forward_consistent=False)

    assert exact.route_complete
    assert not failed.route_complete
    assert compare_product_synthesis_values(failed, exact) is Dominance.WORSE
    assert compare_product_synthesis_values(exact, failed) is Dominance.BETTER
