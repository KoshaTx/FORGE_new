from __future__ import annotations

from dataclasses import dataclass, field

import pytest

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
    RouteKnowledgeError,
    RouteStepProposal,
    RouteTarget,
    SynthesisAssessment,
)


@dataclass
class MappingKnowledgeSource:
    records: dict[str, KnowledgeResult]
    failures: set[str] = field(default_factory=set)
    calls: list[str] = field(default_factory=list)

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        self.calls.append(target.canonical_smiles)
        if target.canonical_smiles in self.failures:
            raise RouteKnowledgeError("qualified lookup failed")
        return self.records[target.canonical_smiles]


def _limits(**overrides: int) -> PlannerBudgetLimits:
    values = {
        "maximum_depth": 4,
        "maximum_logical_planner_calls": 4,
        "maximum_expansions": 8,
        "maximum_product_candidates": 8,
        "maximum_verifier_calls": 8,
        "maximum_elapsed_milliseconds": 10_000,
    }
    values.update(overrides)
    return PlannerBudgetLimits(**values)


def _evidence(
    evidence_id: str,
    *,
    tier: EvidenceTier = EvidenceTier.EXACT_SOURCE,
    availability: AvailabilityState = AvailabilityState.UNASSESSED,
    exact_substrate: bool = True,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        tier=tier,
        source_sha256=(evidence_id[0] * 64),
        source_locator=f"source:{evidence_id}",
        exact_substrate=exact_substrate,
        forward_verification=(
            ForwardVerificationState.NOT_APPLICABLE
            if tier is EvidenceTier.ACCEPTED_TERMINAL
            else ForwardVerificationState.VERIFIED_EXACT_PRODUCT_UNIQUE
        ),
        availability=availability,
    )


def _target(smiles: str, role: str = "isocyanide_tail") -> RouteTarget:
    return RouteTarget(role=role, canonical_smiles=smiles)


def _terminal(smiles: str, evidence_id: str) -> tuple[str, KnowledgeResult]:
    evidence = _evidence(
        evidence_id,
        tier=EvidenceTier.ACCEPTED_TERMINAL,
        availability=AvailabilityState.CURRENT_CLOSED,
    )
    return smiles, KnowledgeResult(
        disposition=KnowledgeDisposition.TERMINAL,
        evidence=(evidence,),
        detail="accepted terminal with current L3 evidence",
    )


def _expansion(
    product: str,
    reactants: tuple[str, ...],
    evidence_id: str,
    *,
    evidence: EvidenceRecord | None = None,
) -> tuple[str, KnowledgeResult]:
    record = evidence or _evidence(evidence_id)
    proposal = RouteStepProposal(
        reaction_id=f"reaction:{evidence_id}",
        reactants=tuple(_target(smiles) for smiles in reactants),
        evidence=(record,),
        forward_product_count=1,
        verifier_calls_required=1,
        product_candidates_considered=1,
    )
    return product, KnowledgeResult(
        disposition=KnowledgeDisposition.EXPAND,
        evidence=(record,),
        proposal=proposal,
        detail="exact-source expansion",
    )


def test_recursive_assessment_closes_only_through_exact_steps_and_current_terminals() -> None:
    source = MappingKnowledgeSource(
        dict(
            [
                _expansion("CCN=C", ("CCN", "CO"), "a"),
                _terminal("CCN", "b"),
                _expansion("CO", ("C",), "c"),
                _terminal("C", "d"),
            ]
        )
    )
    budget = PlannerBudgetLedger(_limits())
    result = RecursiveRouteAssessor(source).assess(_target("CCN=C"), budget)

    assert result.outcome is AssessmentOutcome.COMPLETE
    assert [child.outcome for child in result.route_tree.children] == [
        AssessmentOutcome.COMPLETE,
        AssessmentOutcome.COMPLETE,
    ]
    assert result.route_tree.children[1].children[0].outcome is AssessmentOutcome.COMPLETE
    assert {evidence_id for event in result.trace for evidence_id in event.evidence_ids} == {
        "a",
        "b",
        "c",
        "d",
    }
    assert budget.to_dict() == {
        "limits": _limits().to_dict(),
        "logical_planner_calls": 1,
        "physical_cache_hits": 0,
        "physical_cache_misses": 0,
        "expansions": 2,
        "product_candidates": 2,
        "verifier_calls": 2,
        "elapsed_milliseconds": 0,
        "exhaustion_events": [],
    }
    assert SynthesisAssessment.from_dict(result.to_dict()) == result


@pytest.mark.parametrize(
    ("disposition", "outcome"),
    [
        (KnowledgeDisposition.INCOMPATIBLE, AssessmentOutcome.INCOMPATIBLE),
        (KnowledgeDisposition.OUTSIDE_SUPPORT, AssessmentOutcome.OUTSIDE_SUPPORT),
        (KnowledgeDisposition.MISSING_KNOWLEDGE, AssessmentOutcome.MISSING_KNOWLEDGE),
        (KnowledgeDisposition.INVALID_INPUT, AssessmentOutcome.INVALID_INPUT),
        (KnowledgeDisposition.EXECUTION_ERROR, AssessmentOutcome.EXECUTION_ERROR),
    ],
)
def test_direct_failure_outcomes_remain_distinct(
    disposition: KnowledgeDisposition,
    outcome: AssessmentOutcome,
) -> None:
    source = MappingKnowledgeSource(
        {
            "CC": KnowledgeResult(
                disposition=disposition,
                evidence=(),
                detail=f"explicit {disposition.value}",
            )
        }
    )
    result = RecursiveRouteAssessor(source).assess(
        _target("CC"),
        PlannerBudgetLedger(_limits()),
    )

    assert result.outcome is outcome
    assert result.route_tree.detail == f"explicit {disposition.value}"


def test_budget_exhaustion_is_not_relabelled_as_incompatibility() -> None:
    source = MappingKnowledgeSource(dict([_expansion("CC", ("C",), "e")]))
    budget = PlannerBudgetLedger(_limits(maximum_expansions=0))
    result = RecursiveRouteAssessor(source).assess(_target("CC"), budget)

    assert result.outcome is AssessmentOutcome.BUDGET_EXHAUSTED
    assert result.outcome is not AssessmentOutcome.INCOMPATIBLE
    assert budget.exhaustion_events == ["maximum_expansions"]
    assert budget.product_candidates == 0
    assert budget.verifier_calls == 0


def test_family_projection_cannot_promote_a_route_to_complete() -> None:
    family = _evidence(
        "f",
        tier=EvidenceTier.FAMILY_PROJECTED,
        exact_substrate=False,
    )
    source = MappingKnowledgeSource(
        dict([_expansion("CC", ("C",), "f", evidence=family), _terminal("C", "a")])
    )
    result = RecursiveRouteAssessor(source).assess(
        _target("CC"),
        PlannerBudgetLedger(_limits()),
    )

    assert result.outcome is AssessmentOutcome.OUTSIDE_SUPPORT
    assert source.calls == ["CC"]
    assert result.trace[-1].evidence_ids == ("f",)


@pytest.mark.parametrize(
    ("availability", "outcome"),
    [
        (AvailabilityState.UNAVAILABLE, AssessmentOutcome.OUTSIDE_SUPPORT),
        (AvailabilityState.EXPIRED, AssessmentOutcome.MISSING_KNOWLEDGE),
        (AvailabilityState.UNASSESSED, AssessmentOutcome.MISSING_KNOWLEDGE),
    ],
)
def test_noncurrent_terminal_evidence_cannot_close_l3(
    availability: AvailabilityState,
    outcome: AssessmentOutcome,
) -> None:
    evidence = _evidence(
        "a",
        tier=EvidenceTier.ACCEPTED_TERMINAL,
        availability=availability,
    )
    source = MappingKnowledgeSource(
        {
            "C": KnowledgeResult(
                disposition=KnowledgeDisposition.TERMINAL,
                evidence=(evidence,),
                detail="terminal evidence",
            )
        }
    )
    result = RecursiveRouteAssessor(source).assess(
        _target("C"),
        PlannerBudgetLedger(_limits()),
    )

    assert result.outcome is outcome


def test_recursive_cycle_fails_outside_tree_support_without_hanging() -> None:
    source = MappingKnowledgeSource(dict([_expansion("CC", ("CC",), "a")]))
    result = RecursiveRouteAssessor(source).assess(
        _target("CC"),
        PlannerBudgetLedger(_limits()),
    )

    assert result.outcome is AssessmentOutcome.OUTSIDE_SUPPORT
    assert result.route_tree.children[0].outcome is AssessmentOutcome.OUTSIDE_SUPPORT
    assert any(event.action.value == "cycle" for event in result.trace)


def test_expected_knowledge_execution_failure_remains_distinct() -> None:
    source = MappingKnowledgeSource({}, failures={"CC"})
    result = RecursiveRouteAssessor(source).assess(
        _target("CC"),
        PlannerBudgetLedger(_limits()),
    )

    assert result.outcome is AssessmentOutcome.EXECUTION_ERROR
    assert "qualified lookup failed" in result.route_tree.detail
