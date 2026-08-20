"""Structured pre-prospective synthesis values without success probabilities.

The records in this module summarize typed route assessments for later
guidance experiments.  They preserve blocker taxonomy and expose only a
Pareto-style partial order.  They deliberately do not define a scalar, a route
likelihood, or a probability of experimental synthesis success.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from forge.synthesis.engine.planner import (
    AssessmentOutcome,
    AvailabilityState,
    EvidenceTier,
    ForwardVerificationState,
    RouteTarget,
    SynthesisAssessment,
    SynthesisRouteNode,
)

SYNTHESIS_VALUE_SCHEMA_VERSION = "forge.synthesis_value.v1"
PRODUCT_SYNTHESIS_VALUE_SCHEMA_VERSION = "forge.product_synthesis_value.v1"


class SynthesisValueError(ValueError):
    """Raised when a synthesis-value record violates its frozen semantics."""


class BurdenKnowledge(str, Enum):
    """Whether an operational burden has been normalized rather than omitted."""

    KNOWN = "known"
    UNKNOWN = "unknown"


class EvidenceSupport(str, Enum):
    """Weakest evidence class retained anywhere in an assessment."""

    MISSING = "missing"
    PROVENANCE_ONLY = "provenance_only"
    FAMILY_PROJECTED = "family_projected"
    EXACT_IDENTITY = "exact_identity"
    NOT_APPLICABLE = "not_applicable"


class ForwardConsistency(str, Enum):
    """Worst forward-verification state retained by an assessment."""

    FAILED = "failed"
    UNVERIFIED = "unverified"
    NOT_APPLICABLE = "not_applicable"
    EXACT_UNIQUE = "exact_unique"


class Dominance(str, Enum):
    """Partial-order relationship between two structured values."""

    BETTER = "better"
    WORSE = "worse"
    EQUAL = "equal"
    INCOMPARABLE = "incomparable"


@dataclass(frozen=True)
class BurdenEstimate:
    """A nonnegative normalized burden or an explicit unknown state."""

    knowledge: BurdenKnowledge
    count: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.knowledge, BurdenKnowledge):
            raise SynthesisValueError("burden knowledge has an unsupported state")
        if self.knowledge is BurdenKnowledge.KNOWN:
            if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 0:
                raise SynthesisValueError("known burden requires a nonnegative integer")
        elif self.count is not None:
            raise SynthesisValueError("unknown burden cannot carry a numeric count")

    @classmethod
    def unknown(cls) -> BurdenEstimate:
        return cls(knowledge=BurdenKnowledge.UNKNOWN, count=None)

    @classmethod
    def known(cls, count: int) -> BurdenEstimate:
        return cls(knowledge=BurdenKnowledge.KNOWN, count=count)

    def to_dict(self) -> dict[str, Any]:
        return {"knowledge": self.knowledge.value, "count": self.count}

    @classmethod
    def from_dict(cls, value: Any) -> BurdenEstimate:
        if not isinstance(value, dict):
            raise SynthesisValueError("burden estimate must be an object")
        try:
            knowledge = BurdenKnowledge(value.get("knowledge"))
        except ValueError as exc:
            raise SynthesisValueError("burden estimate has an unsupported state") from exc
        return cls(knowledge=knowledge, count=value.get("count"))


@dataclass(frozen=True)
class ComponentSynthesisValue:
    """Evidence-weighted route-completion record for one component."""

    target: RouteTarget
    assessment_outcome: AssessmentOutcome
    forward_consistency: ForwardConsistency
    evidence_support: EvidenceSupport
    route_step_count: int
    maximum_route_depth: int
    leaf_count: int
    current_terminal_leaf_count: int
    unavailable_terminal_leaf_count: int
    unassessed_terminal_leaf_count: int
    missing_knowledge_leaf_count: int
    outside_support_leaf_count: int
    incompatible_leaf_count: int
    budget_exhausted_leaf_count: int
    invalid_input_leaf_count: int
    execution_error_leaf_count: int
    protection_burden: BurdenEstimate
    purification_burden: BurdenEstimate

    def __post_init__(self) -> None:
        if not isinstance(self.target, RouteTarget):
            raise SynthesisValueError("component value target must be a RouteTarget")
        if not isinstance(self.assessment_outcome, AssessmentOutcome):
            raise SynthesisValueError("component value outcome is unsupported")
        if not isinstance(self.forward_consistency, ForwardConsistency):
            raise SynthesisValueError("forward-consistency state is unsupported")
        if not isinstance(self.evidence_support, EvidenceSupport):
            raise SynthesisValueError("evidence-support state is unsupported")
        integer_fields = (
            "route_step_count",
            "maximum_route_depth",
            "leaf_count",
            "current_terminal_leaf_count",
            "unavailable_terminal_leaf_count",
            "unassessed_terminal_leaf_count",
            "missing_knowledge_leaf_count",
            "outside_support_leaf_count",
            "incompatible_leaf_count",
            "budget_exhausted_leaf_count",
            "invalid_input_leaf_count",
            "execution_error_leaf_count",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SynthesisValueError(f"{name} must be a nonnegative integer")
        if self.leaf_count == 0:
            raise SynthesisValueError("component route tree must contain at least one leaf")
        categorized = sum(
            (
                self.current_terminal_leaf_count,
                self.unavailable_terminal_leaf_count,
                self.unassessed_terminal_leaf_count,
                self.missing_knowledge_leaf_count,
                self.outside_support_leaf_count,
                self.incompatible_leaf_count,
                self.budget_exhausted_leaf_count,
                self.invalid_input_leaf_count,
                self.execution_error_leaf_count,
            )
        )
        if categorized != self.leaf_count:
            raise SynthesisValueError("leaf outcome counts must partition all route leaves")
        if self.assessment_outcome is AssessmentOutcome.COMPLETE:
            if self.current_terminal_leaf_count != self.leaf_count:
                raise SynthesisValueError("complete assessment must close every terminal leaf")
            if self.forward_consistency not in {
                ForwardConsistency.EXACT_UNIQUE,
                ForwardConsistency.NOT_APPLICABLE,
            }:
                raise SynthesisValueError("complete assessment cannot retain unverified steps")
        if not isinstance(self.protection_burden, BurdenEstimate) or not isinstance(
            self.purification_burden, BurdenEstimate
        ):
            raise SynthesisValueError("burden fields must be BurdenEstimate records")

    @property
    def route_complete(self) -> bool:
        return self.assessment_outcome is AssessmentOutcome.COMPLETE

    @property
    def all_terminal_leaves_current(self) -> bool:
        return self.current_terminal_leaf_count == self.leaf_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SYNTHESIS_VALUE_SCHEMA_VERSION,
            "target": self.target.to_dict(),
            "assessment_outcome": self.assessment_outcome.value,
            "forward_consistency": self.forward_consistency.value,
            "evidence_support": self.evidence_support.value,
            "route_step_count": self.route_step_count,
            "maximum_route_depth": self.maximum_route_depth,
            "leaf_count": self.leaf_count,
            "current_terminal_leaf_count": self.current_terminal_leaf_count,
            "unavailable_terminal_leaf_count": self.unavailable_terminal_leaf_count,
            "unassessed_terminal_leaf_count": self.unassessed_terminal_leaf_count,
            "missing_knowledge_leaf_count": self.missing_knowledge_leaf_count,
            "outside_support_leaf_count": self.outside_support_leaf_count,
            "incompatible_leaf_count": self.incompatible_leaf_count,
            "budget_exhausted_leaf_count": self.budget_exhausted_leaf_count,
            "invalid_input_leaf_count": self.invalid_input_leaf_count,
            "execution_error_leaf_count": self.execution_error_leaf_count,
            "protection_burden": self.protection_burden.to_dict(),
            "purification_burden": self.purification_burden.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> ComponentSynthesisValue:
        if not isinstance(value, dict) or value.get("schema_version") != (
            SYNTHESIS_VALUE_SCHEMA_VERSION
        ):
            raise SynthesisValueError("unsupported component synthesis-value schema")
        try:
            outcome = AssessmentOutcome(value.get("assessment_outcome"))
            forward = ForwardConsistency(value.get("forward_consistency"))
            evidence = EvidenceSupport(value.get("evidence_support"))
        except ValueError as exc:
            raise SynthesisValueError("component value contains an unsupported state") from exc
        return cls(
            target=RouteTarget.from_dict(value.get("target")),
            assessment_outcome=outcome,
            forward_consistency=forward,
            evidence_support=evidence,
            route_step_count=value.get("route_step_count"),
            maximum_route_depth=value.get("maximum_route_depth"),
            leaf_count=value.get("leaf_count"),
            current_terminal_leaf_count=value.get("current_terminal_leaf_count"),
            unavailable_terminal_leaf_count=value.get("unavailable_terminal_leaf_count"),
            unassessed_terminal_leaf_count=value.get("unassessed_terminal_leaf_count"),
            missing_knowledge_leaf_count=value.get("missing_knowledge_leaf_count"),
            outside_support_leaf_count=value.get("outside_support_leaf_count"),
            incompatible_leaf_count=value.get("incompatible_leaf_count"),
            budget_exhausted_leaf_count=value.get("budget_exhausted_leaf_count"),
            invalid_input_leaf_count=value.get("invalid_input_leaf_count"),
            execution_error_leaf_count=value.get("execution_error_leaf_count"),
            protection_burden=BurdenEstimate.from_dict(value.get("protection_burden")),
            purification_burden=BurdenEstimate.from_dict(value.get("purification_burden")),
        )


def _walk(node: SynthesisRouteNode, depth: int = 0):
    yield node, depth
    for child in node.children:
        yield from _walk(child, depth + 1)


def _forward_state(nodes: tuple[SynthesisRouteNode, ...]) -> ForwardConsistency:
    steps = [node.step for node in nodes if node.step is not None]
    if not steps:
        return ForwardConsistency.NOT_APPLICABLE
    states = {evidence.forward_verification for step in steps for evidence in step.evidence}
    if states & {
        ForwardVerificationState.MISMATCHED,
        ForwardVerificationState.AMBIGUOUS,
    }:
        return ForwardConsistency.FAILED
    if any(not step.is_exact_source_forward_verified for step in steps) or states & {
        ForwardVerificationState.NOT_RUN,
    }:
        return ForwardConsistency.UNVERIFIED
    return ForwardConsistency.EXACT_UNIQUE


def _evidence_support(nodes: tuple[SynthesisRouteNode, ...]) -> EvidenceSupport:
    tiers = {evidence.tier for node in nodes for evidence in node.evidence}
    if not tiers:
        return EvidenceSupport.MISSING
    if EvidenceTier.PROVENANCE_ONLY in tiers:
        return EvidenceSupport.PROVENANCE_ONLY
    if EvidenceTier.FAMILY_PROJECTED in tiers:
        return EvidenceSupport.FAMILY_PROJECTED
    if EvidenceTier.EXACT_SOURCE in tiers:
        return EvidenceSupport.EXACT_IDENTITY
    return EvidenceSupport.NOT_APPLICABLE


def component_synthesis_value_from_assessment(
    assessment: SynthesisAssessment,
    *,
    protection_burden: BurdenEstimate | None = None,
    purification_burden: BurdenEstimate | None = None,
) -> ComponentSynthesisValue:
    """Translate one typed assessment without inventing missing burden data."""

    nodes_with_depth = tuple(_walk(assessment.route_tree))
    nodes = tuple(node for node, _ in nodes_with_depth)
    leaves = tuple(node for node in nodes if not node.children)
    availability_counts: dict[AvailabilityState, int] = {state: 0 for state in AvailabilityState}
    outcome_counts: dict[AssessmentOutcome, int] = {outcome: 0 for outcome in AssessmentOutcome}
    for leaf in leaves:
        if leaf.outcome is AssessmentOutcome.COMPLETE:
            current = any(
                evidence.tier is EvidenceTier.ACCEPTED_TERMINAL
                and evidence.availability is AvailabilityState.CURRENT_CLOSED
                for evidence in leaf.evidence
            )
            if not current:
                raise SynthesisValueError(
                    "complete terminal leaf lacks current accepted-terminal evidence"
                )
            availability_counts[AvailabilityState.CURRENT_CLOSED] += 1
            continue
        terminal_availability = {
            evidence.availability
            for evidence in leaf.evidence
            if evidence.tier is EvidenceTier.ACCEPTED_TERMINAL
        }
        if AvailabilityState.UNAVAILABLE in terminal_availability:
            availability_counts[AvailabilityState.UNAVAILABLE] += 1
        elif terminal_availability & {
            AvailabilityState.EXPIRED,
            AvailabilityState.UNASSESSED,
        }:
            availability_counts[AvailabilityState.UNASSESSED] += 1
        else:
            outcome_counts[leaf.outcome] += 1
    return ComponentSynthesisValue(
        target=assessment.target,
        assessment_outcome=assessment.outcome,
        forward_consistency=_forward_state(nodes),
        evidence_support=_evidence_support(nodes),
        route_step_count=sum(node.step is not None for node in nodes),
        maximum_route_depth=max(depth for _, depth in nodes_with_depth),
        leaf_count=len(leaves),
        current_terminal_leaf_count=availability_counts[AvailabilityState.CURRENT_CLOSED],
        unavailable_terminal_leaf_count=availability_counts[AvailabilityState.UNAVAILABLE],
        unassessed_terminal_leaf_count=availability_counts[AvailabilityState.UNASSESSED],
        missing_knowledge_leaf_count=outcome_counts[AssessmentOutcome.MISSING_KNOWLEDGE],
        outside_support_leaf_count=outcome_counts[AssessmentOutcome.OUTSIDE_SUPPORT],
        incompatible_leaf_count=outcome_counts[AssessmentOutcome.INCOMPATIBLE],
        budget_exhausted_leaf_count=outcome_counts[AssessmentOutcome.BUDGET_EXHAUSTED],
        invalid_input_leaf_count=outcome_counts[AssessmentOutcome.INVALID_INPUT],
        execution_error_leaf_count=outcome_counts[AssessmentOutcome.EXECUTION_ERROR],
        protection_burden=protection_burden or BurdenEstimate.unknown(),
        purification_burden=purification_burden or BurdenEstimate.unknown(),
    )


_FORWARD_RANK = {
    ForwardConsistency.FAILED: 0,
    ForwardConsistency.UNVERIFIED: 1,
    ForwardConsistency.NOT_APPLICABLE: 2,
    ForwardConsistency.EXACT_UNIQUE: 2,
}
_EVIDENCE_RANK = {
    EvidenceSupport.MISSING: 0,
    EvidenceSupport.PROVENANCE_ONLY: 1,
    EvidenceSupport.FAMILY_PROJECTED: 2,
    EvidenceSupport.EXACT_IDENTITY: 3,
    EvidenceSupport.NOT_APPLICABLE: 3,
}


def _burden_axes(value: BurdenEstimate) -> tuple[int, int]:
    if value.knowledge is BurdenKnowledge.UNKNOWN:
        return 0, 0
    assert value.count is not None
    return 1, -value.count


def _dominance_axes(value: ComponentSynthesisValue) -> tuple[int, ...]:
    unresolved = sum(
        (
            value.unavailable_terminal_leaf_count,
            value.unassessed_terminal_leaf_count,
            value.missing_knowledge_leaf_count,
            value.outside_support_leaf_count,
            value.incompatible_leaf_count,
            value.budget_exhausted_leaf_count,
            value.invalid_input_leaf_count,
            value.execution_error_leaf_count,
        )
    )
    return (
        int(value.route_complete),
        _FORWARD_RANK[value.forward_consistency],
        _EVIDENCE_RANK[value.evidence_support],
        int(value.all_terminal_leaves_current),
        -unresolved,
        -value.route_step_count,
        -value.maximum_route_depth,
        *_burden_axes(value.protection_burden),
        *_burden_axes(value.purification_burden),
    )


def compare_component_synthesis_values(
    left: ComponentSynthesisValue,
    right: ComponentSynthesisValue,
) -> Dominance:
    """Return a closure-gated Pareto relationship preserving blocker taxonomy."""

    if not isinstance(left, ComponentSynthesisValue) or not isinstance(
        right, ComponentSynthesisValue
    ):
        raise SynthesisValueError("component comparison requires structured values")
    if left.target.role != right.target.role:
        raise SynthesisValueError("component comparisons require the same Ugi role")
    if left.route_complete != right.route_complete:
        return Dominance.BETTER if left.route_complete else Dominance.WORSE
    if (
        left.assessment_outcome is not right.assessment_outcome
        and not left.route_complete
        and not right.route_complete
    ):
        return Dominance.INCOMPARABLE
    left_axes = _dominance_axes(left)
    right_axes = _dominance_axes(right)
    better = any(a > b for a, b in zip(left_axes, right_axes, strict=True))
    worse = any(a < b for a, b in zip(left_axes, right_axes, strict=True))
    if better and worse:
        return Dominance.INCOMPARABLE
    if better:
        return Dominance.BETTER
    if worse:
        return Dominance.WORSE
    return Dominance.EQUAL


@dataclass(frozen=True)
class ProductSynthesisValue:
    """Role-preserving aggregation without a scalar product score."""

    product_smiles: str
    l1_forward_consistent: bool
    components: tuple[tuple[str, ComponentSynthesisValue], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.product_smiles, str) or not self.product_smiles:
            raise SynthesisValueError("product SMILES must be nonempty")
        if not isinstance(self.l1_forward_consistent, bool):
            raise SynthesisValueError("product L1 forward state must be boolean")
        if not isinstance(self.components, tuple) or not self.components:
            raise SynthesisValueError("product value requires component records")
        roles = [role for role, _ in self.components]
        if any(not isinstance(role, str) or not role for role in roles):
            raise SynthesisValueError("product component roles must be nonempty")
        if len(set(roles)) != len(roles):
            raise SynthesisValueError("product component roles must be unique")
        if any(not isinstance(value, ComponentSynthesisValue) for _, value in self.components):
            raise SynthesisValueError("product components must contain structured values")

    @property
    def route_complete(self) -> bool:
        return self.l1_forward_consistent and all(
            value.route_complete for _, value in self.components
        )

    def by_role(self) -> dict[str, ComponentSynthesisValue]:
        return dict(self.components)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PRODUCT_SYNTHESIS_VALUE_SCHEMA_VERSION,
            "product_smiles": self.product_smiles,
            "l1_forward_consistent": self.l1_forward_consistent,
            "components": [
                {"role": role, "value": value.to_dict()} for role, value in self.components
            ],
            "route_complete": self.route_complete,
            "scalar_value": None,
            "success_probability": None,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ProductSynthesisValue:
        if not isinstance(value, dict) or value.get("schema_version") != (
            PRODUCT_SYNTHESIS_VALUE_SCHEMA_VERSION
        ):
            raise SynthesisValueError("unsupported product synthesis-value schema")
        components = value.get("components")
        if not isinstance(components, list):
            raise SynthesisValueError("product components must be a list")
        record = cls(
            product_smiles=value.get("product_smiles"),
            l1_forward_consistent=value.get("l1_forward_consistent"),
            components=tuple(
                (
                    item.get("role"),
                    ComponentSynthesisValue.from_dict(item.get("value")),
                )
                for item in components
                if isinstance(item, dict)
            ),
        )
        if len(record.components) != len(components):
            raise SynthesisValueError("product component entry is malformed")
        if value.get("route_complete") != record.route_complete:
            raise SynthesisValueError("serialized product closure flag is inconsistent")
        if value.get("scalar_value") is not None or value.get("success_probability") is not None:
            raise SynthesisValueError("pre-prospective value cannot contain scalar probabilities")
        return record


def compare_product_synthesis_values(
    left: ProductSynthesisValue,
    right: ProductSynthesisValue,
) -> Dominance:
    """Compare exact L1 and closure lexicographically, then roles by Pareto order."""

    left_by_role = left.by_role()
    right_by_role = right.by_role()
    if set(left_by_role) != set(right_by_role):
        raise SynthesisValueError("product comparisons require identical role sets")
    if left.l1_forward_consistent != right.l1_forward_consistent:
        return Dominance.BETTER if left.l1_forward_consistent else Dominance.WORSE
    if left.route_complete != right.route_complete:
        return Dominance.BETTER if left.route_complete else Dominance.WORSE
    relationships = {
        compare_component_synthesis_values(left_by_role[role], right_by_role[role])
        for role in left_by_role
    }
    if Dominance.INCOMPARABLE in relationships or (
        Dominance.BETTER in relationships and Dominance.WORSE in relationships
    ):
        return Dominance.INCOMPARABLE
    if Dominance.BETTER in relationships:
        return Dominance.BETTER
    if Dominance.WORSE in relationships:
        return Dominance.WORSE
    return Dominance.EQUAL
