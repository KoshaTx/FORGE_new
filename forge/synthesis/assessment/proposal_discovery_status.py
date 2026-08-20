"""Source-neutral discovery status for proposal-only upstream route hypotheses.

This layer is additive to the exact-scope :mod:`l2_forward_resolver`.  It first
asks that resolver whether a proposal is one exact known route.  Proposals with
no exact admission are then tested against the same registry-owned forward
transforms without projecting their evidence scope.  A successful projection
establishes graph consistency only; it cannot create route evidence, closure,
success probability, or synthesis value.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from rdkit import Chem, rdBase

from forge.core.hashing import sha256_json as _sha256_payload
from forge.synthesis.assessment.l2_forward_resolver import (
    L2ForwardExecutor,
    L2ForwardResolution,
    L2ForwardResolutionStatus,
    execute_qualified_forward,
)
from forge.synthesis.engine.proposal_engine import SingleStepRetrosynthesisProposal
from forge.synthesis.engine.qualified_forward import QualifiedForwardError, QualifiedForwardReaction


class ProposalDiscoveryError(ValueError):
    """Raised when a discovery-resolution input or result is malformed."""


class ProposalDiscoveryStatus(str, Enum):
    """Mutually exclusive, non-evidentiary proposal discovery states."""

    EXACT_KNOWN_ROUTE = "exact_known_route"
    KNOWN_FAMILY_FORWARD_CONSISTENT_PROJECTION = "known_family_forward_consistent_projection"
    NEW_FAMILY_HYPOTHESIS_RETAINED = "new_family_hypothesis_retained"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"


class SourceNeutralRouteAdjudicationStatus(str, Enum):
    """Downstream state after identical independent route contracts are applied."""

    QUALIFIED_ROUTE = "qualified_route"
    UNQUALIFIED_HYPOTHESIS_MISSING_EVIDENCE = "unqualified_hypothesis_missing_evidence"
    REJECTED_BY_INDEPENDENT_CONTRACT = "rejected_by_independent_contract"


class FamilyForwardTransform(Protocol):
    """Registry-owned transform interface consumed without chemistry redefinition."""

    reaction: QualifiedForwardReaction
    transform_sha256: str

    @property
    def reaction_id(self) -> str: ...


class ExactKnownRouteResolver(Protocol):
    """Existing exact-scope resolver interface used as the first authority."""

    @property
    def transforms(self) -> tuple[FamilyForwardTransform, ...]: ...

    @property
    def resolver_config_sha256(self) -> str: ...

    def resolve(
        self,
        proposal: SingleStepRetrosynthesisProposal,
        *,
        maximum_forward_calls: int | None = None,
    ) -> L2ForwardResolution: ...


def _canonical_connected(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProposalDiscoveryError(f"{label} must be nonempty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise ProposalDiscoveryError(f"{label} must be one valid connected molecule")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


@dataclass(frozen=True)
class FamilyProjectionTrace:
    """One registry-owned family transform attempted without evidence projection."""

    reaction_id: str
    transform_sha256: str
    role_ordered_reactants: tuple[str, ...]
    products: tuple[str, ...]
    target_reconstructed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "reaction_id": self.reaction_id,
            "transform_sha256": self.transform_sha256,
            "role_ordered_reactants": list(self.role_ordered_reactants),
            "products": list(self.products),
            "target_reconstructed": self.target_reconstructed,
            "graph_consistency_only": True,
            "evidence_projected": False,
        }


@dataclass(frozen=True)
class ProposalDiscoveryResolution:
    """Discovery status that is structurally unable to carry route authority."""

    proposal_sha256: str
    target_smiles: str
    status: ProposalDiscoveryStatus
    exact_resolution_status: L2ForwardResolutionStatus
    exact_resolver_config_sha256: str
    exact_forward_calls: int
    family_projection_calls: int
    maximum_family_projection_calls: int
    traces: tuple[FamilyProjectionTrace, ...]
    accepted_trace_index: int | None = None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ProposalDiscoveryStatus):
            raise ProposalDiscoveryError("unsupported proposal discovery status")
        if not isinstance(self.exact_resolution_status, L2ForwardResolutionStatus):
            raise ProposalDiscoveryError("unsupported exact resolver status")
        if (
            self.exact_forward_calls < 0
            or self.family_projection_calls < 0
            or self.maximum_family_projection_calls < 1
        ):
            raise ProposalDiscoveryError("invalid family-projection call count")
        if self.forward_calls > self.maximum_family_projection_calls:
            raise ProposalDiscoveryError("forward calls exceed the frozen budget")
        if self.accepted_trace_index is not None:
            if (
                self.status
                is not ProposalDiscoveryStatus.KNOWN_FAMILY_FORWARD_CONSISTENT_PROJECTION
                or self.accepted_trace_index < 0
                or self.accepted_trace_index >= len(self.traces)
                or not self.traces[self.accepted_trace_index].target_reconstructed
            ):
                raise ProposalDiscoveryError("accepted projection trace is inconsistent")
        elif self.status is ProposalDiscoveryStatus.KNOWN_FAMILY_FORWARD_CONSISTENT_PROJECTION:
            raise ProposalDiscoveryError("known-family projection requires one accepted trace")

    @property
    def graph_consistent(self) -> bool:
        return self.status in {
            ProposalDiscoveryStatus.EXACT_KNOWN_ROUTE,
            ProposalDiscoveryStatus.KNOWN_FAMILY_FORWARD_CONSISTENT_PROJECTION,
        }

    @property
    def retained_for_discovery(self) -> bool:
        return self.status is ProposalDiscoveryStatus.NEW_FAMILY_HYPOTHESIS_RETAINED

    @property
    def forward_calls(self) -> int:
        return self.exact_forward_calls + self.family_projection_calls

    @property
    def may_enter_synthesis_value(self) -> bool:
        return False

    @property
    def resolution_sha256(self) -> str:
        return _sha256_payload(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": "forge.proposal_discovery_resolution.v1",
            "proposal_sha256": self.proposal_sha256,
            "target_smiles": self.target_smiles,
            "status": self.status.value,
            "exact_resolution_status": self.exact_resolution_status.value,
            "exact_resolver_config_sha256": self.exact_resolver_config_sha256,
            "exact_forward_calls": self.exact_forward_calls,
            "family_projection_calls": self.family_projection_calls,
            "forward_calls": self.forward_calls,
            "maximum_family_projection_calls": self.maximum_family_projection_calls,
            "traces": [trace.to_dict() for trace in self.traces],
            "accepted_trace_index": self.accepted_trace_index,
            "rejection_reason": self.rejection_reason,
            "graph_consistent": self.graph_consistent,
            "retained_for_discovery": self.retained_for_discovery,
            "chemical_incompatibility_asserted": False,
            "local_novelty_is_rejection_basis": False,
            "downstream_source_neutral_adjudication_required": True,
            "proposal_source_used_to_select_family": False,
            "model_score_used": False,
            "model_reaction_class_used": False,
            "substrate_scope_state": "unassessed",
            "evidence_tier": None,
            "success_probability": None,
            "route_closure_authorized": False,
            "may_enter_synthesis_value": False,
        }
        if include_hash:
            value["resolution_sha256"] = self.resolution_sha256
        return value


@dataclass(frozen=True)
class SourceNeutralRouteAdjudication:
    """Machine-readable proof that proposal source cannot change route state/value."""

    proposal_sha256: str
    proposal_discovery_status: ProposalDiscoveryStatus
    status: SourceNeutralRouteAdjudicationStatus
    independent_checks: Mapping[str, bool]
    adjudicated_route_state: str | None
    adjudicated_route_value: Mapping[str, Any] | None
    rejection_basis: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "forge.source_neutral_route_adjudication.v1",
            "proposal_sha256": self.proposal_sha256,
            "proposal_discovery_status": self.proposal_discovery_status.value,
            "status": self.status.value,
            "independent_checks": dict(sorted(self.independent_checks.items())),
            "adjudicated_route_state": self.adjudicated_route_state,
            "adjudicated_route_value": self.adjudicated_route_value,
            "rejection_basis": self.rejection_basis,
            "proposal_source_used_to_set_route_state": False,
            "proposal_source_used_to_set_route_value": False,
            "local_novelty_used_as_chemical_incompatibility": False,
        }


def adjudicate_source_neutral_route(
    discovery: ProposalDiscoveryResolution,
    *,
    independent_forward_consistent: bool,
    substrate_scope_qualified: bool,
    evidence_qualified: bool,
    operationally_compatible: bool,
    l3_terminal_closed: bool,
    qualified_route_state: str | None = None,
    qualified_route_value: Mapping[str, Any] | None = None,
) -> SourceNeutralRouteAdjudication:
    """Apply source-blind route gates after proposal discovery.

    The discovery status is retained only as provenance.  A locally novel
    proposal that later passes every independent contract receives the supplied
    route state and value unchanged.  Missing scope/evidence/L3 support remains
    an unqualified hypothesis, not an assertion of chemical incompatibility.
    """

    if not isinstance(discovery, ProposalDiscoveryResolution):
        raise ProposalDiscoveryError("source-neutral adjudication requires a discovery receipt")
    checks = {
        "independent_forward_consistent": independent_forward_consistent,
        "substrate_scope_qualified": substrate_scope_qualified,
        "evidence_qualified": evidence_qualified,
        "operationally_compatible": operationally_compatible,
        "l3_terminal_closed": l3_terminal_closed,
    }
    if any(not isinstance(value, bool) for value in checks.values()):
        raise ProposalDiscoveryError("source-neutral route checks must be booleans")
    if all(checks.values()):
        if not isinstance(qualified_route_state, str) or not qualified_route_state:
            raise ProposalDiscoveryError("qualified route state is required after all checks pass")
        if not isinstance(qualified_route_value, Mapping):
            raise ProposalDiscoveryError("qualified route value is required after all checks pass")
        return SourceNeutralRouteAdjudication(
            proposal_sha256=discovery.proposal_sha256,
            proposal_discovery_status=discovery.status,
            status=SourceNeutralRouteAdjudicationStatus.QUALIFIED_ROUTE,
            independent_checks=checks,
            adjudicated_route_state=qualified_route_state,
            adjudicated_route_value=dict(qualified_route_value),
            rejection_basis=None,
        )
    if independent_forward_consistent and operationally_compatible:
        status = SourceNeutralRouteAdjudicationStatus.UNQUALIFIED_HYPOTHESIS_MISSING_EVIDENCE
        rejection_basis = None
    else:
        status = SourceNeutralRouteAdjudicationStatus.REJECTED_BY_INDEPENDENT_CONTRACT
        rejection_basis = (
            "independent_forward_contract_failed"
            if not independent_forward_consistent
            else "independent_operational_contract_failed"
        )
    if qualified_route_state is not None or qualified_route_value is not None:
        raise ProposalDiscoveryError("unqualified route cannot carry route state or value")
    return SourceNeutralRouteAdjudication(
        proposal_sha256=discovery.proposal_sha256,
        proposal_discovery_status=discovery.status,
        status=status,
        independent_checks=checks,
        adjudicated_route_state=None,
        adjudicated_route_value=None,
        rejection_basis=rejection_basis,
    )


class SourceNeutralProposalDiscoveryResolver:
    """Classify exact, projected, new-family, ambiguous, and rejected proposals."""

    def __init__(
        self,
        *,
        exact_resolver: ExactKnownRouteResolver,
        maximum_family_projection_calls: int = 50,
        maximum_products_per_assignment: int = 64,
        executor: L2ForwardExecutor = execute_qualified_forward,
    ) -> None:
        if not hasattr(exact_resolver, "resolve") or not hasattr(exact_resolver, "transforms"):
            raise ProposalDiscoveryError("an exact known-route resolver is required")
        if maximum_family_projection_calls < 1 or maximum_products_per_assignment < 1:
            raise ProposalDiscoveryError("projection budgets must be positive")
        if not callable(executor):
            raise ProposalDiscoveryError("family projection executor must be callable")
        self._exact_resolver = exact_resolver
        self._transforms = tuple(
            sorted(exact_resolver.transforms, key=lambda transform: transform.reaction_id)
        )
        self._maximum_family_projection_calls = maximum_family_projection_calls
        self._maximum_products_per_assignment = maximum_products_per_assignment
        self._executor = executor

    @property
    def exact_resolver_config_sha256(self) -> str:
        return self._exact_resolver.resolver_config_sha256

    @property
    def transforms(self) -> tuple[FamilyForwardTransform, ...]:
        """Registry transforms exposed for source-neutral semantic audits."""

        return self._transforms

    def resolve(
        self,
        proposal: SingleStepRetrosynthesisProposal,
        *,
        maximum_forward_calls: int | None = None,
    ) -> ProposalDiscoveryResolution:
        if not isinstance(proposal, SingleStepRetrosynthesisProposal):
            raise ProposalDiscoveryError("discovery resolver requires a typed proposal")
        call_budget = self._maximum_family_projection_calls
        if maximum_forward_calls is not None:
            if maximum_forward_calls < 1:
                raise ProposalDiscoveryError("maximum_forward_calls must be positive")
            call_budget = min(call_budget, maximum_forward_calls)

        exact = self._exact_resolver.resolve(
            proposal,
            maximum_forward_calls=call_budget,
        )
        target = _canonical_connected(proposal.request.target.canonical_smiles, label="target")
        common = {
            "proposal_sha256": proposal.proposal_sha256,
            "target_smiles": target,
            "exact_resolution_status": exact.status,
            "exact_resolver_config_sha256": self.exact_resolver_config_sha256,
            "exact_forward_calls": exact.forward_calls,
            "maximum_family_projection_calls": call_budget,
        }
        if exact.status is L2ForwardResolutionStatus.EXACT_UNIQUE:
            return ProposalDiscoveryResolution(
                status=ProposalDiscoveryStatus.EXACT_KNOWN_ROUTE,
                family_projection_calls=0,
                traces=(),
                **common,
            )
        if exact.status is L2ForwardResolutionStatus.REJECT_FORWARD_MISMATCH:
            return ProposalDiscoveryResolution(
                status=ProposalDiscoveryStatus.REJECTED,
                family_projection_calls=0,
                traces=(),
                rejection_reason="exact_admission_forward_mismatch",
                **common,
            )
        if exact.status is not L2ForwardResolutionStatus.CENSOR_NO_VERIFIER:
            return ProposalDiscoveryResolution(
                status=ProposalDiscoveryStatus.AMBIGUOUS,
                family_projection_calls=0,
                traces=(),
                rejection_reason=f"exact_resolver:{exact.status.value}",
                **common,
            )

        reactants = tuple(proposal.reactant_smiles)
        assignments: list[tuple[FamilyForwardTransform, tuple[str, ...]]] = []
        for transform in self._transforms:
            if len(transform.reaction.role_names) != len(reactants):
                continue
            assignments.extend(
                (transform, assignment)
                for assignment in sorted(set(itertools.permutations(reactants)))
            )
        assignments.sort(key=lambda item: (item[0].reaction_id, item[1]))
        if len(assignments) > call_budget:
            return ProposalDiscoveryResolution(
                status=ProposalDiscoveryStatus.AMBIGUOUS,
                family_projection_calls=0,
                traces=(),
                rejection_reason="family_projection_budget_exhausted_before_execution",
                **common,
            )

        traces: list[FamilyProjectionTrace] = []
        for transform, assignment in assignments:
            try:
                products = self._executor(
                    transform.reaction,
                    assignment,
                    max_products=self._maximum_products_per_assignment,
                )
            except (QualifiedForwardError, RuntimeError, ValueError):
                return ProposalDiscoveryResolution(
                    status=ProposalDiscoveryStatus.AMBIGUOUS,
                    family_projection_calls=len(traces) + 1,
                    traces=tuple(traces),
                    rejection_reason="family_projection_execution_error",
                    **common,
                )
            canonical_products = tuple(
                sorted(
                    {_canonical_connected(product, label="forward product") for product in products}
                )
            )
            traces.append(
                FamilyProjectionTrace(
                    reaction_id=transform.reaction_id,
                    transform_sha256=transform.transform_sha256,
                    role_ordered_reactants=assignment,
                    products=canonical_products,
                    target_reconstructed=canonical_products == (target,),
                )
            )

        if any(len(trace.products) > 1 for trace in traces):
            return ProposalDiscoveryResolution(
                status=ProposalDiscoveryStatus.AMBIGUOUS,
                family_projection_calls=len(traces),
                traces=tuple(traces),
                rejection_reason="family_projection_multiple_products",
                **common,
            )
        matches = [index for index, trace in enumerate(traces) if trace.target_reconstructed]
        if len(matches) == 1:
            return ProposalDiscoveryResolution(
                status=ProposalDiscoveryStatus.KNOWN_FAMILY_FORWARD_CONSISTENT_PROJECTION,
                family_projection_calls=len(traces),
                traces=tuple(traces),
                accepted_trace_index=matches[0],
                **common,
            )
        if len(matches) > 1:
            return ProposalDiscoveryResolution(
                status=ProposalDiscoveryStatus.AMBIGUOUS,
                family_projection_calls=len(traces),
                traces=tuple(traces),
                rejection_reason="multiple_known_family_assignments_reconstruct_target",
                **common,
            )
        return ProposalDiscoveryResolution(
            status=ProposalDiscoveryStatus.NEW_FAMILY_HYPOTHESIS_RETAINED,
            family_projection_calls=len(traces),
            traces=tuple(traces),
            **common,
        )


__all__ = [
    "FamilyForwardTransform",
    "FamilyProjectionTrace",
    "ProposalDiscoveryError",
    "ProposalDiscoveryResolution",
    "ProposalDiscoveryStatus",
    "SourceNeutralRouteAdjudication",
    "SourceNeutralRouteAdjudicationStatus",
    "SourceNeutralProposalDiscoveryResolver",
    "adjudicate_source_neutral_route",
]
