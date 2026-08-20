"""Proposal-only contract for learned single-step retrosynthesis engines.

Learned engines can broaden bounded route search, but their ranked outputs are
not experimental evidence and cannot close a FORGE route.  This module keeps
that distinction structural: a :class:`SingleStepRetrosynthesisProposal` has
no ``EvidenceRecord`` field and cannot be converted into the evidence-bearing
``RouteStepProposal`` used by the recursive assessor.

Admission here means only "eligible to enter bounded search".  Exact forward
reconstruction, Ugi-root qualification and the operational screen are required
before admission.  Substrate-scope evidence and terminal-material closure must
still be established independently before a route can contribute synthesis
value.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from rdkit import Chem

from forge.route.assessment.ugi3_support_boundary import (
    UGI_COMPONENT_ROLES,
    AuthenticatedInternalRoleRegistry,
    TargetQualification,
    validate_root_qualification,
)
from forge.route.engine.planner import ForwardVerificationState, RouteTarget

PROPOSAL_ENGINE_SCHEMA_VERSION = "forge.route_proposal_engine.v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PROPOSAL_REACTANT_IDENTITY_POLICY = "rdkit_canonical_constitutional_connected_v1"


class ProposalEngineError(ValueError):
    """Raised when proposal metadata or admission state is malformed."""


def _nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProposalEngineError(f"{label} must be a nonempty string")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ProposalEngineError(f"{label} must contain 64 lowercase hexadecimal characters")
    return value


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _identity_to_dict(
    identity: tuple[str, str, tuple[str, ...]],
) -> dict[str, Any]:
    role, canonical_smiles, context = identity
    return {
        "role": role,
        "canonical_smiles": canonical_smiles,
        "product_context_smiles": list(context),
    }


def _validate_target_identity(
    value: Any,
    *,
    label: str,
) -> tuple[str, str, tuple[str, ...]]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ProposalEngineError(f"{label} is malformed")
    role, canonical_smiles, context = value
    if not isinstance(role, str) or not role.strip():
        raise ProposalEngineError(f"{label} role must be nonempty")
    if not isinstance(canonical_smiles, str) or not canonical_smiles.strip():
        raise ProposalEngineError(f"{label} canonical SMILES must be nonempty")
    if not isinstance(context, tuple) or any(
        not isinstance(item, str) or not item for item in context
    ):
        raise ProposalEngineError(f"{label} product context is malformed")
    return value


def _canonical_connected_constitution(smiles: Any, *, label: str) -> str:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ProposalEngineError(f"{label} must be a nonempty SMILES string")
    try:
        molecule = Chem.MolFromSmiles(smiles)
    except Exception as error:  # pragma: no cover - defensive RDKit boundary
        raise ProposalEngineError(f"{label} could not be parsed") from error
    if molecule is None:
        raise ProposalEngineError(f"{label} could not be parsed")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ProposalEngineError(f"{label} must be one connected constitutional reactant")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


class OperationalCompatibility(str, Enum):
    """Compatibility with the frozen laboratory and route-search policy."""

    PASSED = "passed"
    REVIEW_REQUIRED = "review_required"
    REJECTED = "rejected"
    NOT_ASSESSED = "not_assessed"


class SubstrateScopeState(str, Enum):
    """Evidence state for applying the proposed step to this exact target."""

    EXACT_SUBSTRATE = "exact_substrate"
    LIPID_LIKE_ANALOGUE = "lipid_like_analogue"
    GENERAL_ANALOGUE = "general_analogue"
    FAMILY_ONLY = "family_only"
    UNASSESSED = "unassessed"
    CONTRADICTED = "contradicted"


class ProposalDisposition(str, Enum):
    """Whether a learned proposal may enter bounded route search."""

    ADMIT_TO_BOUNDED_SEARCH = "admit_to_bounded_search"
    REJECT = "reject"
    CENSOR = "censor"


class ProposalTargetKind(str, Enum):
    """Whether a request assesses the exact L1 root or an authenticated child."""

    ROOT = "root"
    INTERNAL = "internal"


class ProposalReason(str, Enum):
    """Stable reason codes retained beside every proposal decision."""

    PROPOSAL_ONLY = "proposal_only_not_route_evidence"
    ROOT_NOT_EXACT_L1 = "root_not_exact_l1"
    ROOT_ROLE_UNSUPPORTED = "root_role_unsupported"
    ROOT_HANDLE_UNQUALIFIED = "root_handle_unqualified"
    ROOT_QUALIFICATION_MISMATCH = "root_qualification_mismatch"
    INTERNAL_LINEAGE_MISMATCH = "internal_lineage_mismatch"
    PROPOSAL_CONTENT_MISMATCH = "proposal_content_mismatch"
    TARGET_CONTEXT_MISMATCH = "target_context_mismatch"
    SCREEN_POLICY_MISMATCH = "screen_policy_mismatch"
    FORWARD_EXACT_UNIQUE = "forward_exact_unique"
    FORWARD_NOT_RUN = "forward_not_run"
    FORWARD_AMBIGUOUS = "forward_ambiguous"
    FORWARD_MISMATCH = "forward_mismatch"
    OPERATIONAL_PASSED = "operational_passed"
    OPERATIONAL_REVIEW_REQUIRED = "operational_review_required"
    OPERATIONAL_REJECTED = "operational_rejected"
    OPERATIONAL_NOT_ASSESSED = "operational_not_assessed"
    SUBSTRATE_SCOPE_CONTRADICTED = "substrate_scope_contradicted"
    SUBSTRATE_SCOPE_EXACT_EVIDENCE = "substrate_scope_exact_independent_evidence"
    SUBSTRATE_SCOPE_REQUIRES_EVIDENCE = "substrate_scope_requires_independent_evidence"


@dataclass(frozen=True)
class RootQualificationReceipt:
    """Caller-owned, content-addressed qualification for one exact Ugi route root.

    The learned backend may read this receipt but cannot originate or amend it.
    Its artifact hashes bind the request to the independently qualified terminal,
    L1 transform and support-boundary record used by production route assessment.
    """

    route_root: RouteTarget
    qualification: TargetQualification
    terminal_sha256: str
    generator_checkpoint_sha256: str
    l1_reaction_sha256: str
    qualification_artifact_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.route_root, RouteTarget):
            raise ProposalEngineError("root qualification route_root is malformed")
        if not isinstance(self.qualification, TargetQualification):
            raise ProposalEngineError("root qualification record is malformed")
        for name in (
            "terminal_sha256",
            "generator_checkpoint_sha256",
            "l1_reaction_sha256",
            "qualification_artifact_sha256",
        ):
            _sha256(getattr(self, name), label=name)
        invalid = validate_root_qualification(self.route_root, self.qualification)
        if invalid is not None:
            raise ProposalEngineError(invalid.detail)
        if not self.qualification.role_handle_qualified:
            raise ProposalEngineError("root handle is not qualified")

    def _content_dict(self) -> dict[str, Any]:
        qualification = self.qualification
        return {
            "route_root": self.route_root.to_dict(),
            "qualification": {
                "exact_l1_eligible": qualification.exact_l1_eligible,
                "supported_ugi_role": qualification.supported_ugi_role,
                "role_handle_qualified": qualification.role_handle_qualified,
                "molecular_support_state": qualification.molecular_support_state.value,
                "declared_exclusion_code": qualification.declared_exclusion_code,
                "declared_exclusion_policy_locator": (
                    qualification.declared_exclusion_policy_locator
                ),
            },
            "terminal_sha256": self.terminal_sha256,
            "generator_checkpoint_sha256": self.generator_checkpoint_sha256,
            "l1_reaction_sha256": self.l1_reaction_sha256,
            "qualification_artifact_sha256": self.qualification_artifact_sha256,
        }

    @property
    def receipt_sha256(self) -> str:
        return _content_sha256(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True)
class InternalTargetLineage:
    """Content-addressed proof that an internal target is one screened child."""

    route_root_identity: tuple[str, str, tuple[str, ...]]
    parent_target_identity: tuple[str, str, tuple[str, ...]]
    target_identity: tuple[str, str, tuple[str, ...]]
    parent_depth: int
    child_index: int
    parent_canonical_reactants: tuple[str, ...]
    parent_proposal_sha256: str
    parent_screen_sha256: str
    authenticated_internal_roles: AuthenticatedInternalRoleRegistry
    internal_role_registry_sha256: str

    def __post_init__(self) -> None:
        _validate_target_identity(self.route_root_identity, label="lineage route root")
        _validate_target_identity(self.parent_target_identity, label="lineage parent target")
        _validate_target_identity(self.target_identity, label="lineage target")
        for name in ("parent_depth", "child_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProposalEngineError(f"{name} must be a nonnegative integer")
        if not isinstance(self.parent_canonical_reactants, tuple) or not (
            self.parent_canonical_reactants
        ):
            raise ProposalEngineError("lineage parent reactants must be a nonempty tuple")
        canonical = tuple(
            _canonical_connected_constitution(value, label="lineage parent reactant")
            for value in self.parent_canonical_reactants
        )
        object.__setattr__(self, "parent_canonical_reactants", canonical)
        if self.child_index >= len(canonical):
            raise ProposalEngineError("lineage child_index is outside parent reactants")
        for name in ("parent_proposal_sha256", "parent_screen_sha256"):
            _sha256(getattr(self, name), label=name)
        if not isinstance(
            self.authenticated_internal_roles,
            AuthenticatedInternalRoleRegistry,
        ):
            raise ProposalEngineError("authenticated internal-role registry is malformed")
        _sha256(self.internal_role_registry_sha256, label="internal_role_registry_sha256")
        expected_registry_sha = _content_sha256(
            {"roles": sorted(self.authenticated_internal_roles.roles)}
        )
        if self.internal_role_registry_sha256 != expected_registry_sha:
            raise ProposalEngineError("internal-role registry hash does not match its roles")
        target_role, target_smiles, _ = self.target_identity
        if target_role not in self.authenticated_internal_roles:
            raise ProposalEngineError("internal target role is not authenticated")
        if target_smiles != canonical[self.child_index]:
            raise ProposalEngineError(
                "internal target is not the selected canonical parent reactant"
            )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "route_root_identity": _identity_to_dict(self.route_root_identity),
            "parent_target_identity": _identity_to_dict(self.parent_target_identity),
            "target_identity": _identity_to_dict(self.target_identity),
            "parent_depth": self.parent_depth,
            "child_index": self.child_index,
            "parent_canonical_reactants": list(self.parent_canonical_reactants),
            "parent_proposal_sha256": self.parent_proposal_sha256,
            "parent_screen_sha256": self.parent_screen_sha256,
            "authenticated_internal_roles": sorted(self.authenticated_internal_roles.roles),
            "internal_role_registry_sha256": self.internal_role_registry_sha256,
        }

    @property
    def lineage_sha256(self) -> str:
        return _content_sha256(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "lineage_sha256": self.lineage_sha256}


@dataclass(frozen=True)
class ProposalBackendManifest:
    """Immutable identity and provenance for one learned proposal backend."""

    backend_id: str
    implementation_version: str
    checkpoint_sha256: str
    checkpoint_license: str
    training_corpus_id: str
    training_corpus_snapshot_sha256: str
    source_locator: str

    def __post_init__(self) -> None:
        for name in (
            "backend_id",
            "implementation_version",
            "checkpoint_license",
            "training_corpus_id",
            "source_locator",
        ):
            _nonempty(getattr(self, name), label=name)
        _sha256(self.checkpoint_sha256, label="checkpoint_sha256")
        _sha256(
            self.training_corpus_snapshot_sha256,
            label="training_corpus_snapshot_sha256",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROPOSAL_ENGINE_SCHEMA_VERSION,
            "backend_id": self.backend_id,
            "implementation_version": self.implementation_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_license": self.checkpoint_license,
            "training_corpus_id": self.training_corpus_id,
            "training_corpus_snapshot_sha256": self.training_corpus_snapshot_sha256,
            "source_locator": self.source_locator,
        }


@dataclass(frozen=True)
class ProposalRequest:
    """One target within a route rooted at an exact-L1 Ugi component."""

    route_root: RouteTarget
    target: RouteTarget
    depth: int
    target_kind: ProposalTargetKind
    root_qualification: RootQualificationReceipt
    operational_policy_id: str
    operational_policy_sha256: str
    internal_lineage: InternalTargetLineage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.route_root, RouteTarget) or not isinstance(self.target, RouteTarget):
            raise ProposalEngineError("proposal request targets must be RouteTarget records")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 0:
            raise ProposalEngineError("proposal request depth must be a nonnegative integer")
        if not isinstance(self.target_kind, ProposalTargetKind):
            raise ProposalEngineError("proposal target kind is unsupported")
        if not isinstance(self.root_qualification, RootQualificationReceipt):
            raise ProposalEngineError("proposal root qualification is malformed")
        if self.route_root.identity != self.root_qualification.route_root.identity:
            raise ProposalEngineError("proposal route root and qualification receipt disagree")
        _nonempty(self.operational_policy_id, label="operational_policy_id")
        _sha256(self.operational_policy_sha256, label="operational_policy_sha256")
        if self.target_kind is ProposalTargetKind.ROOT:
            if self.depth != 0 or self.target.identity != self.route_root.identity:
                raise ProposalEngineError(
                    "root proposal target must equal the depth-zero route root"
                )
            if self.internal_lineage is not None:
                raise ProposalEngineError("root proposal target cannot carry internal lineage")
        else:
            if self.depth == 0 or not isinstance(self.internal_lineage, InternalTargetLineage):
                raise ProposalEngineError("internal proposal target requires typed lineage")
            lineage = self.internal_lineage
            if (
                lineage.route_root_identity != self.route_root.identity
                or lineage.target_identity != self.target.identity
                or lineage.parent_depth + 1 != self.depth
            ):
                raise ProposalEngineError("internal proposal target disagrees with its lineage")

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_root": self.route_root.to_dict(),
            "target": self.target.to_dict(),
            "depth": self.depth,
            "target_kind": self.target_kind.value,
            "root_qualification": self.root_qualification.to_dict(),
            "operational_policy_id": self.operational_policy_id,
            "operational_policy_sha256": self.operational_policy_sha256,
            "internal_lineage": (
                None if self.internal_lineage is None else self.internal_lineage.to_dict()
            ),
        }


@dataclass(frozen=True)
class SingleStepRetrosynthesisProposal:
    """One ranked learned disconnection with deliberately no evidence tier."""

    proposal_id: str
    backend: ProposalBackendManifest
    request: ProposalRequest
    reactant_smiles: tuple[str, ...]
    rank: int
    model_score: float | None = None
    predicted_reaction_class: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.proposal_id, label="proposal_id")
        if not isinstance(self.backend, ProposalBackendManifest):
            raise ProposalEngineError("proposal backend manifest is malformed")
        if not isinstance(self.request, ProposalRequest):
            raise ProposalEngineError("proposal request is malformed")
        if not isinstance(self.reactant_smiles, tuple) or not self.reactant_smiles:
            raise ProposalEngineError("proposal must contain at least one reactant")
        canonical_reactants = tuple(
            _canonical_connected_constitution(value, label="proposal reactant")
            for value in self.reactant_smiles
        )
        object.__setattr__(self, "reactant_smiles", canonical_reactants)
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ProposalEngineError("proposal rank must be a positive integer")
        if self.model_score is not None and (
            isinstance(self.model_score, bool)
            or not isinstance(self.model_score, (int, float))
            or not math.isfinite(float(self.model_score))
        ):
            raise ProposalEngineError("proposal model score must be finite or null")
        if self.predicted_reaction_class is not None:
            _nonempty(self.predicted_reaction_class, label="predicted_reaction_class")

    def _content_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "backend": self.backend.to_dict(),
            "request": self.request.to_dict(),
            "reactant_smiles": list(self.reactant_smiles),
            "reactant_identity_policy": PROPOSAL_REACTANT_IDENTITY_POLICY,
            "rank": self.rank,
            "model_score": self.model_score,
            "predicted_reaction_class": self.predicted_reaction_class,
        }

    @property
    def proposal_sha256(self) -> str:
        return _content_sha256(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROPOSAL_ENGINE_SCHEMA_VERSION,
            **self._content_dict(),
            "proposal_sha256": self.proposal_sha256,
            "evidence_tier": None,
            "success_probability": None,
            "route_closure_authorized": False,
        }


class SingleStepProposalEngine(Protocol):
    """Backend interface; implementations return ranked proposal-only records."""

    @property
    def manifest(self) -> ProposalBackendManifest: ...

    def propose(
        self,
        request: ProposalRequest,
        *,
        maximum_proposals: int,
    ) -> tuple[SingleStepRetrosynthesisProposal, ...]: ...


def validate_proposal_batch(
    backend: ProposalBackendManifest,
    request: ProposalRequest,
    proposals: tuple[SingleStepRetrosynthesisProposal, ...],
    *,
    maximum_proposals: int,
) -> tuple[SingleStepRetrosynthesisProposal, ...]:
    """Validate one backend response before any proposal is screened.

    The caller-owned request and backend manifest are authoritative.  A model
    response cannot replace either object with self-reported metadata.
    """

    if not isinstance(backend, ProposalBackendManifest) or not isinstance(request, ProposalRequest):
        raise ProposalEngineError("proposal batch requires typed backend and request")
    if (
        isinstance(maximum_proposals, bool)
        or not isinstance(maximum_proposals, int)
        or maximum_proposals < 1
    ):
        raise ProposalEngineError("maximum_proposals must be a positive integer")
    if not isinstance(proposals, tuple):
        raise ProposalEngineError("proposal backend output must be a tuple")
    if len(proposals) > maximum_proposals:
        raise ProposalEngineError("proposal backend exceeded maximum_proposals")
    if any(not isinstance(value, SingleStepRetrosynthesisProposal) for value in proposals):
        raise ProposalEngineError("proposal backend returned a malformed record")
    if any(value.backend != backend for value in proposals):
        raise ProposalEngineError("proposal backend output changed the frozen manifest")
    if any(value.request != request for value in proposals):
        raise ProposalEngineError("proposal backend output changed the caller-owned request")
    expected_ranks = tuple(range(1, len(proposals) + 1))
    if tuple(value.rank for value in proposals) != expected_ranks:
        raise ProposalEngineError("proposal ranks must be unique, contiguous and ordered")
    proposal_ids = tuple(value.proposal_id for value in proposals)
    if len(set(proposal_ids)) != len(proposal_ids):
        raise ProposalEngineError("proposal IDs must be unique within one response")
    reactant_sets = tuple(tuple(sorted(value.reactant_smiles)) for value in proposals)
    if len(set(reactant_sets)) != len(reactant_sets):
        raise ProposalEngineError("proposal reactant sets must be unique within one response")
    return proposals


@dataclass(frozen=True)
class ProposalScreen:
    """Independent checks performed after one learned proposal is emitted."""

    proposal_sha256: str
    request_target_identity: tuple[str, str, tuple[str, ...]]
    verifier_id: str
    verifier_version: str
    forward_transform_sha256: str
    forward_verification: ForwardVerificationState
    forward_product_count: int
    forward_product_smiles: tuple[str, ...]
    operational_policy_id: str
    operational_policy_sha256: str
    operational_compatibility: OperationalCompatibility
    substrate_scope: SubstrateScopeState
    substrate_scope_evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        _sha256(self.proposal_sha256, label="proposal_sha256")
        _validate_target_identity(
            self.request_target_identity,
            label="screen request target identity",
        )
        _nonempty(self.verifier_id, label="verifier_id")
        _nonempty(self.verifier_version, label="verifier_version")
        _sha256(self.forward_transform_sha256, label="forward_transform_sha256")
        if not isinstance(self.forward_verification, ForwardVerificationState):
            raise ProposalEngineError("forward verification state is unsupported")
        if (
            isinstance(self.forward_product_count, bool)
            or not isinstance(self.forward_product_count, int)
            or self.forward_product_count < 0
        ):
            raise ProposalEngineError("forward_product_count must be nonnegative")
        if not isinstance(self.forward_product_smiles, tuple):
            raise ProposalEngineError("forward_product_smiles must be a tuple")
        canonical_products = tuple(
            _canonical_connected_constitution(value, label="forward product")
            for value in self.forward_product_smiles
        )
        object.__setattr__(self, "forward_product_smiles", canonical_products)
        if self.forward_product_count != len(canonical_products):
            raise ProposalEngineError("forward_product_count disagrees with forward_product_smiles")
        _nonempty(self.operational_policy_id, label="operational_policy_id")
        _sha256(self.operational_policy_sha256, label="operational_policy_sha256")
        if not isinstance(self.operational_compatibility, OperationalCompatibility):
            raise ProposalEngineError("operational compatibility state is unsupported")
        if not isinstance(self.substrate_scope, SubstrateScopeState):
            raise ProposalEngineError("substrate scope state is unsupported")
        if self.substrate_scope is SubstrateScopeState.EXACT_SUBSTRATE:
            _sha256(
                self.substrate_scope_evidence_sha256,
                label="substrate_scope_evidence_sha256",
            )
        elif self.substrate_scope_evidence_sha256 is not None:
            _sha256(
                self.substrate_scope_evidence_sha256,
                label="substrate_scope_evidence_sha256",
            )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "proposal_sha256": self.proposal_sha256,
            "request_target_identity": _identity_to_dict(self.request_target_identity),
            "verifier_id": self.verifier_id,
            "verifier_version": self.verifier_version,
            "forward_transform_sha256": self.forward_transform_sha256,
            "forward_verification": self.forward_verification.value,
            "forward_product_count": self.forward_product_count,
            "forward_product_smiles": list(self.forward_product_smiles),
            "operational_policy_id": self.operational_policy_id,
            "operational_policy_sha256": self.operational_policy_sha256,
            "operational_compatibility": self.operational_compatibility.value,
            "substrate_scope": self.substrate_scope.value,
            "substrate_scope_evidence_sha256": self.substrate_scope_evidence_sha256,
        }

    @property
    def screen_sha256(self) -> str:
        return _content_sha256(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "screen_sha256": self.screen_sha256}


@dataclass(frozen=True)
class ProposalDecision:
    """Fail-closed admission result; never an evidence or route-closure record."""

    proposal_id: str
    proposal_sha256: str
    screen_sha256: str
    disposition: ProposalDisposition
    reasons: tuple[ProposalReason, ...]
    can_enter_bounded_search: bool
    can_close_route: bool = False
    evidence_tier: None = None
    success_probability: None = None

    def __post_init__(self) -> None:
        _nonempty(self.proposal_id, label="proposal decision proposal_id")
        _sha256(self.proposal_sha256, label="proposal decision proposal_sha256")
        _sha256(self.screen_sha256, label="proposal decision screen_sha256")
        if not isinstance(self.disposition, ProposalDisposition):
            raise ProposalEngineError("proposal disposition is unsupported")
        if not self.reasons or any(
            not isinstance(reason, ProposalReason) for reason in self.reasons
        ):
            raise ProposalEngineError("proposal decision requires typed reasons")
        if not isinstance(self.can_enter_bounded_search, bool):
            raise ProposalEngineError("can_enter_bounded_search must be boolean")
        if self.can_close_route is not False:
            raise ProposalEngineError("learned proposals cannot authorize route closure")
        if self.evidence_tier is not None or self.success_probability is not None:
            raise ProposalEngineError(
                "learned proposals cannot carry evidence or success probability"
            )
        if self.can_enter_bounded_search != (
            self.disposition is ProposalDisposition.ADMIT_TO_BOUNDED_SEARCH
        ):
            raise ProposalEngineError("search admission disagrees with proposal disposition")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROPOSAL_ENGINE_SCHEMA_VERSION,
            "proposal_id": self.proposal_id,
            "proposal_sha256": self.proposal_sha256,
            "screen_sha256": self.screen_sha256,
            "disposition": self.disposition.value,
            "reasons": [reason.value for reason in self.reasons],
            "can_enter_bounded_search": self.can_enter_bounded_search,
            "can_close_route": False,
            "evidence_tier": None,
            "success_probability": None,
        }


def build_internal_target_lineage(
    *,
    parent_proposal: SingleStepRetrosynthesisProposal,
    parent_screen: ProposalScreen,
    parent_decision: ProposalDecision,
    child_index: int,
    target: RouteTarget,
    authenticated_internal_roles: AuthenticatedInternalRoleRegistry,
) -> InternalTargetLineage:
    """Build lineage only from one admitted, content-matched parent proposal."""

    if not isinstance(parent_proposal, SingleStepRetrosynthesisProposal):
        raise ProposalEngineError("lineage parent proposal is malformed")
    if not isinstance(parent_screen, ProposalScreen) or not isinstance(
        parent_decision,
        ProposalDecision,
    ):
        raise ProposalEngineError("lineage parent screen or decision is malformed")
    if not isinstance(target, RouteTarget):
        raise ProposalEngineError("lineage target is malformed")
    if not isinstance(
        authenticated_internal_roles,
        AuthenticatedInternalRoleRegistry,
    ):
        raise ProposalEngineError("lineage internal-role registry is malformed")
    if (
        parent_decision.disposition is not ProposalDisposition.ADMIT_TO_BOUNDED_SEARCH
        or parent_decision.proposal_sha256 != parent_proposal.proposal_sha256
        or parent_screen.proposal_sha256 != parent_proposal.proposal_sha256
        or parent_decision.screen_sha256 != parent_screen.screen_sha256
    ):
        raise ProposalEngineError("lineage parent was not admitted under the bound screen")
    registry_sha256 = _content_sha256({"roles": sorted(authenticated_internal_roles.roles)})
    return InternalTargetLineage(
        route_root_identity=parent_proposal.request.route_root.identity,
        parent_target_identity=parent_proposal.request.target.identity,
        target_identity=target.identity,
        parent_depth=parent_proposal.request.depth,
        child_index=child_index,
        parent_canonical_reactants=parent_proposal.reactant_smiles,
        parent_proposal_sha256=parent_proposal.proposal_sha256,
        parent_screen_sha256=parent_screen.screen_sha256,
        authenticated_internal_roles=authenticated_internal_roles,
        internal_role_registry_sha256=registry_sha256,
    )


def screen_learned_proposal(
    proposal: SingleStepRetrosynthesisProposal,
    screen: ProposalScreen,
) -> ProposalDecision:
    """Admit only exact-forward, role-qualified, operationally screened proposals.

    A positive result permits bounded search only.  It does not establish
    substrate scope, experimental evidence, terminal availability or route
    closure.
    """

    if not isinstance(proposal, SingleStepRetrosynthesisProposal) or not isinstance(
        screen, ProposalScreen
    ):
        raise ProposalEngineError("proposal screening requires typed inputs")
    reasons: list[ProposalReason] = [ProposalReason.PROPOSAL_ONLY]
    request = proposal.request

    if screen.proposal_sha256 != proposal.proposal_sha256:
        reasons.append(ProposalReason.PROPOSAL_CONTENT_MISMATCH)
        return _decision(proposal, screen, ProposalDisposition.CENSOR, reasons)
    if request.route_root.role not in UGI_COMPONENT_ROLES:
        reasons.append(ProposalReason.ROOT_ROLE_UNSUPPORTED)
        return _decision(proposal, screen, ProposalDisposition.REJECT, reasons)
    qualification = request.root_qualification
    if qualification.route_root.identity != request.route_root.identity:
        reasons.append(ProposalReason.ROOT_QUALIFICATION_MISMATCH)
        return _decision(proposal, screen, ProposalDisposition.REJECT, reasons)
    if not qualification.qualification.exact_l1_eligible:
        reasons.append(ProposalReason.ROOT_NOT_EXACT_L1)
        return _decision(proposal, screen, ProposalDisposition.REJECT, reasons)
    if not qualification.qualification.role_handle_qualified:
        reasons.append(ProposalReason.ROOT_HANDLE_UNQUALIFIED)
        return _decision(proposal, screen, ProposalDisposition.REJECT, reasons)
    if request.target_kind is ProposalTargetKind.INTERNAL:
        lineage = request.internal_lineage
        if lineage is None or (
            lineage.route_root_identity != request.route_root.identity
            or lineage.target_identity != request.target.identity
            or lineage.parent_depth + 1 != request.depth
        ):
            reasons.append(ProposalReason.INTERNAL_LINEAGE_MISMATCH)
            return _decision(proposal, screen, ProposalDisposition.REJECT, reasons)
    if screen.request_target_identity != request.target.identity:
        reasons.append(ProposalReason.TARGET_CONTEXT_MISMATCH)
        return _decision(proposal, screen, ProposalDisposition.CENSOR, reasons)
    if (
        screen.operational_policy_id != request.operational_policy_id
        or screen.operational_policy_sha256 != request.operational_policy_sha256
    ):
        reasons.append(ProposalReason.SCREEN_POLICY_MISMATCH)
        return _decision(proposal, screen, ProposalDisposition.CENSOR, reasons)

    if screen.forward_verification is ForwardVerificationState.NOT_RUN:
        reasons.append(ProposalReason.FORWARD_NOT_RUN)
        return _decision(proposal, screen, ProposalDisposition.CENSOR, reasons)
    if screen.forward_verification is ForwardVerificationState.AMBIGUOUS:
        reasons.append(ProposalReason.FORWARD_AMBIGUOUS)
        return _decision(proposal, screen, ProposalDisposition.REJECT, reasons)
    if screen.forward_verification is ForwardVerificationState.MISMATCHED:
        reasons.append(ProposalReason.FORWARD_MISMATCH)
        return _decision(proposal, screen, ProposalDisposition.REJECT, reasons)
    if (
        screen.forward_verification is not ForwardVerificationState.VERIFIED_EXACT_PRODUCT_UNIQUE
        or screen.forward_product_count != 1
        or screen.forward_product_smiles != (request.target.canonical_smiles,)
    ):
        reasons.append(ProposalReason.FORWARD_MISMATCH)
        return _decision(proposal, screen, ProposalDisposition.REJECT, reasons)
    reasons.append(ProposalReason.FORWARD_EXACT_UNIQUE)

    if screen.operational_compatibility is OperationalCompatibility.REJECTED:
        reasons.append(ProposalReason.OPERATIONAL_REJECTED)
        return _decision(proposal, screen, ProposalDisposition.REJECT, reasons)
    if screen.operational_compatibility is OperationalCompatibility.NOT_ASSESSED:
        reasons.append(ProposalReason.OPERATIONAL_NOT_ASSESSED)
        return _decision(proposal, screen, ProposalDisposition.CENSOR, reasons)
    if screen.operational_compatibility is OperationalCompatibility.REVIEW_REQUIRED:
        reasons.append(ProposalReason.OPERATIONAL_REVIEW_REQUIRED)
        return _decision(proposal, screen, ProposalDisposition.CENSOR, reasons)
    else:
        reasons.append(ProposalReason.OPERATIONAL_PASSED)

    if screen.substrate_scope is SubstrateScopeState.CONTRADICTED:
        reasons.append(ProposalReason.SUBSTRATE_SCOPE_CONTRADICTED)
        return _decision(proposal, screen, ProposalDisposition.REJECT, reasons)
    if screen.substrate_scope is SubstrateScopeState.EXACT_SUBSTRATE:
        reasons.append(ProposalReason.SUBSTRATE_SCOPE_EXACT_EVIDENCE)
    else:
        reasons.append(ProposalReason.SUBSTRATE_SCOPE_REQUIRES_EVIDENCE)

    return _decision(
        proposal,
        screen,
        ProposalDisposition.ADMIT_TO_BOUNDED_SEARCH,
        reasons,
    )


def _decision(
    proposal: SingleStepRetrosynthesisProposal,
    screen: ProposalScreen,
    disposition: ProposalDisposition,
    reasons: list[ProposalReason],
) -> ProposalDecision:
    return ProposalDecision(
        proposal_id=proposal.proposal_id,
        proposal_sha256=proposal.proposal_sha256,
        screen_sha256=screen.screen_sha256,
        disposition=disposition,
        reasons=tuple(reasons),
        can_enter_bounded_search=(disposition is ProposalDisposition.ADMIT_TO_BOUNDED_SEARCH),
    )


__all__ = [
    "InternalTargetLineage",
    "OperationalCompatibility",
    "PROPOSAL_ENGINE_SCHEMA_VERSION",
    "PROPOSAL_REACTANT_IDENTITY_POLICY",
    "ProposalBackendManifest",
    "ProposalDecision",
    "ProposalDisposition",
    "ProposalEngineError",
    "ProposalReason",
    "ProposalRequest",
    "ProposalScreen",
    "ProposalTargetKind",
    "RootQualificationReceipt",
    "SingleStepProposalEngine",
    "SingleStepRetrosynthesisProposal",
    "SubstrateScopeState",
    "build_internal_target_lineage",
    "screen_learned_proposal",
    "validate_proposal_batch",
]
