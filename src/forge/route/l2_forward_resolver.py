"""Independent, fail-closed L2 forward resolution for model proposals.

The resolver is deliberately downstream of every proposal source.  It never
uses a learned reaction-class label or model score to choose a transform.  It
enumerates every exact-scope-admitted upstream L2 transform and unique
reactant-role assignment, executes all admissible assignments under a frozen
budget, and accepts only one uniquely target-reconstructing assignment.

Forward graph reconstruction is a proposal screen, not reaction evidence.
Successful resolution cannot set substrate scope, close a route, or enter a
synthesis value without the independent evidence and L3 systems.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from rdkit import Chem, rdBase

from forge.core.hashing import sha256_file as _sha256_file
from forge.core.hashing import sha256_json as _sha256_payload
from forge.route.planner import ForwardVerificationState
from forge.route.proposal_engine import SingleStepRetrosynthesisProposal
from forge.route.qualified_forward import (
    QualifiedForwardError,
    QualifiedForwardReaction,
    load_qualified_forward_reaction,
    unique_forward_products,
)

L2_FORWARD_RESOLVER_SCHEMA_VERSION = "forge.l2_forward_resolver.v1"
L2_FORWARD_RESOLVER_CONFIG_SCHEMA_VERSION = "forge.l2_forward_resolver_config.v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_TRANSFORM_STATUSES = {
    "qualified_for_exact_source_forward_verification_only",
    "qualified_for_one_exact_substrate_product_pair_only",
}


class L2ForwardResolverError(ValueError):
    """Raised when the resolver configuration or request is malformed."""


class L2ForwardResolutionStatus(str, Enum):
    """Mutually exclusive proposal-only resolution states."""

    EXACT_UNIQUE = "exact_unique"
    CENSOR_NO_VERIFIER = "censor_no_verifier"
    REJECT_FORWARD_MISMATCH = "reject_forward_mismatch"
    CENSOR_AMBIGUOUS_FORWARD_PRODUCTS = "censor_ambiguous_forward_products"
    CENSOR_AMBIGUOUS_TRANSFORM_ASSIGNMENT = "censor_ambiguous_transform_assignment"
    CENSOR_BUDGET_EXHAUSTED = "censor_budget_exhausted"
    CENSOR_EXECUTION_ERROR = "censor_execution_error"


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise L2ForwardResolverError(f"{label} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise L2ForwardResolverError(f"{label} must be a nonempty string")
    return value


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise L2ForwardResolverError(f"{label} must be a positive integer")
    return value


def _canonical_connected_constitution(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise L2ForwardResolverError(f"{label} must be nonempty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise L2ForwardResolverError(f"{label} must be one valid connected molecule")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


@dataclass(frozen=True)
class ExactPairAdmission:
    """One exact reactant-role assignment admitted only for one target."""

    admission_id: str
    role_ordered_reactants: tuple[str, ...]
    target_smiles: str
    source_record_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.admission_id, label="admission_id")
        if not isinstance(self.role_ordered_reactants, tuple) or not (self.role_ordered_reactants):
            raise L2ForwardResolverError("exact-pair admission requires reactants")
        canonical_reactants = tuple(
            _canonical_connected_constitution(value, label="admitted reactant")
            for value in self.role_ordered_reactants
        )
        object.__setattr__(self, "role_ordered_reactants", canonical_reactants)
        object.__setattr__(
            self,
            "target_smiles",
            _canonical_connected_constitution(self.target_smiles, label="admitted target"),
        )
        if not isinstance(self.source_record_sha256s, tuple) or not (self.source_record_sha256s):
            raise L2ForwardResolverError("exact-pair admission requires source record hashes")
        source_hashes = tuple(
            sorted(
                {
                    _require_sha256(value, label="source_record_sha256")
                    for value in self.source_record_sha256s
                }
            )
        )
        object.__setattr__(self, "source_record_sha256s", source_hashes)

    @property
    def identity(self) -> tuple[tuple[str, ...], str]:
        return self.role_ordered_reactants, self.target_smiles

    def to_dict(self) -> dict[str, Any]:
        return {
            "admission_id": self.admission_id,
            "role_ordered_reactants": list(self.role_ordered_reactants),
            "target_smiles": self.target_smiles,
            "source_record_sha256s": list(self.source_record_sha256s),
            "scope_only_not_route_evidence": True,
        }


@dataclass(frozen=True)
class AdmittedL2ForwardTransform:
    """One upstream L2 transform plus its exact admitted substrate pairs."""

    reaction: QualifiedForwardReaction
    transform_sha256: str
    registry_id: str
    registry_sha256: str
    qualification_status: str
    admissions: tuple[ExactPairAdmission, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.reaction, QualifiedForwardReaction):
            raise L2ForwardResolverError("L2 transform requires a compiled reaction")
        _require_sha256(self.transform_sha256, label="transform_sha256")
        _nonempty(self.registry_id, label="registry_id")
        _require_sha256(self.registry_sha256, label="registry_sha256")
        if self.qualification_status not in _ALLOWED_TRANSFORM_STATUSES:
            raise L2ForwardResolverError(
                "only exact-source or exact-pair upstream L2 transforms are admitted"
            )
        if not isinstance(self.admissions, tuple) or not self.admissions:
            raise L2ForwardResolverError("L2 transform has no exact-pair admission")
        if any(not isinstance(value, ExactPairAdmission) for value in self.admissions):
            raise L2ForwardResolverError("L2 transform admission is malformed")
        identities = tuple(value.identity for value in self.admissions)
        if len(set(identities)) != len(identities):
            raise L2ForwardResolverError(
                "duplicate exact-pair admissions must be merged before resolver construction"
            )
        if any(
            len(value.role_ordered_reactants) != len(self.reaction.role_names)
            for value in self.admissions
        ):
            raise L2ForwardResolverError("admitted pair arity disagrees with transform roles")

    @property
    def reaction_id(self) -> str:
        return self.reaction.reaction_id

    def admission_for(
        self,
        role_ordered_reactants: tuple[str, ...],
        target_smiles: str,
    ) -> ExactPairAdmission | None:
        identity = role_ordered_reactants, target_smiles
        for admission in self.admissions:
            if admission.identity == identity:
                return admission
        return None


class L2ForwardExecutor(Protocol):
    """Injectable sparse forward executor used by the resolver."""

    def __call__(
        self,
        reaction: QualifiedForwardReaction,
        role_ordered_reactants: tuple[str, ...],
        *,
        max_products: int,
    ) -> tuple[str, ...]: ...


def execute_qualified_forward(
    reaction: QualifiedForwardReaction,
    role_ordered_reactants: tuple[str, ...],
    *,
    max_products: int,
) -> tuple[str, ...]:
    """Production executor using the shared qualified-forward implementation."""

    return unique_forward_products(
        reaction,
        role_ordered_reactants,
        max_products=max_products,
        isomeric_smiles=False,
    )


@dataclass(frozen=True)
class L2ForwardAssignmentTrace:
    """One independently selected, scope-admitted forward execution."""

    reaction_id: str
    transform_sha256: str
    role_names: tuple[str, ...]
    role_ordered_reactants: tuple[str, ...]
    admission_id: str
    source_record_sha256s: tuple[str, ...]
    products: tuple[str, ...]
    target_reconstructed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "reaction_id": self.reaction_id,
            "transform_sha256": self.transform_sha256,
            "role_names": list(self.role_names),
            "role_ordered_reactants": list(self.role_ordered_reactants),
            "admission_id": self.admission_id,
            "source_record_sha256s": list(self.source_record_sha256s),
            "products": list(self.products),
            "target_reconstructed": self.target_reconstructed,
            "scope_only_not_route_evidence": True,
        }


@dataclass(frozen=True)
class L2ForwardResolution:
    """Fail-closed resolution result with no evidence or value authority."""

    proposal_sha256: str
    target_smiles: str
    resolver_id: str
    resolver_version: str
    resolver_config_sha256: str
    status: L2ForwardResolutionStatus
    forward_verification: ForwardVerificationState
    scope_checks: int
    eligible_assignments: int
    forward_calls: int
    maximum_forward_calls: int
    maximum_products_per_assignment: int
    traces: tuple[L2ForwardAssignmentTrace, ...]
    accepted_trace_index: int | None = None

    def __post_init__(self) -> None:
        _require_sha256(self.proposal_sha256, label="proposal_sha256")
        _canonical_connected_constitution(self.target_smiles, label="resolution target")
        _nonempty(self.resolver_id, label="resolver_id")
        _nonempty(self.resolver_version, label="resolver_version")
        _require_sha256(self.resolver_config_sha256, label="resolver_config_sha256")
        if not isinstance(self.status, L2ForwardResolutionStatus):
            raise L2ForwardResolverError("resolution status is unsupported")
        if not isinstance(self.forward_verification, ForwardVerificationState):
            raise L2ForwardResolverError("forward verification state is unsupported")
        for name in ("scope_checks", "eligible_assignments", "forward_calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise L2ForwardResolverError(f"{name} must be a nonnegative integer")
        _positive_integer(self.maximum_forward_calls, label="maximum_forward_calls")
        _positive_integer(
            self.maximum_products_per_assignment,
            label="maximum_products_per_assignment",
        )
        if self.forward_calls > self.maximum_forward_calls:
            raise L2ForwardResolverError("forward calls exceed the frozen resolver budget")
        if not isinstance(self.traces, tuple) or any(
            not isinstance(value, L2ForwardAssignmentTrace) for value in self.traces
        ):
            raise L2ForwardResolverError("resolution traces are malformed")
        if self.forward_calls < len(self.traces):
            raise L2ForwardResolverError("forward-call count is smaller than completed traces")
        if self.status is L2ForwardResolutionStatus.EXACT_UNIQUE:
            if (
                self.forward_verification
                is not ForwardVerificationState.VERIFIED_EXACT_PRODUCT_UNIQUE
                or self.accepted_trace_index is None
                or self.accepted_trace_index < 0
                or self.accepted_trace_index >= len(self.traces)
                or not self.traces[self.accepted_trace_index].target_reconstructed
                or len(self.traces[self.accepted_trace_index].products) != 1
            ):
                raise L2ForwardResolverError("exact resolution lacks one valid accepted trace")
        elif self.accepted_trace_index is not None:
            raise L2ForwardResolverError("non-exact resolution cannot select a trace")

    @property
    def accepted_trace(self) -> L2ForwardAssignmentTrace | None:
        if self.accepted_trace_index is None:
            return None
        return self.traces[self.accepted_trace_index]

    @property
    def may_enter_synthesis_value(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        accepted = self.accepted_trace
        return {
            "schema_version": L2_FORWARD_RESOLVER_SCHEMA_VERSION,
            "proposal_sha256": self.proposal_sha256,
            "target_smiles": self.target_smiles,
            "resolver_id": self.resolver_id,
            "resolver_version": self.resolver_version,
            "resolver_config_sha256": self.resolver_config_sha256,
            "status": self.status.value,
            "forward_verification": self.forward_verification.value,
            "scope_checks": self.scope_checks,
            "eligible_assignments": self.eligible_assignments,
            "forward_calls": self.forward_calls,
            "maximum_forward_calls": self.maximum_forward_calls,
            "maximum_products_per_assignment": self.maximum_products_per_assignment,
            "traces": [value.to_dict() for value in self.traces],
            "accepted_assignment": None if accepted is None else accepted.to_dict(),
            "proposal_model_score_used": False,
            "proposal_model_class_used": False,
            "substrate_scope_state": "unassessed",
            "evidence_tier": None,
            "success_probability": None,
            "route_closure_authorized": False,
            "may_enter_synthesis_value": False,
        }


@dataclass(frozen=True)
class _EligibleAssignment:
    transform: AdmittedL2ForwardTransform
    reactants: tuple[str, ...]
    admission: ExactPairAdmission


class IndependentL2ForwardResolver:
    """Resolve proposal-only reactants against every admitted upstream L2 step."""

    def __init__(
        self,
        *,
        resolver_id: str,
        resolver_version: str,
        resolver_config_sha256: str,
        transforms: Sequence[AdmittedL2ForwardTransform],
        maximum_forward_calls: int,
        maximum_products_per_assignment: int,
        executor: L2ForwardExecutor = execute_qualified_forward,
    ) -> None:
        self._resolver_id = _nonempty(resolver_id, label="resolver_id")
        self._resolver_version = _nonempty(resolver_version, label="resolver_version")
        self._resolver_config_sha256 = _require_sha256(
            resolver_config_sha256,
            label="resolver_config_sha256",
        )
        if not isinstance(transforms, Sequence) or not transforms:
            raise L2ForwardResolverError("resolver requires admitted upstream L2 transforms")
        ordered = tuple(sorted(transforms, key=lambda value: value.reaction_id))
        if any(not isinstance(value, AdmittedL2ForwardTransform) for value in ordered):
            raise L2ForwardResolverError("resolver transform is malformed")
        if len({value.reaction_id for value in ordered}) != len(ordered):
            raise L2ForwardResolverError("resolver reaction IDs must be unique")
        self._transforms = ordered
        self._maximum_forward_calls = _positive_integer(
            maximum_forward_calls,
            label="maximum_forward_calls",
        )
        self._maximum_products_per_assignment = _positive_integer(
            maximum_products_per_assignment,
            label="maximum_products_per_assignment",
        )
        if not callable(executor):
            raise L2ForwardResolverError("resolver executor must be callable")
        self._executor = executor

    @property
    def transforms(self) -> tuple[AdmittedL2ForwardTransform, ...]:
        return self._transforms

    @property
    def resolver_id(self) -> str:
        """Stable resolver identity recorded in every resolution receipt."""

        return self._resolver_id

    @property
    def resolver_version(self) -> str:
        """Version of the independent resolver contract."""

        return self._resolver_version

    @property
    def resolver_config_sha256(self) -> str:
        """Authenticated configuration digest used to construct this resolver."""

        return self._resolver_config_sha256

    def resolve(
        self,
        proposal: SingleStepRetrosynthesisProposal,
        *,
        maximum_forward_calls: int | None = None,
    ) -> L2ForwardResolution:
        if not isinstance(proposal, SingleStepRetrosynthesisProposal):
            raise L2ForwardResolverError("resolver requires a typed proposal-only record")
        call_budget = self._maximum_forward_calls
        if maximum_forward_calls is not None:
            requested_budget = _positive_integer(
                maximum_forward_calls,
                label="maximum_forward_calls override",
            )
            call_budget = min(call_budget, requested_budget)
        target = _canonical_connected_constitution(
            proposal.request.target.canonical_smiles,
            label="proposal target",
        )
        proposal_reactants = tuple(proposal.reactant_smiles)
        eligible: list[_EligibleAssignment] = []
        scope_checks = 0
        for transform in self._transforms:
            if len(transform.reaction.role_names) != len(proposal_reactants):
                continue
            assignments = tuple(sorted(set(itertools.permutations(proposal_reactants))))
            for assignment in assignments:
                scope_checks += 1
                admission = transform.admission_for(assignment, target)
                if admission is not None:
                    eligible.append(
                        _EligibleAssignment(
                            transform=transform,
                            reactants=assignment,
                            admission=admission,
                        )
                    )

        if not eligible:
            return self._result(
                proposal=proposal,
                target=target,
                status=L2ForwardResolutionStatus.CENSOR_NO_VERIFIER,
                forward_verification=ForwardVerificationState.NOT_RUN,
                scope_checks=scope_checks,
                eligible_assignments=0,
                traces=(),
                maximum_forward_calls=call_budget,
            )
        eligible.sort(
            key=lambda value: (
                value.transform.reaction_id,
                value.reactants,
                value.admission.admission_id,
            )
        )
        if len(eligible) > call_budget:
            return self._result(
                proposal=proposal,
                target=target,
                status=L2ForwardResolutionStatus.CENSOR_BUDGET_EXHAUSTED,
                forward_verification=ForwardVerificationState.NOT_RUN,
                scope_checks=scope_checks,
                eligible_assignments=len(eligible),
                traces=(),
                maximum_forward_calls=call_budget,
            )

        traces: list[L2ForwardAssignmentTrace] = []
        for assignment in eligible:
            try:
                products = self._executor(
                    assignment.transform.reaction,
                    assignment.reactants,
                    max_products=self._maximum_products_per_assignment,
                )
            except QualifiedForwardError as exc:
                message = str(exc)
                status = (
                    L2ForwardResolutionStatus.CENSOR_BUDGET_EXHAUSTED
                    if "reached max_products" in message
                    else L2ForwardResolutionStatus.CENSOR_EXECUTION_ERROR
                )
                return self._result(
                    proposal=proposal,
                    target=target,
                    status=status,
                    forward_verification=ForwardVerificationState.NOT_RUN,
                    scope_checks=scope_checks,
                    eligible_assignments=len(eligible),
                    traces=tuple(traces),
                    forward_calls=len(traces) + 1,
                    maximum_forward_calls=call_budget,
                )
            except Exception:  # pragma: no cover - defensive RDKit/external boundary
                return self._result(
                    proposal=proposal,
                    target=target,
                    status=L2ForwardResolutionStatus.CENSOR_EXECUTION_ERROR,
                    forward_verification=ForwardVerificationState.NOT_RUN,
                    scope_checks=scope_checks,
                    eligible_assignments=len(eligible),
                    traces=tuple(traces),
                    forward_calls=len(traces) + 1,
                    maximum_forward_calls=call_budget,
                )
            canonical_products = tuple(
                sorted(
                    {
                        _canonical_connected_constitution(value, label="forward product")
                        for value in products
                    }
                )
            )
            traces.append(
                L2ForwardAssignmentTrace(
                    reaction_id=assignment.transform.reaction_id,
                    transform_sha256=assignment.transform.transform_sha256,
                    role_names=assignment.transform.reaction.role_names,
                    role_ordered_reactants=assignment.reactants,
                    admission_id=assignment.admission.admission_id,
                    source_record_sha256s=assignment.admission.source_record_sha256s,
                    products=canonical_products,
                    target_reconstructed=(canonical_products == (target,)),
                )
            )

        if any(len(trace.products) > 1 for trace in traces):
            return self._result(
                proposal=proposal,
                target=target,
                status=L2ForwardResolutionStatus.CENSOR_AMBIGUOUS_FORWARD_PRODUCTS,
                forward_verification=ForwardVerificationState.AMBIGUOUS,
                scope_checks=scope_checks,
                eligible_assignments=len(eligible),
                traces=tuple(traces),
                maximum_forward_calls=call_budget,
            )
        matches = [index for index, trace in enumerate(traces) if trace.target_reconstructed]
        if not matches:
            return self._result(
                proposal=proposal,
                target=target,
                status=L2ForwardResolutionStatus.REJECT_FORWARD_MISMATCH,
                forward_verification=ForwardVerificationState.MISMATCHED,
                scope_checks=scope_checks,
                eligible_assignments=len(eligible),
                traces=tuple(traces),
                maximum_forward_calls=call_budget,
            )
        if len(matches) > 1:
            return self._result(
                proposal=proposal,
                target=target,
                status=L2ForwardResolutionStatus.CENSOR_AMBIGUOUS_TRANSFORM_ASSIGNMENT,
                forward_verification=ForwardVerificationState.AMBIGUOUS,
                scope_checks=scope_checks,
                eligible_assignments=len(eligible),
                traces=tuple(traces),
                maximum_forward_calls=call_budget,
            )
        return self._result(
            proposal=proposal,
            target=target,
            status=L2ForwardResolutionStatus.EXACT_UNIQUE,
            forward_verification=ForwardVerificationState.VERIFIED_EXACT_PRODUCT_UNIQUE,
            scope_checks=scope_checks,
            eligible_assignments=len(eligible),
            traces=tuple(traces),
            accepted_trace_index=matches[0],
            maximum_forward_calls=call_budget,
        )

    def _result(
        self,
        *,
        proposal: SingleStepRetrosynthesisProposal,
        target: str,
        status: L2ForwardResolutionStatus,
        forward_verification: ForwardVerificationState,
        scope_checks: int,
        eligible_assignments: int,
        traces: tuple[L2ForwardAssignmentTrace, ...],
        accepted_trace_index: int | None = None,
        forward_calls: int | None = None,
        maximum_forward_calls: int | None = None,
    ) -> L2ForwardResolution:
        return L2ForwardResolution(
            proposal_sha256=proposal.proposal_sha256,
            target_smiles=target,
            resolver_id=self._resolver_id,
            resolver_version=self._resolver_version,
            resolver_config_sha256=self._resolver_config_sha256,
            status=status,
            forward_verification=forward_verification,
            scope_checks=scope_checks,
            eligible_assignments=eligible_assignments,
            forward_calls=len(traces) if forward_calls is None else forward_calls,
            maximum_forward_calls=(
                self._maximum_forward_calls
                if maximum_forward_calls is None
                else maximum_forward_calls
            ),
            maximum_products_per_assignment=self._maximum_products_per_assignment,
            traces=traces,
            accepted_trace_index=accepted_trace_index,
        )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise L2ForwardResolverError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise L2ForwardResolverError(f"{label} must be a JSON object")
    return value


def _verify_artifact(repo_root: Path, record: Mapping[str, Any], *, label: str) -> Path:
    relative = record.get("path")
    if not isinstance(relative, str) or not relative:
        raise L2ForwardResolverError(f"{label} artifact path is malformed")
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise L2ForwardResolverError(f"{label} artifact escapes the repository") from exc
    if not path.is_file():
        raise L2ForwardResolverError(f"{label} artifact is missing: {path}")
    expected = _require_sha256(record.get("sha256"), label=f"{label} artifact SHA-256")
    if _sha256_file(path) != expected:
        raise L2ForwardResolverError(f"{label} artifact SHA-256 mismatch")
    return path


def _reaction_definition(registry: Mapping[str, Any], reaction_id: str) -> dict[str, Any]:
    reactions = registry.get("reactions")
    if not isinstance(reactions, list):
        raise L2ForwardResolverError("L2 registry has no reaction list")
    matches = [
        value
        for value in reactions
        if isinstance(value, dict) and value.get("reaction_id") == reaction_id
    ]
    if len(matches) != 1:
        raise L2ForwardResolverError(f"L2 reaction {reaction_id!r} must resolve exactly once")
    return matches[0]


def load_independent_l2_forward_resolver(
    config_path: Path,
    *,
    repo_root: Path,
    executor: L2ForwardExecutor = execute_qualified_forward,
) -> IndependentL2ForwardResolver:
    """Load a resolver from one hash-pinned, exact-scope-only L2 manifest."""

    config = _load_json(config_path, label="L2 forward resolver config")
    if config.get("schema_version") != L2_FORWARD_RESOLVER_CONFIG_SCHEMA_VERSION:
        raise L2ForwardResolverError("unsupported L2 forward resolver config schema")
    if config.get("status") != "frozen_inactive_not_benchmarked":
        raise L2ForwardResolverError("L2 resolver config must remain inactive and unbenchmarked")
    if config.get("activation_authorized") is not False:
        raise L2ForwardResolverError("L2 resolver config cannot activate Graph2Edits")
    guards = config.get("scope_guards")
    if not isinstance(guards, Mapping) or guards.get("ugi_l1_registry_admitted") is not False:
        raise L2ForwardResolverError("Ugi L1 is forbidden in the L2 resolver")
    for field in (
        "proposal_model_class_may_select_verifier",
        "proposal_model_score_may_select_verifier",
        "model_score_may_enter_v_syn",
        "exact_forward_implies_scope_evidence",
        "sealed_holdouts_accessed",
        "frozen_benchmark_executed",
    ):
        if guards.get(field) is not False:
            raise L2ForwardResolverError(f"unsafe L2 resolver scope guard: {field}")

    artifacts = config.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise L2ForwardResolverError("L2 resolver artifacts are missing")
    artifact_paths = {
        str(label): _verify_artifact(repo_root, record, label=str(label))
        for label, record in artifacts.items()
        if isinstance(record, Mapping)
    }
    if len(artifact_paths) != len(artifacts):
        raise L2ForwardResolverError("L2 resolver artifact record is malformed")

    transform_records = config.get("transforms")
    if not isinstance(transform_records, list) or not transform_records:
        raise L2ForwardResolverError("L2 resolver has no admitted transforms")
    transforms: list[AdmittedL2ForwardTransform] = []
    for index, record in enumerate(transform_records):
        if not isinstance(record, Mapping):
            raise L2ForwardResolverError(f"transform record {index} is malformed")
        registry_label = _nonempty(record.get("registry_artifact"), label="registry_artifact")
        variant_label = _nonempty(record.get("variant_artifact"), label="variant_artifact")
        if registry_label not in artifact_paths or variant_label not in artifact_paths:
            raise L2ForwardResolverError("transform references an unpinned artifact")
        registry_path = artifact_paths[registry_label]
        variant_path = artifact_paths[variant_label]
        registry = _load_json(registry_path, label="L2 reaction registry")
        registry_scope = registry.get("scope")
        if not isinstance(registry_scope, str) or "exact" not in registry_scope:
            raise L2ForwardResolverError("general or L1 reaction registry is not admitted")
        reaction_id = _nonempty(record.get("reaction_id"), label="reaction_id")
        definition = _reaction_definition(registry, reaction_id)
        status = definition.get("status")
        if status not in _ALLOWED_TRANSFORM_STATUSES:
            raise L2ForwardResolverError(f"reaction {reaction_id!r} is not exact-scope upstream L2")
        compiled = load_qualified_forward_reaction(
            registry_path,
            variant_path,
            reaction_id=reaction_id,
        )
        transform_payload = {
            "reaction_id": reaction_id,
            "registry_sha256": _sha256_file(registry_path),
            "variant_sha256": _sha256_file(variant_path),
            "role_names": list(compiled.role_names),
            "qualification_status": status,
        }
        transform_sha256 = _sha256_payload(transform_payload)
        expected_transform_sha = _require_sha256(
            record.get("transform_sha256"),
            label="transform_sha256",
        )
        if transform_sha256 != expected_transform_sha:
            raise L2ForwardResolverError(f"transform receipt mismatch for {reaction_id}")
        raw_admissions = record.get("admissions")
        if not isinstance(raw_admissions, list) or not raw_admissions:
            raise L2ForwardResolverError(f"reaction {reaction_id!r} has no exact admissions")
        admissions = tuple(
            ExactPairAdmission(
                admission_id=value.get("admission_id"),
                role_ordered_reactants=tuple(value.get("role_ordered_reactants", ())),
                target_smiles=value.get("target_smiles"),
                source_record_sha256s=tuple(value.get("source_record_sha256s", ())),
            )
            for value in raw_admissions
            if isinstance(value, Mapping)
        )
        if len(admissions) != len(raw_admissions):
            raise L2ForwardResolverError(f"reaction {reaction_id!r} admission is malformed")
        transforms.append(
            AdmittedL2ForwardTransform(
                reaction=compiled,
                transform_sha256=transform_sha256,
                registry_id=_nonempty(registry.get("registry_version"), label="registry_version"),
                registry_sha256=_sha256_file(registry_path),
                qualification_status=status,
                admissions=admissions,
            )
        )

    budget = config.get("budget")
    if not isinstance(budget, Mapping):
        raise L2ForwardResolverError("L2 resolver budget is missing")
    return IndependentL2ForwardResolver(
        resolver_id=_nonempty(config.get("resolver_id"), label="resolver_id"),
        resolver_version=_nonempty(config.get("resolver_version"), label="resolver_version"),
        resolver_config_sha256=_sha256_file(config_path),
        transforms=transforms,
        maximum_forward_calls=_positive_integer(
            budget.get("maximum_forward_calls_per_proposal"),
            label="maximum_forward_calls_per_proposal",
        ),
        maximum_products_per_assignment=_positive_integer(
            budget.get("maximum_products_per_assignment"),
            label="maximum_products_per_assignment",
        ),
        executor=executor,
    )


def transform_receipt_sha256(
    *,
    reaction_id: str,
    registry_sha256: str,
    variant_sha256: str,
    role_names: Sequence[str],
    qualification_status: str,
) -> str:
    """Public deterministic helper for building frozen resolver manifests."""

    return _sha256_payload(
        {
            "reaction_id": reaction_id,
            "registry_sha256": _require_sha256(registry_sha256, label="registry_sha256"),
            "variant_sha256": _require_sha256(variant_sha256, label="variant_sha256"),
            "role_names": [_nonempty(value, label="role_name") for value in role_names],
            "qualification_status": qualification_status,
        }
    )


__all__ = [
    "AdmittedL2ForwardTransform",
    "ExactPairAdmission",
    "IndependentL2ForwardResolver",
    "L2ForwardAssignmentTrace",
    "L2ForwardResolution",
    "L2ForwardResolutionStatus",
    "L2ForwardResolverError",
    "execute_qualified_forward",
    "load_independent_l2_forward_resolver",
    "transform_receipt_sha256",
]
