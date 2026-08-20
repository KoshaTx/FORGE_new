"""Artifact-derived qualification for the production cumulative Ugi planner.

The route cache context is part of the scientific contract: it must change
whenever executable planner code, admitted evidence, reaction registries,
variants, verification policy, L3 state, or resource limits change.  This
module derives that context from authenticated repository artifacts rather
than accepting semantic placeholder hashes from a caller.

It also derives the exact internal role allow-list used by the support-boundary
wrapper.  Those roles come only from source-owned component/program records
and admitted exact-route configurations; prefixes and handwritten patterns are
never accepted.
"""

from __future__ import annotations

import csv
import gzip
import platform
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import rdBase

from experiments.phase1.synthesis_guidance.sources.ugi3_cumulative_production_source import (
    CumulativeUgi3ProductionPaths,
)
from forge.core.hashing import sha256_file
from forge.core.hashing import sha256_json as _sha256_payload
from forge.core.io import read_json_object
from forge.synthesis.assessment.ugi3_support_boundary import (
    UGI_COMPONENT_ROLES,
    AuthenticatedInternalRoleRegistry,
)
from forge.synthesis.engine.planner import PlannerBudgetLimits
from forge.synthesis.engine.planner_cache import (
    PLANNER_CACHE_KEY_SCHEMA_VERSION,
    PlannerCacheContext,
)
from forge.synthesis.engine.planner_cache_snapshot import planner_cache_context_sha256
from forge.synthesis.terminals.terminal_assessment import (
    DEFAULT_IDENTITY_POLICY,
    DEFAULT_STEREOCHEMISTRY_POLICY,
)
from forge.synthesis.value.contracts import (
    PRODUCT_SYNTHESIS_VALUE_SCHEMA_VERSION,
    SYNTHESIS_VALUE_SCHEMA_VERSION,
)

PRODUCTION_INTERNAL_ROLE_MANIFEST_SCHEMA_VERSION = "forge.ugi3_production_internal_role_manifest.v1"
PRODUCTION_PLANNER_CONTEXT_QUALIFICATION_SCHEMA_VERSION = (
    "forge.ugi3_production_planner_context_qualification.v1"
)
PRODUCTION_PLANNER_ID = "forge.cumulative_ugi3_recursive_planner.v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EXACT_PROGRAM_STATUS = "exact_source_program"
_ROOT_ROLE_FIELDS = frozenset(UGI_COMPONENT_ROLES)
_BUDGET_FIELDS = tuple(PlannerBudgetLimits.__annotations__)


class Ugi3ProductionPlannerQualificationError(RuntimeError):
    """Raised when the production planner qualification cannot fail closed."""


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise Ugi3ProductionPlannerQualificationError(
            f"{label} must contain 64 lowercase hexadecimal characters"
        )
    return value


def _require_nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Ugi3ProductionPlannerQualificationError(f"{label} must be nonempty")
    return value


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=Ugi3ProductionPlannerQualificationError, label=label)


def _portable(path: Path, *, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


@dataclass(frozen=True, order=True)
class QualifiedArtifact:
    """One current file identity used in a production qualification."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.path, label="qualified artifact path")
        _require_sha256(self.sha256, label=f"qualified artifact {self.path} SHA-256")

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, order=True)
class InternalRoleQualificationRecord:
    """One exact internal route role and its authenticated source record."""

    role: str
    layer: str
    record_id: str
    source: QualifiedArtifact

    def __post_init__(self) -> None:
        _require_nonempty(self.role, label="internal role")
        _require_nonempty(self.layer, label="internal role layer")
        _require_nonempty(self.record_id, label="internal role record_id")
        if self.role in _ROOT_ROLE_FIELDS:
            raise Ugi3ProductionPlannerQualificationError(
                "root Ugi component roles cannot enter the internal-role manifest"
            )
        # Reuse the production exact-role validator rather than duplicating its
        # role grammar or accidentally admitting patterns.
        AuthenticatedInternalRoleRegistry(frozenset({self.role}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "layer": self.layer,
            "record_id": self.record_id,
            "source": self.source.to_dict(),
        }


@dataclass(frozen=True)
class ProductionInternalRoleManifest:
    """Exact internal roles derived from the authenticated cumulative source."""

    source_inputs_sha256: str
    records: tuple[InternalRoleQualificationRecord, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.source_inputs_sha256, label="source_inputs_sha256")
        if not self.records:
            raise Ugi3ProductionPlannerQualificationError(
                "production internal-role manifest must not be empty"
            )
        if self.records != tuple(sorted(set(self.records))):
            raise Ugi3ProductionPlannerQualificationError(
                "internal-role qualification records must be unique and sorted"
            )
        # Constructing the registry is itself the exact membership validation.
        self.registry

    @property
    def registry(self) -> AuthenticatedInternalRoleRegistry:
        return AuthenticatedInternalRoleRegistry(frozenset(record.role for record in self.records))

    @property
    def registry_sha256(self) -> str:
        return authenticated_internal_role_registry_sha256(self.registry)

    @property
    def manifest_sha256(self) -> str:
        return _sha256_payload(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PRODUCTION_INTERNAL_ROLE_MANIFEST_SCHEMA_VERSION,
            "source_inputs_sha256": self.source_inputs_sha256,
            "roles": sorted(self.registry.roles),
            "registry_sha256": self.registry_sha256,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True, order=True)
class PlannerBudgetSource:
    """One authenticated per-component budget contributing to the maximum."""

    layer: str
    source: QualifiedArtifact
    budget_limits: PlannerBudgetLimits

    def __post_init__(self) -> None:
        _require_nonempty(self.layer, label="planner budget layer")
        if not isinstance(self.budget_limits, PlannerBudgetLimits):
            raise Ugi3ProductionPlannerQualificationError(
                "planner budget source must contain PlannerBudgetLimits"
            )
        if self.budget_limits.maximum_logical_planner_calls != 1:
            raise Ugi3ProductionPlannerQualificationError(
                f"{self.layer} must reserve exactly one logical planner call per component"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "source": self.source.to_dict(),
            "budget_limits": self.budget_limits.to_dict(),
        }


@dataclass(frozen=True)
class ProductionPlannerContextQualification:
    """Complete artifact receipt for one production PlannerCacheContext."""

    planner_context: PlannerCacheContext
    source_metadata_sha256: str
    source_inputs_sha256: str
    selected_generator_checkpoint_sha256: str
    graph_support_sha256: str
    internal_role_manifest_sha256: str
    qualified_l1_registry_artifact: QualifiedArtifact
    planner_artifacts: tuple[QualifiedArtifact, ...]
    registry_artifacts: tuple[QualifiedArtifact, ...]
    variant_artifacts: tuple[QualifiedArtifact, ...]
    verifier_artifacts: tuple[QualifiedArtifact, ...]
    value_artifacts: tuple[QualifiedArtifact, ...]
    budget_sources: tuple[PlannerBudgetSource, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.planner_context, PlannerCacheContext):
            raise Ugi3ProductionPlannerQualificationError("planner context is malformed")
        for name in (
            "source_metadata_sha256",
            "source_inputs_sha256",
            "selected_generator_checkpoint_sha256",
            "graph_support_sha256",
            "internal_role_manifest_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if not isinstance(self.qualified_l1_registry_artifact, QualifiedArtifact):
            raise Ugi3ProductionPlannerQualificationError(
                "qualified L1 registry artifact is malformed"
            )
        for name in (
            "planner_artifacts",
            "registry_artifacts",
            "variant_artifacts",
            "verifier_artifacts",
            "value_artifacts",
        ):
            records = getattr(self, name)
            if not records or records != tuple(sorted(set(records))):
                raise Ugi3ProductionPlannerQualificationError(
                    f"{name} must be a nonempty unique sorted tuple"
                )
        if not self.budget_sources or self.budget_sources != tuple(
            sorted(set(self.budget_sources))
        ):
            raise Ugi3ProductionPlannerQualificationError(
                "budget_sources must be a nonempty unique sorted tuple"
            )
        if self.planner_context.l3_snapshot_sha256 != self.source_inputs_sha256:
            raise Ugi3ProductionPlannerQualificationError(
                "planner context L3 snapshot must equal cumulative source inputs"
            )
        if self.planner_context.budget_limits != _maximum_budget(self.budget_sources):
            raise Ugi3ProductionPlannerQualificationError(
                "planner context budget is not the coordinatewise authenticated maximum"
            )

    @property
    def planner_context_sha256(self) -> str:
        return planner_cache_context_sha256(self.planner_context)

    @property
    def qualification_sha256(self) -> str:
        return _sha256_payload(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PRODUCTION_PLANNER_CONTEXT_QUALIFICATION_SCHEMA_VERSION,
            "planner_context": self.planner_context.to_dict(),
            "planner_context_sha256": self.planner_context_sha256,
            "source_metadata_sha256": self.source_metadata_sha256,
            "source_inputs_sha256": self.source_inputs_sha256,
            "selected_generator_checkpoint_sha256": (self.selected_generator_checkpoint_sha256),
            "graph_support_sha256": self.graph_support_sha256,
            "internal_role_manifest_sha256": self.internal_role_manifest_sha256,
            "qualified_l1_registry_artifact": (self.qualified_l1_registry_artifact.to_dict()),
            "planner_artifacts": [item.to_dict() for item in self.planner_artifacts],
            "registry_artifacts": [item.to_dict() for item in self.registry_artifacts],
            "variant_artifacts": [item.to_dict() for item in self.variant_artifacts],
            "verifier_artifacts": [item.to_dict() for item in self.verifier_artifacts],
            "value_artifacts": [item.to_dict() for item in self.value_artifacts],
            "budget_sources": [item.to_dict() for item in self.budget_sources],
            "scope": {
                "scalar_value": None,
                "success_probability": None,
                "biology_used": False,
                "selection_used": False,
                "holdout_accessed": False,
            },
        }


def authenticated_internal_role_registry_sha256(
    registry: AuthenticatedInternalRoleRegistry,
) -> str:
    """Return the canonical exact-membership digest used by the factory."""

    if not isinstance(registry, AuthenticatedInternalRoleRegistry):
        raise Ugi3ProductionPlannerQualificationError("internal role registry is malformed")
    return _sha256_payload(
        {
            "schema": "forge.authenticated_internal_role_registry.v1",
            "roles": sorted(registry.roles),
        }
    )


def _metadata_artifact(
    metadata: Mapping[str, Any],
    key: str,
    *,
    repo_root: Path,
) -> tuple[Path, QualifiedArtifact]:
    inputs = metadata.get("inputs")
    record = inputs.get(key) if isinstance(inputs, Mapping) else None
    if not isinstance(record, Mapping):
        raise Ugi3ProductionPlannerQualificationError(
            f"cumulative source input record is missing: {key}"
        )
    raw_path = record.get("path")
    expected = record.get("sha256")
    if not isinstance(raw_path, str):
        raise Ugi3ProductionPlannerQualificationError(
            f"cumulative source input path is malformed: {key}"
        )
    _require_sha256(expected, label=f"cumulative source input {key} SHA-256")
    candidate = Path(raw_path)
    path = candidate if candidate.is_absolute() else repo_root / candidate
    if sha256_file(path) != expected:
        raise Ugi3ProductionPlannerQualificationError(
            f"cumulative source input changed after authentication: {key}"
        )
    return path, QualifiedArtifact(_portable(path, repo_root=repo_root), expected)


def _current_artifact(repo_root: Path, relative_path: str) -> QualifiedArtifact:
    path = repo_root / relative_path
    return QualifiedArtifact(relative_path, sha256_file(path))


def _metadata_artifacts_by_input_names(
    metadata: Mapping[str, Any],
    *,
    repo_root: Path,
    names: frozenset[str],
) -> tuple[QualifiedArtifact, ...]:
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise Ugi3ProductionPlannerQualificationError(
            "cumulative source input manifest is malformed"
        )
    selected: dict[tuple[str, str], QualifiedArtifact] = {}
    for key in sorted(inputs):
        if ":input:" not in key or key.rsplit(":input:", 1)[1] not in names:
            continue
        _, artifact = _metadata_artifact(metadata, key, repo_root=repo_root)
        selected[(artifact.path, artifact.sha256)] = artifact
    if not selected:
        raise Ugi3ProductionPlannerQualificationError(
            f"no cumulative source inputs matched {sorted(names)}"
        )
    return tuple(sorted(selected.values()))


def _append_role(
    records: list[InternalRoleQualificationRecord],
    *,
    role: Any,
    layer: str,
    record_id: str,
    source: QualifiedArtifact,
) -> None:
    if not isinstance(role, str) or role in _ROOT_ROLE_FIELDS:
        return
    records.append(
        InternalRoleQualificationRecord(
            role=role,
            layer=layer,
            record_id=record_id,
            source=source,
        )
    )


def derive_production_internal_role_manifest(
    *,
    repo_root: Path,
    source_metadata: Mapping[str, Any],
) -> ProductionInternalRoleManifest:
    """Derive the exact internal role allow-list from authenticated source records."""

    source_inputs_sha256 = _require_sha256(
        source_metadata.get("inputs_sha256"),
        label="cumulative source inputs_sha256",
    )
    records: list[InternalRoleQualificationRecord] = []

    program_path, program_artifact = _metadata_artifact(
        source_metadata,
        "exact_evidence_base:input:component_program",
        repo_root=repo_root,
    )
    try:
        with gzip.open(program_path, "rt", newline="") as handle:
            programs = list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        raise Ugi3ProductionPlannerQualificationError(
            "could not read authenticated exact component-program ledger"
        ) from error
    exact_program_ids: list[str] = []
    for row in programs:
        if row.get("program_status") != _EXACT_PROGRAM_STATUS:
            continue
        component_id = row.get("component_id")
        if not isinstance(component_id, str) or not component_id:
            raise Ugi3ProductionPlannerQualificationError(
                "exact component-program record lacks component_id"
            )
        exact_program_ids.append(component_id)
        _append_role(
            records,
            role=f"ugi3_upstream_material:{component_id}",
            layer="exact_evidence_base",
            record_id=component_id,
            source=program_artifact,
        )
    if not exact_program_ids or len(exact_program_ids) != len(set(exact_program_ids)):
        raise Ugi3ProductionPlannerQualificationError(
            "exact component-program identities are missing or duplicated"
        )

    layer_metadata = source_metadata.get("layer_metadata")
    if not isinstance(layer_metadata, Mapping):
        raise Ugi3ProductionPlannerQualificationError(
            "cumulative source layer metadata is malformed"
        )
    hybrid = layer_metadata.get("bounded_hybrid_search")
    if not isinstance(hybrid, Mapping):
        raise Ugi3ProductionPlannerQualificationError(
            "authenticated hybrid layer metadata is missing"
        )
    retrieved_component_id = hybrid.get("exact_retrieval_component_id")
    retrieval_record_id = hybrid.get("exact_retrieval_source_record_id")
    if not isinstance(retrieved_component_id, str) or not isinstance(retrieval_record_id, str):
        raise Ugi3ProductionPlannerQualificationError(
            "authenticated exact retrieval identity is malformed"
        )
    _, transfer_artifact = _metadata_artifact(
        source_metadata,
        "bounded_hybrid_search:input:transfer_ledger",
        repo_root=repo_root,
    )
    _append_role(
        records,
        role=f"ugi3_retrieved_upstream_material:{retrieved_component_id}",
        layer="bounded_hybrid_search",
        record_id=retrieval_record_id,
        source=transfer_artifact,
    )

    targeted = layer_metadata.get("targeted_aldehyde_exact_overlay")
    targeted_records = targeted.get("records") if isinstance(targeted, Mapping) else None
    if not isinstance(targeted_records, list):
        raise Ugi3ProductionPlannerQualificationError(
            "authenticated targeted exact-overlay records are missing"
        )
    _, targeted_artifact = _metadata_artifact(
        source_metadata,
        "targeted_aldehyde_exact_overlay:input:evidence_pack",
        repo_root=repo_root,
    )
    selected_route_count = 0
    for raw in targeted_records:
        if not isinstance(raw, Mapping) or raw.get("channel") != "targeted_exact_route":
            continue
        component_id = raw.get("component_id")
        route_id = raw.get("route_id")
        if not isinstance(component_id, str) or not isinstance(route_id, str):
            raise Ugi3ProductionPlannerQualificationError(
                "selected targeted exact route identity is malformed"
            )
        selected_route_count += 1
        _append_role(
            records,
            role=f"ugi3_targeted_upstream:{component_id}:{route_id}",
            layer="targeted_aldehyde_exact_overlay",
            record_id=route_id,
            source=targeted_artifact,
        )
    if selected_route_count != targeted.get("selected_recursive_routes"):
        raise Ugi3ProductionPlannerQualificationError(
            "targeted exact role derivation disagrees with authenticated route count"
        )

    for layer, config_key in (
        ("exact_c18_route", "frozen:exact_c18_config"),
        ("exact_c16_route", "frozen:exact_c16_config"),
    ):
        config_path, config_artifact = _metadata_artifact(
            source_metadata,
            config_key,
            repo_root=repo_root,
        )
        config = _load_json(config_path, label=f"{layer} config")
        steps = config.get("steps")
        if not isinstance(steps, list) or not steps:
            raise Ugi3ProductionPlannerQualificationError(
                f"{layer} lacks authenticated route steps"
            )
        for index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise Ugi3ProductionPlannerQualificationError(f"{layer} step {index} is malformed")
            roles: list[Any] = []
            if "reactant_role" in step:
                roles.append(step.get("reactant_role"))
            reactant_roles = step.get("reactant_roles")
            if reactant_roles is not None:
                if not isinstance(reactant_roles, list):
                    raise Ugi3ProductionPlannerQualificationError(
                        f"{layer} step {index} reactant_roles are malformed"
                    )
                roles.extend(reactant_roles)
            roles.append(step.get("product_role"))
            for role in roles:
                _append_role(
                    records,
                    role=role,
                    layer=layer,
                    record_id=f"steps[{index}]",
                    source=config_artifact,
                )
        terminals = config.get("terminal_procurement")
        terminal_records = terminals if isinstance(terminals, list) else [terminals]
        for index, terminal in enumerate(terminal_records):
            if not isinstance(terminal, Mapping):
                raise Ugi3ProductionPlannerQualificationError(
                    f"{layer} terminal procurement record is malformed"
                )
            _append_role(
                records,
                role=terminal.get("role"),
                layer=layer,
                record_id=f"terminal_procurement[{index}]",
                source=config_artifact,
            )

    return ProductionInternalRoleManifest(
        source_inputs_sha256=source_inputs_sha256,
        records=tuple(sorted(set(records))),
    )


def _budget_from_mapping(value: Any, *, label: str) -> PlannerBudgetLimits:
    if not isinstance(value, Mapping) or set(value) != set(_BUDGET_FIELDS):
        raise Ugi3ProductionPlannerQualificationError(
            f"{label} must contain the complete planner budget schema"
        )
    try:
        return PlannerBudgetLimits(**{name: value[name] for name in _BUDGET_FIELDS})
    except (TypeError, ValueError) as error:
        raise Ugi3ProductionPlannerQualificationError(f"{label} is invalid") from error


def _maximum_budget(sources: Sequence[PlannerBudgetSource]) -> PlannerBudgetLimits:
    return PlannerBudgetLimits(
        **{
            name: max(getattr(source.budget_limits, name) for source in sources)
            for name in _BUDGET_FIELDS
        }
    )


def _budget_sources(
    *,
    repo_root: Path,
    source_metadata: Mapping[str, Any],
    source_paths: CumulativeUgi3ProductionPaths,
) -> tuple[PlannerBudgetSource, ...]:
    definitions = (
        (
            "exact_evidence_base",
            "frozen:exact_config",
            source_paths.exact_config,
            ("diagnostic_policy", "budget_limits"),
        ),
        (
            "bounded_hybrid_search",
            "frozen:hybrid_config",
            source_paths.hybrid_config,
            ("search_policy", "per_component_budget"),
        ),
        (
            "exact_c18_route",
            "frozen:exact_c18_config",
            source_paths.exact_c18_config,
            ("planner_budget",),
        ),
        (
            "exact_c16_route",
            "frozen:exact_c16_config",
            source_paths.exact_c16_config,
            ("planner_budget",),
        ),
    )
    records: list[PlannerBudgetSource] = []
    for layer, metadata_key, expected_path, field_path in definitions:
        path, artifact = _metadata_artifact(
            source_metadata,
            metadata_key,
            repo_root=repo_root,
        )
        if path.resolve() != expected_path.resolve():
            raise Ugi3ProductionPlannerQualificationError(
                f"{layer} budget source differs from the authenticated cumulative path"
            )
        value: Any = _load_json(path, label=f"{layer} config")
        for field in field_path:
            value = value.get(field) if isinstance(value, Mapping) else None
        records.append(
            PlannerBudgetSource(
                layer=layer,
                source=artifact,
                budget_limits=_budget_from_mapping(value, label=f"{layer} budget"),
            )
        )
    return tuple(sorted(records))


def _bundle_sha256(schema: str, artifacts: Sequence[QualifiedArtifact]) -> str:
    if not artifacts:
        raise Ugi3ProductionPlannerQualificationError(f"{schema} artifact bundle is empty")
    return _sha256_payload(
        {"schema": schema, "artifacts": [artifact.to_dict() for artifact in artifacts]}
    )


def build_production_planner_context_qualification(
    *,
    repo_root: Path,
    assessment_as_of_utc: str,
    selected_generator_checkpoint_sha256: str,
    graph_support_sha256: str,
    l1_reaction_sha256: str,
    source_metadata: Mapping[str, Any],
    internal_role_manifest: ProductionInternalRoleManifest,
    source_paths: CumulativeUgi3ProductionPaths | None = None,
) -> ProductionPlannerContextQualification:
    """Build the production cache context from actual code and artifacts."""

    repo = repo_root.resolve()
    _require_nonempty(assessment_as_of_utc, label="assessment_as_of_utc")
    _require_sha256(
        selected_generator_checkpoint_sha256,
        label="selected_generator_checkpoint_sha256",
    )
    _require_sha256(graph_support_sha256, label="graph_support_sha256")
    _require_sha256(l1_reaction_sha256, label="l1_reaction_sha256")
    source_inputs_sha256 = _require_sha256(
        source_metadata.get("inputs_sha256"),
        label="cumulative source inputs_sha256",
    )
    if internal_role_manifest.source_inputs_sha256 != source_inputs_sha256:
        raise Ugi3ProductionPlannerQualificationError(
            "internal-role manifest and cumulative source use different input manifests"
        )

    l1_records: set[tuple[str, str]] = set()
    inputs = source_metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise Ugi3ProductionPlannerQualificationError(
            "cumulative source input manifest is malformed"
        )
    for key in sorted(inputs):
        if key.endswith(":input:l1_variant"):
            _, artifact = _metadata_artifact(source_metadata, key, repo_root=repo)
            l1_records.add((artifact.path, artifact.sha256))
    if len(l1_records) != 1:
        raise Ugi3ProductionPlannerQualificationError(
            "cumulative source must bind one exact L1 variant identity"
        )
    _, authenticated_l1_sha256 = next(iter(l1_records))
    if authenticated_l1_sha256 != l1_reaction_sha256:
        raise Ugi3ProductionPlannerQualificationError(
            "qualified L1 reverifier does not use the frozen Ugi variant SHA-256"
        )

    paths = source_paths or CumulativeUgi3ProductionPaths.from_repo(repo)
    budgets = _budget_sources(
        repo_root=repo,
        source_metadata=source_metadata,
        source_paths=paths,
    )
    budget_limits = _maximum_budget(budgets)
    if budget_limits.maximum_logical_planner_calls != 1:
        raise Ugi3ProductionPlannerQualificationError(
            "production budget must retain one logical planner call per component"
        )

    planner_artifacts = tuple(
        sorted(
            {
                _current_artifact(repo, path)
                for path in (
                    "experiments/phase1/synthesis_guidance/planner_qualification.py",
                    "experiments/phase1/synthesis_guidance/schedule/ugi_production_terminal_route_evaluator.py",
                    "experiments/phase1/synthesis_guidance/sources/ugi3_cumulative_production_source.py",
                    "forge/corpus/ugi_generated_terminal_support.py",
                    "forge/synthesis/assessment/ugi3_support_boundary.py",
                    "forge/synthesis/engine/planner.py",
                    "forge/synthesis/engine/planner_cache.py",
                    "forge/synthesis/engine/planner_cache_snapshot.py",
                    "forge/synthesis/terminals/terminal_assessment.py",
                )
            }
        )
    )
    registry_artifacts = _metadata_artifacts_by_input_names(
        source_metadata,
        repo_root=repo,
        names=frozenset({"upstream_reaction_registry", "upstream_registry", "reaction_registry"}),
    )
    variant_artifacts = _metadata_artifacts_by_input_names(
        source_metadata,
        repo_root=repo,
        names=frozenset(
            {
                "oxidation_variant",
                "reduction_variant",
                "zipper_variant",
                "chain_variant",
                "deprotection_variant",
            }
        ),
    )
    verifier_artifacts = tuple(
        sorted(
            set(
                _metadata_artifacts_by_input_names(
                    source_metadata,
                    repo_root=repo,
                    names=frozenset({"qualified_forward_source"}),
                )
            )
            | {
                _current_artifact(repo, "data/vendor/qualified_reactions_v1.json"),
                _current_artifact(repo, "forge/synthesis/sources/ugi3_exact_evidence_source.py"),
                _current_artifact(repo, "forge/synthesis/engine/ugi3_hybrid_search.py"),
                _current_artifact(repo, "forge/synthesis/evidence/ugi3_targeted_exact_overlay.py"),
                _current_artifact(repo, "forge/synthesis/evidence/ugi3_exact_c18_route.py"),
                _current_artifact(repo, "forge/synthesis/evidence/ugi3_exact_c16_route.py"),
            }
        )
    )
    value_artifacts = (_current_artifact(repo, "forge/synthesis/value/contracts.py"),)
    qualified_l1_registry_artifact = _current_artifact(
        repo, "data/vendor/qualified_reactions_v1.json"
    )

    window = source_metadata.get("unified_l3_window")
    if not isinstance(window, Mapping):
        raise Ugi3ProductionPlannerQualificationError(
            "cumulative source unified L3 window is malformed"
        )
    hybrid_config = _load_json(paths.hybrid_config, label="bounded hybrid config")
    combined_l3 = hybrid_config.get("search_policy", {}).get("combined_l3_context")
    region = combined_l3.get("region") if isinstance(combined_l3, Mapping) else None
    _require_nonempty(region, label="production L3 region")

    source_metadata_sha256 = _sha256_payload(source_metadata)
    planner_sha256 = _bundle_sha256("forge.production_planner_code_bundle.v1", planner_artifacts)
    registry_sha256 = _bundle_sha256(
        "forge.production_upstream_registry_bundle.v1", registry_artifacts
    )
    variant_sha256 = _bundle_sha256("forge.production_route_variant_bundle.v1", variant_artifacts)
    verifier_sha256 = _bundle_sha256(
        "forge.production_forward_verifier_bundle.v1", verifier_artifacts
    )
    value_policy_sha256 = _sha256_payload(
        {
            "schema": "forge.production_structured_nonscalar_synthesis_value.v1",
            "artifacts": [item.to_dict() for item in value_artifacts],
            "component_schema": SYNTHESIS_VALUE_SCHEMA_VERSION,
            "product_schema": PRODUCT_SYNTHESIS_VALUE_SCHEMA_VERSION,
            "scalar_value": None,
            "success_probability": None,
        }
    )
    search_policy_sha256 = _sha256_payload(
        {
            "schema": "forge.production_cumulative_ugi3_search_policy.v1",
            "source_metadata_sha256": source_metadata_sha256,
            "source_inputs_sha256": source_inputs_sha256,
            "layer_order": source_metadata.get("layer_order"),
            "evidence_policy": source_metadata.get("evidence_policy"),
            "scope": source_metadata.get("scope"),
            "selected_generator_checkpoint_sha256": (selected_generator_checkpoint_sha256),
            "graph_support_sha256": graph_support_sha256,
            "internal_role_manifest_sha256": internal_role_manifest.manifest_sha256,
            "budget_sources": [item.to_dict() for item in budgets],
        }
    )
    context = PlannerCacheContext(
        planner_id=PRODUCTION_PLANNER_ID,
        planner_sha256=planner_sha256,
        search_policy_sha256=search_policy_sha256,
        value_policy_sha256=value_policy_sha256,
        l1_reaction_sha256=l1_reaction_sha256,
        upstream_reaction_registry_sha256=registry_sha256,
        variant_registry_sha256=variant_sha256,
        verifier_sha256=verifier_sha256,
        l3_snapshot_sha256=source_inputs_sha256,
        l3_region=region,
        l3_accessed_at_utc=_require_nonempty(
            window.get("accessed_utc"), label="production L3 accessed UTC"
        ),
        l3_expires_at_utc=_require_nonempty(
            window.get("expires_utc"), label="production L3 expires UTC"
        ),
        software_versions=tuple(
            sorted(
                (
                    ("forge_planner_cache_key", PLANNER_CACHE_KEY_SCHEMA_VERSION),
                    ("forge_route_schema", "forge.synthesis_assessment.v1"),
                    ("python", platform.python_version()),
                    ("rdkit", rdBase.rdkitVersion),
                )
            )
        ),
        identity_policy=DEFAULT_IDENTITY_POLICY,
        stereochemistry_policy=DEFAULT_STEREOCHEMISTRY_POLICY,
        budget_limits=budget_limits,
    )
    return ProductionPlannerContextQualification(
        planner_context=context,
        source_metadata_sha256=source_metadata_sha256,
        source_inputs_sha256=source_inputs_sha256,
        selected_generator_checkpoint_sha256=selected_generator_checkpoint_sha256,
        graph_support_sha256=graph_support_sha256,
        internal_role_manifest_sha256=internal_role_manifest.manifest_sha256,
        qualified_l1_registry_artifact=qualified_l1_registry_artifact,
        planner_artifacts=planner_artifacts,
        registry_artifacts=registry_artifacts,
        variant_artifacts=variant_artifacts,
        verifier_artifacts=verifier_artifacts,
        value_artifacts=value_artifacts,
        budget_sources=budgets,
    )


__all__ = [
    "PRODUCTION_INTERNAL_ROLE_MANIFEST_SCHEMA_VERSION",
    "PRODUCTION_PLANNER_CONTEXT_QUALIFICATION_SCHEMA_VERSION",
    "PRODUCTION_PLANNER_ID",
    "InternalRoleQualificationRecord",
    "PlannerBudgetSource",
    "ProductionInternalRoleManifest",
    "ProductionPlannerContextQualification",
    "QualifiedArtifact",
    "Ugi3ProductionPlannerQualificationError",
    "authenticated_internal_role_registry_sha256",
    "build_production_planner_context_qualification",
    "derive_production_internal_role_manifest",
]
