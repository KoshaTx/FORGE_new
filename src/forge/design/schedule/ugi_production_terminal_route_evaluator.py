"""Terminal-aware production planner factory for generated Ugi candidates.

The zero-guidance composer owns matched-cache binding and the typed terminal
assessment.  This module owns the narrower terminal-aware planner seam: it
authenticates the cumulative source, requalifies the selected-checkpoint native
candidate, wraps the source with that terminal's exact root qualifications and
returns a cached recursive planner using the already-bound arm overlay.

It does not bind caches, run the product assessment, define a scalar, guide
generation, invoke biology or select a candidate.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.design.ugi_generated_terminal_support import (
    DeclaredGraphSupportContext,
    QualifiedGeneratedUgiTerminalSupport,
    UgiGeneratedTerminalSupportError,
    declared_graph_support_context_sha256,
    qualify_locked_generated_ugi_terminal_support,
)
from forge.design.ugi_held_component_gate import load_ugi_reaction_contract
from forge.design.ugi_matched_budget_orchestration import (
    LockedMatchedTerminal,
    MatchedAssessmentContext,
    RouteComputeUsage,
)
from forge.route.assessment.ugi3_support_boundary import AuthenticatedInternalRoleRegistry
from forge.route.engine.planner import (
    PlannerBudgetLedger,
    RecursiveRouteAssessor,
    RoutePlanner,
    RouteTarget,
    SynthesisAssessment,
)
from forge.route.engine.planner_cache import (
    CachedRoutePlanner,
    PlannerCacheContext,
    PlannerCacheError,
)
from forge.route.engine.planner_cache_snapshot import (
    OverlayFilePlannerCache,
    planner_cache_context_sha256,
    require_current_l3_context,
)
from forge.route.engine.ugi3_production_planner_qualification import (
    ProductionInternalRoleManifest,
    ProductionPlannerContextQualification,
    Ugi3ProductionPlannerQualificationError,
    authenticated_internal_role_registry_sha256,
    build_production_planner_context_qualification,
    derive_production_internal_role_manifest,
)
from forge.route.sources.ugi3_cumulative_production_source import (
    CumulativeProductionUgi3Source,
    CumulativeUgi3ProductionPaths,
    Ugi3CumulativeProductionSourceError,
    load_cumulative_production_ugi3_source,
)
from forge.route.sources.ugi3_source_qualified_cumulative_inputs import (
    source_qualified_cumulative_paths,
)
from forge.route.terminals.terminal_assessment import (
    QualifiedUgiL1Reverifier,
    required_three_role_route_reservation,
)

PRODUCTION_TERMINAL_PLANNER_FACTORY_SCHEMA_VERSION = (
    "forge.ugi_production_terminal_aware_planner_factory.v1"
)
_EXPECTED_SOURCE_SCOPE = {
    "zero_guidance_annotation_rehearsal": True,
    "support_boundary_wrapped": False,
    "synthesis_scalar_computed": False,
    "generator_guidance_authorized": False,
    "biology_used": False,
    "selection_used": False,
    "holdout_accessed": False,
}
_L3_INTERVAL_SEMANTICS = "accessed_inclusive_expires_exclusive"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class UgiProductionTerminalRouteEvaluatorError(RuntimeError):
    """Raised when the production terminal-aware planner seam fails closed."""


CandidateRecordResolver = Callable[[LockedMatchedTerminal], Mapping[str, Any]]


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise UgiProductionTerminalRouteEvaluatorError(
            f"{label} must contain 64 lowercase hexadecimal characters"
        )
    return value


def _require_nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise UgiProductionTerminalRouteEvaluatorError(f"{label} must be a nonempty string")
    return value


def _graph_support_sha256(value: DeclaredGraphSupportContext) -> str:
    return declared_graph_support_context_sha256(value)


def _support_to_dict(value: QualifiedGeneratedUgiTerminalSupport) -> dict[str, Any]:
    """Serialize every typed support fact used to qualify the route roots."""

    return {
        "terminal_sha256": value.terminal_sha256,
        "generator_checkpoint_sha256": value.generator_checkpoint_sha256,
        "product_smiles": value.product_smiles,
        "l1_reverification": value.l1_reverification.to_dict(),
        "handle_rechecks": [
            {
                "role": item.role,
                "raw_handle_matches": item.raw_handle_matches,
                "symmetry_distinct_handle_sites": item.symmetry_distinct_handle_sites,
                "forbidden_substructure_match": item.forbidden_substructure_match,
                "passes_registry_handle_policy": item.passes_registry_handle_policy,
            }
            for item in value.handle_rechecks
        ],
        "root_targets": [item.to_dict() for item in value.root_targets],
        "root_qualifications": [
            {
                "exact_l1_eligible": item.exact_l1_eligible,
                "supported_ugi_role": item.supported_ugi_role,
                "role_handle_qualified": item.role_handle_qualified,
                "molecular_support_state": item.molecular_support_state.value,
                "declared_exclusion_code": item.declared_exclusion_code,
                "declared_exclusion_policy_locator": item.declared_exclusion_policy_locator,
            }
            for item in value.root_qualifications
        ],
    }


def _support_sha256(value: QualifiedGeneratedUgiTerminalSupport) -> str:
    return _sha256_payload(_support_to_dict(value))


@dataclass(frozen=True)
class ProductionTerminalSupportAudit:
    """Immutable per-terminal qualification retained on the returned planner."""

    terminal_sha256: str
    selected_generator_checkpoint_sha256: str
    source_inputs_sha256: str
    graph_support_sha256: str
    internal_role_registry_sha256: str
    internal_role_manifest_sha256: str
    support_sha256: str
    planner_context_sha256: str
    planner_context_qualification_sha256: str
    assessment_as_of_utc: str
    cache_clone_id: str
    support: QualifiedGeneratedUgiTerminalSupport

    def __post_init__(self) -> None:
        for name in (
            "terminal_sha256",
            "selected_generator_checkpoint_sha256",
            "source_inputs_sha256",
            "graph_support_sha256",
            "internal_role_registry_sha256",
            "internal_role_manifest_sha256",
            "support_sha256",
            "planner_context_sha256",
            "planner_context_qualification_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        _require_nonempty(self.assessment_as_of_utc, label="assessment_as_of_utc")
        _require_nonempty(self.cache_clone_id, label="cache_clone_id")
        if not isinstance(self.support, QualifiedGeneratedUgiTerminalSupport):
            raise UgiProductionTerminalRouteEvaluatorError("terminal support is malformed")
        if self.support.terminal_sha256 != self.terminal_sha256:
            raise UgiProductionTerminalRouteEvaluatorError(
                "support audit and qualification refer to different terminals"
            )
        if self.support.generator_checkpoint_sha256 != (self.selected_generator_checkpoint_sha256):
            raise UgiProductionTerminalRouteEvaluatorError(
                "support audit and qualification use different generator checkpoints"
            )
        if self.support_sha256 != _support_sha256(self.support):
            raise UgiProductionTerminalRouteEvaluatorError("terminal support checksum mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "forge.production_terminal_support_audit.v1",
            "terminal_sha256": self.terminal_sha256,
            "selected_generator_checkpoint_sha256": (self.selected_generator_checkpoint_sha256),
            "source_inputs_sha256": self.source_inputs_sha256,
            "graph_support_sha256": self.graph_support_sha256,
            "internal_role_registry_sha256": self.internal_role_registry_sha256,
            "internal_role_manifest_sha256": self.internal_role_manifest_sha256,
            "support_sha256": self.support_sha256,
            "planner_context_sha256": self.planner_context_sha256,
            "planner_context_qualification_sha256": (self.planner_context_qualification_sha256),
            "assessment_as_of_utc": self.assessment_as_of_utc,
            "cache_clone_id": self.cache_clone_id,
            "support": _support_to_dict(self.support),
            "scalar_value": None,
            "success_probability": None,
        }


@dataclass(frozen=True)
class ProductionQualifiedRoutePlanner(RoutePlanner):
    """Cached recursive planner carrying its complete terminal support audit."""

    delegate: CachedRoutePlanner
    support_audit: ProductionTerminalSupportAudit

    def __post_init__(self) -> None:
        if not isinstance(self.delegate, CachedRoutePlanner):
            raise UgiProductionTerminalRouteEvaluatorError(
                "production planner delegate must be CachedRoutePlanner"
            )
        if not isinstance(self.support_audit, ProductionTerminalSupportAudit):
            raise UgiProductionTerminalRouteEvaluatorError(
                "production planner support audit is malformed"
            )

    def assess(
        self,
        target: RouteTarget,
        budget: PlannerBudgetLedger,
    ) -> SynthesisAssessment:
        return self.delegate.assess(target, budget)


@dataclass(frozen=True)
class ProductionUgiTerminalAwarePlannerFactory:
    """Build one per-terminal planner using a composer-bound cache overlay."""

    assessment_as_of_utc: str
    selected_generator_checkpoint_sha256: str
    source: CumulativeProductionUgi3Source
    source_metadata_sha256: str
    source_inputs_sha256: str
    graph_support: DeclaredGraphSupportContext
    graph_support_sha256: str
    l1_reverifier: QualifiedUgiL1Reverifier
    internal_role_manifest: ProductionInternalRoleManifest
    authenticated_internal_roles: AuthenticatedInternalRoleRegistry
    internal_role_registry_sha256: str
    planner_context_qualification: ProductionPlannerContextQualification
    planner_context: PlannerCacheContext
    planner_context_sha256: str
    candidate_record_resolver: CandidateRecordResolver

    def __post_init__(self) -> None:
        _require_nonempty(self.assessment_as_of_utc, label="assessment_as_of_utc")
        for name in (
            "selected_generator_checkpoint_sha256",
            "source_metadata_sha256",
            "source_inputs_sha256",
            "graph_support_sha256",
            "internal_role_registry_sha256",
            "planner_context_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if not isinstance(self.source, CumulativeProductionUgi3Source):
            raise UgiProductionTerminalRouteEvaluatorError(
                "source must be the authenticated cumulative production Ugi source"
            )
        if not isinstance(self.graph_support, DeclaredGraphSupportContext):
            raise UgiProductionTerminalRouteEvaluatorError(
                "graph support must be a DeclaredGraphSupportContext"
            )
        if not isinstance(self.l1_reverifier, QualifiedUgiL1Reverifier):
            raise UgiProductionTerminalRouteEvaluatorError(
                "L1 reverifier must be a QualifiedUgiL1Reverifier"
            )
        if not isinstance(
            self.internal_role_manifest,
            ProductionInternalRoleManifest,
        ):
            raise UgiProductionTerminalRouteEvaluatorError("internal role manifest is malformed")
        if not isinstance(
            self.authenticated_internal_roles,
            AuthenticatedInternalRoleRegistry,
        ):
            raise UgiProductionTerminalRouteEvaluatorError(
                "internal roles must be an AuthenticatedInternalRoleRegistry"
            )
        if not isinstance(
            self.planner_context_qualification,
            ProductionPlannerContextQualification,
        ):
            raise UgiProductionTerminalRouteEvaluatorError(
                "planner context qualification is malformed"
            )
        if not isinstance(self.planner_context, PlannerCacheContext):
            raise UgiProductionTerminalRouteEvaluatorError("planner context is malformed")
        if not callable(self.candidate_record_resolver):
            raise UgiProductionTerminalRouteEvaluatorError(
                "candidate_record_resolver must be callable"
            )
        self._require_dependencies_unchanged()

    @property
    def qualification_sha256(self) -> str:
        return _sha256_payload(
            {
                "schema_version": PRODUCTION_TERMINAL_PLANNER_FACTORY_SCHEMA_VERSION,
                "assessment_as_of_utc": self.assessment_as_of_utc,
                "selected_generator_checkpoint_sha256": (self.selected_generator_checkpoint_sha256),
                "source_metadata_sha256": self.source_metadata_sha256,
                "source_inputs_sha256": self.source_inputs_sha256,
                "graph_support_sha256": self.graph_support_sha256,
                "l1_reaction_sha256": self.l1_reverifier.l1_reaction_sha256,
                "internal_role_registry_sha256": self.internal_role_registry_sha256,
                "internal_role_manifest_sha256": (self.internal_role_manifest.manifest_sha256),
                "planner_context_sha256": self.planner_context_sha256,
                "planner_context_qualification_sha256": (
                    self.planner_context_qualification.qualification_sha256
                ),
                "scope": {
                    "binds_cache": False,
                    "runs_terminal_assessment": False,
                    "scalarizes": False,
                    "guides": False,
                    "biology": False,
                    "selection": False,
                    "holdout": False,
                },
            }
        )

    def _require_dependencies_unchanged(self) -> None:
        metadata = self.source.metadata
        if _sha256_payload(metadata) != self.source_metadata_sha256:
            raise UgiProductionTerminalRouteEvaluatorError(
                "cumulative source metadata changed after factory construction"
            )
        if metadata.get("inputs_sha256") != self.source_inputs_sha256:
            raise UgiProductionTerminalRouteEvaluatorError(
                "cumulative source input manifest changed after factory construction"
            )
        if _graph_support_sha256(self.graph_support) != self.graph_support_sha256:
            raise UgiProductionTerminalRouteEvaluatorError(
                "declared graph-support context changed after factory construction"
            )
        if self.graph_support.generator_checkpoint_sha256 != (
            self.selected_generator_checkpoint_sha256
        ):
            raise UgiProductionTerminalRouteEvaluatorError(
                "declared graph support is not bound to the selected generator checkpoint"
            )
        if self.internal_role_manifest.registry != self.authenticated_internal_roles:
            raise UgiProductionTerminalRouteEvaluatorError(
                "authenticated internal roles differ from their source-derived manifest"
            )
        if authenticated_internal_role_registry_sha256(self.authenticated_internal_roles) != (
            self.internal_role_registry_sha256
        ):
            raise UgiProductionTerminalRouteEvaluatorError(
                "authenticated internal-role registry changed after factory construction"
            )
        if self.internal_role_manifest.source_inputs_sha256 != self.source_inputs_sha256:
            raise UgiProductionTerminalRouteEvaluatorError(
                "internal-role manifest is not bound to the cumulative source inputs"
            )
        if self.planner_context_qualification.planner_context != self.planner_context:
            raise UgiProductionTerminalRouteEvaluatorError(
                "planner context differs from its artifact-derived qualification"
            )
        if self.planner_context_qualification.source_metadata_sha256 != (
            self.source_metadata_sha256
        ):
            raise UgiProductionTerminalRouteEvaluatorError(
                "planner context qualification is not bound to the cumulative source metadata"
            )
        if self.planner_context_qualification.internal_role_manifest_sha256 != (
            self.internal_role_manifest.manifest_sha256
        ):
            raise UgiProductionTerminalRouteEvaluatorError(
                "planner context qualification is not bound to the internal-role manifest"
            )
        if planner_cache_context_sha256(self.planner_context) != self.planner_context_sha256:
            raise UgiProductionTerminalRouteEvaluatorError(
                "planner context changed after factory construction"
            )
        if self.planner_context.l1_reaction_sha256 != (self.l1_reverifier.l1_reaction_sha256):
            raise UgiProductionTerminalRouteEvaluatorError(
                "planner context and qualified L1 reverifier use different reaction hashes"
            )
        if self.planner_context.l3_snapshot_sha256 != self.source_inputs_sha256:
            raise UgiProductionTerminalRouteEvaluatorError(
                "planner L3 snapshot is not bound to the cumulative source input manifest"
            )
        if self.planner_context.budget_limits.maximum_logical_planner_calls != 1:
            raise UgiProductionTerminalRouteEvaluatorError(
                "each isolated component planner must own exactly one logical planner call"
            )
        try:
            require_current_l3_context(
                self.planner_context,
                assessment_at_utc=self.assessment_as_of_utc,
            )
        except PlannerCacheError as error:
            raise UgiProductionTerminalRouteEvaluatorError(
                "planner L3 context is not current at the explicit assessment time"
            ) from error

    def _require_runtime_context(
        self,
        cache: OverlayFilePlannerCache,
        planner_context: PlannerCacheContext,
        assessment_context: MatchedAssessmentContext,
    ) -> None:
        if not isinstance(cache, OverlayFilePlannerCache):
            raise UgiProductionTerminalRouteEvaluatorError(
                "production planner requires the composer's bound cache overlay"
            )
        if not isinstance(planner_context, PlannerCacheContext) or (
            planner_cache_context_sha256(planner_context) != self.planner_context_sha256
        ):
            raise UgiProductionTerminalRouteEvaluatorError(
                "runtime planner context differs from the factory qualification"
            )
        if not isinstance(assessment_context, MatchedAssessmentContext):
            raise UgiProductionTerminalRouteEvaluatorError(
                "runtime assessment context is malformed"
            )
        if cache.clone_id != assessment_context.cache_clone_id:
            raise UgiProductionTerminalRouteEvaluatorError(
                "bound cache overlay and runtime assessment use different clone IDs"
            )
        if not isinstance(assessment_context.unit_reservation, RouteComputeUsage) or not isinstance(
            assessment_context.remaining_budget,
            RouteComputeUsage,
        ):
            raise UgiProductionTerminalRouteEvaluatorError("runtime route budgets are malformed")
        required = required_three_role_route_reservation(planner_context.budget_limits)
        if assessment_context.unit_reservation != required:
            raise UgiProductionTerminalRouteEvaluatorError(
                "unit reservation must equal three planner calls plus the exact "
                "1 + 3 * per-role verifier ceiling"
            )
        if not required.fits_within(assessment_context.remaining_budget):
            raise UgiProductionTerminalRouteEvaluatorError(
                "remaining arm budget cannot atomically reserve the exact three-role ceiling"
            )

    def build_planner(
        self,
        cache: OverlayFilePlannerCache,
        planner_context: PlannerCacheContext,
        assessment_context: MatchedAssessmentContext,
        terminal: LockedMatchedTerminal,
    ) -> ProductionQualifiedRoutePlanner:
        """Return a terminal-qualified planner without binding or assessing it."""

        self._require_dependencies_unchanged()
        self._require_runtime_context(cache, planner_context, assessment_context)
        if not isinstance(terminal, LockedMatchedTerminal):
            raise UgiProductionTerminalRouteEvaluatorError(
                "terminal-aware planning requires a LockedMatchedTerminal"
            )
        if terminal.generator_checkpoint_sha256 != (self.selected_generator_checkpoint_sha256):
            raise UgiProductionTerminalRouteEvaluatorError(
                "terminal was not generated by the selected checkpoint"
            )
        try:
            candidate_record = self.candidate_record_resolver(terminal)
        except Exception as error:
            raise UgiProductionTerminalRouteEvaluatorError(
                "native candidate record could not be resolved for the locked terminal"
            ) from error
        if not isinstance(candidate_record, Mapping):
            raise UgiProductionTerminalRouteEvaluatorError(
                "candidate record resolver returned an unsupported value"
            )
        try:
            support = qualify_locked_generated_ugi_terminal_support(
                terminal,
                candidate_record=candidate_record,
                graph_support=self.graph_support,
                l1_reverifier=self.l1_reverifier,
            )
            source = support.wrap_source(
                self.source,
                authenticated_internal_roles=self.authenticated_internal_roles,
            )
        except UgiGeneratedTerminalSupportError as error:
            raise UgiProductionTerminalRouteEvaluatorError(
                "generated terminal failed exact per-terminal support qualification"
            ) from error
        support_sha256 = _support_sha256(support)
        audit = ProductionTerminalSupportAudit(
            terminal_sha256=terminal.terminal_sha256,
            selected_generator_checkpoint_sha256=(self.selected_generator_checkpoint_sha256),
            source_inputs_sha256=self.source_inputs_sha256,
            graph_support_sha256=self.graph_support_sha256,
            internal_role_registry_sha256=self.internal_role_registry_sha256,
            internal_role_manifest_sha256=self.internal_role_manifest.manifest_sha256,
            support_sha256=support_sha256,
            planner_context_sha256=self.planner_context_sha256,
            planner_context_qualification_sha256=(
                self.planner_context_qualification.qualification_sha256
            ),
            assessment_as_of_utc=self.assessment_as_of_utc,
            cache_clone_id=cache.clone_id,
            support=support,
        )
        return ProductionQualifiedRoutePlanner(
            delegate=CachedRoutePlanner(
                RecursiveRouteAssessor(source),
                cache,
                planner_context,
            ),
            support_audit=audit,
        )

    def __call__(
        self,
        cache: OverlayFilePlannerCache,
        planner_context: PlannerCacheContext,
        assessment_context: MatchedAssessmentContext,
        terminal: LockedMatchedTerminal,
    ) -> ProductionQualifiedRoutePlanner:
        return self.build_planner(cache, planner_context, assessment_context, terminal)


def build_production_ugi_terminal_aware_planner_factory(
    *,
    repo_root: Path,
    assessment_as_of_utc: str,
    expected_cumulative_source_inputs_sha256: str,
    selected_generator_checkpoint_sha256: str,
    graph_support: DeclaredGraphSupportContext,
    l1_reverifier: QualifiedUgiL1Reverifier,
    candidate_record_resolver: CandidateRecordResolver,
    source_paths: CumulativeUgi3ProductionPaths | None = None,
) -> ProductionUgiTerminalAwarePlannerFactory:
    """Authenticate production dependencies and return the terminal-aware seam."""

    _require_nonempty(assessment_as_of_utc, label="assessment_as_of_utc")
    _require_sha256(
        expected_cumulative_source_inputs_sha256,
        label="expected_cumulative_source_inputs_sha256",
    )
    _require_sha256(
        selected_generator_checkpoint_sha256,
        label="selected_generator_checkpoint_sha256",
    )
    if not isinstance(repo_root, Path):
        raise UgiProductionTerminalRouteEvaluatorError("repo_root must be a Path")
    repo_root = repo_root.resolve()
    if source_paths is None:
        source_paths = source_qualified_cumulative_paths(
            repo_root,
            repo_root / "results/phase1/ugi3_source_qualified_cumulative_inputs_v1",
        )
    try:
        source, metadata = load_cumulative_production_ugi3_source(
            repo_root=repo_root,
            assessment_as_of_utc=assessment_as_of_utc,
            paths=source_paths,
        )
    except Ugi3CumulativeProductionSourceError as error:
        raise UgiProductionTerminalRouteEvaluatorError(
            "cumulative production Ugi source failed authentication"
        ) from error
    if metadata.get("inputs_sha256") != expected_cumulative_source_inputs_sha256:
        raise UgiProductionTerminalRouteEvaluatorError(
            "cumulative production source input manifest differs from the expected pin"
        )
    if metadata.get("assessment_as_of_utc") != assessment_as_of_utc:
        raise UgiProductionTerminalRouteEvaluatorError(
            "cumulative source did not preserve the explicit assessment time"
        )
    if metadata.get("scope") != _EXPECTED_SOURCE_SCOPE:
        raise UgiProductionTerminalRouteEvaluatorError(
            "cumulative source scope exceeds zero-guidance route annotation"
        )
    window = metadata.get("unified_l3_window")
    if not isinstance(window, Mapping) or window.get("interval_semantics") != (
        _L3_INTERVAL_SEMANTICS
    ):
        raise UgiProductionTerminalRouteEvaluatorError(
            "cumulative source L3 window is not accessed-inclusive and expiry-exclusive"
        )
    qualified_l1_registry_path = repo_root / "data/vendor/qualified_reactions_v1.json"
    try:
        registry_reaction = load_ugi_reaction_contract(qualified_l1_registry_path)
        same_definition = l1_reverifier.reaction_contract.definition == registry_reaction.definition
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise UgiProductionTerminalRouteEvaluatorError(
            "qualified L1 reaction registry could not be authenticated"
        ) from error
    if not same_definition:
        raise UgiProductionTerminalRouteEvaluatorError(
            "qualified L1 reverifier differs from the vendored reaction registry"
        )
    try:
        internal_role_manifest = derive_production_internal_role_manifest(
            repo_root=repo_root,
            source_metadata=metadata,
        )
        planner_context_qualification = build_production_planner_context_qualification(
            repo_root=repo_root,
            assessment_as_of_utc=assessment_as_of_utc,
            selected_generator_checkpoint_sha256=selected_generator_checkpoint_sha256,
            graph_support_sha256=_graph_support_sha256(graph_support),
            l1_reaction_sha256=l1_reverifier.l1_reaction_sha256,
            source_metadata=metadata,
            internal_role_manifest=internal_role_manifest,
            source_paths=source_paths,
        )
    except Ugi3ProductionPlannerQualificationError as error:
        raise UgiProductionTerminalRouteEvaluatorError(
            "production planner context or internal-role qualification failed"
        ) from error
    authenticated_internal_roles = internal_role_manifest.registry
    planner_context = planner_context_qualification.planner_context
    if planner_context.l3_accessed_at_utc != window.get(
        "accessed_utc"
    ) or planner_context.l3_expires_at_utc != window.get("expires_utc"):
        raise UgiProductionTerminalRouteEvaluatorError(
            "artifact-derived planner context and cumulative source use different L3 windows"
        )
    return ProductionUgiTerminalAwarePlannerFactory(
        assessment_as_of_utc=assessment_as_of_utc,
        selected_generator_checkpoint_sha256=selected_generator_checkpoint_sha256,
        source=source,
        source_metadata_sha256=_sha256_payload(metadata),
        source_inputs_sha256=expected_cumulative_source_inputs_sha256,
        graph_support=graph_support,
        graph_support_sha256=_graph_support_sha256(graph_support),
        l1_reverifier=l1_reverifier,
        internal_role_manifest=internal_role_manifest,
        authenticated_internal_roles=authenticated_internal_roles,
        internal_role_registry_sha256=authenticated_internal_role_registry_sha256(
            authenticated_internal_roles
        ),
        planner_context_qualification=planner_context_qualification,
        planner_context=planner_context,
        planner_context_sha256=planner_cache_context_sha256(planner_context),
        candidate_record_resolver=candidate_record_resolver,
    )


__all__ = [
    "CandidateRecordResolver",
    "PRODUCTION_TERMINAL_PLANNER_FACTORY_SCHEMA_VERSION",
    "ProductionQualifiedRoutePlanner",
    "ProductionTerminalSupportAudit",
    "ProductionUgiTerminalAwarePlannerFactory",
    "UgiProductionTerminalRouteEvaluatorError",
    "build_production_ugi_terminal_aware_planner_factory",
]
