"""Typed, nonselecting Ugi terminal-to-route-assessor integration.

This module binds one sealed, chemically valid and exact-L1 Ugi terminal to
three independently budgeted component route assessments.  It returns only
structured pre-prospective diagnostics.  It does not define a scalar synthesis
value, resample particles, select candidates or invoke a biological oracle.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rdkit import Chem, rdBase

from forge.bio.ugi_semantic_annotations import ROLE_NAMES
from forge.product.ugi_chemistry_flow import (
    UgiChemistrySample,
    chemistry_sample_to_molecule,
)
from forge.product.ugi_chemistry_interface import ChemistryTopologyCondition
from forge.product.ugi_generated_components import (
    generated_ugi_component_smiles,
    precursor_components_from_product_semantics,
)
from forge.product.ugi_held_component_gate import exact_forward_reconstructs_ugi_product
from forge.product.ugi_matched_budget_orchestration import (
    LockedMatchedTerminal,
    MatchedArm,
    MatchedAssessmentContext,
    RouteComputeUsage,
)
from forge.route.planner import (
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RoutePlanner,
    RouteTarget,
    SynthesisAssessment,
)
from forge.route.planner_cache import (
    PlannerCacheContext,
    PlannerCacheKey,
)
from forge.route.planner_cache_snapshot import (
    planner_cache_context_sha256,
    require_current_l3_context,
)
from forge.value.synthesis import (
    ComponentSynthesisValue,
    ProductSynthesisValue,
    component_synthesis_value_from_assessment,
)

VALIDATED_TERMINAL_SCHEMA_VERSION = "forge.validated_ugi_terminal_payload.v1"
TERMINAL_ROUTE_ASSESSMENT_SCHEMA_VERSION = "forge.ugi_terminal_route_assessment.v1"
TARGET_CONTEXT_POLICY = "component_intrinsic_empty_product_context.v1"
DEFAULT_IDENTITY_POLICY = "canonical_constitutional_smiles"
DEFAULT_STEREOCHEMISTRY_POLICY = "phase1_stereo_free"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class UgiTerminalRouteAssessmentError(RuntimeError):
    """Raised when the typed terminal-to-route boundary fails closed."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _require_nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise UgiTerminalRouteAssessmentError(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise UgiTerminalRouteAssessmentError(
            f"{label} must contain 64 lowercase hexadecimal characters"
        )
    return value


def _require_nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UgiTerminalRouteAssessmentError(f"{label} must be a nonnegative integer")
    return value


def _canonical_constitution(smiles: Any, *, label: str) -> str:
    _require_nonempty(smiles, label=label)
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise UgiTerminalRouteAssessmentError(
            f"{label} must be one valid connected molecular constitution"
        )
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _route_usage_to_dict(value: RouteComputeUsage) -> dict[str, int]:
    return {
        "logical_planner_calls": value.logical_planner_calls,
        "physical_planner_calls": value.physical_planner_calls,
        "logical_verifier_calls": value.logical_verifier_calls,
        "physical_verifier_calls": value.physical_verifier_calls,
    }


def _route_usage_from_dict(value: Any) -> RouteComputeUsage:
    if not isinstance(value, dict) or set(value) != {
        "logical_planner_calls",
        "physical_planner_calls",
        "logical_verifier_calls",
        "physical_verifier_calls",
    }:
        raise UgiTerminalRouteAssessmentError("route usage has an unsupported schema")
    return RouteComputeUsage(**value)


def _planner_context_from_dict(value: Any) -> PlannerCacheContext:
    if not isinstance(value, dict):
        raise UgiTerminalRouteAssessmentError("planner context must be an object")
    expected = {
        "planner_id",
        "planner_sha256",
        "search_policy_sha256",
        "value_policy_sha256",
        "l1_reaction_sha256",
        "upstream_reaction_registry_sha256",
        "variant_registry_sha256",
        "verifier_sha256",
        "l3_snapshot_sha256",
        "l3_region",
        "l3_accessed_at_utc",
        "l3_expires_at_utc",
        "software_versions",
        "identity_policy",
        "stereochemistry_policy",
        "budget_limits",
    }
    if set(value) != expected:
        raise UgiTerminalRouteAssessmentError("planner context has an unsupported schema")
    versions = value.get("software_versions")
    if not isinstance(versions, list) or any(
        not isinstance(item, list) or len(item) != 2 for item in versions
    ):
        raise UgiTerminalRouteAssessmentError("planner software versions are malformed")
    return PlannerCacheContext(
        planner_id=value.get("planner_id"),
        planner_sha256=value.get("planner_sha256"),
        search_policy_sha256=value.get("search_policy_sha256"),
        value_policy_sha256=value.get("value_policy_sha256"),
        l1_reaction_sha256=value.get("l1_reaction_sha256"),
        upstream_reaction_registry_sha256=value.get("upstream_reaction_registry_sha256"),
        variant_registry_sha256=value.get("variant_registry_sha256"),
        verifier_sha256=value.get("verifier_sha256"),
        l3_snapshot_sha256=value.get("l3_snapshot_sha256"),
        l3_region=value.get("l3_region"),
        l3_accessed_at_utc=value.get("l3_accessed_at_utc"),
        l3_expires_at_utc=value.get("l3_expires_at_utc"),
        software_versions=tuple((str(item[0]), str(item[1])) for item in versions),
        identity_policy=value.get("identity_policy"),
        stereochemistry_policy=value.get("stereochemistry_policy"),
        budget_limits=PlannerBudgetLimits.from_dict(value.get("budget_limits")),
    )


def _require_current_planner_context(
    context: PlannerCacheContext,
    *,
    assessment_at_utc: str,
) -> None:
    try:
        require_current_l3_context(
            context,
            assessment_at_utc=assessment_at_utc,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise UgiTerminalRouteAssessmentError(
            "planner context is not current at the recorded assessment time"
        ) from error


@dataclass(frozen=True)
class UgiRoleComponent:
    """One exact precursor role and its canonical constitutional identity."""

    role: str
    canonical_smiles: str

    def __post_init__(self) -> None:
        if self.role not in ROLE_NAMES:
            raise UgiTerminalRouteAssessmentError(f"unsupported Ugi role: {self.role!r}")
        canonical = _canonical_constitution(
            self.canonical_smiles,
            label=f"{self.role} component",
        )
        if self.canonical_smiles != canonical:
            raise UgiTerminalRouteAssessmentError(
                f"{self.role} component must already be canonical constitutional SMILES"
            )

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "canonical_smiles": self.canonical_smiles}

    @classmethod
    def from_dict(cls, value: Any) -> UgiRoleComponent:
        if not isinstance(value, dict) or set(value) != {"role", "canonical_smiles"}:
            raise UgiTerminalRouteAssessmentError("Ugi component has an unsupported schema")
        return cls(role=value.get("role"), canonical_smiles=value.get("canonical_smiles"))


@dataclass(frozen=True)
class ExactL1ForwardVerification:
    """Exact qualified Ugi reconstruction state retained at terminal admission."""

    exact_product_reconstructed: bool
    maximum_outcomes: int
    maximum_outcomes_saturated: bool
    outcome_count: int

    def __post_init__(self) -> None:
        if self.exact_product_reconstructed is not True:
            raise UgiTerminalRouteAssessmentError(
                "validated terminal payload requires exact L1 product reconstruction"
            )
        if (
            isinstance(self.maximum_outcomes, bool)
            or not isinstance(self.maximum_outcomes, int)
            or self.maximum_outcomes <= 0
        ):
            raise UgiTerminalRouteAssessmentError("maximum_outcomes must be a positive integer")
        _require_nonnegative_integer(self.outcome_count, label="L1 outcome_count")
        if self.outcome_count == 0:
            raise UgiTerminalRouteAssessmentError(
                "exact L1 reconstruction cannot contain zero forward outcomes"
            )
        if not isinstance(self.maximum_outcomes_saturated, bool):
            raise UgiTerminalRouteAssessmentError("maximum_outcomes_saturated must be boolean")
        expected_saturation = self.outcome_count >= self.maximum_outcomes
        if self.maximum_outcomes_saturated != expected_saturation:
            raise UgiTerminalRouteAssessmentError(
                "L1 saturation flag disagrees with outcome_count and maximum_outcomes"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_product_reconstructed": self.exact_product_reconstructed,
            "maximum_outcomes": self.maximum_outcomes,
            "maximum_outcomes_saturated": self.maximum_outcomes_saturated,
            "outcome_count": self.outcome_count,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ExactL1ForwardVerification:
        if not isinstance(value, dict) or set(value) != {
            "exact_product_reconstructed",
            "maximum_outcomes",
            "maximum_outcomes_saturated",
            "outcome_count",
        }:
            raise UgiTerminalRouteAssessmentError("L1 verification has an unsupported schema")
        return cls(**value)


@dataclass(frozen=True)
class QualifiedUgiL1Reverifier:
    """Qualified reaction contract bound to the L1 artifact it reverifies.

    This object exists so route admission never trusts a serialized exact-L1
    flag.  The caller must bind the compiled qualified reaction to the same
    content hash frozen in the planner and terminal contracts.
    """

    reaction_contract: Any
    l1_reaction_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.l1_reaction_sha256, label="l1_reaction_sha256")
        try:
            role_order = tuple(
                role.name for role in self.reaction_contract.definition.reactant_roles
            )
        except (AttributeError, TypeError) as error:
            raise UgiTerminalRouteAssessmentError(
                "qualified L1 reverifier reaction contract is malformed"
            ) from error
        if role_order != ROLE_NAMES:
            raise UgiTerminalRouteAssessmentError(
                "qualified L1 reverifier role order differs from the frozen Ugi adapter"
            )

    def require_exact(
        self,
        payload: ValidatedUgiTerminalPayload,
    ) -> ExactL1ForwardVerification:
        """Independently rerun the qualified forward transform and fail closed."""

        if not isinstance(payload, ValidatedUgiTerminalPayload):
            raise UgiTerminalRouteAssessmentError(
                "qualified L1 reverification requires a validated payload record"
            )
        if payload.l1_reaction_sha256 != self.l1_reaction_sha256:
            raise UgiTerminalRouteAssessmentError(
                "L1 reverifier and terminal payload use different reaction hashes"
            )
        exact, saturated, outcome_count = exact_forward_reconstructs_ugi_product(
            self.reaction_contract,
            {component.role: component.canonical_smiles for component in payload.components},
            payload.product_smiles,
            maximum_outcomes=(payload.l1_forward_verification.maximum_outcomes),
        )
        if not exact:
            raise UgiTerminalRouteAssessmentError(
                "independent qualified L1 reverification did not reconstruct the terminal"
            )
        stored = payload.l1_forward_verification
        if saturated != stored.maximum_outcomes_saturated or outcome_count != stored.outcome_count:
            raise UgiTerminalRouteAssessmentError(
                "independent qualified L1 result disagrees with serialized verification"
            )
        return ExactL1ForwardVerification(
            exact_product_reconstructed=True,
            maximum_outcomes=stored.maximum_outcomes,
            maximum_outcomes_saturated=saturated,
            outcome_count=outcome_count,
        )


@dataclass(frozen=True)
class ValidatedUgiTerminalPayload:
    """Canonical exact-L1 terminal payload suitable for route assessment."""

    product_smiles: str
    components: tuple[UgiRoleComponent, ...]
    l1_forward_verification: ExactL1ForwardVerification
    l1_reaction_sha256: str
    component_recovery_contract_sha256: str
    identity_policy: str = DEFAULT_IDENTITY_POLICY
    stereochemistry_policy: str = DEFAULT_STEREOCHEMISTRY_POLICY

    def __post_init__(self) -> None:
        canonical_product = _canonical_constitution(
            self.product_smiles,
            label="terminal product",
        )
        if self.product_smiles != canonical_product:
            raise UgiTerminalRouteAssessmentError(
                "terminal product must already be canonical constitutional SMILES"
            )
        if (
            not isinstance(self.components, tuple)
            or tuple(component.role for component in self.components) != ROLE_NAMES
        ):
            raise UgiTerminalRouteAssessmentError(
                "terminal payload must contain exactly the three Ugi roles in frozen order"
            )
        if any(not isinstance(component, UgiRoleComponent) for component in self.components):
            raise UgiTerminalRouteAssessmentError(
                "terminal components must be UgiRoleComponent records"
            )
        if not isinstance(self.l1_forward_verification, ExactL1ForwardVerification):
            raise UgiTerminalRouteAssessmentError(
                "terminal payload requires typed exact-L1 verification"
            )
        _require_sha256(self.l1_reaction_sha256, label="l1_reaction_sha256")
        _require_sha256(
            self.component_recovery_contract_sha256,
            label="component_recovery_contract_sha256",
        )
        _require_nonempty(self.identity_policy, label="identity_policy")
        _require_nonempty(self.stereochemistry_policy, label="stereochemistry_policy")

    @classmethod
    def from_recovered_components(
        cls,
        *,
        product_smiles: str,
        components_by_role: Mapping[str, str],
        l1_reaction: Any,
        l1_reaction_sha256: str,
        component_recovery_contract_sha256: str,
        maximum_outcomes: int = 64,
        identity_policy: str = DEFAULT_IDENTITY_POLICY,
        stereochemistry_policy: str = DEFAULT_STEREOCHEMISTRY_POLICY,
    ) -> ValidatedUgiTerminalPayload:
        """Canonicalize exact recovered roles and independently verify L1."""

        if not isinstance(components_by_role, Mapping) or set(components_by_role) != set(
            ROLE_NAMES
        ):
            raise UgiTerminalRouteAssessmentError(
                "recovered components must contain exactly the three frozen Ugi roles"
            )
        role_order = tuple(role.name for role in l1_reaction.definition.reactant_roles)
        if role_order != ROLE_NAMES:
            raise UgiTerminalRouteAssessmentError(
                "qualified L1 reaction role order differs from the frozen Ugi adapter"
            )
        if (
            isinstance(maximum_outcomes, bool)
            or not isinstance(maximum_outcomes, int)
            or maximum_outcomes <= 0
        ):
            raise UgiTerminalRouteAssessmentError("maximum_outcomes must be a positive integer")
        canonical_product = _canonical_constitution(product_smiles, label="terminal product")
        canonical_components = {
            role: _canonical_constitution(
                components_by_role[role],
                label=f"recovered {role} component",
            )
            for role in ROLE_NAMES
        }
        exact, saturated, outcome_count = exact_forward_reconstructs_ugi_product(
            l1_reaction,
            canonical_components,
            canonical_product,
            maximum_outcomes=maximum_outcomes,
        )
        if not exact:
            raise UgiTerminalRouteAssessmentError(
                "recovered Ugi components do not exactly reconstruct the terminal product"
            )
        return cls(
            product_smiles=canonical_product,
            components=tuple(
                UgiRoleComponent(role=role, canonical_smiles=canonical_components[role])
                for role in ROLE_NAMES
            ),
            l1_forward_verification=ExactL1ForwardVerification(
                exact_product_reconstructed=True,
                maximum_outcomes=maximum_outcomes,
                maximum_outcomes_saturated=saturated,
                outcome_count=outcome_count,
            ),
            l1_reaction_sha256=l1_reaction_sha256,
            component_recovery_contract_sha256=component_recovery_contract_sha256,
            identity_policy=identity_policy,
            stereochemistry_policy=stereochemistry_policy,
        )

    @classmethod
    def from_product_semantics(
        cls,
        *,
        product: Chem.Mol,
        origin_states: Sequence[int],
        core_position_states: Sequence[int],
        l1_reaction: Any,
        l1_reaction_sha256: str,
        component_recovery_contract_sha256: str,
        maximum_outcomes: int = 64,
    ) -> ValidatedUgiTerminalPayload:
        """Recover the exact three Ugi roles from product atom semantics."""

        try:
            components = precursor_components_from_product_semantics(
                product,
                origin_states,
                core_position_states,
            )
            product_smiles = Chem.MolToSmiles(
                product,
                canonical=True,
                isomericSmiles=False,
            )
        except (RuntimeError, ValueError) as error:
            raise UgiTerminalRouteAssessmentError(
                "exact Ugi precursor recovery from product semantics failed"
            ) from error
        return cls.from_recovered_components(
            product_smiles=product_smiles,
            components_by_role=components,
            l1_reaction=l1_reaction,
            l1_reaction_sha256=l1_reaction_sha256,
            component_recovery_contract_sha256=component_recovery_contract_sha256,
            maximum_outcomes=maximum_outcomes,
        )

    @classmethod
    def from_generated_semantics(
        cls,
        *,
        condition: ChemistryTopologyCondition,
        sample: UgiChemistrySample,
        atom_vocabulary: tuple[Any, ...],
        l1_reaction: Any,
        l1_reaction_sha256: str,
        component_recovery_contract_sha256: str,
        maximum_outcomes: int = 64,
    ) -> ValidatedUgiTerminalPayload:
        """Build the payload directly from a completed generated terminal."""

        try:
            product = chemistry_sample_to_molecule(condition, sample, atom_vocabulary)
            components = generated_ugi_component_smiles(condition, sample, atom_vocabulary)
            product_smiles = Chem.MolToSmiles(
                product,
                canonical=True,
                isomericSmiles=False,
            )
        except (RuntimeError, ValueError) as error:
            raise UgiTerminalRouteAssessmentError(
                "generated terminal chemistry or exact component recovery failed"
            ) from error
        return cls.from_recovered_components(
            product_smiles=product_smiles,
            components_by_role=components,
            l1_reaction=l1_reaction,
            l1_reaction_sha256=l1_reaction_sha256,
            component_recovery_contract_sha256=component_recovery_contract_sha256,
            maximum_outcomes=maximum_outcomes,
        )

    def by_role(self) -> dict[str, UgiRoleComponent]:
        return {component.role: component for component in self.components}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VALIDATED_TERMINAL_SCHEMA_VERSION,
            "product_smiles": self.product_smiles,
            "components": [component.to_dict() for component in self.components],
            "l1_forward_verification": self.l1_forward_verification.to_dict(),
            "l1_reaction_sha256": self.l1_reaction_sha256,
            "component_recovery_contract_sha256": self.component_recovery_contract_sha256,
            "identity_policy": self.identity_policy,
            "stereochemistry_policy": self.stereochemistry_policy,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return (_stable_json(self.to_dict()) + "\n").encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> ValidatedUgiTerminalPayload:
        expected = {
            "schema_version",
            "product_smiles",
            "components",
            "l1_forward_verification",
            "l1_reaction_sha256",
            "component_recovery_contract_sha256",
            "identity_policy",
            "stereochemistry_policy",
        }
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or value.get("schema_version") != VALIDATED_TERMINAL_SCHEMA_VERSION
        ):
            raise UgiTerminalRouteAssessmentError(
                "validated Ugi terminal payload has an unsupported schema"
            )
        components = value.get("components")
        if not isinstance(components, list):
            raise UgiTerminalRouteAssessmentError("terminal components must be a list")
        return cls(
            product_smiles=value.get("product_smiles"),
            components=tuple(UgiRoleComponent.from_dict(item) for item in components),
            l1_forward_verification=ExactL1ForwardVerification.from_dict(
                value.get("l1_forward_verification")
            ),
            l1_reaction_sha256=value.get("l1_reaction_sha256"),
            component_recovery_contract_sha256=value.get("component_recovery_contract_sha256"),
            identity_policy=value.get("identity_policy"),
            stereochemistry_policy=value.get("stereochemistry_policy"),
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> ValidatedUgiTerminalPayload:
        if not isinstance(payload, bytes) or not payload:
            raise UgiTerminalRouteAssessmentError("terminal payload bytes must be nonempty")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UgiTerminalRouteAssessmentError(
                "terminal payload bytes are invalid JSON"
            ) from error
        record = cls.from_dict(value)
        if payload != record.canonical_bytes:
            raise UgiTerminalRouteAssessmentError("terminal payload bytes are not canonical")
        return record


@dataclass(frozen=True)
class PlannerBudgetReceipt:
    """Immutable snapshot of every logical and physical planner counter."""

    limits: PlannerBudgetLimits
    logical_planner_calls: int
    physical_cache_hits: int
    physical_cache_misses: int
    expansions: int
    product_candidates: int
    verifier_calls: int
    elapsed_milliseconds: int
    exhaustion_events: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.limits, PlannerBudgetLimits):
            raise UgiTerminalRouteAssessmentError("budget receipt limits are malformed")
        for name in (
            "logical_planner_calls",
            "physical_cache_hits",
            "physical_cache_misses",
            "expansions",
            "product_candidates",
            "verifier_calls",
            "elapsed_milliseconds",
        ):
            _require_nonnegative_integer(getattr(self, name), label=name)
        if not isinstance(self.exhaustion_events, tuple) or any(
            not isinstance(value, str) or not value for value in self.exhaustion_events
        ):
            raise UgiTerminalRouteAssessmentError(
                "budget exhaustion events must be nonempty strings"
            )
        if self.expansions > self.limits.maximum_expansions:
            raise UgiTerminalRouteAssessmentError("expansion usage exceeds its limit")
        if self.logical_planner_calls > self.limits.maximum_logical_planner_calls:
            raise UgiTerminalRouteAssessmentError("logical planner usage exceeds its limit")
        if self.physical_cache_hits + self.physical_cache_misses > (self.logical_planner_calls):
            raise UgiTerminalRouteAssessmentError(
                "physical cache decisions exceed logical planner calls"
            )
        if self.product_candidates > self.limits.maximum_product_candidates:
            raise UgiTerminalRouteAssessmentError("product-candidate usage exceeds its limit")
        if self.verifier_calls > self.limits.maximum_verifier_calls:
            raise UgiTerminalRouteAssessmentError("verifier usage exceeds its limit")
        if self.elapsed_milliseconds > self.limits.maximum_elapsed_milliseconds:
            raise UgiTerminalRouteAssessmentError("elapsed usage exceeds its limit")

    @classmethod
    def from_ledger(cls, ledger: PlannerBudgetLedger) -> PlannerBudgetReceipt:
        return cls(
            limits=ledger.limits,
            logical_planner_calls=ledger.logical_planner_calls,
            physical_cache_hits=ledger.physical_cache_hits,
            physical_cache_misses=ledger.physical_cache_misses,
            expansions=ledger.expansions,
            product_candidates=ledger.product_candidates,
            verifier_calls=ledger.verifier_calls,
            elapsed_milliseconds=ledger.elapsed_milliseconds,
            exhaustion_events=tuple(ledger.exhaustion_events),
        )

    @property
    def fresh(self) -> bool:
        return (
            all(
                getattr(self, name) == 0
                for name in (
                    "logical_planner_calls",
                    "physical_cache_hits",
                    "physical_cache_misses",
                    "expansions",
                    "product_candidates",
                    "verifier_calls",
                    "elapsed_milliseconds",
                )
            )
            and not self.exhaustion_events
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "limits": self.limits.to_dict(),
            "logical_planner_calls": self.logical_planner_calls,
            "physical_cache_hits": self.physical_cache_hits,
            "physical_cache_misses": self.physical_cache_misses,
            "expansions": self.expansions,
            "product_candidates": self.product_candidates,
            "verifier_calls": self.verifier_calls,
            "elapsed_milliseconds": self.elapsed_milliseconds,
            "exhaustion_events": list(self.exhaustion_events),
        }

    @classmethod
    def from_dict(cls, value: Any) -> PlannerBudgetReceipt:
        expected = {
            "limits",
            "logical_planner_calls",
            "physical_cache_hits",
            "physical_cache_misses",
            "expansions",
            "product_candidates",
            "verifier_calls",
            "elapsed_milliseconds",
            "exhaustion_events",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise UgiTerminalRouteAssessmentError("budget receipt has an unsupported schema")
        events = value.get("exhaustion_events")
        if not isinstance(events, list):
            raise UgiTerminalRouteAssessmentError("budget exhaustion events must be a list")
        return cls(
            limits=PlannerBudgetLimits.from_dict(value.get("limits")),
            logical_planner_calls=value.get("logical_planner_calls"),
            physical_cache_hits=value.get("physical_cache_hits"),
            physical_cache_misses=value.get("physical_cache_misses"),
            expansions=value.get("expansions"),
            product_candidates=value.get("product_candidates"),
            verifier_calls=value.get("verifier_calls"),
            elapsed_milliseconds=value.get("elapsed_milliseconds"),
            exhaustion_events=tuple(events),
        )


@dataclass(frozen=True)
class RoleRouteAssessmentReceipt:
    """One role's isolated route assessment, value and cache provenance."""

    role: str
    target: RouteTarget
    cache_key_sha256: str
    cache_key_payload_json: str
    budget_before: PlannerBudgetReceipt
    budget_after: PlannerBudgetReceipt
    assessment: SynthesisAssessment
    component_value: ComponentSynthesisValue

    def __post_init__(self) -> None:
        if self.role not in ROLE_NAMES or self.target.role != self.role:
            raise UgiTerminalRouteAssessmentError("role receipt target and role disagree")
        if self.target.product_context_smiles:
            raise UgiTerminalRouteAssessmentError(
                "v1 component-intrinsic route targets cannot carry product context"
            )
        _require_sha256(self.cache_key_sha256, label="cache_key_sha256")
        try:
            cache_key = PlannerCacheKey(
                digest=self.cache_key_sha256,
                payload_json=self.cache_key_payload_json,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise UgiTerminalRouteAssessmentError(
                "role receipt cache-key provenance is malformed"
            ) from error
        if cache_key.payload.get("target") != self.target.to_dict():
            raise UgiTerminalRouteAssessmentError(
                "role receipt cache key does not own its route target"
            )
        if not self.budget_before.fresh:
            raise UgiTerminalRouteAssessmentError(
                "every component assessment must begin with a fresh budget ledger"
            )
        if self.budget_before.limits != self.budget_after.limits:
            raise UgiTerminalRouteAssessmentError(
                "component budget limits changed during assessment"
            )
        if self.budget_after.logical_planner_calls != 1:
            raise UgiTerminalRouteAssessmentError(
                "each role must consume exactly one logical planner call"
            )
        if self.budget_after.physical_cache_hits + self.budget_after.physical_cache_misses != 1:
            raise UgiTerminalRouteAssessmentError(
                "each role must record exactly one physical cache hit or miss"
            )
        if self.assessment.target != self.target:
            raise UgiTerminalRouteAssessmentError("route assessment target changed")
        if self.component_value.target != self.target:
            raise UgiTerminalRouteAssessmentError("component value target changed")

    @property
    def assessment_sha256(self) -> str:
        return _sha256_payload(self.assessment.to_dict())

    @property
    def component_value_sha256(self) -> str:
        return _sha256_payload(self.component_value.to_dict())

    @property
    def realized_route_usage(self) -> RouteComputeUsage:
        return RouteComputeUsage(
            logical_planner_calls=self.budget_after.logical_planner_calls,
            physical_planner_calls=self.budget_after.physical_cache_misses,
            logical_verifier_calls=self.budget_after.verifier_calls,
            physical_verifier_calls=self.budget_after.verifier_calls,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "target": self.target.to_dict(),
            "cache_key_sha256": self.cache_key_sha256,
            "cache_key_payload_json": self.cache_key_payload_json,
            "budget_before": self.budget_before.to_dict(),
            "budget_after": self.budget_after.to_dict(),
            "assessment_sha256": self.assessment_sha256,
            "assessment": self.assessment.to_dict(),
            "component_value_sha256": self.component_value_sha256,
            "component_value": self.component_value.to_dict(),
            "realized_route_usage": _route_usage_to_dict(self.realized_route_usage),
        }

    @classmethod
    def from_dict(cls, value: Any) -> RoleRouteAssessmentReceipt:
        expected = {
            "role",
            "target",
            "cache_key_sha256",
            "cache_key_payload_json",
            "budget_before",
            "budget_after",
            "assessment_sha256",
            "assessment",
            "component_value_sha256",
            "component_value",
            "realized_route_usage",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise UgiTerminalRouteAssessmentError("role receipt has an unsupported schema")
        record = cls(
            role=value.get("role"),
            target=RouteTarget.from_dict(value.get("target")),
            cache_key_sha256=value.get("cache_key_sha256"),
            cache_key_payload_json=value.get("cache_key_payload_json"),
            budget_before=PlannerBudgetReceipt.from_dict(value.get("budget_before")),
            budget_after=PlannerBudgetReceipt.from_dict(value.get("budget_after")),
            assessment=SynthesisAssessment.from_dict(value.get("assessment")),
            component_value=ComponentSynthesisValue.from_dict(value.get("component_value")),
        )
        if value.get("assessment_sha256") != record.assessment_sha256:
            raise UgiTerminalRouteAssessmentError("role assessment checksum mismatch")
        if value.get("component_value_sha256") != record.component_value_sha256:
            raise UgiTerminalRouteAssessmentError("component value checksum mismatch")
        if _route_usage_from_dict(value.get("realized_route_usage")) != (
            record.realized_route_usage
        ):
            raise UgiTerminalRouteAssessmentError("role route usage is inconsistent")
        return record


def required_three_role_route_reservation(
    limits: PlannerBudgetLimits,
) -> RouteComputeUsage:
    """Return the fail-closed worst-case reservation for one Ugi product."""

    if not isinstance(limits, PlannerBudgetLimits):
        raise UgiTerminalRouteAssessmentError("component planner limits are malformed")
    role_count = len(ROLE_NAMES)
    verifier_ceiling = 1 + role_count * limits.maximum_verifier_calls
    return RouteComputeUsage(
        logical_planner_calls=role_count,
        physical_planner_calls=role_count,
        logical_verifier_calls=verifier_ceiling,
        physical_verifier_calls=verifier_ceiling,
    )


def _sum_route_usage(values: Sequence[RouteComputeUsage]) -> RouteComputeUsage:
    output = RouteComputeUsage()
    for value in values:
        output = output.plus(value)
    return output


def _component_budget_policy_sha256(limits: PlannerBudgetLimits) -> str:
    return _sha256_payload(
        {
            "schema": "forge.ugi_component_budget_policy.v1",
            "roles": list(ROLE_NAMES),
            "target_context_policy": TARGET_CONTEXT_POLICY,
            "per_role_limits": limits.to_dict(),
            "product_l1_reverification": {
                "logical_verifier_calls": 1,
                "physical_verifier_calls": 1,
                "cache_policy": "uncached",
            },
            "ledger_isolation": "fresh_ledger_per_role",
            "outer_reservation": "atomic_three_role_worst_case",
        }
    )


@dataclass(frozen=True)
class UgiTerminalRouteAssessmentReceipt:
    """Complete nonselecting product-level route diagnostic."""

    unit_id: str
    terminal_id: str
    terminal_sha256: str
    generation_trace_sha256: str
    morphology_program_sha256: str
    checkpoint_index: int
    generator_checkpoint_sha256: str
    closure_checkpoint_sha256: str
    payload: ValidatedUgiTerminalPayload
    assessment_at_utc: str
    arm: MatchedArm
    route_seed: int
    planner_context: PlannerCacheContext
    cache_snapshot_sha256: str
    cache_clone_id: str
    post_hoc_lock_manifest_sha256: str | None
    target_context_policy: str
    component_budget_policy_sha256: str
    unit_route_reservation: RouteComputeUsage
    l1_reverification: ExactL1ForwardVerification
    l1_reverification_usage: RouteComputeUsage
    role_assessments: tuple[RoleRouteAssessmentReceipt, ...]
    product_value: ProductSynthesisValue

    def __post_init__(self) -> None:
        for name in ("unit_id", "terminal_id", "assessment_at_utc", "cache_clone_id"):
            _require_nonempty(getattr(self, name), label=name)
        for name in (
            "terminal_sha256",
            "generation_trace_sha256",
            "morphology_program_sha256",
            "generator_checkpoint_sha256",
            "closure_checkpoint_sha256",
            "cache_snapshot_sha256",
            "component_budget_policy_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        _require_nonnegative_integer(self.checkpoint_index, label="checkpoint_index")
        _require_nonnegative_integer(self.route_seed, label="route_seed")
        if not isinstance(self.arm, MatchedArm):
            raise UgiTerminalRouteAssessmentError("assessment arm is unsupported")
        if not isinstance(self.payload, ValidatedUgiTerminalPayload):
            raise UgiTerminalRouteAssessmentError("assessment payload is not validated")
        if self.terminal_sha256 != self.payload.sha256:
            raise UgiTerminalRouteAssessmentError(
                "terminal hash does not own the canonical validated payload"
            )
        if not isinstance(self.planner_context, PlannerCacheContext):
            raise UgiTerminalRouteAssessmentError("planner context is malformed")
        _require_current_planner_context(
            self.planner_context,
            assessment_at_utc=self.assessment_at_utc,
        )
        if self.payload.l1_reaction_sha256 != self.planner_context.l1_reaction_sha256:
            raise UgiTerminalRouteAssessmentError(
                "payload and planner contexts use different L1 reaction hashes"
            )
        if self.payload.identity_policy != self.planner_context.identity_policy:
            raise UgiTerminalRouteAssessmentError("payload and planner identity policies disagree")
        if self.payload.stereochemistry_policy != self.planner_context.stereochemistry_policy:
            raise UgiTerminalRouteAssessmentError(
                "payload and planner stereochemistry policies disagree"
            )
        if self.target_context_policy != TARGET_CONTEXT_POLICY:
            raise UgiTerminalRouteAssessmentError("target-context policy changed")
        if self.l1_reverification != self.payload.l1_forward_verification:
            raise UgiTerminalRouteAssessmentError(
                "independent L1 reverification disagrees with the validated payload"
            )
        if self.l1_reverification_usage != RouteComputeUsage(
            logical_verifier_calls=1,
            physical_verifier_calls=1,
        ):
            raise UgiTerminalRouteAssessmentError(
                "independent L1 reverification must record one uncached verifier call"
            )
        expected_budget_policy = _component_budget_policy_sha256(self.planner_context.budget_limits)
        if self.component_budget_policy_sha256 != expected_budget_policy:
            raise UgiTerminalRouteAssessmentError(
                "component budget policy checksum is inconsistent"
            )
        required = required_three_role_route_reservation(self.planner_context.budget_limits)
        if not required.fits_within(self.unit_route_reservation):
            raise UgiTerminalRouteAssessmentError(
                "unit reservation cannot bound three isolated component assessments"
            )
        if self.arm is MatchedArm.POST_HOC:
            _require_sha256(
                self.post_hoc_lock_manifest_sha256,
                label="post_hoc_lock_manifest_sha256",
            )
        elif self.post_hoc_lock_manifest_sha256 is not None:
            raise UgiTerminalRouteAssessmentError(
                "guided receipt cannot carry a post-hoc lock manifest"
            )
        if tuple(item.role for item in self.role_assessments) != ROLE_NAMES:
            raise UgiTerminalRouteAssessmentError(
                "product receipt must retain exactly three role assessments in frozen order"
            )
        if tuple(role for role, _ in self.product_value.components) != ROLE_NAMES:
            raise UgiTerminalRouteAssessmentError(
                "product value must retain exactly three Ugi roles in frozen order"
            )
        payload_by_role = self.payload.by_role()
        for item in self.role_assessments:
            if item.target.canonical_smiles != payload_by_role[item.role].canonical_smiles:
                raise UgiTerminalRouteAssessmentError(
                    "route assessment target differs from its validated payload component"
                )
            fresh_ledger = PlannerBudgetLedger(self.planner_context.budget_limits)
            expected_cache_key = PlannerCacheKey.build(
                item.target,
                self.planner_context,
                fresh_ledger,
            )
            if (
                item.cache_key_sha256 != expected_cache_key.digest
                or item.cache_key_payload_json != expected_cache_key.payload_json
            ):
                raise UgiTerminalRouteAssessmentError(
                    "role cache-key provenance differs from the frozen planner context"
                )
        if self.product_value.product_smiles != self.payload.product_smiles:
            raise UgiTerminalRouteAssessmentError("product value and terminal product disagree")
        if not self.product_value.l1_forward_consistent:
            raise UgiTerminalRouteAssessmentError(
                "validated exact-L1 terminal cannot carry an inconsistent product value"
            )
        if tuple(value for _, value in self.product_value.components) != tuple(
            item.component_value for item in self.role_assessments
        ):
            raise UgiTerminalRouteAssessmentError(
                "product value and role assessment values disagree"
            )
        if not self.realized_route_usage.fits_within(self.unit_route_reservation):
            raise UgiTerminalRouteAssessmentError(
                "realized route usage exceeds the unit reservation"
            )

    @property
    def planner_context_sha256(self) -> str:
        return planner_cache_context_sha256(self.planner_context)

    @property
    def product_value_sha256(self) -> str:
        return _sha256_payload(self.product_value.to_dict())

    @property
    def realized_route_usage(self) -> RouteComputeUsage:
        return self.l1_reverification_usage.plus(
            _sum_route_usage(tuple(item.realized_route_usage for item in self.role_assessments))
        )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TERMINAL_ROUTE_ASSESSMENT_SCHEMA_VERSION,
            "unit_id": self.unit_id,
            "terminal_id": self.terminal_id,
            "terminal_sha256": self.terminal_sha256,
            "generation_trace_sha256": self.generation_trace_sha256,
            "morphology_program_sha256": self.morphology_program_sha256,
            "checkpoint_index": self.checkpoint_index,
            "generator_checkpoint_sha256": self.generator_checkpoint_sha256,
            "closure_checkpoint_sha256": self.closure_checkpoint_sha256,
            "payload": self.payload.to_dict(),
            "assessment_at_utc": self.assessment_at_utc,
            "arm": self.arm.value,
            "route_seed": self.route_seed,
            "planner_context_sha256": self.planner_context_sha256,
            "planner_context": self.planner_context.to_dict(),
            "cache_snapshot_sha256": self.cache_snapshot_sha256,
            "cache_clone_id": self.cache_clone_id,
            "post_hoc_lock_manifest_sha256": self.post_hoc_lock_manifest_sha256,
            "target_context_policy": self.target_context_policy,
            "component_budget_policy_sha256": self.component_budget_policy_sha256,
            "unit_route_reservation": _route_usage_to_dict(self.unit_route_reservation),
            "l1_reverification": self.l1_reverification.to_dict(),
            "l1_reverification_usage": _route_usage_to_dict(self.l1_reverification_usage),
            "role_assessments": [item.to_dict() for item in self.role_assessments],
            "product_value_sha256": self.product_value_sha256,
            "product_value": self.product_value.to_dict(),
            "realized_route_usage": _route_usage_to_dict(self.realized_route_usage),
            "scalar_value": None,
            "success_probability": None,
            "candidate_selected": False,
            "biological_guidance": False,
        }

    @property
    def assessment_sha256(self) -> str:
        return _sha256_payload(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "assessment_sha256": self.assessment_sha256}

    @property
    def canonical_bytes(self) -> bytes:
        return (_stable_json(self.to_dict()) + "\n").encode()

    @classmethod
    def from_dict(cls, value: Any) -> UgiTerminalRouteAssessmentReceipt:
        expected = {
            "schema_version",
            "unit_id",
            "terminal_id",
            "terminal_sha256",
            "generation_trace_sha256",
            "morphology_program_sha256",
            "checkpoint_index",
            "generator_checkpoint_sha256",
            "closure_checkpoint_sha256",
            "payload",
            "assessment_at_utc",
            "arm",
            "route_seed",
            "planner_context_sha256",
            "planner_context",
            "cache_snapshot_sha256",
            "cache_clone_id",
            "post_hoc_lock_manifest_sha256",
            "target_context_policy",
            "component_budget_policy_sha256",
            "unit_route_reservation",
            "l1_reverification",
            "l1_reverification_usage",
            "role_assessments",
            "product_value_sha256",
            "product_value",
            "realized_route_usage",
            "scalar_value",
            "success_probability",
            "candidate_selected",
            "biological_guidance",
            "assessment_sha256",
        }
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or value.get("schema_version") != TERMINAL_ROUTE_ASSESSMENT_SCHEMA_VERSION
        ):
            raise UgiTerminalRouteAssessmentError(
                "terminal route assessment has an unsupported schema"
            )
        if (
            value.get("scalar_value") is not None
            or value.get("success_probability") is not None
            or value.get("candidate_selected") is not False
            or value.get("biological_guidance") is not False
        ):
            raise UgiTerminalRouteAssessmentError(
                "diagnostic receipt cannot contain selection, biology or scalar synthesis claims"
            )
        try:
            arm = MatchedArm(value.get("arm"))
        except ValueError as error:
            raise UgiTerminalRouteAssessmentError("receipt arm is unsupported") from error
        roles = value.get("role_assessments")
        if not isinstance(roles, list):
            raise UgiTerminalRouteAssessmentError("role assessments must be a list")
        record = cls(
            unit_id=value.get("unit_id"),
            terminal_id=value.get("terminal_id"),
            terminal_sha256=value.get("terminal_sha256"),
            generation_trace_sha256=value.get("generation_trace_sha256"),
            morphology_program_sha256=value.get("morphology_program_sha256"),
            checkpoint_index=value.get("checkpoint_index"),
            generator_checkpoint_sha256=value.get("generator_checkpoint_sha256"),
            closure_checkpoint_sha256=value.get("closure_checkpoint_sha256"),
            payload=ValidatedUgiTerminalPayload.from_dict(value.get("payload")),
            assessment_at_utc=value.get("assessment_at_utc"),
            arm=arm,
            route_seed=value.get("route_seed"),
            planner_context=_planner_context_from_dict(value.get("planner_context")),
            cache_snapshot_sha256=value.get("cache_snapshot_sha256"),
            cache_clone_id=value.get("cache_clone_id"),
            post_hoc_lock_manifest_sha256=value.get("post_hoc_lock_manifest_sha256"),
            target_context_policy=value.get("target_context_policy"),
            component_budget_policy_sha256=value.get("component_budget_policy_sha256"),
            unit_route_reservation=_route_usage_from_dict(value.get("unit_route_reservation")),
            l1_reverification=ExactL1ForwardVerification.from_dict(value.get("l1_reverification")),
            l1_reverification_usage=_route_usage_from_dict(value.get("l1_reverification_usage")),
            role_assessments=tuple(RoleRouteAssessmentReceipt.from_dict(item) for item in roles),
            product_value=ProductSynthesisValue.from_dict(value.get("product_value")),
        )
        if value.get("planner_context_sha256") != record.planner_context_sha256:
            raise UgiTerminalRouteAssessmentError("planner context checksum mismatch")
        if value.get("product_value_sha256") != record.product_value_sha256:
            raise UgiTerminalRouteAssessmentError("product value checksum mismatch")
        if _route_usage_from_dict(value.get("realized_route_usage")) != (
            record.realized_route_usage
        ):
            raise UgiTerminalRouteAssessmentError("realized route usage is inconsistent")
        if value.get("assessment_sha256") != record.assessment_sha256:
            raise UgiTerminalRouteAssessmentError("terminal route assessment checksum mismatch")
        return record

    @classmethod
    def from_bytes(cls, payload: bytes) -> UgiTerminalRouteAssessmentReceipt:
        if not isinstance(payload, bytes) or not payload:
            raise UgiTerminalRouteAssessmentError("assessment bytes must be nonempty")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UgiTerminalRouteAssessmentError("assessment bytes are invalid JSON") from error
        record = cls.from_dict(value)
        if payload != record.canonical_bytes:
            raise UgiTerminalRouteAssessmentError("assessment bytes are not canonical")
        return record


def _validate_locked_terminal(
    terminal: LockedMatchedTerminal,
    context: PlannerCacheContext,
) -> ValidatedUgiTerminalPayload:
    if not isinstance(terminal, LockedMatchedTerminal):
        raise UgiTerminalRouteAssessmentError(
            "route assessment requires a LockedMatchedTerminal wrapper"
        )
    if not terminal.terminal_locked:
        raise UgiTerminalRouteAssessmentError("route assessment requires a sealed terminal")
    if not terminal.terminal_valid:
        raise UgiTerminalRouteAssessmentError("route assessment requires a valid terminal")
    if not terminal.exact_l1:
        raise UgiTerminalRouteAssessmentError("route assessment requires exact L1 reconstruction")
    if not isinstance(terminal.payload, ValidatedUgiTerminalPayload):
        raise UgiTerminalRouteAssessmentError(
            "sealed terminal payload must be ValidatedUgiTerminalPayload"
        )
    payload = terminal.payload
    if terminal.terminal_bytes != payload.canonical_bytes:
        raise UgiTerminalRouteAssessmentError(
            "sealed terminal bytes do not match the canonical validated payload"
        )
    if payload.l1_reaction_sha256 != context.l1_reaction_sha256:
        raise UgiTerminalRouteAssessmentError(
            "terminal and planner contexts use different L1 reaction hashes"
        )
    if payload.identity_policy != context.identity_policy:
        raise UgiTerminalRouteAssessmentError("terminal and planner identity policies differ")
    if payload.stereochemistry_policy != context.stereochemistry_policy:
        raise UgiTerminalRouteAssessmentError(
            "terminal and planner stereochemistry policies differ"
        )
    return payload


def assess_locked_ugi_terminal_routes(
    terminal: LockedMatchedTerminal,
    *,
    l1_reverifier: QualifiedUgiL1Reverifier,
    planner: RoutePlanner,
    planner_context: PlannerCacheContext,
    assessment_context: MatchedAssessmentContext,
    assessment_at_utc: str,
) -> UgiTerminalRouteAssessmentReceipt:
    """Assess one sealed exact-L1 terminal without selection or scalarization.

    The v1 ``RoutePlanner`` contract is deterministic and therefore consumes no
    random seed; ``route_seed`` is retained as outer matched-run provenance.  A
    future stochastic planner must receive an explicit seed and bind it into its
    cache identity.  Likewise, the outer matched runner owns cache-snapshot and
    clone binding because the minimal planner protocol cannot introspect its
    backing cache.
    """

    if not isinstance(planner_context, PlannerCacheContext):
        raise UgiTerminalRouteAssessmentError("planner_context is malformed")
    if not isinstance(assessment_context, MatchedAssessmentContext):
        raise UgiTerminalRouteAssessmentError("assessment_context is malformed")
    if not isinstance(l1_reverifier, QualifiedUgiL1Reverifier):
        raise UgiTerminalRouteAssessmentError("l1_reverifier is malformed")
    _require_current_planner_context(
        planner_context,
        assessment_at_utc=assessment_at_utc,
    )
    payload = _validate_locked_terminal(terminal, planner_context)
    if l1_reverifier.l1_reaction_sha256 != planner_context.l1_reaction_sha256:
        raise UgiTerminalRouteAssessmentError(
            "L1 reverifier and planner context use different reaction hashes"
        )
    if not isinstance(assessment_context.arm, MatchedArm):
        raise UgiTerminalRouteAssessmentError("assessment arm is unsupported")
    if not isinstance(assessment_context.remaining_budget, RouteComputeUsage):
        raise UgiTerminalRouteAssessmentError("remaining route budget is malformed")
    if not isinstance(assessment_context.unit_reservation, RouteComputeUsage):
        raise UgiTerminalRouteAssessmentError("unit route reservation is malformed")
    _require_sha256(
        assessment_context.cache_snapshot_sha256,
        label="cache_snapshot_sha256",
    )
    _require_nonempty(assessment_context.cache_clone_id, label="cache_clone_id")
    _require_nonnegative_integer(assessment_context.route_seed, label="route_seed")
    if assessment_context.arm is MatchedArm.POST_HOC:
        _require_sha256(
            assessment_context.post_hoc_lock_manifest_sha256,
            label="post_hoc_lock_manifest_sha256",
        )
    elif assessment_context.post_hoc_lock_manifest_sha256 is not None:
        raise UgiTerminalRouteAssessmentError(
            "guided assessment cannot carry a post-hoc lock manifest"
        )

    required = required_three_role_route_reservation(planner_context.budget_limits)
    if not required.fits_within(assessment_context.unit_reservation):
        raise UgiTerminalRouteAssessmentError(
            "unit route reservation cannot bound all three isolated component assessments"
        )
    if not required.fits_within(assessment_context.remaining_budget):
        raise UgiTerminalRouteAssessmentError(
            "remaining route budget cannot atomically reserve all three component assessments"
        )

    l1_reverification = l1_reverifier.require_exact(payload)
    l1_reverification_usage = RouteComputeUsage(
        logical_verifier_calls=1,
        physical_verifier_calls=1,
    )

    role_receipts: list[RoleRouteAssessmentReceipt] = []
    components = payload.by_role()
    ledgers: list[PlannerBudgetLedger] = []
    for role in ROLE_NAMES:
        target = RouteTarget(
            role=role,
            canonical_smiles=components[role].canonical_smiles,
            product_context_smiles=(),
        )
        ledger = PlannerBudgetLedger(planner_context.budget_limits)
        ledgers.append(ledger)
        before = PlannerBudgetReceipt.from_ledger(ledger)
        cache_key = PlannerCacheKey.build(target, planner_context, ledger)
        assessment = planner.assess(target, ledger)
        after = PlannerBudgetReceipt.from_ledger(ledger)
        role_receipts.append(
            RoleRouteAssessmentReceipt(
                role=role,
                target=target,
                cache_key_sha256=cache_key.digest,
                cache_key_payload_json=cache_key.payload_json,
                budget_before=before,
                budget_after=after,
                assessment=assessment,
                component_value=component_synthesis_value_from_assessment(assessment),
            )
        )

    realized = l1_reverification_usage.plus(
        _sum_route_usage(tuple(item.realized_route_usage for item in role_receipts))
    )
    if not realized.fits_within(assessment_context.unit_reservation):
        raise UgiTerminalRouteAssessmentError(
            "realized route usage exceeded the frozen unit reservation"
        )
    if not realized.fits_within(assessment_context.remaining_budget):
        raise UgiTerminalRouteAssessmentError(
            "realized route usage exceeded the remaining arm budget"
        )
    component_values = tuple((item.role, item.component_value) for item in role_receipts)
    product_value = ProductSynthesisValue(
        product_smiles=payload.product_smiles,
        l1_forward_consistent=True,
        components=component_values,
    )
    receipt = UgiTerminalRouteAssessmentReceipt(
        unit_id=terminal.unit_id,
        terminal_id=terminal.terminal_id,
        terminal_sha256=terminal.terminal_sha256,
        generation_trace_sha256=terminal.generation_trace_sha256,
        morphology_program_sha256=terminal.morphology_program_sha256,
        checkpoint_index=terminal.checkpoint_index,
        generator_checkpoint_sha256=terminal.generator_checkpoint_sha256,
        closure_checkpoint_sha256=terminal.closure_checkpoint_sha256,
        payload=payload,
        assessment_at_utc=assessment_at_utc,
        arm=assessment_context.arm,
        route_seed=assessment_context.route_seed,
        planner_context=planner_context,
        cache_snapshot_sha256=assessment_context.cache_snapshot_sha256,
        cache_clone_id=assessment_context.cache_clone_id,
        post_hoc_lock_manifest_sha256=(assessment_context.post_hoc_lock_manifest_sha256),
        target_context_policy=TARGET_CONTEXT_POLICY,
        component_budget_policy_sha256=_component_budget_policy_sha256(
            planner_context.budget_limits
        ),
        unit_route_reservation=assessment_context.unit_reservation,
        l1_reverification=l1_reverification,
        l1_reverification_usage=l1_reverification_usage,
        role_assessments=tuple(role_receipts),
        product_value=product_value,
    )
    if not receipt.realized_route_usage.fits_within(required):
        raise UgiTerminalRouteAssessmentError(
            "realized route usage exceeded the derived three-role ceiling"
        )
    return receipt
