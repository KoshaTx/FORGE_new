"""One-sided exact-dossier potential for terminal-only Ugi guidance.

The structured :class:`ProductSynthesisValue` remains authoritative.  This
module supplies only the minimal bounded potential needed by an SMC controller:
one when an exact-L1 product has a strict, current, exact route dossier for all
three Ugi components, and zero for a substantively assessed but noncomplete
terminal.  Computationally censored assessments carry no potential.

The potential is not a synthesis-success probability.  It deliberately ignores
partial closure, route depth, route count, evidence volume and unknown burden.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from forge.bio.ugi_semantic_annotations import ROLE_NAMES
from forge.route.planner import AssessmentOutcome
from forge.value.synthesis import (
    ComponentSynthesisValue,
    EvidenceSupport,
    ForwardConsistency,
    ProductSynthesisValue,
)

UGI_EXACT_CLOSURE_GUIDANCE_SCHEMA_VERSION = "forge.ugi_exact_closure_guidance.v1"


class UgiExactClosureGuidanceError(ValueError):
    """Raised when a synthesis record cannot satisfy the frozen policy."""


class GuidanceDisposition(str, Enum):
    """How one assessment contributes to the SMC potential."""

    SUPPORT_BONUS = "support_bonus"
    NEUTRAL = "neutral"
    CENSOR = "censor"


class GuidanceReason(str, Enum):
    """Nonexclusive chemistry or execution state retained beside the potential."""

    STRICT_EXACT_ROUTE_COMPLETE = "strict_exact_route_complete"
    NONSTRICT_COMPLETE_EVIDENCE = "nonstrict_complete_evidence"
    MISSING_ROUTE_KNOWLEDGE = "missing_route_knowledge"
    FAMILY_OR_PROVENANCE_ONLY = "family_or_provenance_only"
    CHEMICALLY_INCOMPATIBLE = "chemically_incompatible"
    UNAVAILABLE_TERMINAL = "unavailable_terminal"
    OUTSIDE_DECLARED_SUPPORT = "outside_declared_support"
    UNASSESSED_TERMINAL_AVAILABILITY = "unassessed_terminal_availability"
    FAILED_FORWARD_VERIFICATION = "failed_forward_verification"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INVALID_INPUT = "invalid_input"
    EXECUTION_ERROR = "execution_error"
    NONEXACT_L1 = "nonexact_l1"


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


_POLICY_CONTENT = {
    "schema_version": UGI_EXACT_CLOSURE_GUIDANCE_SCHEMA_VERSION,
    "role_order": list(ROLE_NAMES),
    "potential": {
        "strict_exact_current_complete_dossier": 1,
        "substantively_assessed_noncomplete_dossier": 0,
        "computationally_censored_assessment": None,
    },
    "strict_product_rule": "exact L1 and strict closure for every frozen Ugi role",
    "neutral_nonclaims": [
        "missing route knowledge is not evidence of unsynthesizability",
        "outside support is not evidence of unsynthesizability",
        "unavailable or expired materials are not intrinsic chemical impossibility",
        "evidenced incompatibility receives no negative numeric penalty in v1",
    ],
    "excluded_from_potential": [
        "partial route fraction",
        "complete component count",
        "route multiplicity",
        "route step count",
        "maximum route depth",
        "family projected evidence",
        "provenance only evidence",
        "protection burden",
        "purification burden",
        "route model likelihood",
        "planner cache behavior",
    ],
    "interpretation": "binary_route_completion_utility_not_success_probability",
}
UGI_EXACT_CLOSURE_GUIDANCE_POLICY_SHA256 = _sha256_payload(_POLICY_CONTENT)


@dataclass(frozen=True)
class RoleExactClosureDecision:
    """One Ugi role's strict closure state and nonscalar reason."""

    role: str
    disposition: GuidanceDisposition
    reason: GuidanceReason
    strict_complete: bool
    assessment_outcome: AssessmentOutcome

    def __post_init__(self) -> None:
        if self.role not in ROLE_NAMES:
            raise UgiExactClosureGuidanceError(f"unsupported Ugi role: {self.role!r}")
        if not isinstance(self.disposition, GuidanceDisposition):
            raise UgiExactClosureGuidanceError("role disposition is unsupported")
        if not isinstance(self.reason, GuidanceReason):
            raise UgiExactClosureGuidanceError("role reason is unsupported")
        if not isinstance(self.strict_complete, bool):
            raise UgiExactClosureGuidanceError("strict_complete must be boolean")
        if not isinstance(self.assessment_outcome, AssessmentOutcome):
            raise UgiExactClosureGuidanceError("role assessment outcome is unsupported")
        if self.strict_complete != (
            self.disposition is GuidanceDisposition.SUPPORT_BONUS
            and self.reason is GuidanceReason.STRICT_EXACT_ROUTE_COMPLETE
            and self.assessment_outcome is AssessmentOutcome.COMPLETE
        ):
            raise UgiExactClosureGuidanceError(
                "strict completion requires the exact support-bonus state"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "disposition": self.disposition.value,
            "reason": self.reason.value,
            "strict_complete": self.strict_complete,
            "assessment_outcome": self.assessment_outcome.value,
        }


@dataclass(frozen=True)
class ProductExactClosurePotential:
    """Hash-bound binary SMC potential derived from one structured product value."""

    product_value_sha256: str
    disposition: GuidanceDisposition
    roles: tuple[RoleExactClosureDecision, ...]
    strict_complete_role_count: int
    smc_potential: float | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.product_value_sha256, str)
            or len(self.product_value_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.product_value_sha256)
        ):
            raise UgiExactClosureGuidanceError("product value SHA-256 is malformed")
        if not isinstance(self.disposition, GuidanceDisposition):
            raise UgiExactClosureGuidanceError("product disposition is unsupported")
        if tuple(value.role for value in self.roles) != ROLE_NAMES:
            raise UgiExactClosureGuidanceError(
                "product potential requires the three frozen Ugi roles in order"
            )
        observed_count = sum(value.strict_complete for value in self.roles)
        if self.strict_complete_role_count != observed_count:
            raise UgiExactClosureGuidanceError(
                "product strict-complete count disagrees with role decisions"
            )
        if self.disposition is GuidanceDisposition.CENSOR:
            if self.smc_potential is not None:
                raise UgiExactClosureGuidanceError("censored product cannot carry a potential")
            if not any(value.disposition is GuidanceDisposition.CENSOR for value in self.roles):
                raise UgiExactClosureGuidanceError(
                    "product censoring requires at least one censored role"
                )
            return
        if (
            isinstance(self.smc_potential, bool)
            or not isinstance(self.smc_potential, (int, float))
            or not math.isfinite(float(self.smc_potential))
        ):
            raise UgiExactClosureGuidanceError("noncensored product requires a finite potential")
        expected = 1.0 if observed_count == len(ROLE_NAMES) else 0.0
        if float(self.smc_potential) != expected:
            raise UgiExactClosureGuidanceError(
                "SMC potential disagrees with the one-sided exact-dossier policy"
            )
        expected_disposition = (
            GuidanceDisposition.SUPPORT_BONUS if expected == 1 else GuidanceDisposition.NEUTRAL
        )
        if self.disposition is not expected_disposition:
            raise UgiExactClosureGuidanceError(
                "product disposition disagrees with the exact-dossier potential"
            )

    @property
    def strict_route_complete(self) -> bool:
        return self.strict_complete_role_count == len(ROLE_NAMES)

    @property
    def reasons(self) -> tuple[GuidanceReason, ...]:
        return tuple(sorted({value.reason for value in self.roles}, key=lambda item: item.value))

    def require_potential(self) -> float:
        if self.smc_potential is None:
            raise UgiExactClosureGuidanceError(
                "censored synthesis assessment cannot be supplied to SMC"
            )
        return float(self.smc_potential)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": UGI_EXACT_CLOSURE_GUIDANCE_SCHEMA_VERSION,
            "policy_sha256": UGI_EXACT_CLOSURE_GUIDANCE_POLICY_SHA256,
            "product_value_sha256": self.product_value_sha256,
            "disposition": self.disposition.value,
            "roles": [value.to_dict() for value in self.roles],
            "strict_complete_role_count": self.strict_complete_role_count,
            "role_count": len(self.roles),
            "strict_route_complete": self.strict_route_complete,
            "smc_potential": self.smc_potential,
            "route_completion_utility": self.smc_potential,
            "route_completion_utility_not_success_probability": True,
            "success_probability": None,
            "reasons": [reason.value for reason in self.reasons],
        }


@dataclass(frozen=True)
class SMCExactClosureUtilityBridge:
    """Controller input that never imputes a numeric value for censoring."""

    product_value_sha256: str
    route_completion_utility: float | None
    incremental_log_weight: float
    censored: bool
    support_bonus: bool

    def __post_init__(self) -> None:
        if not isinstance(self.product_value_sha256, str) or len(self.product_value_sha256) != 64:
            raise UgiExactClosureGuidanceError("bridge product-value SHA-256 is malformed")
        if self.incremental_log_weight not in {0.0, 1.0}:
            raise UgiExactClosureGuidanceError(
                "binary bridge incremental log-weight must be zero or one"
            )
        if self.censored:
            if self.route_completion_utility is not None:
                raise UgiExactClosureGuidanceError(
                    "censored bridge cannot impute a numeric route-completion utility"
                )
            if self.incremental_log_weight != 0.0 or self.support_bonus:
                raise UgiExactClosureGuidanceError(
                    "censored bridge must be neutral and cannot receive a support bonus"
                )
            return
        if self.route_completion_utility not in {0.0, 1.0}:
            raise UgiExactClosureGuidanceError(
                "noncensored bridge requires a binary route-completion utility"
            )
        if self.incremental_log_weight != self.route_completion_utility:
            raise UgiExactClosureGuidanceError(
                "bridge log-weight disagrees with route-completion utility"
            )
        if self.support_bonus != (self.route_completion_utility == 1.0):
            raise UgiExactClosureGuidanceError(
                "bridge support-bonus flag disagrees with route-completion utility"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_value_sha256": self.product_value_sha256,
            "route_completion_utility": self.route_completion_utility,
            "incremental_log_weight": self.incremental_log_weight,
            "incremental_log_weight_interpretation": (
                "controller_neutral_for_censoring_not_imputed_route_value"
                if self.censored
                else "binary_route_completion_utility"
            ),
            "censored": self.censored,
            "support_bonus": self.support_bonus,
            "success_probability": None,
            "evidence_upgraded": False,
        }


def _is_strict_component_complete(value: ComponentSynthesisValue) -> bool:
    blocker_counts = (
        value.unavailable_terminal_leaf_count,
        value.unassessed_terminal_leaf_count,
        value.missing_knowledge_leaf_count,
        value.outside_support_leaf_count,
        value.incompatible_leaf_count,
        value.budget_exhausted_leaf_count,
        value.invalid_input_leaf_count,
        value.execution_error_leaf_count,
    )
    return (
        value.route_complete
        and value.all_terminal_leaves_current
        and value.forward_consistency
        in {ForwardConsistency.EXACT_UNIQUE, ForwardConsistency.NOT_APPLICABLE}
        and value.evidence_support
        in {EvidenceSupport.EXACT_IDENTITY, EvidenceSupport.NOT_APPLICABLE}
        and not any(blocker_counts)
    )


def _role_decision(role: str, value: ComponentSynthesisValue) -> RoleExactClosureDecision:
    if _is_strict_component_complete(value):
        disposition = GuidanceDisposition.SUPPORT_BONUS
        reason = GuidanceReason.STRICT_EXACT_ROUTE_COMPLETE
        strict_complete = True
    elif value.assessment_outcome is AssessmentOutcome.EXECUTION_ERROR:
        disposition, reason = GuidanceDisposition.CENSOR, GuidanceReason.EXECUTION_ERROR
        strict_complete = False
    elif value.assessment_outcome is AssessmentOutcome.INVALID_INPUT:
        disposition, reason = GuidanceDisposition.CENSOR, GuidanceReason.INVALID_INPUT
        strict_complete = False
    elif value.assessment_outcome is AssessmentOutcome.BUDGET_EXHAUSTED:
        disposition, reason = GuidanceDisposition.CENSOR, GuidanceReason.BUDGET_EXHAUSTED
        strict_complete = False
    else:
        disposition = GuidanceDisposition.NEUTRAL
        strict_complete = False
        if value.route_complete:
            reason = GuidanceReason.NONSTRICT_COMPLETE_EVIDENCE
        elif value.unavailable_terminal_leaf_count:
            reason = GuidanceReason.UNAVAILABLE_TERMINAL
        elif value.unassessed_terminal_leaf_count:
            reason = GuidanceReason.UNASSESSED_TERMINAL_AVAILABILITY
        elif value.forward_consistency is ForwardConsistency.FAILED:
            reason = GuidanceReason.FAILED_FORWARD_VERIFICATION
        elif value.assessment_outcome is AssessmentOutcome.INCOMPATIBLE:
            reason = GuidanceReason.CHEMICALLY_INCOMPATIBLE
        elif value.assessment_outcome is AssessmentOutcome.OUTSIDE_SUPPORT:
            reason = GuidanceReason.OUTSIDE_DECLARED_SUPPORT
        elif value.evidence_support in {
            EvidenceSupport.FAMILY_PROJECTED,
            EvidenceSupport.PROVENANCE_ONLY,
        }:
            reason = GuidanceReason.FAMILY_OR_PROVENANCE_ONLY
        elif value.assessment_outcome is AssessmentOutcome.MISSING_KNOWLEDGE:
            reason = GuidanceReason.MISSING_ROUTE_KNOWLEDGE
        else:  # pragma: no cover - enum exhaustiveness guard
            raise UgiExactClosureGuidanceError(
                f"component {role} has no frozen guidance interpretation"
            )
    return RoleExactClosureDecision(
        role=role,
        disposition=disposition,
        reason=reason,
        strict_complete=strict_complete,
        assessment_outcome=value.assessment_outcome,
    )


def exact_closure_potential_from_product_value(
    value: ProductSynthesisValue,
) -> ProductExactClosurePotential:
    """Derive the frozen binary potential without altering the route receipt."""

    if not isinstance(value, ProductSynthesisValue):
        raise UgiExactClosureGuidanceError(
            "exact closure potential requires a ProductSynthesisValue"
        )
    product_value_sha256 = _sha256_payload(value.to_dict())
    by_role = value.by_role()
    if set(by_role) != set(ROLE_NAMES):
        raise UgiExactClosureGuidanceError(
            "exact closure potential requires exactly the three frozen Ugi roles"
        )
    if not value.l1_forward_consistent:
        roles = tuple(
            RoleExactClosureDecision(
                role=role,
                disposition=GuidanceDisposition.CENSOR,
                reason=GuidanceReason.NONEXACT_L1,
                strict_complete=False,
                assessment_outcome=component.assessment_outcome,
            )
            for role in ROLE_NAMES
            for component in (by_role[role],)
        )
    else:
        roles = tuple(_role_decision(role, by_role[role]) for role in ROLE_NAMES)
    strict_count = sum(item.strict_complete for item in roles)
    if any(item.disposition is GuidanceDisposition.CENSOR for item in roles):
        disposition = GuidanceDisposition.CENSOR
        potential = None
    elif strict_count == len(ROLE_NAMES):
        disposition = GuidanceDisposition.SUPPORT_BONUS
        potential = 1.0
    else:
        disposition = GuidanceDisposition.NEUTRAL
        potential = 0.0
    return ProductExactClosurePotential(
        product_value_sha256=product_value_sha256,
        disposition=disposition,
        roles=roles,
        strict_complete_role_count=strict_count,
        smc_potential=potential,
    )


def exact_closure_guidance_policy_manifest() -> dict[str, Any]:
    """Return the canonical policy definition and its content identity."""

    return {**_POLICY_CONTENT, "policy_sha256": UGI_EXACT_CLOSURE_GUIDANCE_POLICY_SHA256}


def smc_utility_bridge_from_exact_closure(
    potential: ProductExactClosurePotential,
) -> SMCExactClosureUtilityBridge:
    """Map censoring to a neutral controller increment without value imputation."""

    if not isinstance(potential, ProductExactClosurePotential):
        raise UgiExactClosureGuidanceError(
            "SMC utility bridge requires a ProductExactClosurePotential"
        )
    censored = potential.disposition is GuidanceDisposition.CENSOR
    return SMCExactClosureUtilityBridge(
        product_value_sha256=potential.product_value_sha256,
        route_completion_utility=(None if censored else potential.require_potential()),
        incremental_log_weight=(0.0 if censored else potential.require_potential()),
        censored=censored,
        support_bonus=potential.disposition is GuidanceDisposition.SUPPORT_BONUS,
    )


__all__ = [
    "GuidanceDisposition",
    "GuidanceReason",
    "ProductExactClosurePotential",
    "RoleExactClosureDecision",
    "SMCExactClosureUtilityBridge",
    "UGI_EXACT_CLOSURE_GUIDANCE_POLICY_SHA256",
    "UGI_EXACT_CLOSURE_GUIDANCE_SCHEMA_VERSION",
    "UgiExactClosureGuidanceError",
    "exact_closure_guidance_policy_manifest",
    "exact_closure_potential_from_product_value",
    "smc_utility_bridge_from_exact_closure",
]
