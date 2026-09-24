"""Nonselecting, context-strict dossier aggregation for arbitrary assembly roles.

The caller supplies a qualified L1 assessment and its registered role/occurrence
contract. This boundary does not infer chemistry, translate roles, or turn an
identity lookup into evidence. Each route receipt is bound to one exact assembly
context, and every required branch must close before the product can close.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from forge.assembly.families import constitutional_molecule
from forge.core.hashing import sha256_json
from forge.model.compose_lipid_component_diversity import accepted_components
from forge.synthesis.engine.planner import (
    AssessmentOutcome,
    AvailabilityState,
    EvidenceTier,
    SynthesisAssessment,
    SynthesisRouteNode,
)
from forge.synthesis.value.contracts import component_synthesis_value_from_assessment

SCHEMA_VERSION = "forge.compose_lipid_route_assessment.v1"


def _nonempty(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")


def _sha(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _canonical(smiles: str) -> str:
    return constitutional_molecule(smiles)[0]


def _utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("Evidence dates must be explicit ISO-8601 UTC timestamps")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo != timezone.utc:
        raise ValueError("Evidence dates must use UTC")
    return parsed


@dataclass(frozen=True)
class ComponentRequirement:
    """One complete precursor identity with its declared repeated uses/stages."""

    role: str
    canonical_smiles: str
    quantity: int
    stage_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.role, "component role")
        if self.canonical_smiles != _canonical(self.canonical_smiles):
            raise ValueError("Component SMILES must use canonical constitutional identity")
        if type(self.quantity) is not int or self.quantity < 1:
            raise ValueError("Component quantity must be a positive integer")
        if not isinstance(self.stage_ids, tuple) or not self.stage_ids:
            raise ValueError("Component stage IDs must be an explicit nonempty tuple")
        for stage in self.stage_ids:
            _nonempty(stage, "stage ID")
        if len(set(self.stage_ids)) != len(self.stage_ids):
            raise ValueError("Component stage IDs must be unique")


@dataclass(frozen=True)
class ValidatedComposeAssembly:
    """Receipt for one exact, unique L1 tuple and the full supplied role contract."""

    family: str
    canonical_product: str
    program_sha256: str
    l1_assessment_sha256: str
    requirements: tuple[ComponentRequirement, ...]

    def __post_init__(self) -> None:
        _nonempty(self.family, "family")
        if self.canonical_product != _canonical(self.canonical_product):
            raise ValueError("Product must use canonical constitutional identity")
        _sha(self.program_sha256, "program_sha256")
        _sha(self.l1_assessment_sha256, "l1_assessment_sha256")
        if not isinstance(self.requirements, tuple) or not self.requirements:
            raise ValueError("A product requires every registered component role")
        if any(not isinstance(item, ComponentRequirement) for item in self.requirements):
            raise ValueError("Malformed component requirement")
        roles = [item.role for item in self.requirements]
        if len(roles) != len(set(roles)):
            raise ValueError("Roles must be unique; represent identical repeats with quantity")

    @property
    def context_sha256(self) -> str:
        return sha256_json(asdict(self))

    @classmethod
    def from_l1(
        cls,
        *,
        family: str,
        product_smiles: str,
        program_sha256: str,
        assessment: Mapping[str, Any],
        requirements: tuple[ComponentRequirement, ...],
    ) -> ValidatedComposeAssembly:
        """Bind full-source accepted identities to an explicit program contract.

        ``requirements`` must originate from the qualified program's role and
        occurrence metadata, never from a best-effort subset of recovered roles.
        The hash binds the exact L1 result; the caller owns its provenance and
        registered-program authentication, as for the existing Ugi boundary.
        """
        components = accepted_components(assessment)
        required = {item.role: item.canonical_smiles for item in requirements}
        if components != required:
            raise ValueError("L1 precursor tuple differs from complete required roles/identities")
        product = _canonical(product_smiles)
        witnessed = False
        for check in assessment.get("checks", []):
            for candidate in check.get("candidates", []):
                if candidate.get("pass") is not True or candidate.get("components") != required:
                    continue
                replay = candidate.get("replay", {})
                if replay.get("computed_consistency_pass") is not True:
                    continue
                outputs = replay.get("forward_products", [])
                layers = replay.get("forward_layers", [])
                if layers:
                    outputs = layers[-1]
                if product in outputs:
                    quantities = replay.get("declared_quantities")
                    if quantities is not None and quantities != {
                        item.role: item.quantity for item in requirements
                    }:
                        raise ValueError("Repeated quantities differ from the L1 replay")
                    witnessed = True
        if not witnessed:
            raise ValueError("L1 assessment lacks an accepted exact-product replay witness")
        return cls(family, product, program_sha256, sha256_json(assessment), requirements)


@dataclass(frozen=True)
class ScopedRouteReceipt:
    """Assessment already adjudicated for the exact assembly role/context.

    A shared identity is only a retrieval candidate. Reuse requires an explicit
    context qualification receipt; the inherited route tree alone is insufficient.
    The L3 interval is start-inclusive and end-exclusive, and binds every leaf.
    """

    assembly_context_sha256: str
    context_qualification_sha256: str
    requirement: ComponentRequirement
    assessment: SynthesisAssessment
    l3_snapshot_sha256: str
    l3_region: str
    l3_accessed_at_utc: str
    l3_expires_at_utc: str

    def __post_init__(self) -> None:
        for label in (
            "assembly_context_sha256",
            "context_qualification_sha256",
            "l3_snapshot_sha256",
        ):
            _sha(getattr(self, label), label)
        _nonempty(self.l3_region, "L3 region")
        if _utc(self.l3_expires_at_utc) <= _utc(self.l3_accessed_at_utc):
            raise ValueError("L3 evidence interval must be nonempty")
        if not isinstance(self.assessment, SynthesisAssessment):
            raise ValueError("Route receipt needs a typed synthesis assessment")
        target = self.assessment.target
        if (target.role, _canonical(target.canonical_smiles)) != (
            self.requirement.role,
            self.requirement.canonical_smiles,
        ):
            raise ValueError("Route assessment role/identity differs from its requirement")


def _check_tree(node: SynthesisRouteNode, ancestors: frozenset = frozenset()) -> None:
    """Reject inconsistent serialized complete claims before aggregation."""
    if node.target.identity in ancestors:
        raise ValueError("Route tree contains a cycle")
    if node.step is not None:
        if tuple(child.target for child in node.children) != node.step.reactants:
            raise ValueError("Route children differ from declared reactant targets")
        if node.outcome is AssessmentOutcome.COMPLETE and (
            not node.step.is_exact_source_forward_verified
            or any(child.outcome is not AssessmentOutcome.COMPLETE for child in node.children)
        ):
            raise ValueError("Complete branch lacks exact steps or complete children")
    elif node.outcome is AssessmentOutcome.COMPLETE and not any(
        record.tier is EvidenceTier.ACCEPTED_TERMINAL
        and record.exact_substrate
        and record.availability is AvailabilityState.CURRENT_CLOSED
        for record in node.evidence
    ):
        raise ValueError("Complete leaf lacks exact current terminal evidence")
    for child in node.children:
        _check_tree(child, ancestors | {node.target.identity})


def aggregate_compose_routes(
    assembly: ValidatedComposeAssembly,
    receipts: tuple[ScopedRouteReceipt, ...],
    *,
    assessment_at_utc: str,
    region: str,
) -> dict[str, Any]:
    """Retain missing/unassessed/censored branches; close only all required roles."""
    as_of = _utc(assessment_at_utc)
    _nonempty(region, "assessment region")
    required = {item.role: item for item in assembly.requirements}
    by_role = {item.requirement.role: item for item in receipts}
    if len(by_role) != len(receipts) or set(by_role) - set(required):
        raise ValueError("Duplicate or unexpected route-receipt role")
    branches = []
    for role, requirement in required.items():
        receipt = by_role.get(role)
        item: dict[str, Any] = {**asdict(requirement), "route_complete": False}
        if receipt is None:
            item.update(status="not_assessed", assessment=None)
        else:
            if receipt.assembly_context_sha256 != assembly.context_sha256:
                raise ValueError("Route receipt does not match exact assembly context")
            if receipt.requirement != requirement:
                raise ValueError("Role, identity, quantity, or stage context mismatch")
            if receipt.l3_region != region:
                raise ValueError("L3 region differs from the requested assessment region")
            if as_of < _utc(receipt.l3_accessed_at_utc):
                raise ValueError("Assessment precedes the evidence snapshot")
            _check_tree(receipt.assessment.route_tree)
            value = component_synthesis_value_from_assessment(receipt.assessment)
            expired = as_of >= _utc(receipt.l3_expires_at_utc)
            item.update(
                status=("expired_l3_snapshot" if expired else receipt.assessment.outcome.value),
                route_complete=value.route_complete and not expired,
                assessment=receipt.assessment.to_dict(),
                assessment_sha256=sha256_json(receipt.assessment.to_dict()),
                context_qualification_sha256=receipt.context_qualification_sha256,
                l3_snapshot_sha256=receipt.l3_snapshot_sha256,
                l3_accessed_at_utc=receipt.l3_accessed_at_utc,
                l3_expires_at_utc=receipt.l3_expires_at_utc,
            )
        branches.append(item)
    complete = all(item["route_complete"] for item in branches)
    return {
        "schema_version": SCHEMA_VERSION,
        "assembly": asdict(assembly),
        "assembly_context_sha256": assembly.context_sha256,
        "assessment_at_utc": assessment_at_utc,
        "region": region,
        "branches": branches,
        "all_branches_assessed": len(receipts) == len(required),
        "complete_exact_source_dossier": complete,
        "status": "complete" if complete else "incomplete_or_not_assessed",
        "success_probability": None,
        "biological_claim": None,
    }
