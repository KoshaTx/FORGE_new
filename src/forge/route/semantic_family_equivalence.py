"""Audit duplicate registry variants without granting route authority.

Several independently sourced route records can encode the same chemical
transform.  A proposal that reconstructs a target under each of those records
is ambiguous as an *evidence record*, but it need not be ambiguous as a
reaction-family hypothesis.  This module collapses only that semantic
duplication for discovery accounting.  It never projects substrate scope,
creates evidence, closes a route, or defines a synthesis value.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from rdkit.Chem import rdChemReactions

from forge.core.hashing import sha256_json as _sha256_payload
from forge.route.proposal_discovery_status import (
    ProposalDiscoveryResolution,
    ProposalDiscoveryStatus,
)
from forge.route.qualified_forward import QualifiedForwardReaction

SEMANTIC_EQUIVALENCE_SCHEMA_VERSION = "forge.semantic_family_equivalence.v1"
SEMANTIC_EQUIVALENCE_V2_SCHEMA_VERSION = "forge.semantic_family_equivalence.v2"


class SemanticFamilyEquivalenceError(ValueError):
    """Raised when a semantic-equivalence audit input is malformed."""


class FamilyTransform(Protocol):
    reaction: QualifiedForwardReaction

    @property
    def reaction_id(self) -> str: ...


def semantic_transform_key(transform: FamilyTransform) -> str:
    """Return a source-neutral key for one compiled reaction family."""

    if not hasattr(transform, "reaction") or not isinstance(
        transform.reaction, QualifiedForwardReaction
    ):
        raise SemanticFamilyEquivalenceError("family transform is malformed")
    return _sha256_payload(
        {
            "atom_mapped_reaction_smarts": rdChemReactions.ReactionToSmarts(
                transform.reaction.reaction
            ),
            "role_names": list(transform.reaction.role_names),
        }
    )


def semantic_transform_key_v2(transform: FamilyTransform) -> str:
    """Return a chemistry-only key independent of source-specific role labels."""

    if not hasattr(transform, "reaction") or not isinstance(
        transform.reaction, QualifiedForwardReaction
    ):
        raise SemanticFamilyEquivalenceError("family transform is malformed")
    return _sha256_payload(
        {
            "atom_mapped_reaction_smarts": rdChemReactions.ReactionToSmarts(
                transform.reaction.reaction
            ),
            "reactant_template_count": transform.reaction.reaction.GetNumReactantTemplates(),
        }
    )


def audit_semantic_family_equivalence(
    resolution: ProposalDiscoveryResolution,
    transforms: Sequence[FamilyTransform],
) -> dict[str, Any]:
    """Classify whether raw ambiguity is only duplicate family provenance.

    The accepted state is deliberately named ``discovery_only``.  Even when
    all matching registry variants encode one reaction family, independent
    substrate-scope, evidence, operational and L3 checks remain mandatory.
    """

    if not isinstance(resolution, ProposalDiscoveryResolution):
        raise SemanticFamilyEquivalenceError("resolution is malformed")
    by_id = {transform.reaction_id: transform for transform in transforms}
    if len(by_id) != len(tuple(transforms)):
        raise SemanticFamilyEquivalenceError("reaction IDs must be unique")
    matching = tuple(trace for trace in resolution.traces if trace.target_reconstructed)
    groups: dict[tuple[str, tuple[str, ...], tuple[str, ...]], list[str]] = {}
    for trace in matching:
        transform = by_id.get(trace.reaction_id)
        if transform is None:
            raise SemanticFamilyEquivalenceError("resolution trace lacks its transform")
        key = (
            semantic_transform_key(transform),
            trace.role_ordered_reactants,
            trace.products,
        )
        groups.setdefault(key, []).append(trace.reaction_id)

    duplicate_only = (
        resolution.status is ProposalDiscoveryStatus.AMBIGUOUS
        and resolution.rejection_reason == "multiple_known_family_assignments_reconstruct_target"
        and len(matching) > 1
        and len(groups) == 1
    )
    semantic_key = next(iter(groups))[0] if duplicate_only else None
    reaction_ids = sorted(next(iter(groups.values()))) if duplicate_only else []
    return {
        "schema_version": SEMANTIC_EQUIVALENCE_SCHEMA_VERSION,
        "raw_discovery_status": resolution.status.value,
        "raw_rejection_reason": resolution.rejection_reason,
        "matching_trace_count": len(matching),
        "distinct_semantic_assignment_count": len(groups),
        "duplicate_provenance_only": duplicate_only,
        "semantic_resolution_status": (
            "semantically_equivalent_known_family_projection"
            if duplicate_only
            else resolution.status.value
        ),
        "semantic_transform_key": semantic_key,
        "equivalent_reaction_ids": reaction_ids,
        "graph_consistent_discovery_hypothesis": bool(
            resolution.graph_consistent or duplicate_only
        ),
        "evidence_created": False,
        "route_closure_authorized": False,
        "may_enter_synthesis_value": False,
    }


def audit_semantic_family_equivalence_v2(
    resolution: ProposalDiscoveryResolution,
    transforms: Sequence[FamilyTransform],
) -> dict[str, Any]:
    """Collapse duplicate chemistry while retaining all provenance records.

    Version 2 intentionally ignores registry-specific role *names*.  Reactant
    ordering remains encoded in both the atom-mapped reaction SMARTS and the
    role-ordered reactant tuple, so genuinely different assignments remain
    distinct.
    """

    if not isinstance(resolution, ProposalDiscoveryResolution):
        raise SemanticFamilyEquivalenceError("resolution is malformed")
    by_id = {transform.reaction_id: transform for transform in transforms}
    if len(by_id) != len(tuple(transforms)):
        raise SemanticFamilyEquivalenceError("reaction IDs must be unique")
    matching = tuple(trace for trace in resolution.traces if trace.target_reconstructed)
    groups: dict[tuple[str, tuple[str, ...], tuple[str, ...]], list[str]] = {}
    for trace in matching:
        transform = by_id.get(trace.reaction_id)
        if transform is None:
            raise SemanticFamilyEquivalenceError("resolution trace lacks its transform")
        key = (
            semantic_transform_key_v2(transform),
            trace.role_ordered_reactants,
            trace.products,
        )
        groups.setdefault(key, []).append(trace.reaction_id)
    duplicate_only = (
        resolution.status is ProposalDiscoveryStatus.AMBIGUOUS
        and resolution.rejection_reason == "multiple_known_family_assignments_reconstruct_target"
        and len(matching) > 1
        and len(groups) == 1
    )
    semantic_key = next(iter(groups))[0] if duplicate_only else None
    reaction_ids = sorted(next(iter(groups.values()))) if duplicate_only else []
    return {
        "schema_version": SEMANTIC_EQUIVALENCE_V2_SCHEMA_VERSION,
        "raw_discovery_status": resolution.status.value,
        "raw_rejection_reason": resolution.rejection_reason,
        "matching_trace_count": len(matching),
        "distinct_semantic_assignment_count": len(groups),
        "duplicate_provenance_only": duplicate_only,
        "semantic_resolution_status": (
            "semantically_equivalent_known_family_projection"
            if duplicate_only
            else resolution.status.value
        ),
        "semantic_transform_key": semantic_key,
        "equivalent_reaction_ids": reaction_ids,
        "graph_consistent_discovery_hypothesis": bool(
            resolution.graph_consistent or duplicate_only
        ),
        "evidence_created": False,
        "route_closure_authorized": False,
        "may_enter_synthesis_value": False,
    }


__all__ = [
    "SEMANTIC_EQUIVALENCE_SCHEMA_VERSION",
    "SEMANTIC_EQUIVALENCE_V2_SCHEMA_VERSION",
    "SemanticFamilyEquivalenceError",
    "audit_semantic_family_equivalence",
    "audit_semantic_family_equivalence_v2",
    "semantic_transform_key",
    "semantic_transform_key_v2",
]
