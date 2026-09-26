"""Pure identity and non-regression checks for already assessed candidate alternatives.

Callers authenticate source, gate and route receipts before using these functions. These
checks do not establish new chemistry or route evidence, select molecules, or replace the
selector's per-family and pooled diversity floors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

ROUTE_KEYS = ("L2_ready", "combined_primary", "L3_direct_only", "strict_secondary")


class Component(Protocol):
    role: str
    smiles: str


class Candidate(Protocol):
    request: int
    ordinal: int
    exact: bool
    smiles: str
    components: Sequence[Component]


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError("Untyped boolean " + name)
    return value


def _unique_map(rows: Sequence, key: str, value: str, label: str) -> dict:
    result = {}
    for row in rows:
        name = row[key]
        if name in result:
            raise ValueError("Duplicate " + label + ": " + str(name))
        result[name] = row[value]
    return result


def _components(candidate: Candidate) -> dict[str, str]:
    result = {}
    for part in candidate.components:
        if part.role in result:
            raise ValueError("Duplicate candidate component role: " + part.role)
        result[part.role] = part.smiles
    return result


def _contexts(gate: Mapping[str, Any]) -> set[str]:
    return {
        flag["code"] for flag in gate["chemical"]["flags"] if flag["tier"] == "context_required"
    }


def eligibility(
    before: Candidate,
    after: Candidate,
    old_gate: Mapping[str, Any],
    new_gate: Mapping[str, Any],
    old_route: Mapping[str, Any],
    new_route: Mapping[str, Any],
) -> list[str]:
    """Return every regression reason; malformed evidence raises instead of coercing."""
    if before.request != after.request:
        raise ValueError("Alternative belongs to a different request")
    reasons = []
    if _boolean(before.exact, "before.exact") != _boolean(after.exact, "after.exact"):
        reasons.append("changed_exact_status")
    old_design = _boolean(old_gate["qualified_design_pass"], "old design")
    new_design = _boolean(new_gate["qualified_design_pass"], "new design")
    if old_design and not new_design:
        reasons.append("lost_current_design_pass")
    if not _contexts(new_gate) <= _contexts(old_gate):
        reasons.append("new_context_code")
    for key in ROUTE_KEYS:
        old = _boolean(old_route[key], "old route " + key)
        new = _boolean(new_route[key], "new route " + key)
        if old and not new:
            reasons.append("lost_" + key)
    if _components(before).keys() != _components(after).keys():
        reasons.append("changed_role_set")
    return reasons


def bind_route(candidate: Candidate, family: str, request: Mapping, verdict: Mapping) -> bool:
    """Bind an authenticated verdict to its exact request, product and component IDs.

    This is string identity under the upstream canonicalization contract. No molecule
    parsing, normalization, route search or inherited proof is performed here.
    """
    exact = _boolean(candidate.exact, "candidate.exact")
    request_exact = _boolean(request["exact_L1"], "request.exact_L1")
    verdict_exact = _boolean(verdict["exact_L1"], "verdict.exact_L1")
    for key in ROUTE_KEYS:
        _boolean(verdict[key], "route " + key)
    if (
        request["index"] != candidate.request
        or verdict["index"] != candidate.request
        or request["family"] != family
        or verdict["family"] != family
        or (request["canonical_product"] if exact else request["selected_smiles"])
        != candidate.smiles
        or request.get("selected_smiles", candidate.smiles) != candidate.smiles
        or request["selected_ordinal"] != candidate.ordinal
        or request_exact != exact
        or verdict_exact != exact
    ):
        raise ValueError("Route request/product identity mismatch")
    requirements = request["requirements"]
    parts = _components(candidate)
    required = _unique_map(requirements, "role", "identity", "required role")
    declared = _unique_map(requirements, "branch_id", "identity", "required branch")
    observed = _unique_map(verdict["branches"], "branch_id", "identity", "verdict branch")
    if exact:
        if parts != required:
            raise ValueError("Route component identities mismatch")
        if declared != observed:
            raise ValueError("Classifier branches mismatch")
    elif parts or required or declared or observed or any(verdict[k] for k in ROUTE_KEYS):
        raise ValueError("Nonexact request cannot supply a decomposition or positive route proof")
    return True


def require_same_route_context(before: Mapping, after: Mapping) -> None:
    """Preserve source-declared roles, quantities and ordered stage requirements."""

    def signature(request):
        result = {}
        for part in request["requirements"]:
            role = part["role"]
            if role in result:
                raise ValueError("Duplicate required role: " + role)
            quantity = part["quantity"]
            if type(quantity) is not int or quantity < 1:
                raise ValueError("Invalid component quantity for " + role)
            stages = part["stages"]
            if (
                not isinstance(stages, list)
                or not stages
                or not all(isinstance(stage, str) and stage for stage in stages)
            ):
                raise ValueError("Invalid component stages for " + role)
            result[role] = (quantity, tuple(stages))
        return result

    if signature(before) != signature(after):
        raise ValueError("Changed source role, quantity or stage requirements")
