"""Source-declared COMPOSE programs; metadata is not an executable reaction certificate."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ComposeLipidError(ValueError):
    """The imported release or requested use violates its pinned contract."""


def program_catalogue(
    key: Mapping[str, Any], bindings: Mapping[str, Any], registry_ids: set[str]
) -> dict[str, Any]:
    """Keep every source program distinct, even when an existing transform may be reusable."""
    if not key or set(key) != set(bindings):
        raise ComposeLipidError("program bindings must cover the complete decomposition key")
    result = {}
    for family, source in sorted(key.items()):
        roles = source.get("roles")
        if not isinstance(roles, dict) or not roles:
            raise ComposeLipidError(f"{family}: missing source roles")
        if any(type(count) is not int or count < 1 for count in roles.values()):
            raise ComposeLipidError(f"{family}: invalid nominal role count")
        mapping = source.get("field_to_role", {})
        if any(not targets or set(targets) - set(roles) for targets in mapping.values()):
            raise ComposeLipidError(f"{family}: unknown role in source field mapping")
        binding = bindings[family]
        positional = binding.get("precursor_id_order", [])
        if positional and (len(positional) != len(roles) or set(positional) != set(roles)):
            raise ComposeLipidError(f"{family}: positional roles differ from source roles")
        related = binding.get("related_registry_reactions", [])
        if not isinstance(related, list) or set(related) - registry_ids:
            raise ComposeLipidError(f"{family}: unknown related registry reaction")
        result[family] = {
            "program_id": f"compose_lipid_v8:{family}",
            "source_definition": source,
            "precursor_id_order": positional,
            "related_registry_reactions": related,
            "adapter_status": "requires_exact_program_qualification",
            "relation_is_equivalence": False,
            "component_structures_supplied": False,
        }
    return result


def role_metadata(row: Mapping[str, Any], program: Mapping[str, Any]) -> dict[str, Any]:
    """Read opaque labels, retaining missing anchor metadata and unresolved multiplicity."""
    source = program["source_definition"]
    metadata = row["primary_metadata"]
    roles = {
        role: {"source_nominal_count": count, "metadata": {}}
        for role, count in source["roles"].items()
    }
    for field, targets in source.get("field_to_role", {}).items():
        if field in metadata:
            for role in targets:
                roles[role]["metadata"][field] = metadata[field]
    order = program["precursor_id_order"]
    if "precursor_ids" in metadata and order:
        values = metadata["precursor_ids"]
        if not isinstance(values, list) or len(values) != len(order):
            raise ComposeLipidError(f"{row['target_id']}: precursor_ids arity mismatch")
        if any(not isinstance(value, str) or not value for value in values):
            raise ComposeLipidError(f"{row['target_id']}: invalid precursor identifier")
        for role, value in zip(order, values, strict=True):
            roles[role]["metadata"]["precursor_id"] = value
    variables = {field: metadata.get(field) for field in source.get("variable", [])}
    return {
        "roles": roles,
        "variable_multiplicity_metadata": variables,
        "missing_role_metadata": sorted(
            role for role, value in roles.items() if not value["metadata"]
        ),
        "missing_variable_metadata": sorted(
            field for field, value in variables.items() if value is None
        ),
        "counts_are_source_nominal_not_validated_occurrences": True,
        "exact_atom_partition_verified": False,
    }
