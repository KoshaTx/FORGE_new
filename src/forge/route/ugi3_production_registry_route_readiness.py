"""Audit route readiness across the full Phase 1 Ugi component registry.

The audit composes frozen evidence; it does not plan routes.  Structural L1
admission, reaction-handle qualification, source provenance and motif
similarity can never promote a component into a route-complete tier.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.core.hashing import sha256_bytes, sha256_file
from forge.core.io import csv_gz_bytes as _csv_bytes
from forge.core.io import read_json_object
from forge.route.qualified_forward import (
    QualifiedForwardError,
    load_qualified_forward_reaction,
    unique_forward_products,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi3_production_registry_route_readiness_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_production_registry_route_readiness.v1"

ACCEPTED_TERMINAL = "accepted_terminal"
EXACT_CLOSED = "exact_source_forward_verified_l3_closed"
EXACT_OPEN = "exact_source_l3_open"
FAMILY_ONLY = "family_projected_only"
PROVENANCE_ONLY = "handle_qualified_provenance_only"
OUTSIDE_SUPPORT = "outside_route_support"
MISSING_KNOWLEDGE = "missing_route_knowledge"
CATEGORIES = (
    ACCEPTED_TERMINAL,
    EXACT_CLOSED,
    EXACT_OPEN,
    FAMILY_ONLY,
    PROVENANCE_ONLY,
    OUTSIDE_SUPPORT,
    MISSING_KNOWLEDGE,
)
ROUTE_COMPLETE_CATEGORIES = {ACCEPTED_TERMINAL, EXACT_CLOSED}

OLD_TIER_TO_CATEGORY = {
    "accepted_terminal_l3_closed": ACCEPTED_TERMINAL,
    "exact_source_l2_verified_l3_closed": EXACT_CLOSED,
    "exact_source_l2_verified_l3_open": EXACT_OPEN,
    "family_projected_l3_closed_not_exact_source": FAMILY_ONLY,
    "unresolved_component": OUTSIDE_SUPPORT,
}
ALLOWED_ROUTE_STATES = {
    "bounded_family_applicability",
    "exact_source_route",
    "no_upstream_route_reported",
    "not_assessed",
    "reaction_family_precedent",
}
FAMILY_STATES = {"bounded_family_applicability", "reaction_family_precedent"}

COMPONENT_FIELDS = (
    "component_id",
    "role",
    "canonical_smiles",
    "is_current_catalog",
    "constitutional_match_to_original_93",
    "original_component_id",
    "evidence_category",
    "evidence_basis",
    "l2_forward_status",
    "l3_status",
    "route_complete_component",
    "route_evidence_states_json",
    "procurement_evidence_states_json",
    "route_closure_states_json",
    "e2_route_closed_input_flag",
    "source_classes_json",
    "source_record_ids_json",
    "gap_class",
)
GAP_FIELDS = (
    "priority_rank",
    "role",
    "gap_class",
    "evidence_category",
    "component_count",
    "recurring",
    "material_relevance",
    "required_evidence",
)


class ProductionRegistryRouteReadinessError(ValueError):
    """Raised when a frozen route-readiness input violates its contract."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=ProductionRegistryRouteReadinessError, label=label)


def _read_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ProductionRegistryRouteReadinessError(f"{label} has no header")
            return list(reader)
    except (OSError, csv.Error) as exc:
        raise ProductionRegistryRouteReadinessError(f"could not read {label}: {path}") from exc


def _verify_hash(path: Path, expected: Any, *, label: str) -> None:
    observed = sha256_file(path)
    if not isinstance(expected, str) or len(expected) != 64 or observed != expected:
        raise ProductionRegistryRouteReadinessError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )


def _json_string_list(value: str, *, label: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ProductionRegistryRouteReadinessError(f"{label} is invalid JSON") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ProductionRegistryRouteReadinessError(f"{label} must be a list of strings")
    return parsed


def _canonical_constitution(smiles: Any, *, label: str) -> str:
    if not isinstance(smiles, str) or not smiles:
        raise ProductionRegistryRouteReadinessError(f"{label} must be non-empty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ProductionRegistryRouteReadinessError(f"{label} contains invalid SMILES")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _portable(path: Path, *, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _validate_expected(observed: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise ProductionRegistryRouteReadinessError(f"{label} fields mismatch")
        for key, value in expected.items():
            _validate_expected(observed[key], value, label=f"{label}.{key}")
    elif isinstance(expected, list):
        if observed != expected:
            raise ProductionRegistryRouteReadinessError(f"{label} list mismatch")
    elif observed != expected:
        raise ProductionRegistryRouteReadinessError(
            f"{label} mismatch: expected {expected!r}, observed {observed!r}"
        )


def classify_component_evidence(
    registry_row: Mapping[str, str],
    *,
    original_component: Mapping[str, str] | None,
    exact_transfer_forward_verified: bool,
) -> dict[str, str]:
    """Assign the strongest explicit evidence tier without chemical inference."""

    component_id = registry_row.get("component_id", "")
    route_states = set(
        _json_string_list(
            registry_row.get("route_evidence_states_json", ""),
            label=f"{component_id} route evidence",
        )
    )
    unknown = route_states - ALLOWED_ROUTE_STATES
    if unknown:
        raise ProductionRegistryRouteReadinessError(
            f"{component_id} has unknown route evidence states {sorted(unknown)}"
        )
    e2_flag = registry_row.get("e2_route_closed") == "true"

    if original_component is not None:
        tier = original_component.get("evidence_tier", "")
        try:
            category = OLD_TIER_TO_CATEGORY[tier]
        except KeyError as exc:
            raise ProductionRegistryRouteReadinessError(
                f"{component_id} has unsupported original evidence tier {tier!r}"
            ) from exc
        basis = f"original_93_component_dossier:{tier}"
        l2_status = original_component.get("l2_forward_status", "")
        l3_status = original_component.get("l3_terminal_status", "")
    elif e2_flag:
        if not exact_transfer_forward_verified:
            raise ProductionRegistryRouteReadinessError(
                f"{component_id} has an E2 flag without independent exact forward verification"
            )
        if route_states != {"exact_source_route"}:
            raise ProductionRegistryRouteReadinessError(
                f"{component_id} E2 route-evidence contract changed"
            )
        procurement_states = set(
            _json_string_list(
                registry_row.get("procurement_evidence_states_json", ""),
                label=f"{component_id} procurement evidence",
            )
        )
        closure_states = set(
            _json_string_list(
                registry_row.get("route_closure_states_json", ""),
                label=f"{component_id} route closure",
            )
        )
        if procurement_states != {"current_item_level_procurement_closed"} or closure_states != {
            "computationally_complete"
        }:
            raise ProductionRegistryRouteReadinessError(
                f"{component_id} E2 L3-closure contract changed"
            )
        category = EXACT_CLOSED
        basis = "independently_reexecuted_exact_source_transfer_and_frozen_l3_closure"
        l2_status = "exact_source_product_uniquely_forward_verified"
        l3_status = "closed"
    elif "exact_source_route" in route_states:
        category = EXACT_OPEN
        basis = "exact_source_route_record_without_complete_reexecuted_l2_l3_dossier"
        l2_status = "not_forward_verified_by_this_audit"
        l3_status = "open"
    elif route_states & FAMILY_STATES:
        category = FAMILY_ONLY
        basis = "reaction_family_or_bounded_family_evidence_only"
        l2_status = "family_projection_not_exact_source"
        l3_status = "not_route_complete"
    elif "no_upstream_route_reported" in route_states:
        category = OUTSIDE_SUPPORT
        basis = "explicit_no_upstream_route_reported"
        l2_status = "no_assigned_upstream_program"
        l3_status = "not_route_complete"
    elif "not_assessed" in route_states:
        category = MISSING_KNOWLEDGE
        basis = "route_evidence_not_assessed"
        l2_status = "not_assessed"
        l3_status = "unknown"
    elif not route_states:
        category = PROVENANCE_ONLY
        basis = "structural_handle_and_provenance_only_no_route_evidence"
        l2_status = "no_route_evidence_attached"
        l3_status = "unknown"
    else:  # pragma: no cover - set algebra guard
        raise ProductionRegistryRouteReadinessError(f"{component_id} is unclassifiable")

    if category == ACCEPTED_TERMINAL:
        gap = "none"
    elif category == EXACT_CLOSED:
        gap = "none"
    elif category == EXACT_OPEN:
        gap = "l3_terminal_closure_required"
    elif category == FAMILY_ONLY:
        gap = "exact_substrate_route_evidence_and_forward_verification_required"
    elif category == OUTSIDE_SUPPORT:
        gap = "upstream_route_not_reported"
    elif category == MISSING_KNOWLEDGE:
        gap = "route_evidence_not_assessed"
    else:
        gap = "no_route_evidence_attached"

    return {
        "evidence_category": category,
        "evidence_basis": basis,
        "l2_forward_status": l2_status,
        "l3_status": l3_status,
        "route_complete_component": str(category in ROUTE_COMPLETE_CATEGORIES).lower(),
        "gap_class": gap,
    }


def _verify_exact_transfer(
    config: Mapping[str, Any],
    transfer_rows: Sequence[Mapping[str, str]],
    registry_rows: Sequence[Mapping[str, str]],
    executable_registry_path: Path,
    variant_path: Path,
) -> tuple[str, dict[str, Any]]:
    policy = config.get("exact_transfer_verification")
    if not isinstance(policy, dict):
        raise ProductionRegistryRouteReadinessError("exact transfer verification is missing")
    record_id = policy.get("source_record_id")
    matches = [row for row in transfer_rows if row.get("record_id") == record_id]
    if len(matches) != 1:
        raise ProductionRegistryRouteReadinessError(
            f"exact transfer source {record_id!r} must resolve exactly once"
        )
    source = matches[0]
    required_fields = policy.get("required_source_fields")
    if not isinstance(required_fields, dict) or any(
        source.get(field) != value for field, value in required_fields.items()
    ):
        raise ProductionRegistryRouteReadinessError("exact transfer source evidence is incomplete")
    reactant_field = policy.get("reactant_field")
    product_field = policy.get("product_field")
    if not isinstance(reactant_field, str) or not isinstance(product_field, str):
        raise ProductionRegistryRouteReadinessError("exact transfer fields are invalid")
    reactant = _canonical_constitution(source.get(reactant_field), label="transfer reactant")
    expected = _canonical_constitution(source.get(product_field), label="transfer product")
    try:
        compiled = load_qualified_forward_reaction(
            executable_registry_path,
            variant_path,
            reaction_id=str(policy.get("reaction_id")),
        )
        products = unique_forward_products(
            compiled,
            [reactant],
            max_products=int(policy.get("max_products", 0)),
            isomeric_smiles=False,
        )
    except (QualifiedForwardError, TypeError, ValueError) as exc:
        raise ProductionRegistryRouteReadinessError(
            "exact transfer forward execution failed"
        ) from exc
    if products != (expected,):
        raise ProductionRegistryRouteReadinessError(
            "exact transfer did not produce exactly one expected constitutional product"
        )
    registry_matches = [
        row
        for row in registry_rows
        if row.get("l1_structural_admission") == "true"
        and row.get("role") == policy.get("registry_role")
        and _canonical_constitution(row.get("canonical_smiles"), label="registry transfer target")
        == expected
    ]
    if len(registry_matches) != 1 or registry_matches[0].get("e2_route_closed") != "true":
        raise ProductionRegistryRouteReadinessError(
            "exact transfer target must resolve to one E2-flagged admitted component"
        )
    return registry_matches[0]["component_id"], {
        "source_record_id": record_id,
        "reaction_id": compiled.reaction_id,
        "canonical_reactant": reactant,
        "canonical_expected_product": expected,
        "forward_products": list(products),
        "verification_status": "verified_exact_product_unique",
    }


def _gap_required_evidence(category: str) -> str:
    return {
        EXACT_OPEN: "close all terminal leaves and retain executable exact-source steps",
        FAMILY_ONLY: "obtain exact-substrate route evidence and forward-verify every step",
        OUTSIDE_SUPPORT: "attach an explicit supported upstream program or accepted-terminal record",
        MISSING_KNOWLEDGE: "perform targeted route-evidence assessment",
        PROVENANCE_ONLY: "attach explicit route evidence before any route claim",
    }[category]


def build_production_registry_route_readiness(
    config_path: Path,
    component_registry_path: Path,
    expansion_result_path: Path,
    original_component_ledger_path: Path,
    original_result_path: Path,
    hydrophobic_transfer_path: Path,
    executable_registry_path: Path,
    oxidation_variant_path: Path,
) -> tuple[dict[str, Any], bytes, bytes]:
    """Build deterministic component and recurring-gap route-readiness ledgers."""

    config = _load_json(config_path, label="route-readiness config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ProductionRegistryRouteReadinessError("unsupported config schema")
    paths = {
        "component_registry": component_registry_path,
        "expansion_result": expansion_result_path,
        "original_component_ledger": original_component_ledger_path,
        "original_result": original_result_path,
        "hydrophobic_transfer": hydrophobic_transfer_path,
        "executable_registry": executable_registry_path,
        "oxidation_variant": oxidation_variant_path,
    }
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(paths):
        raise ProductionRegistryRouteReadinessError("config inputs mismatch")
    for name, path in paths.items():
        record = inputs[name]
        if not isinstance(record, dict):
            raise ProductionRegistryRouteReadinessError(f"input {name} must be an object")
        _verify_hash(path, record.get("expected_sha256"), label=name)

    expansion_result = _load_json(expansion_result_path, label="component expansion result")
    expansion_artifact = expansion_result.get("artifacts", {}).get("registry", {})
    if expansion_artifact.get("sha256") != sha256_file(component_registry_path):
        raise ProductionRegistryRouteReadinessError(
            "component expansion result does not own the registry"
        )
    original_result = _load_json(original_result_path, label="original dossier result")
    if original_result.get("artifacts", {}).get("component_ledger_sha256") != sha256_file(
        original_component_ledger_path
    ):
        raise ProductionRegistryRouteReadinessError(
            "original dossier result does not own the component ledger"
        )

    all_registry_rows = _read_csv(component_registry_path, label="component registry")
    admitted = [row for row in all_registry_rows if row.get("l1_structural_admission") == "true"]
    expected_total = config.get("expected_counts", {}).get("components")
    if len(admitted) != expected_total:
        raise ProductionRegistryRouteReadinessError(
            f"admitted registry count mismatch: expected {expected_total}, observed {len(admitted)}"
        )
    role_set = {row.get("role") for row in admitted}
    expected_roles = set(config.get("expected_roles", []))
    if role_set != expected_roles:
        raise ProductionRegistryRouteReadinessError("admitted registry roles mismatch")
    registry_keys: set[tuple[str, str]] = set()
    for row in admitted:
        key = (
            row["role"],
            _canonical_constitution(row.get("canonical_smiles"), label=row["component_id"]),
        )
        if key in registry_keys:
            raise ProductionRegistryRouteReadinessError(
                "duplicate admitted role-constitutional identity"
            )
        registry_keys.add(key)

    original_rows = _read_csv(original_component_ledger_path, label="original component ledger")
    original_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in original_rows:
        key = (
            row["role"],
            _canonical_constitution(row.get("canonical_smiles"), label=row["component_id"]),
        )
        if key in original_by_key:
            raise ProductionRegistryRouteReadinessError(
                "duplicate original role-constitutional identity"
            )
        original_by_key[key] = row
    if set(original_by_key) - registry_keys:
        raise ProductionRegistryRouteReadinessError(
            "one or more original components are absent from the production registry"
        )

    transfer_rows = _read_csv(hydrophobic_transfer_path, label="hydrophobic transfer ledger")
    exact_transfer_component, transfer_verification = _verify_exact_transfer(
        config,
        transfer_rows,
        admitted,
        executable_registry_path,
        oxidation_variant_path,
    )
    e2_components = {row["component_id"] for row in admitted if row["e2_route_closed"] == "true"}
    if e2_components != {exact_transfer_component}:
        raise ProductionRegistryRouteReadinessError(
            "the admitted E2 set does not equal the independently verified transfer set"
        )

    component_rows: list[dict[str, str]] = []
    category_counts: Counter[str] = Counter()
    role_category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    gap_counts: Counter[tuple[str, str, str]] = Counter()
    original_matches = 0
    for row in sorted(admitted, key=lambda item: (item["role"], item["canonical_smiles"])):
        key = (
            row["role"],
            _canonical_constitution(row["canonical_smiles"], label=row["component_id"]),
        )
        original = original_by_key.get(key)
        if original is not None:
            original_matches += 1
        classification = classify_component_evidence(
            row,
            original_component=original,
            exact_transfer_forward_verified=row["component_id"] == exact_transfer_component,
        )
        category = classification["evidence_category"]
        if category not in CATEGORIES:
            raise ProductionRegistryRouteReadinessError("classifier returned unknown category")
        category_counts[category] += 1
        role_category_counts[row["role"]][category] += 1
        if classification["gap_class"] != "none":
            gap_counts[(row["role"], classification["gap_class"], category)] += 1
        component_rows.append(
            {
                "component_id": row["component_id"],
                "role": row["role"],
                "canonical_smiles": row["canonical_smiles"],
                "is_current_catalog": row["is_current_catalog"],
                "constitutional_match_to_original_93": str(original is not None).lower(),
                "original_component_id": original["component_id"] if original else "",
                **classification,
                "route_evidence_states_json": row["route_evidence_states_json"],
                "procurement_evidence_states_json": row["procurement_evidence_states_json"],
                "route_closure_states_json": row["route_closure_states_json"],
                "e2_route_closed_input_flag": row["e2_route_closed"],
                "source_classes_json": row["source_classes_json"],
                "source_record_ids_json": row["source_record_ids_json"],
            }
        )
    if original_matches != len(original_rows):
        raise ProductionRegistryRouteReadinessError("not every original component matched once")

    recurring_minimum = config.get("gap_policy", {}).get("recurring_minimum_components")
    if isinstance(recurring_minimum, bool) or not isinstance(recurring_minimum, int):
        raise ProductionRegistryRouteReadinessError("invalid recurring gap threshold")
    ordered_gaps = sorted(
        gap_counts.items(),
        key=lambda item: (item[1], item[0][0], item[0][1]),
    )
    gap_rows: list[dict[str, Any]] = []
    recurring_rank = 0
    for (role, gap_class, category), count in ordered_gaps:
        recurring = count >= recurring_minimum
        if recurring:
            recurring_rank += 1
        gap_rows.append(
            {
                "priority_rank": recurring_rank if recurring else "",
                "role": role,
                "gap_class": gap_class,
                "evidence_category": category,
                "component_count": count,
                "recurring": str(recurring).lower(),
                "material_relevance": (
                    "closing this role-specific class would add route-ready component support; "
                    "product-level expansion requires separately verified L1 assemblies"
                ),
                "required_evidence": _gap_required_evidence(category),
            }
        )

    by_role = {
        role: {
            "components": sum(role_category_counts[role].values()),
            "route_complete_components": sum(
                role_category_counts[role][category] for category in ROUTE_COMPLETE_CATEGORIES
            ),
            "by_evidence_category": {
                category: role_category_counts[role][category] for category in CATEGORIES
            },
        }
        for role in sorted(expected_roles)
    }
    recurring_gaps = [
        {
            "priority_rank": row["priority_rank"],
            "role": row["role"],
            "gap_class": row["gap_class"],
            "evidence_category": row["evidence_category"],
            "component_count": row["component_count"],
        }
        for row in gap_rows
        if row["recurring"] == "true"
    ]
    exact_evidence_absent_by_role = {
        role: role_category_counts[role][FAMILY_ONLY]
        + role_category_counts[role][OUTSIDE_SUPPORT]
        + role_category_counts[role][PROVENANCE_ONLY]
        for role in sorted(expected_roles)
    }
    not_assessed_by_role = {
        role: role_category_counts[role][MISSING_KNOWLEDGE] for role in sorted(expected_roles)
    }
    exact_open_by_role = {
        role: role_category_counts[role][EXACT_OPEN] for role in sorted(expected_roles)
    }
    summary = {
        "components": len(component_rows),
        "original_93_constitutional_matches": original_matches,
        "independently_reexecuted_exact_transfer_components": 1,
        "route_complete_components": sum(category_counts[c] for c in ROUTE_COMPLETE_CATEGORIES),
        "by_evidence_category": {category: category_counts[category] for category in CATEGORIES},
        "by_role": by_role,
        "gap_evidence_status_groups": {
            "exact_substrate_route_evidence_absent_or_unreported": {
                "components": sum(exact_evidence_absent_by_role.values()),
                "by_role": exact_evidence_absent_by_role,
                "includes_categories": [FAMILY_ONLY, OUTSIDE_SUPPORT, PROVENANCE_ONLY],
            },
            "route_evidence_not_yet_assessed": {
                "components": sum(not_assessed_by_role.values()),
                "by_role": not_assessed_by_role,
                "includes_categories": [MISSING_KNOWLEDGE],
            },
            "exact_source_route_present_but_l3_open": {
                "components": sum(exact_open_by_role.values()),
                "by_role": exact_open_by_role,
                "includes_categories": [EXACT_OPEN],
            },
        },
        "smallest_recurring_gap_classes": recurring_gaps,
    }
    _validate_expected(summary, config.get("expected_counts"), label="summary")

    component_bytes = _csv_bytes(component_rows, COMPONENT_FIELDS)
    gap_bytes = _csv_bytes(gap_rows, GAP_FIELDS)
    root = config_path.resolve().parents[2]
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": config.get("task"),
        "inputs": {
            "config": {
                "path": _portable(config_path, root=root),
                "sha256": sha256_file(config_path),
            },
            **{
                name: {"path": _portable(path, root=root), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
        },
        "identity_policy": "role_plus_canonical_constitutional_smiles",
        "evidence_policy": config.get("evidence_policy"),
        "gap_policy": config.get("gap_policy"),
        "exact_transfer_verification": transfer_verification,
        "summary": summary,
        "claims_boundary": {
            "structural_l1_admission_is_route_closure": False,
            "handle_qualification_is_route_evidence": False,
            "motif_similarity_is_route_evidence": False,
            "family_projection_is_exact_source_route_evidence": False,
            "route_complete_component_is_observed_synthesis": False,
            "route_complete_component_is_experimental_success": False,
        },
        "safe_claim": (
            f"Among {summary['components']} L1-admitted production components, "
            f"{summary['route_complete_components']} are either accepted terminals or have an "
            "exact-source, uniquely forward-verified upstream program closed to frozen terminal "
            "evidence. Structural admission and component novelty do not establish route closure, "
            "exact-product synthesis or experimental success."
        ),
        "software": {"rdkit_version": rdBase.rdkitVersion},
        "artifacts": {
            "component_ledger_sha256": sha256_bytes(component_bytes),
            "gap_ledger_sha256": sha256_bytes(gap_bytes),
        },
    }
    return result, component_bytes, gap_bytes
