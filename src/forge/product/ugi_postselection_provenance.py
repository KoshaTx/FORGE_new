"""Exact-identity provenance audit for the frozen Ugi product-plus-L1 sample.

This module composes structural provenance and the already frozen static route-
readiness census.  It does not infer provenance from molecular similarity and it
does not plan, score or complete routes.
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
from forge.potency.ugi_semantic_annotations import ROLE_NAMES

CONFIG_SCHEMA_VERSION = "phase1_ugi_postselection_provenance_audit_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_postselection_provenance_audit.v1"

ORIGINAL = "original_ugi_component"
KNOWN_EXPANSION = "admitted_transferred_or_expanded_known_component"
CATALOG_ABSENT = "graph_absent_from_frozen_424_component_catalog"
COMPONENT_STRATA = (ORIGINAL, KNOWN_EXPANSION, CATALOG_ABSENT)

ORIGINAL_SUBSTRATUM = "original_current_ugi_catalog_component"
TRANSFER_SUBSTRATUM = "admitted_cross_platform_transferred_known_component"
EXPANDED_SUBSTRATUM = "admitted_expanded_known_component"
ABSENT_SUBSTRATUM = "graph_absent_from_frozen_catalog"
COMPONENT_SUBSTRATA = (
    ORIGINAL_SUBSTRATUM,
    TRANSFER_SUBSTRATUM,
    EXPANDED_SUBSTRATUM,
    ABSENT_SUBSTRATUM,
)

PRODUCT_ORIGINAL = "product_uses_only_original_ugi_components"
PRODUCT_KNOWN_EXPANSION = "product_contains_admitted_transferred_or_expanded_known_component"
PRODUCT_CATALOG_ABSENT = "product_contains_catalog_absent_component"
PRODUCT_STRATA = (PRODUCT_ORIGINAL, PRODUCT_KNOWN_EXPANSION, PRODUCT_CATALOG_ABSENT)

OUTSIDE_ROUTE_TIER = "not_assessed_missing_route_knowledge"
PRODUCT_ROUTE_COMPLETE = "all_components_static_route_complete"
PRODUCT_ROUTE_INCOMPLETE = "within_catalog_with_one_or_more_noncomplete_static_route_tiers"
PRODUCT_ROUTE_NOT_ASSESSED = "contains_catalog_absent_component_route_not_assessed"

COMPONENT_FIELDS = (
    "sample_index",
    "structure_id",
    "product_id",
    "role",
    "canonical_component_smiles",
    "structural_provenance_stratum",
    "catalog_provenance_substratum",
    "catalog_membership",
    "registry_component_id",
    "registry_source_classes_json",
    "registry_source_record_ids_json",
    "static_route_readiness_tier",
    "static_route_evidence_category",
    "static_route_complete",
    "static_route_gap_class",
    "static_route_basis",
)

PRODUCT_FIELDS = (
    "sample_index",
    "structure_id",
    "product_id",
    "canonical_product_smiles",
    "product_structural_provenance_stratum",
    "amine_head_structural_provenance_stratum",
    "oxoester_aldehyde_body_tail_structural_provenance_stratum",
    "isocyanide_tail_structural_provenance_stratum",
    "original_component_count",
    "admitted_transferred_or_expanded_known_component_count",
    "catalog_absent_component_count",
    "static_route_complete_component_count",
    "static_route_not_assessed_component_count",
    "product_static_route_diagnostic",
)


class UgiPostselectionProvenanceError(ValueError):
    """Raised when an input or exact-identity invariant fails."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=UgiPostselectionProvenanceError, label=label)


def _read_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise UgiPostselectionProvenanceError(f"{label} has no header")
            return list(reader)
    except (OSError, csv.Error) as exc:
        raise UgiPostselectionProvenanceError(f"could not read {label}: {path}") from exc


def _canonical_constitution(smiles: Any, *, label: str) -> str:
    if not isinstance(smiles, str) or not smiles:
        raise UgiPostselectionProvenanceError(f"{label} must be non-empty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise UgiPostselectionProvenanceError(
            f"{label} must be one valid connected molecular graph"
        )
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _json_string_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, str):
        raise UgiPostselectionProvenanceError(f"{label} must be encoded JSON")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise UgiPostselectionProvenanceError(f"{label} is invalid JSON") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise UgiPostselectionProvenanceError(f"{label} must be a list of strings")
    return sorted(set(parsed))


def _portable(path: Path, *, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _validate_expected(observed: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise UgiPostselectionProvenanceError(f"{label} fields mismatch")
        for key, value in expected.items():
            _validate_expected(observed[key], value, label=f"{label}.{key}")
    elif isinstance(expected, list):
        if observed != expected:
            raise UgiPostselectionProvenanceError(f"{label} list mismatch")
    elif observed != expected:
        raise UgiPostselectionProvenanceError(
            f"{label} mismatch: expected {expected!r}, observed {observed!r}"
        )


def classify_component_provenance(
    registry_row: Mapping[str, str] | None,
    *,
    transfer_source_class: str,
) -> tuple[str, str]:
    """Classify one exact role-identity match without structural inference."""

    if registry_row is None:
        return CATALOG_ABSENT, ABSENT_SUBSTRATUM
    current = registry_row.get("is_current_catalog")
    if current not in {"true", "false"}:
        raise UgiPostselectionProvenanceError("registry current-catalog flag is invalid")
    source_classes = _json_string_list(
        registry_row.get("source_classes_json"),
        label=f"{registry_row.get('component_id', 'component')} source classes",
    )
    if current == "true":
        return ORIGINAL, ORIGINAL_SUBSTRATUM
    if transfer_source_class in source_classes:
        return KNOWN_EXPANSION, TRANSFER_SUBSTRATUM
    return KNOWN_EXPANSION, EXPANDED_SUBSTRATUM


def classify_product_provenance(component_strata: Sequence[str]) -> str:
    """Aggregate exactly three mutually exclusive component strata."""

    if len(component_strata) != len(ROLE_NAMES) or any(
        stratum not in COMPONENT_STRATA for stratum in component_strata
    ):
        raise UgiPostselectionProvenanceError("invalid product component strata")
    if CATALOG_ABSENT in component_strata:
        return PRODUCT_CATALOG_ABSENT
    if KNOWN_EXPANSION in component_strata:
        return PRODUCT_KNOWN_EXPANSION
    return PRODUCT_ORIGINAL


def _route_diagnostic(
    readiness_row: Mapping[str, str] | None,
) -> dict[str, str]:
    if readiness_row is None:
        return {
            "tier": OUTSIDE_ROUTE_TIER,
            "category": "missing_route_knowledge",
            "complete": "false",
            "gap": "route_evidence_not_assessed",
            "basis": "component_absent_from_frozen_424_catalog_no_route_assessment",
        }
    complete = readiness_row.get("route_complete_component")
    if complete not in {"true", "false"}:
        raise UgiPostselectionProvenanceError("route-complete flag is invalid")
    category = readiness_row.get("evidence_category", "")
    if not category:
        raise UgiPostselectionProvenanceError("static route evidence category is missing")
    return {
        "tier": category,
        "category": category,
        "complete": complete,
        "gap": readiness_row.get("gap_class", ""),
        "basis": readiness_row.get("evidence_basis", ""),
    }


def _product_route_diagnostic(route_rows: Sequence[Mapping[str, str]]) -> str:
    if len(route_rows) != len(ROLE_NAMES):
        raise UgiPostselectionProvenanceError("product route diagnostic needs three components")
    if any(row["tier"] == OUTSIDE_ROUTE_TIER for row in route_rows):
        return PRODUCT_ROUTE_NOT_ASSESSED
    if all(row["complete"] == "true" for row in route_rows):
        return PRODUCT_ROUTE_COMPLETE
    return PRODUCT_ROUTE_INCOMPLETE


def build_postselection_provenance_audit(
    config_path: Path,
    production_manifest_path: Path,
    selected_sample_path: Path,
    component_registry_path: Path,
    component_expansion_result_path: Path,
    route_readiness_result_path: Path,
    route_readiness_ledger_path: Path,
) -> tuple[dict[str, Any], bytes, bytes]:
    """Build exact-identity component/product provenance and route diagnostics."""

    config = _load_json(config_path, label="provenance audit config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiPostselectionProvenanceError("unsupported provenance audit config schema")
    paths = {
        "production_manifest": production_manifest_path,
        "selected_sample": selected_sample_path,
        "component_registry": component_registry_path,
        "component_expansion_result": component_expansion_result_path,
        "route_readiness_result": route_readiness_result_path,
        "route_readiness_ledger": route_readiness_ledger_path,
    }
    specifications = config.get("inputs")
    if not isinstance(specifications, dict) or set(specifications) != set(paths):
        raise UgiPostselectionProvenanceError("config input set mismatch")
    inputs: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        specification = specifications[name]
        if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
            raise UgiPostselectionProvenanceError(f"input {name} must define path and sha256")
        observed = sha256_file(path)
        if observed != specification["sha256"]:
            raise UgiPostselectionProvenanceError(
                f"{name} hash mismatch: expected {specification['sha256']}, observed {observed}"
            )
        inputs[name] = {
            "path": _portable(path, root=config_path.parents[2]),
            "bytes": path.stat().st_size,
            "sha256": observed,
        }

    manifest = _load_json(production_manifest_path, label="production manifest")
    sample = _load_json(selected_sample_path, label="selected sample")
    expansion = _load_json(component_expansion_result_path, label="component expansion result")
    readiness = _load_json(route_readiness_result_path, label="route readiness result")
    if manifest.get("status") != "frozen":
        raise UgiPostselectionProvenanceError("production generator is not frozen")
    selected_spec = manifest.get("selection", {}).get("fresh_selected_sample", {})
    if selected_spec.get("sha256") != inputs["selected_sample"]["sha256"]:
        raise UgiPostselectionProvenanceError("production manifest does not own selected sample")
    registry_spec = manifest.get("sampling_dependencies", {}).get("admitted_component_registry", {})
    if registry_spec.get("sha256") != inputs["component_registry"]["sha256"]:
        raise UgiPostselectionProvenanceError("production manifest registry changed")
    if (
        expansion.get("artifacts", {}).get("registry", {}).get("sha256")
        != inputs["component_registry"]["sha256"]
    ):
        raise UgiPostselectionProvenanceError("component expansion result does not own registry")
    if (
        readiness.get("artifacts", {}).get("component_ledger_sha256")
        != inputs["route_readiness_ledger"]["sha256"]
    ):
        raise UgiPostselectionProvenanceError("route-readiness result does not own ledger")

    identity_policy = config.get("identity_policy", {})
    if identity_policy != {
        "component_identity": "exact_role_plus_canonical_constitutional_smiles",
        "stereochemistry": "excluded",
        "similarity_used_for_provenance": False,
    }:
        raise UgiPostselectionProvenanceError("identity policy changed")
    transfer_source_class = config.get("provenance_policy", {}).get("transfer_source_class")
    if not isinstance(transfer_source_class, str) or not transfer_source_class:
        raise UgiPostselectionProvenanceError("transfer source class is missing")

    registry_rows = [
        row
        for row in _read_csv(component_registry_path, label="component registry")
        if row.get("l1_structural_admission") == "true"
    ]
    registry: dict[tuple[str, str], dict[str, str]] = {}
    for row in registry_rows:
        role = row.get("role", "")
        if role not in ROLE_NAMES:
            raise UgiPostselectionProvenanceError(f"unexpected registry role {role!r}")
        key = (
            role,
            _canonical_constitution(row.get("canonical_smiles"), label=row["component_id"]),
        )
        if key in registry:
            raise UgiPostselectionProvenanceError("duplicate role-constitutional registry identity")
        registry[key] = row

    route_rows = _read_csv(route_readiness_ledger_path, label="route-readiness ledger")
    route_registry: dict[tuple[str, str], dict[str, str]] = {}
    for row in route_rows:
        key = (
            row.get("role", ""),
            _canonical_constitution(row.get("canonical_smiles"), label=row["component_id"]),
        )
        if key in route_registry:
            raise UgiPostselectionProvenanceError("duplicate route-readiness exact identity")
        route_registry[key] = row
    if set(route_registry) != set(registry):
        raise UgiPostselectionProvenanceError(
            "route-readiness ledger must cover the admitted registry exactly"
        )

    sample_rows = sample.get("samples")
    if not isinstance(sample_rows, list):
        raise UgiPostselectionProvenanceError("selected sample lacks sample rows")
    component_rows: list[dict[str, Any]] = []
    product_rows: list[dict[str, Any]] = []
    component_counts: Counter[str] = Counter()
    component_substratum_counts: Counter[str] = Counter()
    component_route_counts: Counter[str] = Counter()
    role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    role_unique: dict[tuple[str, str], set[str]] = defaultdict(set)
    product_counts: Counter[str] = Counter()
    product_route_counts: Counter[str] = Counter()
    reconstructed = 0

    for sample_index, sample_row in enumerate(sample_rows):
        if not sample_row.get("component_reconstruction_valid"):
            continue
        components = sample_row.get("component_smiles_by_role")
        if not isinstance(components, dict) or set(components) != set(ROLE_NAMES):
            raise UgiPostselectionProvenanceError(
                f"sample {sample_index} has invalid reconstructed component roles"
            )
        canonical_product = _canonical_constitution(
            sample_row.get("smiles"), label=f"sample {sample_index} product"
        )
        reconstructed += 1
        strata_by_role: dict[str, str] = {}
        route_by_role: dict[str, dict[str, str]] = {}
        for role in ROLE_NAMES:
            canonical_component = _canonical_constitution(
                components[role], label=f"sample {sample_index}/{role}"
            )
            key = (role, canonical_component)
            registry_row = registry.get(key)
            stratum, substratum = classify_component_provenance(
                registry_row, transfer_source_class=transfer_source_class
            )
            readiness_row = route_registry.get(key)
            if (registry_row is None) != (readiness_row is None):
                raise UgiPostselectionProvenanceError(
                    "catalog and route-readiness exact membership disagree"
                )
            route_diagnostic = _route_diagnostic(readiness_row)
            if registry_row is None and route_diagnostic["tier"] != OUTSIDE_ROUTE_TIER:
                raise UgiPostselectionProvenanceError(
                    "catalog-absent component received non-missing route evidence"
                )
            source_classes = (
                _json_string_list(
                    registry_row["source_classes_json"],
                    label=f"sample {sample_index}/{role} source classes",
                )
                if registry_row is not None
                else []
            )
            source_record_ids = (
                _json_string_list(
                    registry_row["source_record_ids_json"],
                    label=f"sample {sample_index}/{role} source records",
                )
                if registry_row is not None
                else []
            )
            component_rows.append(
                {
                    "sample_index": sample_index,
                    "structure_id": sample_row.get("structure_id", ""),
                    "product_id": sample_row.get("product_id", ""),
                    "role": role,
                    "canonical_component_smiles": canonical_component,
                    "structural_provenance_stratum": stratum,
                    "catalog_provenance_substratum": substratum,
                    "catalog_membership": str(registry_row is not None).lower(),
                    "registry_component_id": (
                        registry_row.get("component_id", "") if registry_row is not None else ""
                    ),
                    "registry_source_classes_json": json.dumps(
                        source_classes, separators=(",", ":")
                    ),
                    "registry_source_record_ids_json": json.dumps(
                        source_record_ids, separators=(",", ":")
                    ),
                    "static_route_readiness_tier": route_diagnostic["tier"],
                    "static_route_evidence_category": route_diagnostic["category"],
                    "static_route_complete": route_diagnostic["complete"],
                    "static_route_gap_class": route_diagnostic["gap"],
                    "static_route_basis": route_diagnostic["basis"],
                }
            )
            strata_by_role[role] = stratum
            route_by_role[role] = route_diagnostic
            component_counts[stratum] += 1
            component_substratum_counts[substratum] += 1
            component_route_counts[route_diagnostic["tier"]] += 1
            role_counts[role][stratum] += 1
            role_unique[(role, stratum)].add(canonical_component)

        product_stratum = classify_product_provenance(list(strata_by_role.values()))
        product_route = _product_route_diagnostic(list(route_by_role.values()))
        product_counts[product_stratum] += 1
        product_route_counts[product_route] += 1
        product_rows.append(
            {
                "sample_index": sample_index,
                "structure_id": sample_row.get("structure_id", ""),
                "product_id": sample_row.get("product_id", ""),
                "canonical_product_smiles": canonical_product,
                "product_structural_provenance_stratum": product_stratum,
                **{
                    f"{role}_structural_provenance_stratum": strata_by_role[role]
                    for role in ROLE_NAMES
                },
                "original_component_count": sum(
                    value == ORIGINAL for value in strata_by_role.values()
                ),
                "admitted_transferred_or_expanded_known_component_count": sum(
                    value == KNOWN_EXPANSION for value in strata_by_role.values()
                ),
                "catalog_absent_component_count": sum(
                    value == CATALOG_ABSENT for value in strata_by_role.values()
                ),
                "static_route_complete_component_count": sum(
                    value["complete"] == "true" for value in route_by_role.values()
                ),
                "static_route_not_assessed_component_count": sum(
                    value["tier"] == OUTSIDE_ROUTE_TIER for value in route_by_role.values()
                ),
                "product_static_route_diagnostic": product_route,
            }
        )

    summary = {
        "attempted_draws": len(sample_rows),
        "component_reconstructed_products": reconstructed,
        "component_occurrences": len(component_rows),
        "frozen_catalog_components": len(registry),
        "original_current_catalog_components": sum(
            row.get("is_current_catalog") == "true" for row in registry.values()
        ),
        "component_occurrences_by_structural_provenance": {
            stratum: component_counts[stratum] for stratum in COMPONENT_STRATA
        },
        "component_occurrences_by_catalog_provenance_substratum": {
            stratum: component_substratum_counts[stratum] for stratum in COMPONENT_SUBSTRATA
        },
        "component_occurrences_by_role_and_structural_provenance": {
            role: {stratum: role_counts[role][stratum] for stratum in COMPONENT_STRATA}
            for role in ROLE_NAMES
        },
        "unique_components_by_role_and_structural_provenance": {
            role: {stratum: len(role_unique[(role, stratum)]) for stratum in COMPONENT_STRATA}
            for role in ROLE_NAMES
        },
        "products_by_structural_provenance": {
            stratum: product_counts[stratum] for stratum in PRODUCT_STRATA
        },
        "component_occurrences_by_static_route_readiness_tier": dict(
            sorted(component_route_counts.items())
        ),
        "products_by_static_route_diagnostic": {
            PRODUCT_ROUTE_COMPLETE: product_route_counts[PRODUCT_ROUTE_COMPLETE],
            PRODUCT_ROUTE_INCOMPLETE: product_route_counts[PRODUCT_ROUTE_INCOMPLETE],
            PRODUCT_ROUTE_NOT_ASSESSED: product_route_counts[PRODUCT_ROUTE_NOT_ASSESSED],
        },
    }
    _validate_expected(summary, config.get("expected_summary"), label="summary")

    component_payload = _csv_bytes(component_rows, COMPONENT_FIELDS)
    product_payload = _csv_bytes(product_rows, PRODUCT_FIELDS)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_nonselecting_postselection_audit",
        "task": "Exact-identity provenance strata and static route-readiness composition",
        "config": {
            "path": _portable(config_path, root=config_path.parents[2]),
            "sha256": sha256_file(config_path),
        },
        "inputs": inputs,
        "identity_policy": identity_policy,
        "provenance_policy": config["provenance_policy"],
        "route_diagnostic_policy": config["route_diagnostic_policy"],
        "scope": {
            "postselection_nonselecting": True,
            "can_reopen_architecture_or_checkpoint": False,
            "route_planning_performed": False,
            "route_guidance_performed": False,
            "biological_scoring_performed": False,
            "candidate_selection_performed": False,
            "outside_catalog_components_are_infeasible": False,
        },
        "summary": summary,
        "artifacts": {
            "component_ledger_sha256": sha256_bytes(component_payload),
            "product_ledger_sha256": sha256_bytes(product_payload),
        },
        "safe_claim": (
            "The frozen sample contains exact role-specific constitutional matches to original "
            "and expanded known Ugi components, together with component graphs absent from the "
            "frozen 424-component catalog. Catalog absence is assigned missing route knowledge, "
            "not chemical infeasibility."
        ),
        "claims_boundary": {
            "exact_catalog_match_is_route_closure": False,
            "catalog_absence_is_unsynthesizability": False,
            "registry_provenance_is_experimental_success": False,
            "structural_novelty_is_product_novelty": False,
            "static_route_composition_is_route_planning": False,
        },
    }
    return result, component_payload, product_payload
