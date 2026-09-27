"""Evidence-calibrated AGILE routes projected across every LNPDB lipid.

This module builds an inventory artifact, not a route model. It represents documented
component routes as reusable records and attaches them to exact component identities in
LNPDB. Missing routes, source discrepancies, forward verification, and procurement closure
remain separate states.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import platform
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.core.io import read_json_object
from forge.core.io import stable_json as _stable_json
from forge.synthesis.sources.supervision_inventory import sha256_file

CONFIG_SCHEMA_VERSION = "m0_09_agile_component_routes_config.v2"
ROUTE_ARTIFACT_SCHEMA_VERSION = "m0_09_agile_component_routes.v1"
RESULT_SCHEMA_VERSION = "m0_09_lnpdb_route_awareness.v1"

ROLE_FIELDS = {
    "head": "IL_head_SMILES",
    "linker": "IL_linker_SMILES",
    "tail1": "IL_tail1_SMILES",
    "tail2": "IL_tail2_SMILES",
}
MEASURED_FIELDS = {
    "A": "A_smiles",
    "B": "B_smiles",
    "C": "C_smiles",
}
MISSING_COMPONENT_VALUES = {"", "NA", "N/A", "None", "none", "nan"}
LABEL_PATTERN = re.compile(r"^A(?P<A>\d+)B(?P<B>\d+)C(?P<C>\d+)$")
LNPDB_AGILE_LABEL_PATTERN = re.compile(r"^A(?P<head>\d+)_B(?P<tail1>\d+)_C(?P<tail2>\d+)$")

LIPID_COLUMNS = (
    "lipid_id",
    "canonical_smiles",
    "lnp_ids_json",
    "lipid_names_json",
    "experiment_ids_json",
    "source_ids_json",
    "component_ids_json",
    "component_evidence_json",
    "l1_route_ids_json",
    "l2_route_ids_json",
    "annotated_component_count",
    "route_extracted_component_count",
    "vendor_claim_component_count",
    "unresolved_component_count",
    "unassessed_component_count",
    "route_awareness_status",
    "agile_reference_route_status",
    "procurement_closure_status",
    "execution_closure_status",
)


class RouteAwarenessError(ValueError):
    """Raised when a route-awareness input violates its evidence contract."""


def _canonicalize(smiles: str, *, label: str) -> str:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RouteAwarenessError(f"{label} contains RDKit-invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=RouteAwarenessError, label=label)


def _require_unique_labels(entries: list[dict[str, Any]], expected: set[str], label: str) -> None:
    observed = [str(entry.get("label", "")) for entry in entries]
    if set(observed) != expected or len(observed) != len(expected):
        raise RouteAwarenessError(
            f"{label} labels must be exactly {sorted(expected)}; observed {sorted(observed)}"
        )


def load_route_config(path: Path) -> dict[str, Any]:
    """Load and validate the curated AGILE component-route configuration."""

    config = _load_json(path, label="AGILE route config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise RouteAwarenessError(
            f"unsupported config schema {config.get('schema_version')!r}; "
            f"expected {CONFIG_SCHEMA_VERSION!r}"
        )

    for name, entry in config.get("inputs", {}).items():
        if entry.get("root") not in {"vendor", "m0_results"}:
            raise RouteAwarenessError(f"config inputs.{name}.root is missing or unsupported")
        if not entry.get("asset") or not entry.get("expected_sha256"):
            raise RouteAwarenessError(f"config inputs.{name} must define asset and expected_sha256")
    required_inputs = {
        "agile_measured",
        "agile_si",
        "lnpdb",
        "component_source_ledger",
        "source_queue",
    }
    if set(config.get("inputs", {})) != required_inputs:
        raise RouteAwarenessError(f"config inputs must be exactly {sorted(required_inputs)}")

    amines = config.get("amine_components", [])
    aldehydes = config.get("aldehyde_ester_route_family", {}).get("members", [])
    additional_aldehyde_families = config.get("additional_aldehyde_route_families", [])
    additional_aldehydes = [
        member for family in additional_aldehyde_families for member in family.get("members", [])
    ]
    unresolved = config.get("unresolved_aldehyde_ester_components", [])
    isocyanides = config.get("isocyanide_route_family", {}).get("members", [])
    additional_isocyanides = config.get("additional_isocyanide_route_family", {}).get("members", [])
    _require_unique_labels(amines, {f"A{i}" for i in range(1, 21)}, "amine component")
    _require_unique_labels(
        [*aldehydes, *unresolved],
        {f"B{i}" for i in range(1, 13)},
        "aldehyde-ester component",
    )
    _require_unique_labels(
        isocyanides,
        {f"C{i}" for i in range(1, 6)},
        "isocyanide component",
    )
    if {entry["label"] for entry in unresolved} != {"B5"}:
        raise RouteAwarenessError("B5 must be the sole unresolved AGILE component")
    if not unresolved[0].get("must_not_infer_route"):
        raise RouteAwarenessError("the B5 source discrepancy must prohibit inferred route closure")

    route_ids = [
        member.get("route_id")
        for member in [
            *aldehydes,
            *additional_aldehydes,
            *isocyanides,
            *additional_isocyanides,
        ]
    ]
    if any(not route_id for route_id in route_ids):
        raise RouteAwarenessError("every extracted route member must define route_id")
    if len(route_ids) != len(set(route_ids)):
        raise RouteAwarenessError("AGILE route identifiers must be unique")

    smiles_fields = [
        *((entry, "smiles") for entry in amines),
        *((entry, "target_smiles") for entry in aldehydes),
        *((entry, "alcohol_intermediate_smiles") for entry in aldehydes),
        *((entry, "acid_smiles") for entry in aldehydes),
        *(
            (entry, field)
            for family in additional_aldehyde_families
            for entry in family.get("members", [])
            for field in family.get("member_smiles_fields", [])
        ),
        *((entry, "measured_smiles") for entry in unresolved),
        *((entry, "target_smiles") for entry in isocyanides),
        *((entry, "formamide_intermediate_smiles") for entry in isocyanides),
        *((entry, "amine_smiles") for entry in isocyanides),
        *((entry, "target_smiles") for entry in additional_isocyanides),
        *((entry, "formamide_intermediate_smiles") for entry in additional_isocyanides),
        *((entry, "amine_smiles") for entry in additional_isocyanides),
    ]
    for entry, field in smiles_fields:
        _canonicalize(str(entry.get(field, "")), label=f"{entry.get('label')}.{field}")

    families = [
        ("aldehyde_ester_route_family", config.get("aldehyde_ester_route_family", {}), [1, 2]),
        ("isocyanide_route_family", config.get("isocyanide_route_family", {}), [1, 2]),
        (
            "additional_isocyanide_route_family",
            config.get("additional_isocyanide_route_family", {}),
            [1, 2],
        ),
    ]
    if len(additional_aldehyde_families) != 2:
        raise RouteAwarenessError(
            "additional_aldehyde_route_families must define the two source-reported families"
        )
    for index, family in enumerate(additional_aldehyde_families):
        expected_steps = [1] if family.get("route_kind") == "direct_oxidation" else [1, 2]
        families.append((f"additional_aldehyde_route_families[{index}]", family, expected_steps))

    for family_name, family, expected_steps in families:
        if family.get("route_evidence_status") != "route_extracted":
            raise RouteAwarenessError(f"{family_name} must remain route_extracted")
        if family.get("forward_verification_status") != "not_run_missing_qualified_template":
            raise RouteAwarenessError(f"{family_name} must not imply forward verification")
        steps = family.get("steps", [])
        if [step.get("step_index") for step in steps] != expected_steps:
            raise RouteAwarenessError(
                f"{family_name} must define ordered source steps {expected_steps}"
            )
    return config


def validate_route_inputs(
    config_path: Path,
    config: dict[str, Any],
    vendor_dir: Path,
    m0_results_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    """Hash-check every route-map input and return provenance plus resolved paths."""

    records = [
        {
            "asset": str(config_path),
            "bytes": config_path.stat().st_size,
            "sha256": sha256_file(config_path),
        }
    ]
    roots = {"vendor": vendor_dir, "m0_results": m0_results_dir}
    paths: dict[str, Path] = {}
    for name, entry in sorted(config["inputs"].items()):
        path = roots[entry["root"]] / entry["asset"]
        if not path.exists():
            raise RouteAwarenessError(f"required route-map input not found: {path}")
        actual = sha256_file(path)
        if actual != entry["expected_sha256"]:
            raise RouteAwarenessError(
                f"hash mismatch for {entry['asset']}: "
                f"expected {entry['expected_sha256']}, observed {actual}"
            )
        paths[name] = path
        records.append(
            {
                "asset": entry["asset"],
                "bytes": path.stat().st_size,
                "sha256": actual,
            }
        )
    return records, paths


def _read_csv(path: Path, *, required: set[str], label: str) -> list[dict[str, str]]:
    try:
        handle = path.open(newline="")
    except FileNotFoundError as exc:
        raise RouteAwarenessError(f"{label} not found: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RouteAwarenessError(f"{label} must contain columns {sorted(required)}")
        return list(reader)


def _component_config_by_label(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    components: dict[str, dict[str, Any]] = {}
    for entry in config["amine_components"]:
        components[entry["label"]] = {
            **entry,
            "component_class": "amine",
            "measured_smiles": entry["smiles"],
            "route_ids": [],
            "route_evidence_status": "source_identified",
            "procurement_evidence_status": "vendor_claim_only",
        }
    for entry in config["aldehyde_ester_route_family"]["members"]:
        components[entry["label"]] = {
            **entry,
            "component_class": "aldehyde_ester",
            "measured_smiles": entry["target_smiles"],
            "route_ids": [entry["route_id"]],
            "route_evidence_status": "route_extracted",
            "procurement_evidence_status": "not_assessed",
        }
    for entry in config["unresolved_aldehyde_ester_components"]:
        components[entry["label"]] = {
            **entry,
            "component_class": "aldehyde_ester",
            "route_ids": [],
            "procurement_evidence_status": "not_assessed",
        }
    for entry in config["isocyanide_route_family"]["members"]:
        components[entry["label"]] = {
            **entry,
            "component_class": "isocyanide",
            "measured_smiles": entry["target_smiles"],
            "route_ids": [entry["route_id"]],
            "route_evidence_status": "route_extracted",
            "procurement_evidence_status": "not_assessed",
        }
    return components


def analyze_agile_measured(
    path: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Validate exact AGILE A/B/C identities and the complete measured product grid."""

    required = {"id", "label", "combined_mol_SMILES", *MEASURED_FIELDS.values()}
    rows = _read_csv(path, required=required, label="AGILE measured CSV")
    if len(rows) != 1200:
        raise RouteAwarenessError(f"expected 1200 AGILE measured rows, found {len(rows)}")

    components = _component_config_by_label(config)
    observed_by_label: dict[str, str] = {}
    combinations: set[tuple[str, str, str]] = set()
    product_structures: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        match = LABEL_PATTERN.fullmatch(row["label"])
        if match is None:
            raise RouteAwarenessError(
                f"AGILE measured row {row_number} has unsupported label {row['label']!r}"
            )
        labels = {prefix: f"{prefix}{match.group(prefix)}" for prefix in MEASURED_FIELDS}
        for prefix, field in MEASURED_FIELDS.items():
            component_label = labels[prefix]
            if component_label not in components:
                raise RouteAwarenessError(
                    f"AGILE measured row {row_number} references unknown {component_label}"
                )
            observed = _canonicalize(row[field], label=f"AGILE measured row {row_number} {field}")
            expected = _canonicalize(
                components[component_label]["measured_smiles"],
                label=f"config {component_label}",
            )
            if observed != expected:
                raise RouteAwarenessError(
                    f"AGILE measured {component_label} differs from its curated identity"
                )
            previous = observed_by_label.setdefault(component_label, observed)
            if previous != observed:
                raise RouteAwarenessError(
                    f"AGILE label {component_label} maps to multiple structures"
                )
        combinations.add((labels["A"], labels["B"], labels["C"]))
        product_structures.add(
            _canonicalize(
                row["combined_mol_SMILES"],
                label=f"AGILE measured row {row_number} product",
            )
        )

    expected_combinations = {
        (f"A{a}", f"B{b}", f"C{c}") for a in range(1, 21) for b in range(1, 13) for c in range(1, 6)
    }
    if combinations != expected_combinations:
        raise RouteAwarenessError("AGILE measured rows are not the complete 20 x 12 x 5 grid")
    if len(product_structures) != 1200:
        raise RouteAwarenessError(
            f"expected 1200 unique AGILE products, found {len(product_structures)}"
        )

    return (
        {
            "measured_products": len(rows),
            "unique_product_structures": len(product_structures),
            "component_counts": {"amine": 20, "aldehyde_ester": 12, "isocyanide": 5},
            "complete_cartesian_product": True,
            "exact_extracted_route_components": {
                "aldehyde_ester": 11,
                "isocyanide": 5,
            },
            "vendor_claim_only_components": {"amine": 20},
            "unresolved_source_discrepancies": 1,
        },
        rows,
    )


def _molecule_record(name: str, smiles: str) -> dict[str, str]:
    return {
        "name": name,
        "smiles": smiles,
        "canonical_smiles": _canonicalize(smiles, label=name),
    }


def _expanded_step(step: dict[str, Any]) -> dict[str, Any]:
    internal_fields = {"reactant_member_fields", "product_member_field"}
    return {key: value for key, value in step.items() if key not in internal_fields}


def _route_record(
    *,
    source: dict[str, Any],
    family: dict[str, Any],
    member: dict[str, Any],
    target: dict[str, str],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "route_id": member["route_id"],
        "route_family_id": family["route_family_id"],
        "level": "L2",
        "component_label": member["label"],
        "target": target,
        "source": {
            "source_id": source["source_id"],
            "doi": source["doi"],
            "locator": family["source_locator"],
        },
        "route_evidence_status": family["route_evidence_status"],
        "forward_verification_status": family["forward_verification_status"],
        "procurement_closure_status": "unknown",
        "execution_closure_status": "unknown",
        "analytical_evidence": family["analytical_evidence"],
        "explicit_yield": family["explicit_yield"],
        "steps": steps,
    }


def _diol_aldehyde_routes(
    family: dict[str, Any],
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    diol = family["shared_leaf"]
    for member in family["members"]:
        acid = _molecule_record(member["acid_name"], member["acid_smiles"])
        diol_record = _molecule_record(diol["name"], diol["smiles"])
        intermediate = _molecule_record(
            member["alcohol_intermediate_name"],
            member["alcohol_intermediate_smiles"],
        )
        target = _molecule_record(member["target_name"], member["target_smiles"])
        routes.append(
            _route_record(
                source=source,
                family=family,
                member=member,
                target=target,
                steps=[
                    {
                        **_expanded_step(family["steps"][0]),
                        "reactants": [
                            {
                                **acid,
                                "procurement_evidence_status": "vendor_claim_only",
                            },
                            {
                                **diol_record,
                                "procurement_evidence_status": diol["procurement_evidence_status"],
                            },
                        ],
                        "product": intermediate,
                    },
                    {
                        **_expanded_step(family["steps"][1]),
                        "reactants": [intermediate],
                        "product": target,
                    },
                ],
            )
        )
    return routes


def _direct_aldehyde_routes(
    family: dict[str, Any],
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for member in family["members"]:
        alcohol = _molecule_record(member["alcohol_name"], member["alcohol_smiles"])
        target = _molecule_record(member["target_name"], member["target_smiles"])
        routes.append(
            _route_record(
                source=source,
                family=family,
                member=member,
                target=target,
                steps=[
                    {
                        **_expanded_step(family["steps"][0]),
                        "reactants": [
                            {
                                **alcohol,
                                "procurement_evidence_status": "vendor_claim_only",
                            }
                        ],
                        "product": target,
                    }
                ],
            )
        )
    return routes


def _isocyanide_routes(
    family: dict[str, Any],
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for member in family["members"]:
        amine = _molecule_record(member["amine_name"], member["amine_smiles"])
        formamide = _molecule_record(
            member["formamide_intermediate_name"],
            member["formamide_intermediate_smiles"],
        )
        target = _molecule_record(member["target_name"], member["target_smiles"])
        routes.append(
            _route_record(
                source=source,
                family=family,
                member=member,
                target=target,
                steps=[
                    {
                        **_expanded_step(family["steps"][0]),
                        "reactants": [
                            {
                                **amine,
                                "procurement_evidence_status": "vendor_claim_only",
                            }
                        ],
                        "product": formamide,
                    },
                    {
                        **_expanded_step(family["steps"][1]),
                        "reactants": [formamide],
                        "product": target,
                    },
                ],
            )
        )
    return routes


def build_route_catalog(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand curated route families into explicit, source-linked route DAG records."""

    routes: list[dict[str, Any]] = []
    source = config["source"]

    routes.extend(_diol_aldehyde_routes(config["aldehyde_ester_route_family"], source))
    for family in config["additional_aldehyde_route_families"]:
        if family["route_kind"] == "direct_oxidation":
            routes.extend(_direct_aldehyde_routes(family, source))
        elif family["route_kind"] == "diol_esterification_oxidation":
            routes.extend(_diol_aldehyde_routes(family, source))
        else:
            raise RouteAwarenessError(
                f"unsupported additional aldehyde route kind {family['route_kind']!r}"
            )
    routes.extend(_isocyanide_routes(config["isocyanide_route_family"], source))
    routes.extend(_isocyanide_routes(config["additional_isocyanide_route_family"], source))

    routes.sort(key=lambda route: route["route_id"])
    if any(route["execution_closure_status"] != "unknown" for route in routes):
        raise RouteAwarenessError("source extraction must not imply executable route closure")
    return routes


def _build_exact_evidence_index(
    config: dict[str, Any], routes: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}

    def merge(
        smiles: str,
        *,
        route_ids: list[str],
        route_status: str,
        procurement_status: str,
        source_labels: list[str],
    ) -> None:
        canonical = _canonicalize(smiles, label="curated component evidence")
        entry = evidence.setdefault(
            canonical,
            {
                "route_ids": set(),
                "route_evidence_statuses": set(),
                "procurement_evidence_statuses": set(),
                "source_labels": set(),
            },
        )
        entry["route_ids"].update(route_ids)
        entry["route_evidence_statuses"].add(route_status)
        entry["procurement_evidence_statuses"].add(procurement_status)
        entry["source_labels"].update(source_labels)

    for amine in config["amine_components"]:
        merge(
            amine["smiles"],
            route_ids=[],
            route_status="source_identified",
            procurement_status="vendor_claim_only",
            source_labels=[amine["label"]],
        )
    for route in routes:
        merge(
            route["target"]["canonical_smiles"],
            route_ids=[route["route_id"]],
            route_status="route_extracted",
            procurement_status="not_assessed",
            source_labels=[route["component_label"]],
        )
    for unresolved in config["unresolved_aldehyde_ester_components"]:
        merge(
            unresolved["measured_smiles"],
            route_ids=[],
            route_status="source_discrepancy_unresolved",
            procurement_status="not_assessed",
            source_labels=[unresolved["label"]],
        )

    normalized: dict[str, dict[str, Any]] = {}
    for canonical, entry in sorted(evidence.items()):
        statuses = entry["route_evidence_statuses"]
        if "route_extracted" in statuses:
            route_status = "route_extracted"
        elif "source_discrepancy_unresolved" in statuses:
            route_status = "source_discrepancy_unresolved"
        else:
            route_status = "source_identified"
        procurement_status = (
            "vendor_claim_only"
            if "vendor_claim_only" in entry["procurement_evidence_statuses"]
            else "not_assessed"
        )
        normalized[canonical] = {
            "route_ids": sorted(entry["route_ids"]),
            "route_evidence_status": route_status,
            "procurement_evidence_status": procurement_status,
            "source_labels": sorted(entry["source_labels"]),
        }
    return normalized


def _component_ledger_indices(
    path: Path,
) -> tuple[dict[tuple[str, str], dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    required = {"component_id", "role", "parse_status", "canonical_smiles", "raw_smiles_json"}
    rows = _read_csv(path, required=required, label="component source ledger")
    parsed: dict[tuple[str, str], dict[str, str]] = {}
    raw: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        role = row["role"]
        if row["parse_status"] == "parsed":
            key = (role, row["canonical_smiles"])
            if key in parsed:
                raise RouteAwarenessError(f"duplicate parsed component ledger key: {key}")
            parsed[key] = row
        for value in json.loads(row["raw_smiles_json"]):
            key = (role, value)
            if key in raw and raw[key]["component_id"] != row["component_id"]:
                raise RouteAwarenessError(f"ambiguous raw component ledger key: {key}")
            raw[key] = row
    return parsed, raw


def _source_index(path: Path) -> dict[str, list[str]]:
    required = {"source_id", "experiment_ids_json"}
    rows = _read_csv(path, required=required, label="source queue")
    index: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for experiment_id in json.loads(row["experiment_ids_json"]):
            index[experiment_id].add(row["source_id"])
    return {key: sorted(value) for key, value in sorted(index.items())}


def _resolve_component(
    role: str,
    raw_smiles: str,
    parsed_index: dict[tuple[str, str], dict[str, str]],
    raw_index: dict[tuple[str, str], dict[str, str]],
) -> tuple[dict[str, str], str | None]:
    if raw_smiles.strip() in MISSING_COMPONENT_VALUES:
        raise RouteAwarenessError("missing components must be filtered before resolution")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(raw_smiles)
    canonical = Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule is not None else None
    row = parsed_index.get((role, canonical)) if canonical is not None else None
    if row is None:
        row = raw_index.get((role, raw_smiles))
    if row is None:
        raise RouteAwarenessError(
            f"LNPDB component is absent from component source ledger: {role} {raw_smiles!r}"
        )
    return row, canonical


def _route_awareness_status(component_evidence: list[dict[str, Any]]) -> str:
    statuses = [entry["route_evidence_status"] for entry in component_evidence]
    resolved = {
        "route_extracted",
        "source_identified",
    }
    if "source_discrepancy_unresolved" in statuses:
        return "contains_source_discrepancy"
    if statuses and all(status in resolved for status in statuses):
        return "all_components_source_resolved"
    if any(status in resolved for status in statuses):
        return "partially_source_resolved"
    return "not_assessed"


def _without_stereochemistry(canonical_smiles: str) -> str:
    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:
        raise RouteAwarenessError(
            f"cannot remove stereochemistry from invalid SMILES: {canonical_smiles!r}"
        )
    return Chem.MolToSmiles(molecule, isomericSmiles=False)


def _agile_labels_by_role(rows: list[dict[str, str]]) -> dict[str, str]:
    agile_rows = [row for row in rows if row["Experiment_ID"] == "YX_2024"]
    if not agile_rows:
        return {}
    labels: dict[str, set[str]] = defaultdict(set)
    for row in agile_rows:
        match = LNPDB_AGILE_LABEL_PATTERN.fullmatch(row["IL_name"])
        if match is None:
            raise RouteAwarenessError(
                f"YX_2024 LNPDB name does not encode A/B/C labels: {row['IL_name']!r}"
            )
        labels["head"].add(f"A{match.group('head')}")
        labels["tail1"].add(f"B{match.group('tail1')}")
        labels["tail2"].add(f"C{match.group('tail2')}")
    if any(len(values) != 1 for values in labels.values()):
        raise RouteAwarenessError("one canonical YX_2024 lipid maps to conflicting A/B/C labels")
    return {role: next(iter(values)) for role, values in labels.items()}


def _apply_agile_reference_evidence(
    component: dict[str, Any],
    *,
    source_label: str,
    source_component: dict[str, Any],
) -> dict[str, Any]:
    reference = _canonicalize(
        source_component["measured_smiles"],
        label=f"AGILE measured reference {source_label}",
    )
    observed = component["canonical_smiles"]
    if observed == reference:
        identity_match_basis = "exact_isomeric_structure"
    elif observed and _without_stereochemistry(observed) == _without_stereochemistry(reference):
        identity_match_basis = "source_label_reconciled_missing_or_incomplete_stereochemistry"
    else:
        identity_match_basis = (
            "source_label_reconciled_to_measured_reference_raw_conflict_preserved"
        )

    return {
        **component,
        "route_ids": source_component["route_ids"],
        "route_evidence_status": source_component["route_evidence_status"],
        "procurement_evidence_status": source_component["procurement_evidence_status"],
        "source_labels": [source_label],
        "forward_verification_status": (
            "not_run_missing_qualified_template"
            if source_component["route_ids"]
            else "not_assessed"
        ),
        "identity_match_basis": identity_match_basis,
        "reference_canonical_smiles": reference,
    }


def build_lnpdb_lipid_ledger(
    lnpdb_path: Path,
    component_ledger_path: Path,
    source_queue_path: Path,
    config: dict[str, Any],
    routes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach exact route evidence to every unique LNPDB lipid and component."""

    required = {
        "LNP_ID",
        "Experiment_ID",
        "IL_name",
        "IL_SMILES",
        *ROLE_FIELDS.values(),
    }
    source_rows = _read_csv(lnpdb_path, required=required, label="raw LNPDB")
    parsed_index, raw_index = _component_ledger_indices(component_ledger_path)
    source_index = _source_index(source_queue_path)
    exact_evidence = _build_exact_evidence_index(config, routes)
    component_config = _component_config_by_label(config)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row_number, row in enumerate(source_rows, start=2):
        lnp_id = row["LNP_ID"].strip()
        if not lnp_id:
            raise RouteAwarenessError(f"LNPDB row {row_number} has empty LNP_ID")
        canonical_lipid = _canonicalize(row["IL_SMILES"], label=f"LNPDB row {row_number} IL_SMILES")
        grouped[canonical_lipid].append(row)

    ledger_rows: list[dict[str, Any]] = []
    unique_component_ids: set[str] = set()
    exact_route_component_ids: set[str] = set()
    source_linked_route_component_ids: set[str] = set()
    source_reconciled_route_component_ids: set[str] = set()
    vendor_claim_component_ids: set[str] = set()
    status_counts: dict[str, int] = defaultdict(int)
    agile_status_counts: dict[str, int] = defaultdict(int)
    agile_identity_labels: dict[str, set[str]] = defaultdict(set)
    l1_linked = 0
    for canonical_lipid in sorted(grouped):
        rows = grouped[canonical_lipid]
        lipid_id = f"lnpdb-lipid-{hashlib.sha256(canonical_lipid.encode()).hexdigest()[:16]}"
        agile_labels = _agile_labels_by_role(rows)
        components: list[dict[str, Any]] = []
        agile_component_ids_for_lipid: set[str] = set()
        for role, field in ROLE_FIELDS.items():
            values = {
                row[field].strip()
                for row in rows
                if row[field].strip() not in MISSING_COMPONENT_VALUES
            }
            if not values:
                continue
            resolved = [
                _resolve_component(role, value, parsed_index, raw_index) for value in sorted(values)
            ]
            agile_values = {
                row[field].strip()
                for row in rows
                if row["Experiment_ID"] == "YX_2024"
                and row[field].strip() not in MISSING_COMPONENT_VALUES
            }
            agile_component_ids = {
                _resolve_component(role, value, parsed_index, raw_index)[0]["component_id"]
                for value in agile_values
            }
            agile_component_ids_for_lipid.update(agile_component_ids)
            unique_resolved = {
                component_row["component_id"]: (component_row, canonical)
                for component_row, canonical in resolved
            }
            for component_id in sorted(unique_resolved):
                component_row, canonical = unique_resolved[component_id]
                evidence = exact_evidence.get(
                    canonical,
                    {
                        "route_ids": [],
                        "route_evidence_status": "not_assessed",
                        "procurement_evidence_status": "not_assessed",
                        "source_labels": [],
                    },
                )
                component = {
                    "component_id": component_row["component_id"],
                    "role": role,
                    "parse_status": component_row["parse_status"],
                    "canonical_smiles": canonical or "",
                    "route_ids": evidence["route_ids"],
                    "route_evidence_status": evidence["route_evidence_status"],
                    "procurement_evidence_status": evidence["procurement_evidence_status"],
                    "source_labels": evidence["source_labels"],
                    "forward_verification_status": (
                        "not_run_missing_qualified_template"
                        if evidence["route_ids"]
                        else "not_assessed"
                    ),
                    "execution_closure_status": "unknown",
                    "identity_match_basis": (
                        "exact_isomeric_structure"
                        if canonical in exact_evidence
                        else "not_assessed"
                    ),
                    "reference_canonical_smiles": canonical or "",
                }
                if component_id in agile_component_ids:
                    source_label = agile_labels[role]
                    component = _apply_agile_reference_evidence(
                        component,
                        source_label=source_label,
                        source_component=component_config[source_label],
                    )
                    agile_identity_labels[component["identity_match_basis"]].add(source_label)
                components.append(component)
                unique_component_ids.add(component["component_id"])
                if component["route_ids"]:
                    source_linked_route_component_ids.add(component["component_id"])
                    if component["identity_match_basis"] == "exact_isomeric_structure":
                        exact_route_component_ids.add(component["component_id"])
                    else:
                        source_reconciled_route_component_ids.add(component["component_id"])
                if component["procurement_evidence_status"] == "vendor_claim_only":
                    vendor_claim_component_ids.add(component["component_id"])

        experiments = sorted({row["Experiment_ID"] for row in rows if row["Experiment_ID"]})
        source_ids = sorted(
            {
                source_id
                for experiment in experiments
                for source_id in source_index.get(experiment, [])
            }
        )
        l1_route_ids = [config["assembly"]["route_id"]] if "YX_2024" in experiments else []
        if l1_route_ids:
            l1_linked += 1
        l2_route_ids = sorted(
            {route_id for component in components for route_id in component["route_ids"]}
        )
        status = _route_awareness_status(components)
        status_counts[status] += 1
        agile_components = [
            component
            for component in components
            if component["component_id"] in agile_component_ids_for_lipid
        ]
        agile_status = _route_awareness_status(agile_components) if agile_labels else ""
        if agile_status:
            agile_status_counts[agile_status] += 1
        route_count = sum(
            component["route_evidence_status"] == "route_extracted" for component in components
        )
        vendor_count = sum(
            component["procurement_evidence_status"] == "vendor_claim_only"
            for component in components
        )
        unresolved_count = sum(
            component["route_evidence_status"] == "source_discrepancy_unresolved"
            for component in components
        )
        unassessed_count = sum(
            component["route_evidence_status"] == "not_assessed" for component in components
        )
        ledger_rows.append(
            {
                "lipid_id": lipid_id,
                "canonical_smiles": canonical_lipid,
                "lnp_ids_json": _stable_json(
                    sorted({row["LNP_ID"] for row in rows if row["LNP_ID"]})
                ),
                "lipid_names_json": _stable_json(
                    sorted({row["IL_name"] for row in rows if row["IL_name"]})
                ),
                "experiment_ids_json": _stable_json(experiments),
                "source_ids_json": _stable_json(source_ids),
                "component_ids_json": _stable_json(
                    [component["component_id"] for component in components]
                ),
                "component_evidence_json": _stable_json(components),
                "l1_route_ids_json": _stable_json(l1_route_ids),
                "l2_route_ids_json": _stable_json(l2_route_ids),
                "annotated_component_count": len(components),
                "route_extracted_component_count": route_count,
                "vendor_claim_component_count": vendor_count,
                "unresolved_component_count": unresolved_count,
                "unassessed_component_count": unassessed_count,
                "route_awareness_status": status,
                "agile_reference_route_status": agile_status,
                "procurement_closure_status": "unknown",
                "execution_closure_status": "unknown",
            }
        )

    if len(ledger_rows) != 12837:
        raise RouteAwarenessError(f"expected 12837 unique LNPDB lipids, found {len(ledger_rows)}")
    if any(row["execution_closure_status"] != "unknown" for row in ledger_rows):
        raise RouteAwarenessError("route awareness must not imply whole-lipid execution closure")

    return ledger_rows, {
        "lnpdb_records": len(source_rows),
        "unique_lipids": len(ledger_rows),
        "unique_annotated_components": len(unique_component_ids),
        "components_with_exact_extracted_route": len(exact_route_component_ids),
        "components_with_source_linked_extracted_route": len(source_linked_route_component_ids),
        "components_with_source_reconciled_route": len(source_reconciled_route_component_ids),
        "components_with_historical_vendor_claim": len(vendor_claim_component_ids),
        "lipids_with_source_linked_l1_assembly": l1_linked,
        "route_awareness_status_counts": {key: status_counts[key] for key in sorted(status_counts)},
        "agile_reference_route_status_counts": {
            key: agile_status_counts[key] for key in sorted(agile_status_counts)
        },
        "agile_reference_identity_labels": {
            key: sorted(agile_identity_labels[key]) for key in sorted(agile_identity_labels)
        },
        "forward_verified_routes": 0,
        "procurement_closed_components": 0,
        "execution_closed_lipids": 0,
    }


def _component_catalog(config: dict[str, Any]) -> list[dict[str, Any]]:
    components = _component_config_by_label(config)
    catalog = []
    for label, entry in sorted(components.items(), key=lambda item: (item[0][0], int(item[0][1:]))):
        catalog.append(
            {
                "label": label,
                "component_class": entry["component_class"],
                "measured_smiles": entry["measured_smiles"],
                "canonical_smiles": _canonicalize(
                    entry["measured_smiles"], label=f"component {label}"
                ),
                "route_ids": entry["route_ids"],
                "route_evidence_status": entry["route_evidence_status"],
                "procurement_evidence_status": entry["procurement_evidence_status"],
                "forward_verification_status": (
                    "not_run_missing_qualified_template" if entry["route_ids"] else "not_assessed"
                ),
                "execution_closure_status": "unknown",
                "qa_reason": entry.get("reason", ""),
            }
        )
    return catalog


def build_route_awareness(
    config_path: Path,
    vendor_dir: Path,
    m0_results_dir: Path,
    *,
    generated_utc: str | None = None,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Build route records and the all-LNPDB route-awareness ledger."""

    config = load_route_config(config_path)
    inputs, paths = validate_route_inputs(config_path, config, vendor_dir, m0_results_dir)
    measured_summary, _ = analyze_agile_measured(paths["agile_measured"], config)
    routes = build_route_catalog(config)
    lipid_rows, lnpdb_summary = build_lnpdb_lipid_ledger(
        paths["lnpdb"],
        paths["component_source_ledger"],
        paths["source_queue"],
        config,
        routes,
    )

    if generated_utc is None:
        generated = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    else:
        try:
            parsed = dt.datetime.fromisoformat(generated_utc)
        except ValueError as exc:
            raise RouteAwarenessError(
                f"generated_utc is not valid ISO-8601: {generated_utc!r}"
            ) from exc
        if parsed.tzinfo is None:
            raise RouteAwarenessError("generated_utc must include a timezone")
        generated = parsed.isoformat()

    route_artifact = {
        "schema_version": ROUTE_ARTIFACT_SCHEMA_VERSION,
        "generated_utc": generated,
        "source": config["source"],
        "assembly": config["assembly"],
        "evidence_contract": {
            "route_extracted": "A primary source route and exact target identity were curated.",
            "forward_verified": "Not established by source extraction or structural plausibility.",
            "procurement_closed": "Requires time-stamped exact-identity vendor or internal stock evidence.",
            "execution_closed": "Requires forward-verified route steps and procurement-closed leaves.",
        },
        "measured_library": measured_summary,
        "components": _component_catalog(config),
        "routes": routes,
        "qa_flags": [
            config["unresolved_aldehyde_ester_components"][0],
            {
                "issue": "SI heading says 6-hydroxyphenyl undec-10-enoate",
                "disposition": "Treated as a document typo because the shared route, measured structure, and NMR describe the hexyl product.",
            },
        ],
    }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": "M0-09 L2 supervision inventory addendum",
        "generated_utc": generated,
        "randomness": {"seed": seed, "used": False},
        "inputs": inputs,
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "summary": {
            "agile_measured": measured_summary,
            "route_records": len(routes),
            "lnpdb": lnpdb_summary,
        },
        "decision": {
            "map_every_lnpdb_lipid": True,
            "generator_integration": "Exact component identities can retrieve zero or more source-linked route DAGs. Route proposal and ranking remain downstream work.",
            "whole_graph_generation_preserved": True,
            "model_built": False,
            "reason": "M0-09 inventories route evidence. It does not authorize the L2 model.",
        },
        "limitations": [
            "An empty route list means not yet curated, not unsynthesizable.",
            "Only exact component identity propagates route evidence in this artifact.",
            "Source extraction does not establish forward success on a new substrate.",
            "Vendor claims without exact catalog and time-stamped availability do not close procurement.",
            "The ledger represents known alternatives and explicit unknowns, not every chemically possible route.",
        ],
        "outputs": {},
    }
    return result, route_artifact, lipid_rows


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode())


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _csv_text(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=LIPID_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _bytes_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _deterministic_gzip(content: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(content)
    return buffer.getvalue()


def write_route_awareness(
    result: dict[str, Any],
    route_artifact: dict[str, Any],
    lipid_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Atomically write route records, every-lipid ledger, and hash-linked result."""

    route_path = output_dir / "agile_component_routes.json"
    lipid_path = output_dir / "lnpdb_lipid_route_ledger.csv.gz"
    result_path = output_dir / "route_awareness_result.json"

    route_text = json.dumps(route_artifact, indent=2, sort_keys=True) + "\n"
    lipid_text = _csv_text(lipid_rows)
    lipid_gzip = _deterministic_gzip(lipid_text.encode())
    _atomic_write_text(route_path, route_text)
    _atomic_write_bytes(lipid_path, lipid_gzip)

    result["outputs"] = {
        "agile_component_routes": {
            "asset": route_path.name,
            "records": len(route_artifact["routes"]),
            "sha256": _bytes_sha256(route_text.encode()),
        },
        "lnpdb_lipid_route_ledger": {
            "asset": lipid_path.name,
            "records": len(lipid_rows),
            "sha256": _bytes_sha256(lipid_gzip),
        },
    }
    result_text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(result_path, result_text)
    return result
