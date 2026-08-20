"""Compose Ugi route evidence and terminal closure without overclaiming.

This audit deliberately separates source/program evidence and current terminal
procurement from deterministic reaction-step forward verification.  A product
can have every upstream branch evidence- and terminal-closed while still not
qualifying as a complete forward-verified synthesis dossier.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_bytes, sha256_file
from forge.core.io import csv_gz_bytes as _csv_bytes
from forge.core.io import read_json_object

CONFIG_SCHEMA_VERSION = "phase1_ugi3_dossier_coverage_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_dossier_coverage.v1"
COMPONENT_FIELDS = (
    "component_id",
    "role",
    "canonical_smiles",
    "program_status",
    "evidence_level",
    "structural_program_status",
    "proposed_leaves_json",
    "current_closed_leaves_json",
    "unresolved_leaves_json",
    "component_dossier_status",
    "step_forward_verification_status",
    "next_gap",
)
PRODUCT_FIELDS = (
    "source_row_index",
    "canonical_product_smiles",
    "amine_head_status",
    "aldehyde_status",
    "isocyanide_status",
    "upstream_evidence_tier",
    "l1_exact_graph_reconstruction",
    "complete_forward_verified_dossier",
    "next_gap",
)
ROLE_TO_FIELD = {
    "amine_head": "amine_head_status",
    "oxoester_aldehyde_body_tail": "aldehyde_status",
    "isocyanide_tail": "isocyanide_status",
}


class Ugi3DossierCoverageError(ValueError):
    """Raised when dossier inputs violate their frozen contract."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=Ugi3DossierCoverageError, label=label)


def _read_gzip_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise Ugi3DossierCoverageError(f"{label} has no header")
            return list(reader)
    except (OSError, csv.Error) as exc:
        raise Ugi3DossierCoverageError(f"could not read {label}") from exc


def _verify_hash(path: Path, expected: Any, *, label: str) -> None:
    observed = sha256_file(path)
    if not isinstance(expected, str) or observed != expected:
        raise Ugi3DossierCoverageError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )


def _json_list(value: str, *, label: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise Ugi3DossierCoverageError(f"{label} is invalid JSON") from exc
    if not isinstance(parsed, list):
        raise Ugi3DossierCoverageError(f"{label} must be a list")
    return parsed


def _component_status(
    row: Mapping[str, str], closed_terminals: set[str]
) -> tuple[str, list[str], list[str]]:
    leaves = _json_list(
        row["proposed_leaf_candidates_json"],
        label=f"{row['component_id']} proposed leaves",
    )
    if any(not isinstance(leaf, str) for leaf in leaves):
        raise Ugi3DossierCoverageError("component leaf must be a string")
    closed = sorted(leaf for leaf in leaves if leaf in closed_terminals)
    unresolved = sorted(set(leaves) - closed_terminals)
    if row["route_closure"] == "accepted_terminal":
        return "accepted_terminal", leaves, []
    if unresolved:
        return "incomplete", closed, unresolved
    if row["program_status"] == "exact_source_program":
        return "exact_source_leaf_closed", closed, []
    if row["program_status"] == "reaction_family_projected_program":
        return "family_projected_leaf_closed", closed, []
    return "incomplete", closed, unresolved or leaves


def _validate_expected(observed: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise Ugi3DossierCoverageError(f"{label} fields mismatch")
        for key, value in expected.items():
            _validate_expected(observed[key], value, label=f"{label}.{key}")
    elif observed != expected:
        raise Ugi3DossierCoverageError(
            f"{label} mismatch: expected {expected!r}, observed {observed!r}"
        )


def build_ugi3_dossier_coverage(
    config_path: Path,
    component_program_ledger_path: Path,
    product_ledger_path: Path,
    terminal_procurement_path: Path,
) -> tuple[dict[str, Any], bytes, bytes]:
    """Return the conservative dossier census and deterministic ledgers."""

    config = _load_json(config_path, label="dossier coverage config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3DossierCoverageError("unsupported config schema")
    paths = {
        "component_program_ledger": component_program_ledger_path,
        "product_ledger": product_ledger_path,
        "terminal_procurement": terminal_procurement_path,
    }
    for name, path in paths.items():
        _verify_hash(
            path,
            config["inputs"][name]["expected_sha256"],
            label=name,
        )

    procurement = _load_json(terminal_procurement_path, label="terminal procurement")
    records = procurement.get("records")
    if not isinstance(records, list):
        raise Ugi3DossierCoverageError("procurement records must be a list")
    closed_terminals = {
        record["canonical_smiles"]
        for record in records
        if isinstance(record, dict) and record.get("current_item_level_procurement_closed") is True
    }

    program_rows = _read_gzip_csv(component_program_ledger_path, label="component program ledger")
    component_rows: list[dict[str, Any]] = []
    component_status_by_role_smiles: dict[tuple[str, str], str] = {}
    component_counts: Counter[str] = Counter()
    for row in program_rows:
        status, closed, unresolved = _component_status(row, closed_terminals)
        key = (row["role"], row["canonical_smiles"])
        if key in component_status_by_role_smiles:
            raise Ugi3DossierCoverageError("duplicate role-component program")
        component_status_by_role_smiles[key] = status
        component_counts[status] += 1
        component_rows.append(
            {
                "component_id": row["component_id"],
                "role": row["role"],
                "canonical_smiles": row["canonical_smiles"],
                "program_status": row["program_status"],
                "evidence_level": row["evidence_level"],
                "structural_program_status": row["structural_program_status"],
                "proposed_leaves_json": row["proposed_leaf_candidates_json"],
                "current_closed_leaves_json": json.dumps(closed, separators=(",", ":")),
                "unresolved_leaves_json": json.dumps(unresolved, separators=(",", ":")),
                "component_dossier_status": status,
                "step_forward_verification_status": (
                    "not_applicable" if status == "accepted_terminal" else "not_run"
                ),
                "next_gap": (
                    "none"
                    if status == "accepted_terminal"
                    else (
                        "forward_verify_program_steps"
                        if status != "incomplete"
                        else "resolve_route_or_terminal"
                    )
                ),
            }
        )

    product_rows_raw = _read_gzip_csv(product_ledger_path, label="product ledger")
    product_rows: list[dict[str, Any]] = []
    product_counts: Counter[str] = Counter()
    exact_allowed = {"accepted_terminal", "exact_source_leaf_closed"}
    family_allowed = exact_allowed | {"family_projected_leaf_closed"}
    for row in product_rows_raw:
        candidates = _json_list(
            row["candidate_routes_json"],
            label=f"product {row['source_row_index']} candidates",
        )
        if len(candidates) != 1 or not isinstance(candidates[0], dict):
            raise Ugi3DossierCoverageError("product must contain one exact candidate decomposition")
        candidate = candidates[0]
        components = candidate.get("components")
        if not isinstance(components, dict) or set(components) != set(ROLE_TO_FIELD):
            raise Ugi3DossierCoverageError("product component roles mismatch")
        statuses = {
            role: component_status_by_role_smiles[(role, smiles)]
            for role, smiles in components.items()
        }
        if all(status in exact_allowed for status in statuses.values()):
            tier = "exact_source_upstream_closed"
        elif all(status in family_allowed for status in statuses.values()):
            tier = "family_supported_upstream_closed"
        else:
            tier = "incomplete"
        l1_exact = (
            row["decomposition_status"] == "one_exact_qualified_ugi_decomposition"
            and row["candidate_count"] == "1"
            and isinstance(candidate.get("target_matching_forward_outcomes"), int)
            and candidate["target_matching_forward_outcomes"] >= 1
        )
        if not l1_exact:
            raise Ugi3DossierCoverageError("product lacks exact L1 round trip")
        product_counts[tier] += 1
        output = {
            "source_row_index": row["source_row_index"],
            "canonical_product_smiles": row["canonical_product_smiles"],
            "upstream_evidence_tier": tier,
            "l1_exact_graph_reconstruction": "true",
            "complete_forward_verified_dossier": "false",
            "next_gap": (
                "forward_verify_each_upstream_step"
                if tier != "incomplete"
                else "resolve_upstream_component_or_terminal"
            ),
        }
        for role, field in ROLE_TO_FIELD.items():
            output[field] = statuses[role]
        product_rows.append(output)

    summary = {
        "components": len(component_rows),
        "components_by_status": dict(sorted(component_counts.items())),
        "products": len(product_rows),
        "products_by_upstream_tier": dict(sorted(product_counts.items())),
        "complete_forward_verified_dossiers": 0,
    }
    _validate_expected(summary, config["expected_counts"], label="summary")
    component_bytes = _csv_bytes(component_rows, COMPONENT_FIELDS)
    product_bytes = _csv_bytes(product_rows, PRODUCT_FIELDS)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": config["task"],
        "inputs": {
            "config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            **{
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
        },
        "policy": config["policy"],
        "summary": summary,
        "claims_boundary": {
            "source_exact_leaf_closure_is_step_forward_verification": False,
            "family_projection_is_exact_route_evidence": False,
            "l1_graph_round_trip_is_experimental_success": False,
            "complete_forward_verified_dossier_claimed": False,
        },
        "artifacts": {
            "component_ledger_sha256": sha256_bytes(component_bytes),
            "product_ledger_sha256": sha256_bytes(product_bytes),
        },
    }
    return result, component_bytes, product_bytes
