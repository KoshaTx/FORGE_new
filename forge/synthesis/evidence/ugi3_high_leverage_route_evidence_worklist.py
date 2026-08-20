"""Build a frozen development-only worklist for exact Ugi route evidence.

The worklist is a nonpromoting prioritization view over two already completed
development audits.  It neither plans routes nor reads the sealed saturation
holdout.  Projected program families and leaves are preserved as hypotheses
requiring the exact evidence stated in each worklist record.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = "phase1_ugi3_high_leverage_route_evidence_worklist_config.v1"
MANIFEST_SCHEMA_VERSION = "phase1_ugi3_high_leverage_route_evidence_worklist.v1"
WORKLIST_SCHEMA_VERSION = "phase1_ugi3_high_leverage_route_evidence_component.v1"

_EXPECTED_INPUT_PATHS = {
    "priority_result": "results/phase1/ugi3_fresh_pool_route_priority_v4/result.json",
    "priority_ledger": ("results/phase1/ugi3_fresh_pool_route_priority_v4/priority_ledger.csv.gz"),
    "triage_result": "results/phase1/ugi3_route_gap_triage_v5/result.json",
    "triage_ledger": "results/phase1/ugi3_route_gap_triage_v5/triage_ledger.csv.gz",
}
_EXPECTED_IMPLEMENTATION_PATHS = {
    "builder_source": "src/forge/route/ugi3_high_leverage_route_evidence_worklist.py",
    "runner": "scripts/phase1_build_ugi3_high_leverage_route_evidence_worklist.py",
    "tests": "tests/test_ugi3_high_leverage_route_evidence_worklist.py",
}
_EXISTING_PROGRAMS = frozenset(
    {
        "fatty_acid_diol_esterification_then_alcohol_oxidation",
        "primary_alcohol_oxidation",
        "primary_amine_formylation_then_formamide_dehydration",
    }
)
_HEAD_TRIAGE_CLASS = "head_procurement_or_exact_upstream_route_missing"


class RouteEvidenceWorklistError(ValueError):
    """Raised when an input or worklist invariant is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RouteEvidenceWorklistError(f"{label} could not be loaded") from error
    if not isinstance(value, dict):
        raise RouteEvidenceWorklistError(f"{label} must be a JSON object")
    return value


def _load_gzip_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        raise RouteEvidenceWorklistError(f"{label} could not be loaded") from error


def _require_frozen_inputs(
    config: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Path]:
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(_EXPECTED_INPUT_PATHS):
        raise RouteEvidenceWorklistError("config must contain exactly the four development inputs")
    paths: dict[str, Path] = {}
    for name, expected_relative in _EXPECTED_INPUT_PATHS.items():
        record = inputs.get(name)
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise RouteEvidenceWorklistError(f"{name} input record is malformed")
        relative = record.get("path")
        expected_hash = record.get("sha256")
        if not isinstance(relative, str) or relative != expected_relative:
            raise RouteEvidenceWorklistError(
                f"{name} must use the frozen development path {expected_relative!r}"
            )
        if "holdout" in relative.casefold() or "seal" in relative.casefold():
            raise RouteEvidenceWorklistError("sealed holdout paths are forbidden")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise RouteEvidenceWorklistError(f"{name} SHA-256 is malformed")
        path = repo_root / relative
        if _sha256(path) != expected_hash:
            raise RouteEvidenceWorklistError(f"{name} disagrees with its frozen SHA-256")
        paths[name] = path
    return paths


def _require_frozen_implementation(
    config: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Path]:
    implementation = config.get("implementation")
    if not isinstance(implementation, dict) or set(implementation) != set(
        _EXPECTED_IMPLEMENTATION_PATHS
    ):
        raise RouteEvidenceWorklistError("config implementation manifest is malformed")
    paths: dict[str, Path] = {}
    for name, expected_relative in _EXPECTED_IMPLEMENTATION_PATHS.items():
        record = implementation[name]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise RouteEvidenceWorklistError(f"{name} implementation record is malformed")
        if record.get("path") != expected_relative:
            raise RouteEvidenceWorklistError(f"{name} implementation path changed")
        path = repo_root / expected_relative
        if _sha256(path) != record.get("sha256"):
            raise RouteEvidenceWorklistError(f"{name} implementation SHA-256 changed")
        paths[name] = path
    return paths


def _parse_json_list(raw: str, *, label: str) -> list[str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RouteEvidenceWorklistError(f"{label} is not valid JSON") from error
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise RouteEvidenceWorklistError(f"{label} must be a list of nonempty strings")
    return value


def _missing_evidence(
    row: dict[str, str], leaves: list[str], unresolved: list[str]
) -> list[dict[str, Any]]:
    triage_class = row["triage_class"]
    program = row["program_family"] or None
    target = row["canonical_smiles"]
    if triage_class == _HEAD_TRIAGE_CLASS:
        if row["role"] != "amine_head" or program is not None or leaves:
            raise RouteEvidenceWorklistError("head procurement lane has inconsistent program data")
        return [
            {
                "evidence_type": "current_exact_procurement_or_stocked_terminal",
                "target_smiles": target,
                "state": "not_established_by_development_audits",
            },
            {
                "evidence_type": "exact_upstream_route_if_not_currently_procured",
                "target_smiles": target,
                "state": "not_established_by_development_audits",
            },
        ]
    if program not in _EXISTING_PROGRAMS:
        raise RouteEvidenceWorklistError("selected projected component lacks an existing program")
    missing: list[dict[str, Any]] = []
    if unresolved:
        missing.append(
            {
                "evidence_type": "current_terminal_identity_or_procurement",
                "target_smiles": unresolved,
                "state": "unresolved_projected_leaves",
            }
        )
    missing.append(
        {
            "evidence_type": "exact_substrate_execution_or_bounded_scope_qualification",
            "target_smiles": target,
            "existing_program": program,
            "state": "not_established_by_development_audits",
        }
    )
    return missing


def _proposal_search_policy(row: dict[str, str], unresolved: list[str]) -> dict[str, Any]:
    triage_class = row["triage_class"]
    if triage_class == "projected_family_all_leaves_current_exact_scope_missing":
        return {
            "could_help": False,
            "scope": "exact_evidence_retrieval_or_scope_qualification_is_the_bottleneck",
            "can_supply_evidence": False,
            "can_close_route": False,
            "can_admit_reaction_family": False,
        }
    if triage_class == _HEAD_TRIAGE_CLASS:
        return {
            "could_help": True,
            "scope": "proposal_only_after_current_procurement_and_exact_route_search",
            "can_supply_evidence": False,
            "can_close_route": False,
            "can_admit_reaction_family": False,
        }
    if unresolved:
        return {
            "could_help": True,
            "scope": "proposal_only_for_unresolved_leaves_within_bounded_search",
            "can_supply_evidence": False,
            "can_close_route": False,
            "can_admit_reaction_family": False,
        }
    raise RouteEvidenceWorklistError("proposal-search policy lacks a supported triage state")


def build_high_leverage_route_evidence_worklist(
    config_path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Return the frozen top-prefix worklist without reading any holdout artifact."""

    config = _load_json(config_path, label="worklist config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise RouteEvidenceWorklistError("worklist config schema is unsupported")
    if config.get("development_only") is not True:
        raise RouteEvidenceWorklistError("worklist config must be development-only")
    paths = _require_frozen_inputs(config, repo_root=repo_root)
    implementation_paths = _require_frozen_implementation(config, repo_root=repo_root)
    priority_result = _load_json(paths["priority_result"], label="priority result")
    triage_result = _load_json(paths["triage_result"], label="triage result")
    priority_rows = _load_gzip_csv(paths["priority_ledger"], label="priority ledger")
    triage_rows = _load_gzip_csv(paths["triage_ledger"], label="triage ledger")

    summary = priority_result.get("summary")
    if not isinstance(summary, dict):
        raise RouteEvidenceWorklistError("priority summary is missing")
    missing_products = summary.get("missing_knowledge_one_gap_products")
    audit_component_count = summary.get("greedy_missing_component_counts_for_coverage", {}).get(
        "25pct"
    )
    if missing_products != 802 or audit_component_count != 16:
        raise RouteEvidenceWorklistError("priority audit no longer has the frozen 802/16 frontier")
    if triage_result.get("decision", {}).get("holdout_remains_unrevealed") is not True:
        raise RouteEvidenceWorklistError("triage result does not retain the sealed holdout")

    priority_by_key = {(row["role"], row["canonical_smiles"]): row for row in priority_rows}
    triage_by_key = {(row["role"], row["canonical_smiles"]): row for row in triage_rows}
    if len(priority_by_key) != len(priority_rows) or set(priority_by_key) != set(triage_by_key):
        raise RouteEvidenceWorklistError("priority and triage ledgers do not align one-to-one")

    eligible = sorted(
        (row for row in priority_rows if row["assessment_outcome"] == "missing_knowledge"),
        key=lambda row: int(row["priority_rank"]),
    )
    target_unlock = math.ceil(missing_products * 0.25)
    selected: list[dict[str, str]] = []
    cumulative = 0
    for row in eligible:
        selected.append(row)
        cumulative += int(row["one_gap_product_count"])
        if cumulative >= target_unlock:
            break
    if len(selected) != audit_component_count or cumulative != 207:
        raise RouteEvidenceWorklistError("recomputed 25% worklist differs from the frozen audit")
    prior_unlock = cumulative - int(selected[-1]["one_gap_product_count"])
    if prior_unlock >= target_unlock:
        raise RouteEvidenceWorklistError("selected component prefix is not cardinality-minimal")

    program_only = [
        row
        for row in eligible
        if triage_by_key[(row["role"], row["canonical_smiles"])]["program_family"]
        in _EXISTING_PROGRAMS
    ]
    strict_program_count = 0
    strict_program_unlock = 0
    strict_program_prior_unlock = 0
    for row in program_only:
        strict_program_prior_unlock = strict_program_unlock
        strict_program_unlock += int(row["one_gap_product_count"])
        strict_program_count += 1
        if strict_program_unlock >= target_unlock:
            break
    if (strict_program_count, strict_program_prior_unlock, strict_program_unlock) != (
        18,
        198,
        204,
    ):
        raise RouteEvidenceWorklistError(
            "strict existing-program-only sensitivity differs from the frozen audit"
        )

    records: list[dict[str, Any]] = []
    groups: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for priority in selected:
        key = (priority["role"], priority["canonical_smiles"])
        triage = triage_by_key[key]
        for field in (
            "priority_rank",
            "assessment_outcome",
            "registry_component_id",
            "one_gap_product_count",
        ):
            if priority[field] != triage[field]:
                raise RouteEvidenceWorklistError(f"joined ledgers disagree on {field} for {key}")
        if triage["new_reaction_family_indicated_now"] != "False":
            raise RouteEvidenceWorklistError("selected worklist would admit a new reaction family")
        leaves = _parse_json_list(
            triage["projected_leaves_json"],
            label=f"{key} projected leaves",
        )
        unresolved = _parse_json_list(
            triage["unresolved_projected_leaves_json"],
            label=f"{key} unresolved leaves",
        )
        if not set(unresolved).issubset(leaves):
            raise RouteEvidenceWorklistError("unresolved projected leaves are not required leaves")
        program = triage["program_family"] or None
        record = {
            "schema_version": WORKLIST_SCHEMA_VERSION,
            "worklist_rank": len(records) + 1,
            "source_priority_rank": int(priority["priority_rank"]),
            "role": priority["role"],
            "canonical_smiles": priority["canonical_smiles"],
            "registry_component_id": priority["registry_component_id"] or None,
            "occurrences": int(priority["one_gap_product_count"]),
            "potential_unlock_count": int(priority["one_gap_product_count"]),
            "potential_unlock_semantics": (
                "one-gap development products whose sole noncomplete component is this identity; "
                "not a predicted experimental success count"
            ),
            "existing_upstream_program": program,
            "required_leaves": leaves,
            "current_evidence_state": {
                "assessment_outcome": priority["assessment_outcome"],
                "target_current_terminal": triage["target_current_terminal"] == "True",
                "triage_class": triage["triage_class"],
                "projected_leaf_count": int(triage["projected_leaf_count"]),
                "current_projected_leaf_count": int(triage["current_projected_leaf_count"]),
                "historical_projected_leaf_count": int(triage["historical_projected_leaf_count"]),
                "unresolved_projected_leaves": unresolved,
                "program_is_projection_not_exact_route": program is not None,
            },
            "precise_missing_evidence": _missing_evidence(triage, leaves, unresolved),
            "recommended_action": triage["recommended_action"],
            "learned_proposal_search": _proposal_search_policy(triage, unresolved),
            "route_asserted": False,
            "procurement_asserted": False,
            "new_reaction_family_admitted": False,
        }
        records.append(record)
        groups[(record["role"], program)].append(record)

    grouped = []
    for (role, program), components in sorted(
        groups.items(),
        key=lambda item: (item[0][0], item[0][1] or ""),
    ):
        grouped.append(
            {
                "role": role,
                "existing_upstream_program": program,
                "component_count": len(components),
                "potential_unlock_count": sum(
                    item["potential_unlock_count"] for item in components
                ),
                "component_worklist_ranks": [item["worklist_rank"] for item in components],
            }
        )

    input_manifest = {
        name: {"path": str(path.relative_to(repo_root)), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "frozen_development_only_nonpromoting_evidence_worklist",
        "development_only": True,
        "sealed_holdout_accessed": False,
        "config": {
            "path": str(config_path.relative_to(repo_root)),
            "sha256": _sha256(config_path),
        },
        "inputs": input_manifest,
        "implementation": {
            name: {"path": str(path.relative_to(repo_root)), "sha256": _sha256(path)}
            for name, path in implementation_paths.items()
        },
        "selection": {
            "missing_knowledge_one_gap_products": missing_products,
            "target_fraction": 0.25,
            "target_unlock_count": target_unlock,
            "selected_component_count": len(records),
            "prior_prefix_unlock_count": prior_unlock,
            "potential_unlock_count": cumulative,
            "potential_unlock_fraction": cumulative / missing_products,
            "components_with_existing_upstream_program": sum(
                record["existing_upstream_program"] is not None for record in records
            ),
            "heads_without_existing_program": sum(
                record["existing_upstream_program"] is None for record in records
            ),
            "strict_existing_program_only_sensitivity": {
                "selected_component_count": strict_program_count,
                "prior_prefix_unlock_count": strict_program_prior_unlock,
                "potential_unlock_count": strict_program_unlock,
                "potential_unlock_fraction": strict_program_unlock / missing_products,
                "interpretation": (
                    "excludes the three high-priority heads whose exact upstream program "
                    "is not established"
                ),
            },
        },
        "groups": grouped,
        "components": records,
        "nonclaims": [
            "A projected program is not an exact route for the selected substrate.",
            "Current projected leaves do not establish exact substrate scope.",
            "A missing procurement record is not evidence of unavailability.",
            "Learned proposals are not route evidence and cannot close a route.",
            "This worklist does not admit a reaction family or alter candidate selection.",
        ],
    }


def write_worklist_manifest(value: dict[str, Any], path: Path) -> None:
    """Write one canonical JSON manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "RouteEvidenceWorklistError",
    "WORKLIST_SCHEMA_VERSION",
    "build_high_leverage_route_evidence_worklist",
    "write_worklist_manifest",
]
