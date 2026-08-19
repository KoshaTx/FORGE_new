"""Prioritize bounded L2 evidence gaps in the frozen generated Ugi sample.

This module composes exact component identity, frozen route-readiness evidence,
and graph-derived component architecture.  It does not plan routes and never
promotes structural similarity, provenance, or chemotype membership into route
evidence.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.core.hashing import sha256_bytes, sha256_file
from forge.core.io import csv_gz_bytes as _csv_bytes
from forge.product.ugi_tail_chemotype_audit import (
    ARCHITECTURE_FIELDS,
    architecture_signature,
    component_chemotype_metrics,
)
from forge.route.ugi3_production_registry_route_readiness import (
    ACCEPTED_TERMINAL,
    EXACT_CLOSED,
    EXACT_OPEN,
    FAMILY_ONLY,
    MISSING_KNOWLEDGE,
    OUTSIDE_SUPPORT,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi3_l2_coverage_priority_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_l2_coverage_priority.v1"

ROLE_NAMES = (
    "amine_head",
    "oxoester_aldehyde_body_tail",
    "isocyanide_tail",
)
ORIGINAL = "original_ugi_component"
KNOWN = "admitted_transferred_or_expanded_known_component"
ABSENT = "graph_absent_from_frozen_424_component_catalog"
PROVENANCE_STRATA = (ORIGINAL, KNOWN, ABSENT)

NOT_ASSESSED_TIER = "not_assessed_missing_route_knowledge"
ROUTE_COMPLETE = "route_complete"
EXACT_ROUTE_L3_OPEN = "exact_route_present_l3_open"
EXACT_EVIDENCE_MISSING = "exact_substrate_route_evidence_missing"
EXACT_ROUTE_UNREPORTED = "exact_substrate_route_unreported"
ROUTE_NOT_ASSESSED = "route_evidence_not_assessed"
CATALOG_ABSENT_NOT_ASSESSED = "catalog_absent_route_evidence_not_assessed"

TIER_TO_WORK_STATE = {
    ACCEPTED_TERMINAL: ROUTE_COMPLETE,
    EXACT_CLOSED: ROUTE_COMPLETE,
    EXACT_OPEN: EXACT_ROUTE_L3_OPEN,
    FAMILY_ONLY: EXACT_EVIDENCE_MISSING,
    OUTSIDE_SUPPORT: EXACT_ROUTE_UNREPORTED,
    MISSING_KNOWLEDGE: ROUTE_NOT_ASSESSED,
    NOT_ASSESSED_TIER: CATALOG_ABSENT_NOT_ASSESSED,
}
NONCOMPLETE_STATES = {
    EXACT_ROUTE_L3_OPEN,
    EXACT_EVIDENCE_MISSING,
    EXACT_ROUTE_UNREPORTED,
    ROUTE_NOT_ASSESSED,
    CATALOG_ABSENT_NOT_ASSESSED,
}

DOSSIER_TO_READINESS = {
    "accepted_terminal_l3_closed": ACCEPTED_TERMINAL,
    "exact_source_l2_verified_l3_closed": EXACT_CLOSED,
    "exact_source_l2_verified_l3_open": EXACT_OPEN,
    "family_projected_l3_closed_not_exact_source": FAMILY_ONLY,
    "unresolved_component": OUTSIDE_SUPPORT,
}

COMPONENT_PRIORITY_FIELDS = (
    "priority_rank",
    "role",
    "canonical_smiles",
    "chemotype_id",
    "generated_occurrences",
    "generated_products",
    "single_gap_products_guaranteed_unlocked",
    "structural_provenance_stratum",
    "catalog_provenance_substratum",
    "catalog_membership",
    "static_route_readiness_tier",
    "static_route_evidence_category",
    "route_work_state",
    "gap_class",
    "repeated_component",
    "chemical_incompatibility_evidenced",
)
CHEMOTYPE_PRIORITY_FIELDS = (
    "global_priority_rank",
    "role_priority_rank",
    "role",
    "chemotype_id",
    "architecture_signature",
    "architecture_fields_json",
    "missing_exact_components",
    "repeated_missing_exact_components",
    "generated_gap_occurrences",
    "generated_products_touched",
    "single_gap_products",
    "single_gap_products_per_missing_component",
    "admitted_registry_components_same_chemotype",
    "route_complete_registry_components_same_chemotype",
    "noncomplete_registry_components_same_chemotype",
    "structural_analogy_is_route_evidence",
)
PRODUCT_GAP_FIELDS = (
    "sample_index",
    "structure_id",
    "product_id",
    "product_structural_provenance_stratum",
    "missing_component_count",
    "route_complete_component_count",
    "gap_component_keys_json",
    "gap_work_states_json",
    "baseline_static_route_complete",
    "single_gap_product",
    "chemical_incompatibility_evidenced",
)


class UgiL2CoveragePriorityError(ValueError):
    """Raised when a frozen coverage-priority input violates its contract."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise UgiL2CoveragePriorityError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise UgiL2CoveragePriorityError(f"{label} must be a JSON object")
    return value


def _read_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise UgiL2CoveragePriorityError(f"{label} has no header")
            return list(reader)
    except (OSError, csv.Error) as exc:
        raise UgiL2CoveragePriorityError(f"could not read {label}: {path}") from exc


def _verify_hash(path: Path, expected: Any, *, label: str) -> None:
    observed = sha256_file(path)
    if not isinstance(expected, str) or len(expected) != 64 or observed != expected:
        raise UgiL2CoveragePriorityError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )


def _json_string_list(value: str, *, label: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise UgiL2CoveragePriorityError(f"{label} is invalid JSON") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise UgiL2CoveragePriorityError(f"{label} must be a list of strings")
    return parsed


def _portable(path: Path, *, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _component_key(role: str, smiles: str) -> str:
    return f"{role}\t{smiles}"


def _canonical_constitution(smiles: str, *, label: str) -> str:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise UgiL2CoveragePriorityError(f"{label} must be one valid connected graph")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _chemotype(role: str, smiles: str) -> tuple[str, str]:
    signature = architecture_signature(component_chemotype_metrics(smiles))
    digest = hashlib.sha256(f"{role}\t{signature}".encode()).hexdigest()[:16]
    return f"{role}-{digest}", signature


def route_work_state(tier: str) -> str:
    """Map an explicit static evidence tier without inferring incompatibility."""

    try:
        return TIER_TO_WORK_STATE[tier]
    except KeyError as exc:
        raise UgiL2CoveragePriorityError(f"unsupported static route tier: {tier}") from exc


def _registry_provenance(row: Mapping[str, str], transfer_source_class: str) -> str:
    if row["is_current_catalog"] == "true":
        return ORIGINAL
    sources = _json_string_list(row["source_classes_json"], label="registry source classes")
    if transfer_source_class in sources:
        return "admitted_cross_platform_transferred_known_component"
    return "admitted_expanded_known_component"


def _nested_counts(
    rows: Sequence[Mapping[str, str]],
    *,
    role_field: str,
    provenance_field: str,
    state_field: str,
) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[(row[role_field], row[provenance_field])][row[state_field]] += 1
    output: dict[str, Any] = {}
    for role in ROLE_NAMES:
        role_rows = [row for row in rows if row[role_field] == role]
        by_provenance: dict[str, Any] = {}
        for provenance in sorted({row[provenance_field] for row in role_rows}):
            state_counts = counts[(role, provenance)]
            total = sum(state_counts.values())
            by_provenance[provenance] = {
                "total": total,
                "route_complete": state_counts[ROUTE_COMPLETE],
                "noncomplete": total - state_counts[ROUTE_COMPLETE],
                "by_route_work_state": dict(sorted(state_counts.items())),
            }
        role_state = Counter(row[state_field] for row in role_rows)
        output[role] = {
            "total": len(role_rows),
            "route_complete": role_state[ROUTE_COMPLETE],
            "noncomplete": len(role_rows) - role_state[ROUTE_COMPLETE],
            "by_route_work_state": dict(sorted(role_state.items())),
            "by_provenance": by_provenance,
        }
    return output


def rank_component_priorities(
    component_rows: Sequence[Mapping[str, str]],
    product_gap_sets: Mapping[str, frozenset[str]],
    *,
    minimum_occurrences: int,
) -> list[dict[str, Any]]:
    """Rank exact gaps by guaranteed single-gap unlocks, then repeated reach."""

    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in component_rows:
        if row["route_work_state"] != ROUTE_COMPLETE:
            grouped[_component_key(row["role"], row["canonical_component_smiles"])].append(row)

    rows: list[dict[str, Any]] = []
    for key, occurrences in grouped.items():
        first = occurrences[0]
        invariants = (
            "role",
            "canonical_component_smiles",
            "structural_provenance_stratum",
            "catalog_provenance_substratum",
            "catalog_membership",
            "static_route_readiness_tier",
            "static_route_evidence_category",
            "route_work_state",
            "gap_class",
        )
        for field in invariants:
            if len({row[field] for row in occurrences}) != 1:
                raise UgiL2CoveragePriorityError(f"exact component has inconsistent {field}: {key}")
        chemotype_id, _ = _chemotype(first["role"], first["canonical_component_smiles"])
        product_ids = {row["sample_index"] for row in occurrences}
        single_gap = sum(gaps == frozenset({key}) for gaps in product_gap_sets.values())
        rows.append(
            {
                "priority_rank": 0,
                "role": first["role"],
                "canonical_smiles": first["canonical_component_smiles"],
                "chemotype_id": chemotype_id,
                "generated_occurrences": len(occurrences),
                "generated_products": len(product_ids),
                "single_gap_products_guaranteed_unlocked": single_gap,
                "structural_provenance_stratum": first["structural_provenance_stratum"],
                "catalog_provenance_substratum": first["catalog_provenance_substratum"],
                "catalog_membership": first["catalog_membership"],
                "static_route_readiness_tier": first["static_route_readiness_tier"],
                "static_route_evidence_category": first["static_route_evidence_category"],
                "route_work_state": first["route_work_state"],
                "gap_class": first["gap_class"],
                "repeated_component": len(occurrences) >= minimum_occurrences,
                "chemical_incompatibility_evidenced": False,
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row["single_gap_products_guaranteed_unlocked"]),
            -int(row["generated_products"]),
            str(row["role"]),
            str(row["canonical_smiles"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["priority_rank"] = rank
    return rows


def _single_gap_frontier(
    priorities: Sequence[Mapping[str, Any]],
    product_gap_sets: Mapping[str, frozenset[str]],
    budgets: Sequence[int],
    *,
    minimum_occurrences: int,
) -> list[dict[str, Any]]:
    repeated = [
        row
        for row in priorities
        if int(row["generated_occurrences"]) >= minimum_occurrences
        and int(row["single_gap_products_guaranteed_unlocked"]) > 0
    ]
    output: list[dict[str, Any]] = []
    for budget in budgets:
        selected = repeated[:budget]
        keys = {_component_key(str(row["role"]), str(row["canonical_smiles"])) for row in selected}
        additionally_unlocked = sum(
            bool(gaps) and gaps.issubset(keys) for gaps in product_gap_sets.values()
        )
        guaranteed_single_gap = sum(
            int(row["single_gap_products_guaranteed_unlocked"]) for row in selected
        )
        output.append(
            {
                "target_budget": budget,
                "selected_components": len(selected),
                "guaranteed_single_gap_products_unlocked": guaranteed_single_gap,
                "all_gap_sets_unlocked_by_selected_components": additionally_unlocked,
                "selected_component_keys": sorted(keys),
            }
        )
    return output


def _chemotype_priorities(
    component_rows: Sequence[Mapping[str, str]],
    registry_rows: Sequence[Mapping[str, str]],
    product_gap_sets: Mapping[str, frozenset[str]],
    *,
    minimum_occurrences: int,
) -> list[dict[str, Any]]:
    registry_by_chemotype: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    signatures: dict[str, str] = {}
    for row in registry_rows:
        chemotype_id, signature = _chemotype(row["role"], row["canonical_smiles"])
        registry_by_chemotype[chemotype_id].append(row)
        signatures[chemotype_id] = signature

    missing_by_chemotype: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in component_rows:
        if row["route_work_state"] == ROUTE_COMPLETE:
            continue
        chemotype_id, signature = _chemotype(row["role"], row["canonical_component_smiles"])
        missing_by_chemotype[chemotype_id].append(row)
        signatures[chemotype_id] = signature

    rows: list[dict[str, Any]] = []
    for chemotype_id, occurrences in missing_by_chemotype.items():
        role = occurrences[0]["role"]
        exact_keys = {
            _component_key(row["role"], row["canonical_component_smiles"]) for row in occurrences
        }
        exact_counts = Counter(
            _component_key(row["role"], row["canonical_component_smiles"]) for row in occurrences
        )
        repeated_exact_components = sum(
            count >= minimum_occurrences for count in exact_counts.values()
        )
        if repeated_exact_components == 0:
            continue
        product_ids = {row["sample_index"] for row in occurrences}
        single_gap_products = sum(
            len(gaps) == 1 and next(iter(gaps)) in exact_keys for gaps in product_gap_sets.values()
        )
        registry_matches = registry_by_chemotype.get(chemotype_id, [])
        route_complete_matches = sum(
            row["route_work_state"] == ROUTE_COMPLETE for row in registry_matches
        )
        rows.append(
            {
                "global_priority_rank": 0,
                "role_priority_rank": 0,
                "role": role,
                "chemotype_id": chemotype_id,
                "architecture_signature": signatures[chemotype_id],
                "architecture_fields_json": json.dumps(list(ARCHITECTURE_FIELDS)),
                "missing_exact_components": len(exact_keys),
                "repeated_missing_exact_components": repeated_exact_components,
                "generated_gap_occurrences": len(occurrences),
                "generated_products_touched": len(product_ids),
                "single_gap_products": single_gap_products,
                "single_gap_products_per_missing_component": (
                    single_gap_products / len(exact_keys)
                ),
                "admitted_registry_components_same_chemotype": len(registry_matches),
                "route_complete_registry_components_same_chemotype": route_complete_matches,
                "noncomplete_registry_components_same_chemotype": (
                    len(registry_matches) - route_complete_matches
                ),
                "structural_analogy_is_route_evidence": False,
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["single_gap_products_per_missing_component"]),
            -int(row["single_gap_products"]),
            -int(row["generated_products_touched"]),
            int(row["missing_exact_components"]),
            str(row["role"]),
            str(row["chemotype_id"]),
        )
    )
    role_ranks = Counter()
    for rank, row in enumerate(rows, start=1):
        role_ranks[row["role"]] += 1
        row["global_priority_rank"] = rank
        row["role_priority_rank"] = role_ranks[row["role"]]
    return rows


def _validate_expected(observed: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            raise UgiL2CoveragePriorityError(f"{label} must be a mapping")
        for key, value in expected.items():
            if key not in observed:
                raise UgiL2CoveragePriorityError(f"{label} missing expected key: {key}")
            _validate_expected(observed[key], value, label=f"{label}.{key}")
    elif observed != expected:
        raise UgiL2CoveragePriorityError(
            f"{label} mismatch: expected {expected!r}, observed {observed!r}"
        )


def build_l2_coverage_priority_audit(
    config_path: Path,
    input_paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Build the frozen coverage-priority result and deterministic ledgers."""

    config = _load_json(config_path, label="coverage-priority config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiL2CoveragePriorityError("unsupported coverage-priority config schema")
    specifications = config.get("inputs")
    if not isinstance(specifications, dict) or set(specifications) != set(input_paths):
        raise UgiL2CoveragePriorityError("config and supplied input names must match exactly")

    inputs: dict[str, dict[str, Any]] = {}
    root = config_path.parents[2]
    for name, path in sorted(input_paths.items()):
        specification = specifications[name]
        _verify_hash(path, specification.get("sha256"), label=name)
        inputs[name] = {
            "path": _portable(path, root=root),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    registry_all = _read_csv(input_paths["component_registry"], label="component registry")
    registry = [row for row in registry_all if row.get("l1_structural_admission") == "true"]
    readiness = _read_csv(input_paths["route_readiness_ledger"], label="route readiness ledger")
    dossiers = _read_csv(input_paths["component_dossier_ledger"], label="component dossier ledger")
    component_occurrences = _read_csv(
        input_paths["generated_component_provenance_ledger"],
        label="generated component provenance ledger",
    )
    product_rows = _read_csv(
        input_paths["generated_product_provenance_ledger"],
        label="generated product provenance ledger",
    )
    readiness_result = _load_json(input_paths["route_readiness_result"], label="readiness result")
    dossier_result = _load_json(input_paths["dossier_result"], label="dossier result")
    provenance_result = _load_json(
        input_paths["generated_provenance_result"], label="generated provenance result"
    )

    if readiness_result.get("artifacts", {}).get("component_ledger_sha256") != sha256_file(
        input_paths["route_readiness_ledger"]
    ):
        raise UgiL2CoveragePriorityError("readiness result does not authenticate its ledger")
    if dossier_result.get("artifacts", {}).get("component_ledger_sha256") != sha256_file(
        input_paths["component_dossier_ledger"]
    ):
        raise UgiL2CoveragePriorityError("dossier result does not authenticate its ledger")
    if provenance_result.get("artifacts", {}).get("component_ledger_sha256") != sha256_file(
        input_paths["generated_component_provenance_ledger"]
    ) or provenance_result.get("artifacts", {}).get("product_ledger_sha256") != sha256_file(
        input_paths["generated_product_provenance_ledger"]
    ):
        raise UgiL2CoveragePriorityError("provenance result does not authenticate its ledgers")

    registry_by_id = {row["component_id"]: row for row in registry}
    readiness_by_key = {(row["role"], row["canonical_smiles"]): row for row in readiness}
    if len(registry_by_id) != 424 or len(readiness_by_key) != len(registry):
        raise UgiL2CoveragePriorityError("expected 424 unique admitted registry components")
    if {(row["role"], row["canonical_smiles"]) for row in registry} != set(readiness_by_key):
        raise UgiL2CoveragePriorityError("registry and readiness exact identities differ")

    dossier_by_id = {row["component_id"]: row for row in dossiers}
    original_matches = [
        row for row in readiness if row["constitutional_match_to_original_93"] == "true"
    ]
    if len(original_matches) != 93 or len(dossier_by_id) != 93:
        raise UgiL2CoveragePriorityError("expected 93 original dossier identities")
    for row in original_matches:
        dossier = dossier_by_id.get(row["original_component_id"])
        if dossier is None:
            raise UgiL2CoveragePriorityError("original readiness identity missing from dossier")
        dossier_identity = (
            dossier["role"],
            _canonical_constitution(dossier["canonical_smiles"], label="dossier component"),
        )
        if dossier_identity != (row["role"], row["canonical_smiles"]):
            raise UgiL2CoveragePriorityError(
                "original readiness constitution disagrees with dossier"
            )
        expected_category = DOSSIER_TO_READINESS.get(dossier["evidence_tier"])
        if expected_category != row["evidence_category"]:
            raise UgiL2CoveragePriorityError("readiness tier disagrees with original dossier")

    transfer_class = str(config["provenance_policy"]["transfer_source_class"])
    registry_analysis: list[dict[str, str]] = []
    for row in readiness:
        registry_row = registry_by_id[row["component_id"]]
        registry_analysis.append(
            {
                **row,
                "registry_provenance": _registry_provenance(registry_row, transfer_class),
                "route_work_state": route_work_state(row["evidence_category"]),
            }
        )

    product_by_sample = {row["sample_index"]: row for row in product_rows}
    if len(product_by_sample) != len(product_rows):
        raise UgiL2CoveragePriorityError("generated product ledger has duplicate sample indices")
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    generated_analysis: list[dict[str, str]] = []
    for row in component_occurrences:
        if row["role"] not in ROLE_NAMES or row["sample_index"] not in product_by_sample:
            raise UgiL2CoveragePriorityError("invalid generated component role or sample index")
        state = route_work_state(row["static_route_readiness_tier"])
        augmented = {
            **row,
            "route_work_state": state,
            "gap_class": row["static_route_gap_class"],
        }
        by_sample[row["sample_index"]].append(augmented)
        generated_analysis.append(augmented)
    if len(generated_analysis) != 3 * len(product_rows):
        raise UgiL2CoveragePriorityError("each generated product must have three component rows")

    product_gap_sets: dict[str, frozenset[str]] = {}
    product_gap_rows: list[dict[str, Any]] = []
    for sample_index, product in sorted(product_by_sample.items(), key=lambda item: int(item[0])):
        components = by_sample[sample_index]
        if {row["role"] for row in components} != set(ROLE_NAMES) or len(components) != 3:
            raise UgiL2CoveragePriorityError("generated product roles must occur exactly once")
        gaps = frozenset(
            _component_key(row["role"], row["canonical_component_smiles"])
            for row in components
            if row["route_work_state"] != ROUTE_COMPLETE
        )
        states = sorted(
            row["route_work_state"]
            for row in components
            if row["route_work_state"] != ROUTE_COMPLETE
        )
        product_gap_sets[sample_index] = gaps
        product_gap_rows.append(
            {
                "sample_index": sample_index,
                "structure_id": product["structure_id"],
                "product_id": product["product_id"],
                "product_structural_provenance_stratum": product[
                    "product_structural_provenance_stratum"
                ],
                "missing_component_count": len(gaps),
                "route_complete_component_count": 3 - len(gaps),
                "gap_component_keys_json": json.dumps(sorted(gaps)),
                "gap_work_states_json": json.dumps(states),
                "baseline_static_route_complete": not gaps,
                "single_gap_product": len(gaps) == 1,
                "chemical_incompatibility_evidenced": False,
            }
        )

    priority_policy = config["priority_policy"]
    minimum_occurrences = int(priority_policy["minimum_generated_occurrences"])
    component_priorities = rank_component_priorities(
        generated_analysis,
        product_gap_sets,
        minimum_occurrences=minimum_occurrences,
    )
    chemotype_priorities = _chemotype_priorities(
        generated_analysis,
        registry_analysis,
        product_gap_sets,
        minimum_occurrences=minimum_occurrences,
    )
    frontier = _single_gap_frontier(
        component_priorities,
        product_gap_sets,
        [int(value) for value in priority_policy["single_gap_frontier_budgets"]],
        minimum_occurrences=minimum_occurrences,
    )

    generated_occurrence_summary = _nested_counts(
        generated_analysis,
        role_field="role",
        provenance_field="structural_provenance_stratum",
        state_field="route_work_state",
    )
    registry_summary = _nested_counts(
        registry_analysis,
        role_field="role",
        provenance_field="registry_provenance",
        state_field="route_work_state",
    )
    unique_generated_rows: list[dict[str, str]] = []
    seen_generated: set[str] = set()
    for row in generated_analysis:
        key = _component_key(row["role"], row["canonical_component_smiles"])
        if key not in seen_generated:
            seen_generated.add(key)
            unique_generated_rows.append(row)
    generated_unique_summary = _nested_counts(
        unique_generated_rows,
        role_field="role",
        provenance_field="structural_provenance_stratum",
        state_field="route_work_state",
    )

    gap_count_distribution = Counter(len(gaps) for gaps in product_gap_sets.values())
    route_complete_occurrences = sum(
        row["route_work_state"] == ROUTE_COMPLETE for row in generated_analysis
    )
    summary = {
        "registry_components": len(registry_analysis),
        "registry_route_complete_components": sum(
            row["route_work_state"] == ROUTE_COMPLETE for row in registry_analysis
        ),
        "registry_noncomplete_components": sum(
            row["route_work_state"] != ROUTE_COMPLETE for row in registry_analysis
        ),
        "registry_by_role_and_provenance": registry_summary,
        "generated_products": len(product_rows),
        "generated_component_occurrences": len(generated_analysis),
        "generated_route_complete_component_occurrences": route_complete_occurrences,
        "generated_noncomplete_component_occurrences": (
            len(generated_analysis) - route_complete_occurrences
        ),
        "generated_unique_components": len(unique_generated_rows),
        "generated_occurrences_by_role_and_provenance": generated_occurrence_summary,
        "generated_unique_components_by_role_and_provenance": generated_unique_summary,
        "products_by_noncomplete_component_count": {
            str(count): gap_count_distribution[count] for count in range(4)
        },
        "baseline_static_route_complete_products": gap_count_distribution[0],
        "single_gap_products": gap_count_distribution[1],
        "products_with_two_or_three_gaps": gap_count_distribution[2] + gap_count_distribution[3],
        "unique_noncomplete_exact_components": len(component_priorities),
        "repeated_noncomplete_exact_components": sum(
            int(row["generated_occurrences"]) >= minimum_occurrences for row in component_priorities
        ),
        "unique_noncomplete_chemotypes": len(chemotype_priorities),
        "chemical_incompatibility_evidenced_components": 0,
        "top_exact_component_priorities": component_priorities[
            : int(priority_policy["component_priority_limit"])
        ],
        "top_chemotype_priorities": chemotype_priorities[
            : int(priority_policy["chemotype_priority_limit"])
        ],
        "single_gap_priority_frontier": frontier,
    }
    _validate_expected(summary, config.get("expected_summary", {}), label="summary")

    ledgers = {
        "component_priority_ledger.csv.gz": _csv_bytes(
            component_priorities, COMPONENT_PRIORITY_FIELDS
        ),
        "chemotype_priority_ledger.csv.gz": _csv_bytes(
            chemotype_priorities, CHEMOTYPE_PRIORITY_FIELDS
        ),
        "product_gap_ledger.csv.gz": _csv_bytes(product_gap_rows, PRODUCT_GAP_FIELDS),
    }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_nonselecting_l2_coverage_priority_audit",
        "task": "Bounded L2 evidence-coverage priorities for frozen generated Ugi components",
        "generated_utc": config["generated_utc"],
        "config": {
            "path": _portable(config_path, root=root),
            "sha256": sha256_file(config_path),
        },
        "inputs": inputs,
        "identity_policy": config["identity_policy"],
        "provenance_policy": config["provenance_policy"],
        "route_evidence_policy": config["route_evidence_policy"],
        "chemotype_policy": config["chemotype_policy"],
        "priority_policy": priority_policy,
        "scope": {
            "local_frozen_evidence_only": True,
            "route_planning_performed": False,
            "literature_search_performed": False,
            "synthesis_feasibility_assessed": False,
            "chemical_incompatibility_inferred": False,
            "biological_activity_assessed": False,
            "candidate_selection_performed": False,
        },
        "summary": summary,
        "artifacts": {
            name: {"sha256": sha256_bytes(payload), "bytes": len(payload)}
            for name, payload in sorted(ledgers.items())
        },
        "safe_claim": (
            "The frozen local evidence identifies repeated exact component and graph-derived "
            "chemotype gaps that are efficient priorities for subsequent evidence retrieval and "
            "bounded planning. These priorities are missing-knowledge targets, not inferred routes "
            "or chemical incompatibilities."
        ),
        "claims_boundary": {
            "structural_analogy_is_route_evidence": False,
            "priority_rank_is_synthesis_feasibility": False,
            "catalog_absence_is_chemical_incompatibility": False,
            "family_projection_is_exact_source_evidence": False,
            "single_gap_counterfactual_is_observed_synthesis_success": False,
        },
    }
    return result, ledgers
