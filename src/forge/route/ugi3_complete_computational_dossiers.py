"""Compose Ugi L1, exact-source L2 and frozen L3 evidence fail-closed."""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.data.r1_prime_audit import sha256_bytes, sha256_file

CONFIG_SCHEMA_VERSION = "phase1_ugi3_complete_computational_dossiers_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_complete_computational_dossiers.v1"
VERIFIED_STEP = "verified_exact_product_unique"
EXACT_CLOSED = "exact_source_l2_verified_l3_closed"
FAMILY_CLOSED = "family_projected_l3_closed_not_exact_source"
TERMINAL_CLOSED = "accepted_terminal_l3_closed"
COMPONENT_FIELDS = (
    "component_id",
    "role",
    "canonical_smiles",
    "program_status",
    "evidence_tier",
    "l2_forward_status",
    "step_records_expected",
    "step_records_observed",
    "l3_terminal_status",
    "proposed_leaves_json",
    "closed_leaves_json",
    "unresolved_leaves_json",
    "exact_source_computational_component",
    "family_projected_only",
    "next_gap",
)
PRODUCT_FIELDS = (
    "source_row_index",
    "canonical_product_smiles",
    "l1_forward_status",
    "amine_component_id",
    "amine_evidence_tier",
    "aldehyde_component_id",
    "aldehyde_evidence_tier",
    "isocyanide_component_id",
    "isocyanide_evidence_tier",
    "computational_dossier_tier",
    "complete_exact_source_computational_dossier",
    "observed_exact_product_synthesis_status",
    "experimental_success_status",
    "next_gap",
)
ROLE_FIELDS = {
    "amine_head": ("amine_component_id", "amine_evidence_tier"),
    "oxoester_aldehyde_body_tail": (
        "aldehyde_component_id",
        "aldehyde_evidence_tier",
    ),
    "isocyanide_tail": ("isocyanide_component_id", "isocyanide_evidence_tier"),
}


class Ugi3CompleteDossierError(ValueError):
    """Raised when a pinned dossier input violates its contract."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise Ugi3CompleteDossierError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3CompleteDossierError(f"{label} must be a JSON object")
    return value


def _read_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise Ugi3CompleteDossierError(f"{label} has no header")
            return list(reader)
    except (OSError, csv.Error) as exc:
        raise Ugi3CompleteDossierError(f"could not read {label}") from exc


def _json_list(value: str, *, label: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise Ugi3CompleteDossierError(f"{label} is invalid JSON") from exc
    if not isinstance(parsed, list):
        raise Ugi3CompleteDossierError(f"{label} must be a list")
    return parsed


def _verify_hash(path: Path, expected: Any, *, label: str) -> None:
    observed = sha256_file(path)
    if not isinstance(expected, str) or observed != expected:
        raise Ugi3CompleteDossierError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(text.getvalue().encode())
    return output.getvalue()


def _portable(path: Path, *, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _validate_expected(observed: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise Ugi3CompleteDossierError(f"{label} fields mismatch")
        for key, value in expected.items():
            _validate_expected(observed[key], value, label=f"{label}.{key}")
    elif observed != expected:
        raise Ugi3CompleteDossierError(
            f"{label} mismatch: expected {expected!r}, observed {observed!r}"
        )


def classify_component_program(
    program: Mapping[str, str],
    step_records: Sequence[Mapping[str, str]],
    closed_terminals: set[str],
) -> dict[str, Any]:
    """Classify one component without allowing evidence-tier promotion."""

    component_id = program["component_id"]
    leaves = _json_list(
        program["proposed_leaf_candidates_json"],
        label=f"{component_id} leaves",
    )
    if any(not isinstance(leaf, str) for leaf in leaves):
        raise Ugi3CompleteDossierError(f"{component_id} leaf is not a string")
    closed = sorted(leaf for leaf in leaves if leaf in closed_terminals)
    open_leaves = sorted(set(leaves) - closed_terminals)
    status = program["program_status"]
    expected_steps = _json_list(
        program["program_steps_json"],
        label=f"{component_id} program steps",
    )
    if any(not isinstance(step, dict) for step in expected_steps):
        raise Ugi3CompleteDossierError(f"{component_id} has malformed steps")
    observed = len(step_records)
    l2_status = "not_applicable"
    exact = False
    family_only = False

    if status == "accepted_procurement_terminal":
        if step_records:
            raise Ugi3CompleteDossierError(
                f"accepted terminal {component_id} unexpectedly has step records"
            )
        closed = sorted(leaves)
        open_leaves = []
        tier = TERMINAL_CLOSED
        l3_status = "closed"
        exact = True
        next_gap = "none"
    elif status == "exact_source_program":
        expected_by_index = {str(step.get("step_index")): step for step in expected_steps}
        observed_by_index: dict[str, list[Mapping[str, str]]] = defaultdict(list)
        for record in step_records:
            observed_by_index[record.get("step_index", "")].append(record)
        if len(expected_by_index) != len(expected_steps):
            raise Ugi3CompleteDossierError(f"{component_id} has duplicate expected step indices")
        missing = sorted(set(expected_by_index) - set(observed_by_index))
        ambiguous = sorted(
            index for index, records in observed_by_index.items() if len(records) != 1
        )
        mismatched = False
        unverified = False
        for index, expected in expected_by_index.items():
            records = observed_by_index.get(index, [])
            if len(records) != 1:
                continue
            record = records[0]
            expected_reactants = expected.get("reactants")
            observed_reactants = _json_list(
                record.get("reactants_json", ""),
                label=f"{component_id} step {index} reactants",
            )
            if (
                record.get("component_role") != program["role"]
                or record.get("component_smiles") != program["canonical_smiles"]
                or record.get("transformation") != expected.get("transformation")
                or observed_reactants != expected_reactants
                or record.get("expected_product") != expected.get("product")
            ):
                mismatched = True
            if (
                record.get("verification_status") != VERIFIED_STEP
                or record.get("forward_product_count") != "1"
                or record.get("expected_product_in_outputs") != "true"
            ):
                unverified = True
        if missing:
            tier = "exact_source_l2_missing_step"
            l2_status = "missing_step"
            next_gap = "forward_verify_missing_step"
        elif ambiguous or set(observed_by_index) != set(expected_by_index):
            tier = "exact_source_l2_ambiguous_step"
            l2_status = "ambiguous_step"
            next_gap = "resolve_step_ambiguity"
        elif mismatched:
            tier = "exact_source_l2_step_contract_mismatch"
            l2_status = "contract_mismatch"
            next_gap = "repair_step_provenance"
        elif unverified:
            tier = "exact_source_l2_unverified_step"
            l2_status = "unverified_step"
            next_gap = "qualify_and_verify_step"
        elif open_leaves:
            tier = "exact_source_l2_verified_l3_open"
            l2_status = "all_steps_uniquely_verified"
            next_gap = "close_terminal_procurement"
        else:
            tier = EXACT_CLOSED
            l2_status = "all_steps_uniquely_verified"
            exact = True
            next_gap = "none"
        l3_status = "closed" if not open_leaves else "open"
    elif status == "reaction_family_projected_program":
        family_only = True
        l2_status = "family_projection_not_exact_source"
        if open_leaves:
            tier = "family_projected_l3_open_not_exact_source"
            l3_status = "open"
            next_gap = "obtain_exact_route_evidence_and_close_terminals"
        else:
            tier = FAMILY_CLOSED
            l3_status = "closed"
            next_gap = "obtain_exact_route_evidence_and_forward_verify"
    elif status == "procurement_or_route_search_required":
        open_leaves = sorted(leaves)
        tier = "unresolved_component"
        l2_status = "no_assigned_upstream_program"
        l3_status = "open"
        next_gap = "procurement_or_exact_route_search"
    else:
        raise Ugi3CompleteDossierError(f"{component_id} has unsupported program status {status!r}")

    return {
        "component_id": component_id,
        "role": program["role"],
        "canonical_smiles": program["canonical_smiles"],
        "program_status": status,
        "evidence_tier": tier,
        "l2_forward_status": l2_status,
        "step_records_expected": len(expected_steps),
        "step_records_observed": observed,
        "l3_terminal_status": l3_status,
        "proposed_leaves_json": json.dumps(leaves, separators=(",", ":")),
        "closed_leaves_json": json.dumps(closed, separators=(",", ":")),
        "unresolved_leaves_json": json.dumps(open_leaves, separators=(",", ":")),
        "exact_source_computational_component": str(exact).lower(),
        "family_projected_only": str(family_only).lower(),
        "next_gap": next_gap,
    }


def _validate_l1_candidate(row: Mapping[str, str]) -> tuple[bool, dict[str, Any]]:
    candidates = _json_list(
        row["candidate_routes_json"],
        label=f"product {row['source_row_index']} candidates",
    )
    if len(candidates) != 1 or not isinstance(candidates[0], dict):
        return False, {}
    candidate = candidates[0]
    components = candidate.get("components")
    passed = (
        row.get("decomposition_status") == "one_exact_qualified_ugi_decomposition"
        and row.get("candidate_count") == "1"
        and isinstance(components, dict)
        and set(components) == set(ROLE_FIELDS)
        and candidate.get("distinct_target_reacting_sites") == 1
        and isinstance(candidate.get("target_matching_forward_outcomes"), int)
        and candidate["target_matching_forward_outcomes"] >= 1
        and isinstance(candidate.get("unique_forward_products"), int)
        and candidate["unique_forward_products"] >= 1
    )
    return passed, candidate


def build_complete_computational_dossiers(
    config_path: Path,
    component_program_path: Path,
    product_program_path: Path,
    component_dossier_path: Path,
    product_dossier_path: Path,
    dossier_result_path: Path,
    step_ledger_path: Path,
    step_result_path: Path,
    terminal_procurement_path: Path,
) -> tuple[dict[str, Any], bytes, bytes]:
    """Build row-level exact-source computational dossier classifications."""

    config = _load_json(config_path, label="complete dossier config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3CompleteDossierError("unsupported config schema")
    paths = {
        "component_program": component_program_path,
        "product_program": product_program_path,
        "component_dossier": component_dossier_path,
        "product_dossier": product_dossier_path,
        "dossier_result": dossier_result_path,
        "step_ledger": step_ledger_path,
        "step_result": step_result_path,
        "terminal_procurement": terminal_procurement_path,
    }
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(paths):
        raise Ugi3CompleteDossierError("config inputs mismatch")
    for name, path in paths.items():
        record = inputs[name]
        if not isinstance(record, dict):
            raise Ugi3CompleteDossierError(f"input {name} must be an object")
        _verify_hash(path, record.get("expected_sha256"), label=name)

    dossier_result = _load_json(dossier_result_path, label="dossier result")
    step_result = _load_json(step_result_path, label="step result")
    if dossier_result.get("artifacts", {}).get("component_ledger_sha256") != sha256_file(
        component_dossier_path
    ) or dossier_result.get("artifacts", {}).get("product_ledger_sha256") != sha256_file(
        product_dossier_path
    ):
        raise Ugi3CompleteDossierError("dossier result does not own its ledgers")
    if step_result.get("artifacts", {}).get("step_ledger_sha256") != sha256_file(step_ledger_path):
        raise Ugi3CompleteDossierError("step result does not own its ledger")

    procurement = _load_json(terminal_procurement_path, label="terminal procurement")
    records = procurement.get("records")
    if not isinstance(records, list):
        raise Ugi3CompleteDossierError("procurement records must be a list")
    closed_terminals = {
        record["canonical_smiles"]
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("canonical_smiles"), str)
        and record.get("current_item_level_procurement_closed") is True
    }

    program_rows = _read_csv(component_program_path, label="component program ledger")
    old_component_rows = _read_csv(component_dossier_path, label="component dossier ledger")
    old_components = {row["component_id"]: row for row in old_component_rows}
    if len(old_components) != len(old_component_rows):
        raise Ugi3CompleteDossierError("duplicate prior component dossier")
    step_rows = _read_csv(step_ledger_path, label="step ledger")
    steps_by_component: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in step_rows:
        steps_by_component[row["component_id"]].append(row)

    component_rows: list[dict[str, Any]] = []
    component_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    component_counts: Counter[str] = Counter()
    for program in sorted(program_rows, key=lambda row: row["component_id"]):
        result = classify_component_program(
            program,
            steps_by_component.get(program["component_id"], []),
            closed_terminals,
        )
        prior = old_components.get(program["component_id"])
        if (
            prior is None
            or prior["role"] != program["role"]
            or prior["canonical_smiles"] != program["canonical_smiles"]
            or prior["proposed_leaves_json"] != result["proposed_leaves_json"]
            or prior["current_closed_leaves_json"] != result["closed_leaves_json"]
            or prior["unresolved_leaves_json"] != result["unresolved_leaves_json"]
        ):
            raise Ugi3CompleteDossierError(
                f"prior component dossier disagrees for {program['component_id']}"
            )
        key = (program["role"], program["canonical_smiles"])
        if key in component_by_key:
            raise Ugi3CompleteDossierError("duplicate role-component identity")
        component_by_key[key] = result
        component_counts[result["evidence_tier"]] += 1
        component_rows.append(result)

    product_program_rows = _read_csv(product_program_path, label="product program ledger")
    old_product_rows = _read_csv(product_dossier_path, label="product dossier ledger")
    old_products = {row["source_row_index"]: row for row in old_product_rows}
    if len(old_products) != len(old_product_rows):
        raise Ugi3CompleteDossierError("duplicate prior product dossier")
    product_rows: list[dict[str, Any]] = []
    product_counts: Counter[str] = Counter()
    for row in sorted(product_program_rows, key=lambda item: int(item["source_row_index"])):
        l1_passed, candidate = _validate_l1_candidate(row)
        prior = old_products.get(row["source_row_index"])
        if (
            prior is None
            or prior["canonical_product_smiles"] != row["canonical_product_smiles"]
            or prior["l1_exact_graph_reconstruction"] != "true"
        ):
            raise Ugi3CompleteDossierError(
                f"prior product dossier disagrees for row {row['source_row_index']}"
            )
        components = candidate.get("components") if l1_passed else None
        if not isinstance(components, dict):
            component_results: dict[str, dict[str, Any]] = {}
        else:
            try:
                component_results = {
                    role: component_by_key[(role, smiles)] for role, smiles in components.items()
                }
            except KeyError as exc:
                raise Ugi3CompleteDossierError(
                    f"product {row['source_row_index']} references unknown component"
                ) from exc
        tiers = [result["evidence_tier"] for result in component_results.values()]
        exact_allowed = {TERMINAL_CLOSED, EXACT_CLOSED}
        family_allowed = exact_allowed | {FAMILY_CLOSED}
        if l1_passed and len(tiers) == 3 and all(tier in exact_allowed for tier in tiers):
            dossier_tier = "complete_exact_source_computational_dossier"
            complete = True
            next_gap = "prospective_exact_product_synthesis_and_outcome"
        elif l1_passed and len(tiers) == 3 and all(tier in family_allowed for tier in tiers):
            dossier_tier = "family_projected_support_not_exact_source"
            complete = False
            next_gap = "obtain_exact_upstream_route_evidence"
        else:
            dossier_tier = "incomplete_computational_dossier"
            complete = False
            next_gap = "resolve_l1_l2_or_l3_gap"
        expected_old_tier = {
            "complete_exact_source_computational_dossier": "exact_source_upstream_closed",
            "family_projected_support_not_exact_source": "family_supported_upstream_closed",
            "incomplete_computational_dossier": "incomplete",
        }[dossier_tier]
        if prior["upstream_evidence_tier"] != expected_old_tier:
            raise Ugi3CompleteDossierError(
                f"prior product tier disagrees for row {row['source_row_index']}"
            )
        output: dict[str, Any] = {
            "source_row_index": row["source_row_index"],
            "canonical_product_smiles": row["canonical_product_smiles"],
            "l1_forward_status": (
                "exact_qualified_ugi_target_reconstructed" if l1_passed else "failed"
            ),
            "computational_dossier_tier": dossier_tier,
            "complete_exact_source_computational_dossier": str(complete).lower(),
            "observed_exact_product_synthesis_status": "not_assessed_by_this_gate",
            "experimental_success_status": "not_assessed_by_this_gate",
            "next_gap": next_gap,
        }
        for role, (id_field, tier_field) in ROLE_FIELDS.items():
            component = component_results.get(role)
            output[id_field] = component["component_id"] if component else ""
            output[tier_field] = component["evidence_tier"] if component else ""
        product_counts[dossier_tier] += 1
        product_rows.append(output)

    summary = {
        "components": len(component_rows),
        "components_by_evidence_tier": dict(sorted(component_counts.items())),
        "products": len(product_rows),
        "products_by_dossier_tier": dict(sorted(product_counts.items())),
        "complete_exact_source_computational_dossiers": product_counts[
            "complete_exact_source_computational_dossier"
        ],
    }
    _validate_expected(summary, config.get("expected_counts"), label="summary")
    component_bytes = _csv_bytes(component_rows, COMPONENT_FIELDS)
    product_bytes = _csv_bytes(product_rows, PRODUCT_FIELDS)
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
        "evidence_policy": config.get("evidence_policy"),
        "summary": summary,
        "claims_boundary": {
            "family_projection_is_exact_source_route_evidence": False,
            "complete_computational_dossier_is_observed_exact_product_synthesis": False,
            "complete_computational_dossier_is_experimental_success": False,
            "current_procurement_snapshot_guarantees_future_availability": False,
        },
        "safe_claim": (
            f"{summary['complete_exact_source_computational_dossiers']:,} of "
            f"{summary['products']:,} Ugi products have complete forward-consistent "
            "computational L1/L2/L3 dossiers under exact-source route evidence and the "
            "frozen procurement snapshot. A complete computational dossier does not "
            "establish that the exact product was synthesized or that synthesis, "
            "formulation or biological testing succeeded."
        ),
        "artifacts": {
            "component_ledger_sha256": sha256_bytes(component_bytes),
            "product_ledger_sha256": sha256_bytes(product_bytes),
        },
    }
    return result, component_bytes, product_bytes
