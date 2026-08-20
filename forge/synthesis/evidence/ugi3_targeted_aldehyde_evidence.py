"""Validate targeted exact evidence for high-impact Ugi aldehydes.

This diagnostic is intentionally narrower than a reaction-template learner.  It
admits exact current procurement records and exact-substrate literature routes,
verifies every admitted route with the frozen forward transform, and measures
the resulting deterministic closure of an already frozen generated-product
audit.  Analogue and reaction-family precedent never close a route here.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.core.hashing import sha256_bytes, sha256_file
from forge.core.io import read_json_object
from forge.core.io import stable_json as _stable_json
from forge.synthesis.engine.qualified_forward import (
    QualifiedForwardError,
    load_qualified_forward_reaction,
    unique_forward_products,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi3_targeted_aldehyde_evidence_audit_config.v1"
EVIDENCE_SCHEMA_VERSION = "phase1_ugi3_targeted_aldehyde_evidence.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_targeted_aldehyde_evidence_audit.v1"


class Ugi3TargetedAldehydeEvidenceError(ValueError):
    """Raised when targeted evidence cannot be admitted without inference."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=Ugi3TargetedAldehydeEvidenceError, label=label)


def _read_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise Ugi3TargetedAldehydeEvidenceError(f"{label} has no header")
            return list(reader)
    except (OSError, csv.Error) as exc:
        raise Ugi3TargetedAldehydeEvidenceError(f"could not read {label}") from exc


def _canonical(smiles: Any, *, label: str) -> str:
    if not isinstance(smiles, str) or not smiles:
        raise Ugi3TargetedAldehydeEvidenceError(f"{label} must be nonempty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise Ugi3TargetedAldehydeEvidenceError(f"{label} contains invalid SMILES")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _inchi_key(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:  # pragma: no cover - all callers canonicalize first
        raise Ugi3TargetedAldehydeEvidenceError("cannot compute InChIKey")
    return Chem.MolToInchiKey(molecule)


def _parse_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Ugi3TargetedAldehydeEvidenceError(f"{label} must be ISO-8601 UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Ugi3TargetedAldehydeEvidenceError(f"invalid {label}") from exc
    if parsed.tzinfo != timezone.utc:
        raise Ugi3TargetedAldehydeEvidenceError(f"{label} must resolve to UTC")
    return parsed


def _bool(value: str, *, label: str) -> bool:
    if value == "true" or value == "True":
        return True
    if value == "false" or value == "False":
        return False
    raise Ugi3TargetedAldehydeEvidenceError(f"{label} must be boolean text")


def _json_list(value: str, *, label: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise Ugi3TargetedAldehydeEvidenceError(f"invalid {label}") from exc
    if not isinstance(parsed, list):
        raise Ugi3TargetedAldehydeEvidenceError(f"{label} must be a list")
    return parsed


def _gzip_csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        wrapper = io.TextIOWrapper(compressed, encoding="utf-8", newline="", write_through=True)
        writer = csv.DictWriter(wrapper, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        wrapper.flush()
        wrapper.detach()
    return output.getvalue()


def _portable(path: Path, *, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _validate_sources(
    evidence: Mapping[str, Any],
    *,
    input_paths: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    source_input_names = {
        "kovalerchik_2022_marinedrugs": "kovalerchik_source",
        "mo_2018_chem_eur_j_si": "mo_source",
        "busta_2016_phytochemistry": "busta_source",
    }
    sources = evidence.get("literature_sources")
    if not isinstance(sources, list) or len(sources) != len(source_input_names):
        raise Ugi3TargetedAldehydeEvidenceError("unexpected literature-source count")
    by_id: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise Ugi3TargetedAldehydeEvidenceError("literature source must be an object")
        source_id = source.get("source_id")
        if source_id not in source_input_names or source_id in by_id:
            raise Ugi3TargetedAldehydeEvidenceError("unknown or duplicate literature source")
        observed = sha256_file(input_paths[source_input_names[source_id]])
        if source.get("sha256") != observed:
            raise Ugi3TargetedAldehydeEvidenceError(f"literature source hash mismatch: {source_id}")
        for field in ("title", "doi", "locator", "visual_review"):
            if not isinstance(source.get(field), str) or not source[field]:
                raise Ugi3TargetedAldehydeEvidenceError(
                    f"literature source {source_id} lacks {field}"
                )
        by_id[source_id] = source
    return by_id


def _validate_procurement_snapshot(
    evidence: Mapping[str, Any],
    *,
    assessment_as_of: datetime,
) -> list[dict[str, Any]]:
    snapshot = evidence.get("snapshot")
    if not isinstance(snapshot, dict):
        raise Ugi3TargetedAldehydeEvidenceError("evidence snapshot is missing")
    accessed = _parse_utc(snapshot.get("assessment_as_of_utc"), label="assessment time")
    expiry_days = snapshot.get("procurement_expiry_days")
    if isinstance(expiry_days, bool) or not isinstance(expiry_days, int) or expiry_days < 0:
        raise Ugi3TargetedAldehydeEvidenceError("invalid procurement expiry")
    if assessment_as_of != accessed:
        raise Ugi3TargetedAldehydeEvidenceError("audit and evidence assessment times differ")
    if assessment_as_of > accessed + timedelta(days=expiry_days):
        raise Ugi3TargetedAldehydeEvidenceError("target procurement evidence is expired")
    records = evidence.get("direct_procurement_terminals")
    if not isinstance(records, list):
        raise Ugi3TargetedAldehydeEvidenceError("procurement records must be a list")
    for record in records:
        if not isinstance(record, dict):
            raise Ugi3TargetedAldehydeEvidenceError("procurement record must be an object")
        canonical = _canonical(record.get("canonical_smiles"), label="procurement target")
        identity = record.get("identity")
        vendor = record.get("vendor_evidence")
        if not isinstance(identity, dict) or not isinstance(vendor, dict):
            raise Ugi3TargetedAldehydeEvidenceError("procurement identity/evidence missing")
        if identity.get("inchi_key") != _inchi_key(canonical):
            raise Ugi3TargetedAldehydeEvidenceError("procurement InChIKey mismatch")
        for field in (
            "vendor",
            "product_title",
            "product_code",
            "url",
            "purity",
            "availability_observation",
        ):
            if not isinstance(vendor.get(field), str) or not vendor[field]:
                raise Ugi3TargetedAldehydeEvidenceError(f"procurement record lacks {field}")
        if record.get("current_item_level_procurement_closed") is not True:
            raise Ugi3TargetedAldehydeEvidenceError("nonclosing procurement was admitted")
    return records


def _current_original_terminals(
    procurement: Mapping[str, Any],
    *,
    assessment_as_of: datetime,
) -> dict[str, dict[str, Any]]:
    snapshot = procurement.get("snapshot")
    records = procurement.get("records")
    if not isinstance(snapshot, dict) or not isinstance(records, list):
        raise Ugi3TargetedAldehydeEvidenceError("original procurement snapshot malformed")
    accessed = _parse_utc(snapshot.get("accessed_utc"), label="original L3 accessed time")
    expiry_days = snapshot.get("expiry_days")
    if isinstance(expiry_days, bool) or not isinstance(expiry_days, int):
        raise Ugi3TargetedAldehydeEvidenceError("original L3 expiry is invalid")
    current = assessment_as_of <= accessed + timedelta(days=expiry_days)
    terminals: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise Ugi3TargetedAldehydeEvidenceError("original L3 record malformed")
        canonical = _canonical(record.get("canonical_smiles"), label="original L3 terminal")
        if record.get("current_item_level_procurement_closed") is True and current:
            terminals[canonical] = record
    return terminals


def build_targeted_aldehyde_evidence_audit(
    *,
    config_path: Path,
    input_paths: Mapping[str, Path],
) -> tuple[dict[str, Any], bytes, bytes]:
    """Return the diagnostic result, route ledger and product-impact ledger."""

    config = _load_json(config_path, label="targeted evidence audit config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3TargetedAldehydeEvidenceError("unsupported audit config schema")
    configured_inputs = config.get("inputs")
    if not isinstance(configured_inputs, dict) or set(configured_inputs) != set(input_paths):
        raise Ugi3TargetedAldehydeEvidenceError("configured and supplied inputs differ")
    hashes: dict[str, str] = {}
    for name, path in input_paths.items():
        configured = configured_inputs[name]
        if not isinstance(configured, dict):
            raise Ugi3TargetedAldehydeEvidenceError(f"input {name} is malformed")
        observed = sha256_file(path)
        if observed != configured.get("expected_sha256"):
            raise Ugi3TargetedAldehydeEvidenceError(f"input hash mismatch: {name}")
        hashes[name] = observed

    evidence = _load_json(input_paths["evidence_pack"], label="targeted evidence pack")
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise Ugi3TargetedAldehydeEvidenceError("unsupported evidence-pack schema")
    policy = config.get("audit_policy")
    if not isinstance(policy, dict) or policy.get("diagnostic_only") is not True:
        raise Ugi3TargetedAldehydeEvidenceError("audit must remain diagnostic-only")
    if policy.get("permit_family_projection_as_closure") is not False:
        raise Ugi3TargetedAldehydeEvidenceError("family projection cannot close a target")
    if policy.get("permit_analogue_evidence_as_closure") is not False:
        raise Ugi3TargetedAldehydeEvidenceError("analogue evidence cannot close a target")
    snapshot = evidence.get("snapshot")
    if not isinstance(snapshot, dict):
        raise Ugi3TargetedAldehydeEvidenceError("evidence snapshot missing")
    assessment_as_of = _parse_utc(
        snapshot.get("assessment_as_of_utc"), label="assessment_as_of_utc"
    )
    sources = _validate_sources(evidence, input_paths=input_paths)
    direct = _validate_procurement_snapshot(evidence, assessment_as_of=assessment_as_of)
    if len(direct) != policy.get("expected_direct_procurement_records"):
        raise Ugi3TargetedAldehydeEvidenceError("unexpected direct-procurement count")

    readiness_result = _load_json(input_paths["route_readiness_result"], label="readiness result")
    if (
        readiness_result.get("artifacts", {}).get("component_ledger_sha256")
        != hashes["route_readiness_ledger"]
    ):
        raise Ugi3TargetedAldehydeEvidenceError("readiness result does not own its ledger")
    priority_result = _load_json(input_paths["priority_result"], label="priority result")
    if (
        priority_result.get("artifacts", {}).get("product_gap_ledger.csv.gz", {}).get("sha256")
        != hashes["product_gap_ledger"]
    ):
        raise Ugi3TargetedAldehydeEvidenceError("priority result does not own its ledger")
    readiness_rows = _read_csv(input_paths["route_readiness_ledger"], label="readiness ledger")
    readiness_by_id = {row["component_id"]: row for row in readiness_rows}
    if len(readiness_by_id) != len(readiness_rows):
        raise Ugi3TargetedAldehydeEvidenceError("duplicate readiness component IDs")
    baseline_complete_components = sum(
        _bool(row["route_complete_component"], label="readiness route-complete flag")
        for row in readiness_rows
    )

    procurement = _load_json(input_paths["terminal_procurement"], label="terminal procurement")
    upstream_terminals = _current_original_terminals(procurement, assessment_as_of=assessment_as_of)
    direct_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for record in direct:
        canonical = _canonical(record["canonical_smiles"], label="direct target")
        key = (record["role"], canonical)
        if key in direct_by_key:
            raise Ugi3TargetedAldehydeEvidenceError("duplicate direct terminal")
        direct_by_key[key] = record

    routes = evidence.get("exact_routes")
    if not isinstance(routes, list) or len(routes) != policy.get("expected_exact_route_records"):
        raise Ugi3TargetedAldehydeEvidenceError("unexpected exact-route count")
    route_rows: list[dict[str, Any]] = []
    routes_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for route in routes:
        if not isinstance(route, dict):
            raise Ugi3TargetedAldehydeEvidenceError("route record must be an object")
        if route.get("disposition") != policy.get("required_route_disposition"):
            raise Ugi3TargetedAldehydeEvidenceError("non-exact route was admitted")
        source_id = route.get("source_id")
        if source_id not in sources:
            raise Ugi3TargetedAldehydeEvidenceError("route references unknown source")
        target = route.get("target")
        reactant = route.get("reactant")
        if not isinstance(target, dict) or not isinstance(reactant, dict):
            raise Ugi3TargetedAldehydeEvidenceError("route identity is incomplete")
        target_smiles = _canonical(target.get("canonical_smiles"), label="route target")
        reactant_smiles = _canonical(reactant.get("canonical_smiles"), label="route reactant")
        if target.get("inchi_key") != _inchi_key(target_smiles):
            raise Ugi3TargetedAldehydeEvidenceError("route target InChIKey mismatch")
        if reactant.get("inchi_key") != _inchi_key(reactant_smiles):
            raise Ugi3TargetedAldehydeEvidenceError("route reactant InChIKey mismatch")
        try:
            compiled = load_qualified_forward_reaction(
                input_paths["upstream_registry"],
                input_paths["oxidation_variant"],
                reaction_id=str(route.get("reaction_id")),
            )
            products = unique_forward_products(
                compiled,
                [reactant_smiles],
                max_products=int(policy.get("maximum_forward_products_considered")),
                isomeric_smiles=False,
            )
        except (QualifiedForwardError, TypeError, ValueError) as exc:
            raise Ugi3TargetedAldehydeEvidenceError(
                "exact route failed forward verification"
            ) from exc
        if len(products) != policy.get("required_forward_product_count") or products != (
            target_smiles,
        ):
            raise Ugi3TargetedAldehydeEvidenceError("route did not uniquely reconstruct target")
        terminal = upstream_terminals.get(reactant_smiles)
        if terminal is None:
            raise Ugi3TargetedAldehydeEvidenceError("route lacks current exact L3 closure")
        analytical = route.get("analytical_evidence")
        conditions = route.get("conditions")
        if not isinstance(analytical, list) or not analytical or not isinstance(conditions, dict):
            raise Ugi3TargetedAldehydeEvidenceError("exact route lacks procedure evidence")
        key = (str(route.get("role")), target_smiles)
        routes_by_key[key].append(route)
        route_rows.append(
            {
                "route_id": route["route_id"],
                "component_id": route["component_id"],
                "role": route["role"],
                "target_canonical_smiles": target_smiles,
                "reactant_canonical_smiles": reactant_smiles,
                "source_id": source_id,
                "source_sha256": sources[source_id]["sha256"],
                "source_locator": sources[source_id]["locator"],
                "isolated_yield_percent": route.get("isolated_yield_percent"),
                "analytical_evidence_json": _stable_json(analytical),
                "forward_product_count": len(products),
                "forward_products_json": _stable_json(list(products)),
                "forward_verified_exact_unique": True,
                "upstream_terminal_title": terminal.get("identity", {}).get("title", ""),
                "upstream_terminal_current_closed": True,
                "route_complete": True,
            }
        )

    unresolved = evidence.get("unresolved_predeclared_target")
    if not isinstance(unresolved, dict):
        raise Ugi3TargetedAldehydeEvidenceError("unresolved target is missing")
    if unresolved.get("disposition") != policy.get("required_unresolved_disposition"):
        raise Ugi3TargetedAldehydeEvidenceError("unresolved target must remain an abstention")
    target_records: dict[tuple[str, str], dict[str, Any]] = {}
    for record in [*direct, *routes, unresolved]:
        role = record.get("role")
        identity = record.get("target") if "target" in record else record
        if not isinstance(identity, dict):
            raise Ugi3TargetedAldehydeEvidenceError("target record identity is malformed")
        canonical = _canonical(identity.get("canonical_smiles"), label="target record")
        key = (str(role), canonical)
        target_records.setdefault(key, record)
    if len(target_records) != policy.get("expected_prioritized_targets"):
        raise Ugi3TargetedAldehydeEvidenceError("prioritized target count changed")

    target_rows: list[dict[str, Any]] = []
    newly_closed_keys: set[str] = set()
    for (role, canonical), record in sorted(target_records.items()):
        component_id = str(record.get("component_id"))
        readiness = readiness_by_id.get(component_id)
        if readiness is None:
            raise Ugi3TargetedAldehydeEvidenceError("target is absent from readiness registry")
        if (
            readiness.get("role") != role
            or _canonical(readiness.get("canonical_smiles"), label="readiness target") != canonical
        ):
            raise Ugi3TargetedAldehydeEvidenceError("target and readiness identity disagree")
        ready_before = _bool(
            readiness["route_complete_component"], label="target prior route-complete flag"
        )
        direct_complete = (role, canonical) in direct_by_key
        exact_route_count = len(routes_by_key.get((role, canonical), []))
        route_complete = exact_route_count > 0
        ready_after = direct_complete or route_complete
        if ready_before and ready_after:
            raise Ugi3TargetedAldehydeEvidenceError(
                "targeted evidence redundantly closes a prior target"
            )
        if ready_after:
            newly_closed_keys.add(f"{role}\t{canonical}")
        primary = (
            "direct_procurement"
            if direct_complete
            else "exact_l2_to_current_l3" if route_complete else "abstain_missing_exact_evidence"
        )
        target_rows.append(
            {
                "component_id": component_id,
                "role": role,
                "canonical_smiles": canonical,
                "readiness_category_before": readiness["evidence_category"],
                "route_complete_before": ready_before,
                "direct_procurement_complete": direct_complete,
                "exact_route_count": exact_route_count,
                "all_exact_routes_forward_verified": route_complete,
                "route_complete_after": ready_after,
                "primary_closure_mode": primary,
                "disposition": "admit_exact" if ready_after else "abstain",
            }
        )

    product_rows = _read_csv(input_paths["product_gap_ledger"], label="product-gap ledger")
    if len(product_rows) != policy.get("expected_generated_products"):
        raise Ugi3TargetedAldehydeEvidenceError("generated-product count changed")
    baseline_complete_products = 0
    after_complete_products = 0
    new_complete_products = 0
    remaining_gaps: Counter[int] = Counter()
    closures_by_key: Counter[str] = Counter()
    product_impact_rows: list[dict[str, Any]] = []
    for row in product_rows:
        gaps_raw = _json_list(row["gap_component_keys_json"], label="product gap keys")
        if any(not isinstance(value, str) for value in gaps_raw):
            raise Ugi3TargetedAldehydeEvidenceError("product gap key must be a string")
        gaps = set(gaps_raw)
        baseline_complete = _bool(
            row["baseline_static_route_complete"], label="baseline product closure"
        )
        baseline_complete_products += int(baseline_complete)
        closed_here = sorted(gaps & newly_closed_keys)
        remaining = sorted(gaps - newly_closed_keys)
        complete_after = not remaining
        newly_complete = complete_after and not baseline_complete
        after_complete_products += int(complete_after)
        new_complete_products += int(newly_complete)
        remaining_gaps[len(remaining)] += 1
        if newly_complete:
            for key in closed_here:
                closures_by_key[key] += 1
        product_impact_rows.append(
            {
                "sample_index": row["sample_index"],
                "structure_id": row["structure_id"],
                "product_id": row["product_id"],
                "baseline_missing_component_count": len(gaps),
                "newly_closed_component_keys_json": _stable_json(closed_here),
                "remaining_component_keys_json": _stable_json(remaining),
                "remaining_component_count": len(remaining),
                "baseline_complete": baseline_complete,
                "complete_after_targeted_evidence": complete_after,
                "newly_complete": newly_complete,
            }
        )
    if baseline_complete_products != policy.get("expected_baseline_complete_products"):
        raise Ugi3TargetedAldehydeEvidenceError("baseline product closure changed")
    for row in target_rows:
        key = f"{row['role']}\t{row['canonical_smiles']}"
        row["newly_completed_products_attributed"] = closures_by_key.get(key, 0)

    route_fields = [
        "route_id",
        "component_id",
        "role",
        "target_canonical_smiles",
        "reactant_canonical_smiles",
        "source_id",
        "source_sha256",
        "source_locator",
        "isolated_yield_percent",
        "analytical_evidence_json",
        "forward_product_count",
        "forward_products_json",
        "forward_verified_exact_unique",
        "upstream_terminal_title",
        "upstream_terminal_current_closed",
        "route_complete",
    ]
    product_fields = [
        "sample_index",
        "structure_id",
        "product_id",
        "baseline_missing_component_count",
        "newly_closed_component_keys_json",
        "remaining_component_keys_json",
        "remaining_component_count",
        "baseline_complete",
        "complete_after_targeted_evidence",
        "newly_complete",
    ]
    route_ledger = _gzip_csv_bytes(route_rows, route_fields)
    product_ledger = _gzip_csv_bytes(product_impact_rows, product_fields)
    root = config_path.resolve().parents[2]
    input_records = {
        name: {"path": _portable(path, root=root), "sha256": hashes[name]}
        for name, path in sorted(input_paths.items())
    }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": config["task"],
        "generated_utc": snapshot["assessment_as_of_utc"],
        "inputs": input_records,
        "audit_policy": policy,
        "summary": {
            "prioritized_exact_targets": len(target_rows),
            "admitted_exact_targets": len(newly_closed_keys),
            "abstained_targets": len(target_rows) - len(newly_closed_keys),
            "current_direct_procurement_terminals": len(direct_by_key),
            "exact_source_routes_forward_verified": len(route_rows),
            "exact_routes_with_current_l3_closure": sum(
                row["upstream_terminal_current_closed"] for row in route_rows
            ),
            "registry_route_complete_components_before": baseline_complete_components,
            "registry_route_complete_components_after": baseline_complete_components
            + len(newly_closed_keys),
            "generated_products": len(product_rows),
            "generated_products_complete_before": baseline_complete_products,
            "generated_products_complete_after": after_complete_products,
            "newly_complete_generated_products": new_complete_products,
            "remaining_gap_count_distribution": {
                str(key): value for key, value in sorted(remaining_gaps.items())
            },
            "newly_completed_products_by_target": dict(sorted(closures_by_key.items())),
        },
        "target_adjudications": target_rows,
        "claims_boundary": evidence["claims_boundary"],
        "safe_claim": (
            "Four predeclared high-impact aldehydes gained exact route or current "
            "procurement closure under a fail-closed evidence policy; one target remained "
            "an abstention. This diagnostic increases complete dossiers in the frozen "
            "generated audit but does not qualify a general oxidation template or authorize "
            "production synthesis guidance."
        ),
        "artifacts": {
            "route_verification_ledger.csv.gz": {
                "bytes": len(route_ledger),
                "sha256": sha256_bytes(route_ledger),
            },
            "product_closure_impact_ledger.csv.gz": {
                "bytes": len(product_ledger),
                "sha256": sha256_bytes(product_ledger),
            },
        },
    }
    return result, route_ledger, product_ledger
