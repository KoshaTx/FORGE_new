"""Audit exact head procurement and abstain on homologue-only isocyanide evidence."""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.core.hashing import sha256_bytes, sha256_file
from forge.core.io import read_json_object
from forge.core.io import stable_json as _stable_json

CONFIG_SCHEMA_VERSION = "phase1_ugi3_targeted_role_gap_evidence_audit_config.v1"
EVIDENCE_SCHEMA_VERSION = "phase1_ugi3_targeted_role_gap_evidence.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_targeted_role_gap_evidence_audit.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi3_targeted_role_gap_product_impact.v1"


class Ugi3TargetedRoleGapEvidenceError(ValueError):
    """Raised when role-gap evidence would be promoted beyond its support."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=Ugi3TargetedRoleGapEvidenceError, label=label)


def _read_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise Ugi3TargetedRoleGapEvidenceError(f"{label} has no header")
            return list(reader)
    except (OSError, csv.Error) as exc:
        raise Ugi3TargetedRoleGapEvidenceError(f"could not read {label}") from exc


def _gzip_csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(text.getvalue().encode())
    return output.getvalue()


def _canonical(smiles: Any, *, label: str) -> str:
    if not isinstance(smiles, str) or not smiles:
        raise Ugi3TargetedRoleGapEvidenceError(f"{label} must be nonempty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise Ugi3TargetedRoleGapEvidenceError(f"{label} contains invalid SMILES")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _parse_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Ugi3TargetedRoleGapEvidenceError(f"{label} must be UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Ugi3TargetedRoleGapEvidenceError(f"{label} is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise Ugi3TargetedRoleGapEvidenceError(f"{label} must resolve to UTC")
    return parsed


def _collect_product_smiles(value: Any) -> set[str]:
    products: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "product_smiles" and isinstance(child, str):
                products.add(_canonical(child, label="reviewed product"))
            else:
                products.update(_collect_product_smiles(child))
    elif isinstance(value, list):
        for child in value:
            products.update(_collect_product_smiles(child))
    return products


def _verify_inputs(
    config: Mapping[str, Any],
    input_paths: Mapping[str, Path],
) -> dict[str, str]:
    configured = config.get("inputs")
    if not isinstance(configured, dict) or set(configured) != set(input_paths):
        raise Ugi3TargetedRoleGapEvidenceError("configured and supplied inputs differ")
    hashes: dict[str, str] = {}
    for name, path in input_paths.items():
        record = configured[name]
        if not isinstance(record, dict):
            raise Ugi3TargetedRoleGapEvidenceError(f"input {name} is malformed")
        observed = sha256_file(path)
        if observed != record.get("expected_sha256"):
            raise Ugi3TargetedRoleGapEvidenceError(f"{name} hash mismatch")
        hashes[name] = observed
    return hashes


def build_targeted_role_gap_evidence_audit(
    *,
    config_path: Path,
    input_paths: Mapping[str, Path],
) -> tuple[dict[str, Any], bytes]:
    """Admit one exact terminal and preserve the exact isocyanide abstention."""

    config = _load_json(config_path, label="role-gap audit config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3TargetedRoleGapEvidenceError("unsupported audit config schema")
    input_hashes = _verify_inputs(config, input_paths)
    evidence = _load_json(input_paths["evidence_pack"], label="role-gap evidence pack")
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise Ugi3TargetedRoleGapEvidenceError("unsupported evidence schema")

    head = evidence.get("direct_procurement_terminal")
    unresolved = evidence.get("unresolved_isocyanide_target")
    if not isinstance(head, dict) or not isinstance(unresolved, dict):
        raise Ugi3TargetedRoleGapEvidenceError("role-gap targets are malformed")
    head_smiles = _canonical(head.get("canonical_smiles"), label="head target")
    isocyanide_smiles = _canonical(unresolved.get("canonical_smiles"), label="isocyanide target")
    if (
        head.get("role") != "amine_head"
        or head.get("disposition") != "admit_exact_terminal"
        or head.get("current_item_level_procurement_closed") is not True
    ):
        raise Ugi3TargetedRoleGapEvidenceError("head procurement is not closing exact evidence")
    identity = head.get("identity")
    vendor = head.get("vendor_evidence")
    if not isinstance(identity, dict) or not isinstance(vendor, dict):
        raise Ugi3TargetedRoleGapEvidenceError("head identity or vendor evidence is malformed")
    molecule = Chem.MolFromSmiles(head_smiles)
    assert molecule is not None
    if Chem.MolToInchiKey(molecule) != identity.get("inchi_key"):
        raise Ugi3TargetedRoleGapEvidenceError("head InChI key mismatch")
    if (
        identity.get("cas_rn") != "3529-08-6"
        or vendor.get("marketplace") != "Sigma-Aldrich US"
        or vendor.get("preferred_partner") != "ChemScene LLC"
        or vendor.get("marketplace_product_code") != "CIAH987F216E"
        or "sigmaaldrich.com/US/en/product/chemscenellcpreferredpartner/"
        not in str(vendor.get("url"))
    ):
        raise Ugi3TargetedRoleGapEvidenceError("head marketplace identity is incomplete")
    items = vendor.get("item_level_observations")
    if (
        not isinstance(items, list)
        or len(items) < 1
        or any(
            not isinstance(item, dict)
            or not str(item.get("availability", "")).startswith("Ships in ")
            or not str(item.get("marketplace_sku", "")).startswith("CIAH987F216E-")
            for item in items
        )
    ):
        raise Ugi3TargetedRoleGapEvidenceError("head item-level availability is not closed")
    as_of = _parse_utc(evidence.get("assessment_as_of_utc"), label="assessment_as_of_utc")
    accessed = _parse_utc(vendor.get("accessed_utc"), label="vendor accessed_utc")
    expires = _parse_utc(vendor.get("expires_utc"), label="vendor expires_utc")
    if not (accessed <= as_of < expires):
        raise Ugi3TargetedRoleGapEvidenceError("head availability evidence is expired or future")

    if (
        unresolved.get("role") != "isocyanide_tail"
        or unresolved.get("disposition") != "abstain"
        or unresolved.get("exact_target_found_in_reviewed_product_routes") is not False
    ):
        raise Ugi3TargetedRoleGapEvidenceError("isocyanide target must remain an abstention")
    isocyanide_molecule = Chem.MolFromSmiles(isocyanide_smiles)
    assert isocyanide_molecule is not None
    if Chem.MolToInchiKey(isocyanide_molecule) != unresolved.get("identity", {}).get("inchi_key"):
        raise Ugi3TargetedRoleGapEvidenceError("isocyanide InChI key mismatch")
    precedent = unresolved.get("nonclosing_homologue_evidence")
    if not isinstance(precedent, dict) or precedent.get("classification") != (
        "family_and_neighbouring_homologue_precedent_only"
    ):
        raise Ugi3TargetedRoleGapEvidenceError("isocyanide precedent was promoted")
    if precedent.get("source_sha256") != input_hashes["agile_supplement"]:
        raise Ugi3TargetedRoleGapEvidenceError("AGILE supplement hash mismatch")
    reviews = _load_json(input_paths["paper_reviews"], label="paper review registry")
    reviewed_products = _collect_product_smiles(reviews)
    neighbors = {
        _canonical(value, label="isocyanide homologue")
        for value in precedent.get("exact_neighbor_products", [])
    }
    if isocyanide_smiles in reviewed_products or isocyanide_smiles in neighbors:
        raise Ugi3TargetedRoleGapEvidenceError(
            "exact isocyanide target cannot be recorded as absent"
        )
    if not neighbors or not neighbors.issubset(reviewed_products):
        raise Ugi3TargetedRoleGapEvidenceError("homologue evidence is not source-resolved")

    readiness_rows = _read_csv(input_paths["readiness_ledger"], label="readiness ledger")
    by_id = {row.get("component_id"): row for row in readiness_rows}
    for target, canonical in ((head, head_smiles), (unresolved, isocyanide_smiles)):
        row = by_id.get(target.get("component_id"))
        if (
            row is None
            or row.get("role") != target.get("role")
            or _canonical(row.get("canonical_smiles"), label="readiness target") != canonical
            or row.get("route_complete_component") == "true"
        ):
            raise Ugi3TargetedRoleGapEvidenceError(
                "role-gap target disagrees with frozen readiness registry"
            )

    prior_rows = _read_csv(input_paths["product_impact_ledger"], label="product ledger")
    head_key = f"amine_head\t{head_smiles}"
    isocyanide_key = f"isocyanide_tail\t{isocyanide_smiles}"
    output_rows: list[dict[str, Any]] = []
    head_occurrences = 0
    newly_complete = 0
    isocyanide_occurrences = 0
    isocyanide_single_gap = 0
    remaining_distribution: Counter[int] = Counter()
    for row in prior_rows:
        remaining = json.loads(row["remaining_component_keys_json"])
        if not isinstance(remaining, list) or any(
            not isinstance(value, str) for value in remaining
        ):
            raise Ugi3TargetedRoleGapEvidenceError("product remaining-component list is malformed")
        head_used = head_key in remaining
        head_occurrences += int(head_used)
        updated = [value for value in remaining if value != head_key]
        was_complete = row.get("complete_after_targeted_evidence") == "True"
        now_complete = not updated
        became_complete = not was_complete and now_complete
        newly_complete += int(became_complete)
        isocyanide_occurrences += int(isocyanide_key in updated)
        isocyanide_single_gap += int(updated == [isocyanide_key])
        remaining_distribution[len(updated)] += 1
        output_rows.append(
            {
                "sample_index": row["sample_index"],
                "structure_id": row["structure_id"],
                "product_id": row["product_id"],
                "prior_remaining_component_count": row["remaining_component_count"],
                "head_terminal_applied": head_used,
                "remaining_component_keys_json": _stable_json(updated),
                "remaining_component_count": len(updated),
                "complete_after_aldehyde_evidence": was_complete,
                "complete_after_role_gap_evidence": now_complete,
                "newly_complete_from_head_terminal": became_complete,
            }
        )

    summary = {
        "targets_adjudicated": 2,
        "exact_head_terminals_admitted": 1,
        "isocyanide_abstentions": 1,
        "registry_complete_components_before": 45,
        "registry_complete_components_after": 46,
        "generated_products": len(output_rows),
        "generated_products_complete_before": sum(
            row.get("complete_after_targeted_evidence") == "True" for row in prior_rows
        ),
        "generated_products_complete_after": sum(
            row["complete_after_role_gap_evidence"] for row in output_rows
        ),
        "newly_complete_generated_products": newly_complete,
        "generated_head_occurrences": head_occurrences,
        "unresolved_isocyanide_occurrences_after_head_closure": isocyanide_occurrences,
        "products_one_exact_isocyanide_gap_from_completion": isocyanide_single_gap,
        "remaining_gap_count_distribution": {
            str(key): value for key, value in sorted(remaining_distribution.items())
        },
    }
    expected = config.get("expected_summary")
    if summary != expected:
        raise Ugi3TargetedRoleGapEvidenceError(
            f"summary mismatch: expected {expected!r}, observed {summary!r}"
        )
    fieldnames = [
        "sample_index",
        "structure_id",
        "product_id",
        "prior_remaining_component_count",
        "head_terminal_applied",
        "remaining_component_keys_json",
        "remaining_component_count",
        "complete_after_aldehyde_evidence",
        "complete_after_role_gap_evidence",
        "newly_complete_from_head_terminal",
    ]
    ledger = _gzip_csv_bytes(output_rows, fieldnames)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "config_sha256": sha256_file(config_path),
        "input_sha256": dict(sorted(input_hashes.items())),
        "summary": summary,
        "target_adjudications": [
            {
                "component_id": head["component_id"],
                "role": head["role"],
                "canonical_smiles": head_smiles,
                "disposition": "admit_exact_terminal",
                "evidence_channel": "current_preferred_partner_marketplace_listing",
            },
            {
                "component_id": unresolved["component_id"],
                "role": unresolved["role"],
                "canonical_smiles": isocyanide_smiles,
                "disposition": "abstain",
                "evidence_channel": "nonclosing_neighboring_homologue_precedent",
            },
        ],
        "artifacts": {
            "product_impact_ledger.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "sha256": sha256_bytes(ledger),
            }
        },
        "claims_boundary": evidence["claims_boundary"],
    }
    return result, ledger
