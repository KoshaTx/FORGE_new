"""Audit actual precursor availability beneath optimistic AGILE programs.

This audit distinguishes a mechanically proposed upstream leaf from an exact,
current procurement terminal.  Even when every leaf of a structural program is
available, the program remains a family projection until its exact substrate
steps are supported and forward-qualified.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TextIO

from rdkit import Chem, rdBase

from forge.core.hashing import sha256_file
from forge.core.io import read_json_object
from forge.synthesis.terminals.ugi3_agile_template_saturation_stress import (
    projected_leaf_candidates,
)
from forge.synthesis.terminals.ugi3_virtual_programs import _aldehyde_program, _isocyanide_program

CONFIG_SCHEMA_VERSION = "phase1_ugi3_precursor_leaf_closure_audit_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_precursor_leaf_closure_audit.v1"
LEAF_LEDGER_SCHEMA_VERSION = "phase1_ugi3_precursor_leaf_closure_ledger.v1"
HEAD_LEDGER_SCHEMA_VERSION = "phase1_ugi3_head_terminal_closure_ledger.v1"

HEAD_ROLE = "amine_head"
ALDEHYDE_ROLE = "oxoester_aldehyde_body_tail"
ISOCYANIDE_ROLE = "isocyanide_tail"
TAIL_ROLES = (ALDEHYDE_ROLE, ISOCYANIDE_ROLE)

ROLE_COLUMNS = {
    HEAD_ROLE: "amine_head_smiles",
    ALDEHYDE_ROLE: "oxoester_aldehyde_body_tail_smiles",
    ISOCYANIDE_ROLE: "isocyanide_tail_smiles",
}


class Ugi3PrecursorLeafClosureError(ValueError):
    """Raised when the precursor-leaf audit cannot be reproduced safely."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=Ugi3PrecursorLeafClosureError, label=label)


def _validated_inputs(config: Mapping[str, Any], repo: Path) -> dict[str, Path]:
    specifications = config.get("inputs")
    if not isinstance(specifications, dict) or not specifications:
        raise Ugi3PrecursorLeafClosureError("leaf-audit inputs are missing")
    paths: dict[str, Path] = {}
    for label, specification in specifications.items():
        if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
            raise Ugi3PrecursorLeafClosureError(f"input {label!r} must define path and sha256")
        path = Path(str(specification["path"]))
        if not path.is_absolute():
            path = repo / path
        observed = sha256_file(path)
        if observed != str(specification["sha256"]):
            raise Ugi3PrecursorLeafClosureError(
                f"input hash changed for {label}: expected {specification['sha256']}, "
                f"observed {observed}"
            )
        paths[label] = path
    return paths


def _parse_utc(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Ugi3PrecursorLeafClosureError(f"invalid {label}: {value!r}") from exc
    if parsed.tzinfo is None:
        raise Ugi3PrecursorLeafClosureError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def constitutional_key(smiles: str) -> str:
    """Return the frozen constitution-only identity key for one molecule."""

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise Ugi3PrecursorLeafClosureError(f"invalid molecular identity: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _snapshot_expiry(payload: Mapping[str, Any], *, label: str) -> datetime:
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, dict):
        raise Ugi3PrecursorLeafClosureError(f"{label} lacks a procurement snapshot")
    if "expires_utc" in snapshot:
        return _parse_utc(str(snapshot["expires_utc"]), label=f"{label} expiry")
    accessed = snapshot.get("accessed_utc")
    expiry_days = snapshot.get("expiry_days")
    if not isinstance(accessed, str) or not isinstance(expiry_days, int):
        raise Ugi3PrecursorLeafClosureError(f"{label} lacks a reproducible expiry")
    return _parse_utc(accessed, label=f"{label} access time") + timedelta(days=expiry_days)


def _supplier_fields(record: Mapping[str, Any]) -> tuple[str, str, str]:
    vendor = record.get("vendor_evidence")
    if isinstance(vendor, dict):
        supplier = str(vendor.get("vendor") or vendor.get("marketplace") or "")
        item = str(
            vendor.get("product_code")
            or vendor.get("marketplace_product_code")
            or vendor.get("vendor_sku")
            or ""
        )
        url = str(vendor.get("url") or "")
        return supplier, item, url
    supplier = str(record.get("supplier") or "")
    item = str(record.get("product_number") or "")
    url = str(record.get("source_url") or "")
    return supplier, item, url


def _add_terminal_record(
    terminals: dict[str, dict[str, str]],
    *,
    record: Mapping[str, Any],
    source_path: Path,
    repo: Path,
    expires: datetime,
) -> None:
    smiles = record.get("canonical_smiles")
    if not isinstance(smiles, str) or not smiles:
        raise Ugi3PrecursorLeafClosureError("terminal record lacks canonical_smiles")
    key = constitutional_key(smiles)
    supplier, item, url = _supplier_fields(record)
    candidate = {
        "constitutional_key": key,
        "reported_smiles": smiles,
        "source_path": str(source_path.relative_to(repo)),
        "source_sha256": sha256_file(source_path),
        "supplier": supplier,
        "supplier_item": item,
        "source_url": url,
        "expires_utc": expires.isoformat().replace("+00:00", "Z"),
    }
    existing = terminals.get(key)
    if existing is None or (candidate["source_path"], candidate["supplier_item"]) < (
        existing["source_path"],
        existing["supplier_item"],
    ):
        terminals[key] = candidate


def current_terminal_evidence(
    paths: Mapping[str, Path],
    *,
    assessment_as_of: datetime,
    default_expiry_days: int,
    repo: Path,
) -> dict[str, dict[str, str]]:
    """Load exact, unexpired procurement terminals from frozen evidence packs."""

    terminals: dict[str, dict[str, str]] = {}
    standard_sources = (
        "virtual_terminal_procurement",
        "agile_head_procurement",
    )
    for label in standard_sources:
        path = paths[label]
        payload = _load_json(path, label=label)
        expires = _snapshot_expiry(payload, label=label)
        if expires < assessment_as_of:
            continue
        records = payload.get("records")
        if not isinstance(records, list):
            raise Ugi3PrecursorLeafClosureError(f"{label} lacks records")
        for record in records:
            if not isinstance(record, dict):
                raise Ugi3PrecursorLeafClosureError(f"{label} has a malformed record")
            if record.get("current_item_level_procurement_closed") is True:
                _add_terminal_record(
                    terminals,
                    record=record,
                    source_path=path,
                    repo=repo,
                    expires=expires,
                )

    for label in (
        "high_leverage_head_terminals",
        "second_wave_head_terminals",
        "third_wave_head_terminals",
        "hybrid_high_impact_leaf_terminals",
    ):
        if label not in paths:
            continue
        path = paths[label]
        payload = _load_json(path, label=label)
        expires = _snapshot_expiry(payload, label=label)
        if expires < assessment_as_of:
            continue
        records = payload.get("records")
        if not isinstance(records, list):
            raise Ugi3PrecursorLeafClosureError(f"{label} lacks records")
        for record in records:
            if not isinstance(record, dict):
                raise Ugi3PrecursorLeafClosureError(f"{label} has a malformed record")
            if record.get("disposition") == "admit_exact_terminal":
                _add_terminal_record(
                    terminals,
                    record=record,
                    source_path=path,
                    repo=repo,
                    expires=expires,
                )

    role_gap_path = paths["targeted_role_gap_evidence"]
    role_gap = _load_json(role_gap_path, label="targeted_role_gap_evidence")
    record = role_gap.get("direct_procurement_terminal")
    if not isinstance(record, dict):
        raise Ugi3PrecursorLeafClosureError("role-gap evidence lacks its terminal")
    vendor = record.get("vendor_evidence")
    if not isinstance(vendor, dict) or not isinstance(vendor.get("expires_utc"), str):
        raise Ugi3PrecursorLeafClosureError("role-gap terminal lacks expiry")
    expires = _parse_utc(str(vendor["expires_utc"]), label="role-gap expiry")
    if expires >= assessment_as_of and record.get("current_item_level_procurement_closed") is True:
        _add_terminal_record(
            terminals,
            record=record,
            source_path=role_gap_path,
            repo=repo,
            expires=expires,
        )

    transfer_path = paths["hydrophobic_motif_transfer"]
    transfer = _load_json(transfer_path, label="hydrophobic_motif_transfer")
    programs = transfer.get("programs")
    if not isinstance(programs, dict):
        raise Ugi3PrecursorLeafClosureError("hydrophobic transfer evidence lacks programs")
    for program_id, program in programs.items():
        if (
            not isinstance(program, dict)
            or program.get("route_closure") != "computationally_complete"
        ):
            continue
        exact = program.get("exact_route_evidence")
        terminal = exact.get("terminal_evidence") if isinstance(exact, dict) else None
        reactant = exact.get("reported_reactant") if isinstance(exact, dict) else None
        if not isinstance(terminal, dict) or not isinstance(reactant, str):
            raise Ugi3PrecursorLeafClosureError("complete transfer program lacks terminal evidence")
        accessed = _parse_utc(
            str(terminal.get("accessed_utc")), label="hydrophobic-transfer access time"
        )
        expires = accessed + timedelta(days=default_expiry_days)
        if expires < assessment_as_of:
            continue
        motifs = transfer.get("motifs")
        if not isinstance(motifs, list):
            raise Ugi3PrecursorLeafClosureError("hydrophobic transfer evidence lacks motifs")
        matching = [
            motif
            for motif in motifs
            if isinstance(motif, dict)
            and (
                motif.get("common_precursor_name") == reactant
                or motif.get("program_id") == program_id
            )
            and isinstance(motif.get("common_precursor_smiles"), str)
        ]
        if len(matching) != 1:
            raise Ugi3PrecursorLeafClosureError(
                f"could not resolve exact transfer terminal identity for {reactant!r}"
            )
        record = {
            "canonical_smiles": matching[0]["common_precursor_smiles"],
            "vendor_evidence": terminal,
        }
        _add_terminal_record(
            terminals,
            record=record,
            source_path=transfer_path,
            repo=repo,
            expires=expires,
        )
    return terminals


def _add_historical_leaf_use(
    records: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    role: str,
    smiles: str,
    evidence_pack: Path,
    repo: Path,
    source_id: str,
    source_locator: str,
    route_or_reaction_id: str,
    evidence_basis: str,
    source_asset_ids: Sequence[str] = (),
    source_component_id: str = "",
    reported_yield_percent: int | float | None = None,
) -> None:
    """Record exact historical use of a projected leaf without implying availability."""

    key = (role, constitutional_key(smiles))
    record: dict[str, Any] = {
        "evidence_pack": str(evidence_pack.relative_to(repo)),
        "evidence_pack_sha256": sha256_file(evidence_pack),
        "source_id": source_id,
        "source_locator": source_locator,
        "route_or_reaction_id": route_or_reaction_id,
        "evidence_basis": evidence_basis,
        "source_asset_ids": sorted(source_asset_ids),
        "source_component_id": source_component_id,
        "reported_yield_percent": reported_yield_percent,
    }
    bucket = records.setdefault(key, [])
    if record not in bucket:
        bucket.append(record)


def historical_leaf_source_use(
    paths: Mapping[str, Path], *, repo: Path
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Load exact source use of projected leaves from frozen evidence packs.

    This axis records that the same molecular identity was used as a reactant in
    a reported source procedure.  It neither establishes current procurement nor
    validates the mechanically projected transformation on a new product.
    """

    records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    agile_path = paths["agile_component_routes"]
    agile = _load_json(agile_path, label="agile_component_routes")
    agile_source = agile.get("source")
    if not isinstance(agile_source, dict):
        raise Ugi3PrecursorLeafClosureError("AGILE route evidence lacks source metadata")
    agile_source_id = str(agile_source.get("source_id") or "agile_2024")

    aldehyde_families: list[Mapping[str, Any]] = []
    primary_aldehyde_family = agile.get("aldehyde_ester_route_family")
    if isinstance(primary_aldehyde_family, dict):
        aldehyde_families.append(primary_aldehyde_family)
    additional_aldehyde = agile.get("additional_aldehyde_route_families")
    if not isinstance(additional_aldehyde, list):
        raise Ugi3PrecursorLeafClosureError("AGILE additional aldehyde routes are malformed")
    aldehyde_families.extend(family for family in additional_aldehyde if isinstance(family, dict))
    for family in aldehyde_families:
        if family.get("route_evidence_status") != "route_extracted":
            continue
        family_id = str(family.get("route_family_id") or "")
        locator = str(family.get("source_locator") or "")
        shared_leaf = family.get("shared_leaf")
        if isinstance(shared_leaf, dict) and isinstance(shared_leaf.get("smiles"), str):
            _add_historical_leaf_use(
                records,
                role=ALDEHYDE_ROLE,
                smiles=str(shared_leaf["smiles"]),
                evidence_pack=agile_path,
                repo=repo,
                source_id=agile_source_id,
                source_locator=locator,
                route_or_reaction_id=family_id,
                evidence_basis="exact_reported_library_execution",
            )
        members = family.get("members")
        if not isinstance(members, list):
            raise Ugi3PrecursorLeafClosureError(f"AGILE route family {family_id!r} lacks members")
        for member in members:
            if not isinstance(member, dict):
                raise Ugi3PrecursorLeafClosureError(
                    f"AGILE route family {family_id!r} has a malformed member"
                )
            for field in ("acid_smiles", "alcohol_smiles"):
                smiles = member.get(field)
                if isinstance(smiles, str):
                    _add_historical_leaf_use(
                        records,
                        role=ALDEHYDE_ROLE,
                        smiles=smiles,
                        evidence_pack=agile_path,
                        repo=repo,
                        source_id=agile_source_id,
                        source_locator=locator,
                        route_or_reaction_id=str(member.get("route_id") or family_id),
                        evidence_basis="exact_reported_library_execution",
                        source_component_id=str(member.get("label") or ""),
                    )

    isocyanide_families: list[Mapping[str, Any]] = []
    primary_isocyanide = agile.get("isocyanide_route_family")
    additional_isocyanide = agile.get("additional_isocyanide_route_family")
    if isinstance(primary_isocyanide, dict):
        isocyanide_families.append(primary_isocyanide)
    if isinstance(additional_isocyanide, dict):
        isocyanide_families.append(additional_isocyanide)
    for family in isocyanide_families:
        if family.get("route_evidence_status") != "route_extracted":
            continue
        family_id = str(family.get("route_family_id") or "")
        locator = str(family.get("source_locator") or "")
        members = family.get("members")
        if not isinstance(members, list):
            raise Ugi3PrecursorLeafClosureError(f"AGILE route family {family_id!r} lacks members")
        for member in members:
            if not isinstance(member, dict) or not isinstance(member.get("amine_smiles"), str):
                raise Ugi3PrecursorLeafClosureError(
                    f"AGILE route family {family_id!r} has a malformed member"
                )
            _add_historical_leaf_use(
                records,
                role=ISOCYANIDE_ROLE,
                smiles=str(member["amine_smiles"]),
                evidence_pack=agile_path,
                repo=repo,
                source_id=agile_source_id,
                source_locator=locator,
                route_or_reaction_id=str(member.get("route_id") or family_id),
                evidence_basis="exact_reported_library_execution",
                source_component_id=str(member.get("label") or ""),
            )

    transfer_path = paths["hydrophobic_motif_transfer"]
    transfer = _load_json(transfer_path, label="hydrophobic_motif_transfer")
    sources = transfer.get("sources")
    motifs = transfer.get("motifs")
    if not isinstance(sources, dict) or not isinstance(motifs, list):
        raise Ugi3PrecursorLeafClosureError("hydrophobic transfer source-use evidence is malformed")
    for motif in motifs:
        if not isinstance(motif, dict) or motif.get("disposition") != "propose_ugi_aldehyde":
            continue
        smiles = motif.get("common_precursor_smiles")
        source_key = motif.get("source_id")
        reaction_id = motif.get("source_reaction_id")
        if not isinstance(smiles, str) or not isinstance(source_key, str):
            raise Ugi3PrecursorLeafClosureError("transferred historical leaf lacks identity")
        source = sources.get(source_key)
        if not isinstance(source, dict):
            raise Ugi3PrecursorLeafClosureError(
                f"transferred historical leaf has unknown source {source_key!r}"
            )
        _add_historical_leaf_use(
            records,
            role=ALDEHYDE_ROLE,
            smiles=smiles,
            evidence_pack=transfer_path,
            repo=repo,
            source_id=str(source.get("platform_id") or source_key),
            source_locator=str(source.get("source_locator") or ""),
            route_or_reaction_id=str(reaction_id or ""),
            evidence_basis="exact_executed_series_member",
            source_asset_ids=[str(value) for value in source.get("source_asset_ids", [])],
            source_component_id=str(motif.get("source_component_id") or ""),
            reported_yield_percent=motif.get("source_reported_yield_percent"),
        )

    for bucket in records.values():
        bucket.sort(
            key=lambda record: (
                str(record["source_id"]),
                str(record["route_or_reaction_id"]),
                str(record["source_component_id"]),
            )
        )
    return records


def _read_component_rows(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise Ugi3PrecursorLeafClosureError("could not read component registry") from exc
    return [row for row in rows if row.get("l1_structural_admission") == "true"]


def _component_programs(
    rows: Sequence[Mapping[str, str]],
) -> tuple[
    dict[tuple[str, str], tuple[str, tuple[str, ...]]],
    list[str],
    dict[tuple[str, str], dict[str, list[str]]],
]:
    programs: dict[tuple[str, str], tuple[str, tuple[str, ...]]] = {}
    heads: list[str] = []
    provenance: dict[tuple[str, str], dict[str, list[str]]] = {}
    for row in rows:
        role = str(row.get("role"))
        target = str(row.get("canonical_smiles"))
        try:
            provenance[(role, target)] = {
                "source_classes": list(json.loads(str(row.get("source_classes_json")))),
                "source_platforms": list(json.loads(str(row.get("source_platforms_json")))),
                "source_record_ids": list(json.loads(str(row.get("source_record_ids_json")))),
            }
        except (json.JSONDecodeError, TypeError) as exc:
            raise Ugi3PrecursorLeafClosureError(
                f"component provenance is malformed for {(role, target)!r}"
            ) from exc
        if role == HEAD_ROLE:
            heads.append(target)
            continue
        if role == ALDEHYDE_ROLE:
            family, steps = _aldehyde_program(target)
        elif role == ISOCYANIDE_ROLE:
            family, steps = _isocyanide_program(target)
        else:
            raise Ugi3PrecursorLeafClosureError(f"unexpected admitted role: {role!r}")
        programs[(role, target)] = (family, projected_leaf_candidates(steps))
    return programs, sorted(heads), provenance


def _product_counts(
    handle: TextIO,
    *,
    programs: Mapping[tuple[str, str], tuple[str, tuple[str, ...]]],
    current_terminal_keys: set[str],
    current_heads: set[str],
    historical_leaf_keys: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    parent_counts: Counter[tuple[str, str]] = Counter()
    head_counts: Counter[str] = Counter()
    two_tail_leaf_closed = 0
    head_and_two_tail_leaf_closed = 0
    two_tail_historically_used_leaf_closed = 0
    product_count = 0
    parent_closed = {
        key: all(constitutional_key(leaf) in current_terminal_keys for leaf in leaves)
        for key, (_, leaves) in programs.items()
    }
    historical_keys = historical_leaf_keys or set()
    parent_historical_closed = {
        key: all((key[0], constitutional_key(leaf)) in historical_keys for leaf in leaves)
        for key, (_, leaves) in programs.items()
    }
    reader = csv.DictReader(handle)
    if reader.fieldnames is None or not set(ROLE_COLUMNS.values()).issubset(reader.fieldnames):
        raise Ugi3PrecursorLeafClosureError("expanded product ledger lacks component columns")
    for row_number, row in enumerate(reader, start=2):
        head = row[ROLE_COLUMNS[HEAD_ROLE]]
        head_counts[head] += 1
        tail_states = []
        historical_tail_states = []
        for role in TAIL_ROLES:
            target = row[ROLE_COLUMNS[role]]
            key = (role, target)
            if key not in programs:
                raise Ugi3PrecursorLeafClosureError(
                    f"product row {row_number} has an unprojected component: {key!r}"
                )
            parent_counts[key] += 1
            tail_states.append(parent_closed[key])
            historical_tail_states.append(parent_historical_closed[key])
        if all(tail_states):
            two_tail_leaf_closed += 1
            if head in current_heads:
                head_and_two_tail_leaf_closed += 1
        if all(historical_tail_states):
            two_tail_historically_used_leaf_closed += 1
        product_count += 1
    return {
        "product_count": product_count,
        "parent_counts": parent_counts,
        "head_counts": head_counts,
        "parent_closed": parent_closed,
        "two_tail_leaf_closed": two_tail_leaf_closed,
        "head_and_two_tail_leaf_closed": head_and_two_tail_leaf_closed,
        "parent_historical_closed": parent_historical_closed,
        "two_tail_historically_used_leaf_closed": two_tail_historically_used_leaf_closed,
    }


def build_precursor_leaf_closure_audit(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the exact-current starting-material audit and priority ledgers."""

    config = _load_json(config_path, label="precursor leaf audit config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3PrecursorLeafClosureError("unsupported precursor leaf audit config")
    policy = config.get("admission_policy")
    if not isinstance(policy, dict):
        raise Ugi3PrecursorLeafClosureError("precursor leaf audit policy is missing")
    required_true = (
        "current_terminal_requires_exact_constitutional_identity",
        "current_terminal_requires_region_specific_item_level_availability",
    )
    required_false = (
        "expired_procurement_evidence_closes",
        "starting_material_closure_establishes_exact_l2_route",
        "starting_material_closure_establishes_substrate_scope",
        "starting_material_closure_establishes_ugi_product_success",
        "missing_procurement_evidence_is_unavailability",
    )
    if any(policy.get(field) is not True for field in required_true) or any(
        policy.get(field) is not False for field in required_false
    ):
        raise Ugi3PrecursorLeafClosureError("precursor leaf safeguards changed")

    assessment_raw = config.get("assessment_as_of_utc")
    if not isinstance(assessment_raw, str):
        raise Ugi3PrecursorLeafClosureError("assessment_as_of_utc is required")
    assessment_as_of = _parse_utc(assessment_raw, label="assessment time")
    paths = _validated_inputs(config, repo)
    stress = _load_json(paths["template_saturation_stress"], label="template stress result")
    if (
        stress.get("component_census", {}).get(
            "total_unique_projected_tail_leaf_candidates_by_role_sum"
        )
        != 122
    ):
        raise Ugi3PrecursorLeafClosureError("template-stress leaf anchor changed")

    default_expiry_days = config.get("procurement_evidence_default_expiry_days")
    if not isinstance(default_expiry_days, int) or default_expiry_days <= 0:
        raise Ugi3PrecursorLeafClosureError("default procurement expiry must be positive")
    terminals = current_terminal_evidence(
        paths,
        assessment_as_of=assessment_as_of,
        default_expiry_days=default_expiry_days,
        repo=repo,
    )
    historical_leaf_use = historical_leaf_source_use(paths, repo=repo)
    component_rows = _read_component_rows(paths["component_registry"])
    programs, heads, provenance = _component_programs(component_rows)
    current_terminal_keys = set(terminals)
    current_heads = {head for head in heads if constitutional_key(head) in current_terminal_keys}

    try:
        with gzip.open(paths["expanded_products"], "rt", newline="") as handle:
            product_audit = _product_counts(
                handle,
                programs=programs,
                current_terminal_keys=current_terminal_keys,
                current_heads=current_heads,
                historical_leaf_keys=set(historical_leaf_use),
            )
    except (OSError, csv.Error) as exc:
        raise Ugi3PrecursorLeafClosureError("could not audit expanded products") from exc

    leaf_parents: dict[tuple[str, str], set[str]] = defaultdict(set)
    leaf_families: dict[tuple[str, str], set[str]] = defaultdict(set)
    for (role, target), (family, leaves) in programs.items():
        for leaf in leaves:
            key = (role, constitutional_key(leaf))
            leaf_parents[key].add(target)
            leaf_families[key].add(family)

    leaf_rows: list[dict[str, Any]] = []
    unresolved_by_role: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for (role, leaf_key), parents in sorted(leaf_parents.items()):
        evidence = terminals.get(leaf_key)
        historical_records = historical_leaf_use.get((role, leaf_key), [])
        product_leverage = sum(product_audit["parent_counts"][(role, parent)] for parent in parents)
        source_classes = sorted(
            {value for parent in parents for value in provenance[(role, parent)]["source_classes"]}
        )
        source_platforms = sorted(
            {
                value
                for parent in parents
                for value in provenance[(role, parent)]["source_platforms"]
            }
        )
        source_record_ids = sorted(
            {
                value
                for parent in parents
                for value in provenance[(role, parent)]["source_record_ids"]
            }
        )
        if evidence is None:
            unresolved_by_role[role].append((product_leverage, leaf_key))
        leaf_rows.append(
            {
                "schema_version": LEAF_LEDGER_SCHEMA_VERSION,
                "role": role,
                "leaf_constitutional_smiles": leaf_key,
                "program_families_json": json.dumps(sorted(leaf_families[(role, leaf_key)])),
                "parent_component_count": len(parents),
                "parent_components_json": json.dumps(sorted(parents)),
                "parent_source_classes_json": json.dumps(source_classes),
                "parent_source_platforms_json": json.dumps(source_platforms),
                "parent_source_record_ids_json": json.dumps(source_record_ids),
                "enumerated_parent_product_occurrences": product_leverage,
                "current_item_level_terminal": evidence is not None,
                "historical_exact_source_used": bool(historical_records),
                "historical_source_use_record_count": len(historical_records),
                "historical_source_use_evidence_json": json.dumps(
                    historical_records, sort_keys=True
                ),
                "terminal_evidence_source": "" if evidence is None else evidence["source_path"],
                "terminal_evidence_sha256": "" if evidence is None else evidence["source_sha256"],
                "supplier": "" if evidence is None else evidence["supplier"],
                "supplier_item": "" if evidence is None else evidence["supplier_item"],
                "source_url": "" if evidence is None else evidence["source_url"],
                "expires_utc": "" if evidence is None else evidence["expires_utc"],
                "unresolved_priority_rank_within_role": "",
                "disposition": (
                    "exact_current_procurement_terminal"
                    if evidence is not None
                    else "missing_current_procurement_evidence"
                ),
            }
        )
    rank_lookup: dict[tuple[str, str], int] = {}
    for role, values in unresolved_by_role.items():
        for rank, (_, leaf_key) in enumerate(
            sorted(values, key=lambda item: (-item[0], item[1])), start=1
        ):
            rank_lookup[(role, leaf_key)] = rank
    for row in leaf_rows:
        key = (str(row["role"]), str(row["leaf_constitutional_smiles"]))
        if key in rank_lookup:
            row["unresolved_priority_rank_within_role"] = rank_lookup[key]

    head_rows: list[dict[str, Any]] = []
    unresolved_heads = sorted(
        ((product_audit["head_counts"][head], head) for head in heads if head not in current_heads),
        key=lambda item: (-item[0], item[1]),
    )
    head_rank = {head: rank for rank, (_, head) in enumerate(unresolved_heads, start=1)}
    for head in heads:
        key = constitutional_key(head)
        evidence = terminals.get(key)
        head_provenance = provenance[(HEAD_ROLE, head)]
        head_rows.append(
            {
                "schema_version": HEAD_LEDGER_SCHEMA_VERSION,
                "role": HEAD_ROLE,
                "component_smiles": head,
                "constitutional_key": key,
                "source_classes_json": json.dumps(sorted(head_provenance["source_classes"])),
                "source_platforms_json": json.dumps(sorted(head_provenance["source_platforms"])),
                "source_record_ids_json": json.dumps(sorted(head_provenance["source_record_ids"])),
                "enumerated_product_occurrences": product_audit["head_counts"][head],
                "current_item_level_terminal": evidence is not None,
                "terminal_evidence_source": "" if evidence is None else evidence["source_path"],
                "terminal_evidence_sha256": "" if evidence is None else evidence["source_sha256"],
                "supplier": "" if evidence is None else evidence["supplier"],
                "supplier_item": "" if evidence is None else evidence["supplier_item"],
                "source_url": "" if evidence is None else evidence["source_url"],
                "expires_utc": "" if evidence is None else evidence["expires_utc"],
                "unresolved_priority_rank": head_rank.get(head, ""),
                "disposition": (
                    "exact_current_procurement_terminal"
                    if evidence is not None
                    else "missing_current_procurement_or_route_evidence"
                ),
            }
        )

    leaf_counts: dict[str, Counter[str]] = defaultdict(Counter)
    leaf_evidence_matrix: dict[str, Counter[str]] = defaultdict(Counter)
    unresolved_leaf_provenance: Counter[str] = Counter()
    for row in leaf_rows:
        leaf_counts[str(row["role"])][str(row["disposition"])] += 1
        evidence_state = (
            "current_and_historical"
            if row["current_item_level_terminal"] and row["historical_exact_source_used"]
            else (
                "current_only"
                if row["current_item_level_terminal"]
                else (
                    "historical_only"
                    if row["historical_exact_source_used"]
                    else "neither_current_nor_historical"
                )
            )
        )
        leaf_evidence_matrix[str(row["role"])][evidence_state] += 1
        if row["disposition"] == "missing_current_procurement_evidence":
            classes = json.loads(str(row["parent_source_classes_json"]))
            provenance_class = (
                "bounded_capability_only"
                if classes
                and set(classes) <= {"bounded_aldehyde_capability", "bounded_isocyanide_capability"}
                else "has_observed_or_transferred_parent_provenance"
            )
            unresolved_leaf_provenance[provenance_class] += 1
    parent_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for (role, _), closed in product_audit["parent_closed"].items():
        parent_counts[role]["all_projected_leaves_current" if closed else "leaf_gap_present"] += 1

    product_count = int(product_audit["product_count"])
    fully_leaf_closed = int(product_audit["head_and_two_tail_leaf_closed"])
    stress_product_census = stress.get("product_census")
    if not isinstance(stress_product_census, dict):
        raise Ugi3PrecursorLeafClosureError("template-stress product census is missing")
    syntactic_products = stress_product_census.get("products_with_two_tail_structural_programs")
    strict_exact_products = stress_product_census.get("strict_exact_route_complete_products")
    if syntactic_products != product_count or not isinstance(strict_exact_products, int):
        raise Ugi3PrecursorLeafClosureError("template-stress evidence ladder changed")
    boundary_audits = config.get("source_boundary_audits")
    if not isinstance(boundary_audits, list):
        raise Ugi3PrecursorLeafClosureError("source-boundary audits are missing")
    validated_boundary_audits: list[dict[str, Any]] = []
    for audit in boundary_audits:
        if not isinstance(audit, dict):
            raise Ugi3PrecursorLeafClosureError("source-boundary audit is malformed")
        source_input = audit.get("source_input")
        if not isinstance(source_input, str) or source_input not in paths:
            raise Ugi3PrecursorLeafClosureError("source-boundary audit has an unknown input")
        projected_leaf = audit.get("mechanically_projected_leaf_smiles")
        reported_reactant = audit.get("reported_exact_reactant_smiles")
        if not isinstance(projected_leaf, str) or not isinstance(reported_reactant, str):
            raise Ugi3PrecursorLeafClosureError("source-boundary audit lacks exact identities")
        if audit.get("projected_leaf_reported_as_starting_material") is not False:
            raise Ugi3PrecursorLeafClosureError("source-boundary nonpromotion safeguard changed")
        normalized = dict(audit)
        normalized["reported_exact_reactant_constitutional_key"] = constitutional_key(
            reported_reactant
        )
        normalized["mechanically_projected_leaf_constitutional_key"] = constitutional_key(
            projected_leaf
        )
        normalized["source_asset"] = {
            "path": str(paths[source_input].relative_to(repo)),
            "bytes": paths[source_input].stat().st_size,
            "sha256": sha256_file(paths[source_input]),
        }
        validated_boundary_audits.append(normalized)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_exact_current_precursor_leaf_census",
        "task": config.get("task"),
        "assessment": {
            "as_of_utc": assessment_raw,
            "region": config.get("region"),
            "identity": "constitution_only_stereo_free",
        },
        "inputs": {
            label: {
                "path": str(path.relative_to(repo)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for label, path in sorted(paths.items())
        },
        "terminal_identity_census": {
            "unique_current_terminal_identities_across_all_evidence_packs": len(terminals),
            "admitted_heads_with_current_terminal_identity": len(current_heads),
            "admitted_heads_without_current_terminal_identity": len(heads) - len(current_heads),
            "projected_tail_leaves_by_role_and_disposition": {
                role: dict(sorted(counts.items())) for role, counts in sorted(leaf_counts.items())
            },
            "unresolved_tail_leaves_by_parent_provenance": dict(
                sorted(unresolved_leaf_provenance.items())
            ),
            "projected_tail_leaves_by_role_current_x_historical_source_use": {
                role: dict(sorted(counts.items()))
                for role, counts in sorted(leaf_evidence_matrix.items())
            },
        },
        "component_program_census": {
            role: dict(sorted(counts.items())) for role, counts in sorted(parent_counts.items())
        },
        "source_boundary_audits": validated_boundary_audits,
        "product_census": {
            "expanded_unique_products": product_count,
            "products_with_both_tail_programs_leaf_closed": int(
                product_audit["two_tail_leaf_closed"]
            ),
            "fraction_with_both_tail_programs_leaf_closed": (
                product_audit["two_tail_leaf_closed"] / product_count
            ),
            "products_with_head_and_both_tail_programs_leaf_closed": fully_leaf_closed,
            "fraction_with_head_and_both_tail_programs_leaf_closed": (
                fully_leaf_closed / product_count
            ),
            "products_with_both_tail_programs_supported_by_historically_source_used_leaves": int(
                product_audit["two_tail_historically_used_leaf_closed"]
            ),
            "fraction_with_both_tail_programs_supported_by_historically_source_used_leaves": (
                product_audit["two_tail_historically_used_leaf_closed"] / product_count
            ),
        },
        "cross_audit_evidence_ladder": {
            "mechanical_tail_programs_attached": {
                "products": syntactic_products,
                "fraction": syntactic_products / product_count,
                "claim_level": "structural_program_only",
            },
            "current_head_and_projected_tail_leaf_identities_verified": {
                "products": fully_leaf_closed,
                "fraction": fully_leaf_closed / product_count,
                "claim_level": "starting_material_identity_closure_only",
            },
            "strict_exact_route_complete_in_frozen_routeability_census": {
                "products": strict_exact_products,
                "fraction": strict_exact_products / product_count,
                "claim_level": "exact_computational_route_dossier",
            },
            "interpretation": (
                "Template attachment covers the universe, whereas current starting-material "
                "identity and exact-route evidence are much narrower. These are evidence tiers, "
                "not synthesis-success probabilities."
            ),
        },
        "interpretation": {
            "starting_material_closure_is_not_exact_l2_route_evidence": True,
            "mechanical_programs_still_require_exact_substrate_scope_or_prospective_execution": True,
            "leaf_gaps_are_missing_evidence_not_proven_unavailability": True,
            "historical_source_use_is_not_current_procurement": True,
            "source_observation_of_a_parent_component_is_not_source_use_of_its_projected_leaf": True,
            "priority_action": (
                "refresh or locate item-level procurement for the ranked unresolved leaves and "
                "heads, then separately qualify exact-substrate steps"
            ),
        },
        "nonclaims": [
            "A current starting material does not validate the projected reaction on the exact substrate.",
            "A fully leaf-closed structural program is not an exact route-complete component dossier.",
            "Missing current procurement evidence is not proof that a material cannot be obtained.",
            "Historical source use does not establish current procurement or inventory.",
            "A source-observed isocyanide does not establish source use of the amine projected beneath it.",
            "This audit does not establish Ugi conversion, isolation, yield, purity or purification success.",
        ],
        "artifacts": {
            "config": {
                "path": str(config_path.relative_to(repo)),
                "sha256": sha256_file(config_path),
            },
            "audit_source": {
                "path": "src/forge/route/ugi3_precursor_leaf_closure.py",
                "sha256": sha256_file(Path(__file__)),
            },
        },
    }
    return result, leaf_rows, head_rows
