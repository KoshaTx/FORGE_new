"""Audit exact current terminals for two high-leverage generated Ugi heads."""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.core.io import stable_json as _stable_json
from forge.data.r1_prime_audit import sha256_bytes, sha256_file
from forge.route.planner import (
    AvailabilityState,
    EvidenceRecord,
    EvidenceTier,
    ForwardVerificationState,
    KnowledgeDisposition,
    KnowledgeResult,
    RouteKnowledgeSource,
    RouteTarget,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi3_high_leverage_head_terminal_audit_config.v1"
EVIDENCE_SCHEMA_VERSION = "phase1_ugi3_high_leverage_head_terminals.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_high_leverage_head_terminal_audit.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi3_high_leverage_head_terminal_product_impact.v1"


class Ugi3HighLeverageHeadTerminalError(ValueError):
    """Raised when a head-terminal decision cannot be reproduced exactly."""


class HighLeverageHeadTerminalOverlay:
    """Override the two authenticated head identities and delegate all others."""

    def __init__(
        self,
        *,
        base_source: RouteKnowledgeSource,
        terminal_results: dict[tuple[str, str], KnowledgeResult],
    ):
        self._base_source = base_source
        self._terminal_results = dict(terminal_results)

    @property
    def targets(self) -> tuple[RouteTarget, ...]:
        return tuple(
            RouteTarget(role=role, canonical_smiles=smiles)
            for role, smiles in sorted(self._terminal_results)
        )

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        key = (
            target.role,
            _canonical(target.canonical_smiles, label="head-terminal route target"),
        )
        result = self._terminal_results.get(key)
        return result if result is not None else self._base_source.lookup(target)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise Ugi3HighLeverageHeadTerminalError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3HighLeverageHeadTerminalError(f"{label} must be an object")
    return value


def _read_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise Ugi3HighLeverageHeadTerminalError(f"{label} has no header")
            return list(reader)
    except (OSError, csv.Error) as exc:
        raise Ugi3HighLeverageHeadTerminalError(f"could not read {label}") from exc


def _canonical(smiles: Any, *, label: str) -> str:
    if not isinstance(smiles, str) or not smiles:
        raise Ugi3HighLeverageHeadTerminalError(f"{label} must be nonempty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise Ugi3HighLeverageHeadTerminalError(f"{label} contains invalid SMILES")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _parse_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Ugi3HighLeverageHeadTerminalError(f"{label} must be UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Ugi3HighLeverageHeadTerminalError(f"{label} is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise Ugi3HighLeverageHeadTerminalError(f"{label} must resolve to UTC")
    return parsed


def _parse_json_list(value: str, *, label: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise Ugi3HighLeverageHeadTerminalError(f"{label} is invalid JSON") from exc
    if not isinstance(parsed, list):
        raise Ugi3HighLeverageHeadTerminalError(f"{label} must be a list")
    return parsed


def _gzip_csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(buffer.getvalue().encode())
    return output.getvalue()


def _validate_inputs(config: dict[str, Any], input_paths: dict[str, Path]) -> None:
    declared = config.get("inputs")
    if not isinstance(declared, dict) or set(declared) != set(input_paths):
        raise Ugi3HighLeverageHeadTerminalError("configured input set is not exact")
    for label, path in input_paths.items():
        record = declared.get(label)
        if not isinstance(record, dict):
            raise Ugi3HighLeverageHeadTerminalError(f"input {label} is malformed")
        asset = record.get("asset")
        if not isinstance(asset, str) or Path(asset).resolve() != path.resolve():
            raise Ugi3HighLeverageHeadTerminalError(f"input {label} path changed")
        if record.get("expected_sha256") != sha256_file(path):
            raise Ugi3HighLeverageHeadTerminalError(f"input {label} hash changed")


def build_high_leverage_head_terminal_audit(
    *,
    config_path: Path,
    input_paths: dict[str, Path],
) -> tuple[dict[str, Any], bytes]:
    """Reproduce the exact-identity and generated-product closure audit."""

    config = _load_json(config_path, label="head-terminal audit config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3HighLeverageHeadTerminalError("unsupported audit config schema")
    _validate_inputs(config, input_paths)
    evidence = _load_json(input_paths["evidence_pack"], label="head-terminal evidence")
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise Ugi3HighLeverageHeadTerminalError("unsupported evidence schema")
    snapshot = evidence.get("snapshot")
    records = evidence.get("records")
    if not isinstance(snapshot, dict) or not isinstance(records, list) or len(records) != 2:
        raise Ugi3HighLeverageHeadTerminalError("evidence snapshot must contain two records")
    accessed = _parse_utc(snapshot.get("accessed_utc"), label="accessed_utc")
    expires = _parse_utc(snapshot.get("expires_utc"), label="expires_utc")
    if expires <= accessed:
        raise Ugi3HighLeverageHeadTerminalError("availability snapshot does not expire later")

    readiness = _read_csv(input_paths["readiness_ledger"], label="readiness ledger")
    readiness_by_id = {row["component_id"]: row for row in readiness}
    admitted_keys: set[str] = set()
    admitted_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise Ugi3HighLeverageHeadTerminalError("evidence record must be an object")
        component_id = record.get("component_id")
        row = readiness_by_id.get(component_id)
        if row is None:
            raise Ugi3HighLeverageHeadTerminalError("evidence target is absent from registry")
        canonical = _canonical(record.get("canonical_smiles"), label="evidence target")
        if (
            row["role"] != "amine_head"
            or row["canonical_smiles"] != canonical
            or record.get("role") != row["role"]
            or record.get("disposition") != "admit_exact_terminal"
            or record.get("availability") != "available_to_ship_today"
            or not isinstance(record.get("available_skus"), list)
            or not record["available_skus"]
            or not isinstance(record.get("source_url"), str)
            or not record["source_url"].startswith("https://www.sigmaaldrich.com/US/")
        ):
            raise Ugi3HighLeverageHeadTerminalError("head evidence is not exact and current")
        if row["route_complete_component"] != "false":
            raise Ugi3HighLeverageHeadTerminalError("target was already route complete")
        key = f"amine_head\t{canonical}"
        if key in admitted_keys or component_id in admitted_ids:
            raise Ugi3HighLeverageHeadTerminalError("duplicate head evidence target")
        admitted_keys.add(key)
        admitted_ids.add(component_id)

    prior = _read_csv(input_paths["prior_product_impact"], label="prior product impact")
    prior_result = _load_json(input_paths["prior_result"], label="prior role-gap result")
    prior_summary = prior_result.get("summary")
    if not isinstance(prior_summary, dict):
        raise Ugi3HighLeverageHeadTerminalError("prior role-gap result lacks its summary")
    registry_complete_before = prior_summary.get("registry_complete_components_after")
    if isinstance(registry_complete_before, bool) or not isinstance(registry_complete_before, int):
        raise Ugi3HighLeverageHeadTerminalError("prior registry closure count is invalid")
    rows: list[dict[str, Any]] = []
    newly_by_key: Counter[str] = Counter()
    occurrences: Counter[str] = Counter()
    gap_distribution: Counter[int] = Counter()
    complete_before = 0
    complete_after = 0
    for row in prior:
        gaps = _parse_json_list(
            row["remaining_component_keys_json"], label="remaining component keys"
        )
        if any(not isinstance(key, str) for key in gaps):
            raise Ugi3HighLeverageHeadTerminalError("component gap key must be text")
        gap_set = set(gaps)
        matched = sorted(gap_set & admitted_keys)
        for key in matched:
            occurrences[key] += 1
        remaining = sorted(gap_set - admitted_keys)
        before = row["complete_after_role_gap_evidence"].lower() == "true"
        after = not remaining
        newly = after and not before
        complete_before += int(before)
        complete_after += int(after)
        gap_distribution[len(remaining)] += 1
        if newly:
            for key in matched:
                newly_by_key[key] += 1
        rows.append(
            {
                "sample_index": row["sample_index"],
                "structure_id": row["structure_id"],
                "product_id": row["product_id"],
                "prior_remaining_component_count": row["remaining_component_count"],
                "newly_closed_head_keys_json": _stable_json(matched),
                "remaining_component_keys_json": _stable_json(remaining),
                "remaining_component_count": len(remaining),
                "complete_before_head_terminals": before,
                "complete_after_head_terminals": after,
                "newly_complete_from_head_terminals": newly,
            }
        )

    fields = [
        "sample_index",
        "structure_id",
        "product_id",
        "prior_remaining_component_count",
        "newly_closed_head_keys_json",
        "remaining_component_keys_json",
        "remaining_component_count",
        "complete_before_head_terminals",
        "complete_after_head_terminals",
        "newly_complete_from_head_terminals",
    ]
    ledger = _gzip_csv_bytes(rows, fields)
    summary = {
        "exact_head_terminals_admitted": len(admitted_keys),
        "registry_complete_components_before": registry_complete_before,
        "registry_complete_components_after": registry_complete_before + len(admitted_keys),
        "generated_products": len(rows),
        "generated_products_complete_before": complete_before,
        "generated_products_complete_after": complete_after,
        "newly_complete_generated_products": complete_after - complete_before,
        "generated_target_occurrences": sum(occurrences.values()),
        "target_occurrences": dict(sorted(occurrences.items())),
        "newly_complete_products_by_target": dict(sorted(newly_by_key.items())),
        "remaining_gap_count_distribution": {
            str(key): value for key, value in sorted(gap_distribution.items())
        },
    }
    if summary != config.get("expected_summary"):
        raise Ugi3HighLeverageHeadTerminalError("head-terminal summary changed")
    return (
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "assessment_as_of_utc": snapshot["accessed_utc"],
            "availability_expires_utc": snapshot["expires_utc"],
            "summary": summary,
            "policy": config.get("audit_policy"),
            "inputs": {
                label: {"path": str(path), "sha256": sha256_file(path)}
                for label, path in sorted(input_paths.items())
            },
            "artifacts": {
                "product_impact_ledger.csv.gz": {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "sha256": sha256_bytes(ledger),
                }
            },
            "limitations": [
                "These exact records close current L3 procurement only; they do not establish experimental Ugi success.",
                "The source pages were rendered through a web index because direct HTTP archival failed; candidate lock requires a fresh availability check.",
                "This bounded audit does not authorize production synthesis guidance.",
            ],
        },
        ledger,
    )


def load_high_leverage_head_terminal_overlay(
    *,
    base_source: RouteKnowledgeSource,
    audit_config_path: Path,
    audit_input_paths: dict[str, Path],
    stored_audit_result_path: Path,
    stored_product_ledger_path: Path,
) -> tuple[HighLeverageHeadTerminalOverlay, dict[str, Any]]:
    """Authenticate the audit and expose its exact terminal decisions."""

    fresh_result, fresh_ledger = build_high_leverage_head_terminal_audit(
        config_path=audit_config_path,
        input_paths=audit_input_paths,
    )
    stored_result = _load_json(stored_audit_result_path, label="stored head-terminal result")
    if fresh_result != stored_result:
        raise Ugi3HighLeverageHeadTerminalError("stored head-terminal result is not reproducible")
    ledger_hash = sha256_bytes(fresh_ledger)
    if sha256_file(stored_product_ledger_path) != ledger_hash:
        raise Ugi3HighLeverageHeadTerminalError("stored head-terminal ledger hash mismatch")
    if (
        stored_result.get("artifacts", {}).get("product_impact_ledger.csv.gz", {}).get("sha256")
        != ledger_hash
    ):
        raise Ugi3HighLeverageHeadTerminalError("stored result does not own its ledger")

    evidence_path = audit_input_paths["evidence_pack"]
    evidence = _load_json(evidence_path, label="head-terminal evidence")
    records = evidence.get("records")
    if not isinstance(records, list):
        raise Ugi3HighLeverageHeadTerminalError("head-terminal evidence records are missing")
    terminal_results: dict[tuple[str, str], KnowledgeResult] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("disposition") != "admit_exact_terminal":
            raise Ugi3HighLeverageHeadTerminalError("head-terminal disposition changed")
        key = (
            str(record.get("role")),
            _canonical(record.get("canonical_smiles"), label="head-terminal target"),
        )
        evidence_record = EvidenceRecord(
            evidence_id=f"high-leverage-head-terminal:{record.get('component_id')}",
            tier=EvidenceTier.ACCEPTED_TERMINAL,
            source_sha256=sha256_file(evidence_path),
            source_locator=f"{evidence_path}#records[{index}]",
            exact_substrate=True,
            forward_verification=ForwardVerificationState.NOT_APPLICABLE,
            availability=AvailabilityState.CURRENT_CLOSED,
        )
        if key in terminal_results:
            raise Ugi3HighLeverageHeadTerminalError("duplicate head-terminal overlay key")
        terminal_results[key] = KnowledgeResult(
            disposition=KnowledgeDisposition.TERMINAL,
            evidence=(evidence_record,),
            detail="high-leverage head has exact current supplier evidence",
        )
    if len(terminal_results) != 2:
        raise Ugi3HighLeverageHeadTerminalError("overlay must contain exactly two terminals")
    overlay = HighLeverageHeadTerminalOverlay(
        base_source=base_source,
        terminal_results=terminal_results,
    )
    return overlay, {
        "schema_version": "phase1_ugi3_high_leverage_head_terminal_overlay.v1",
        "evidence_pack_sha256": sha256_file(evidence_path),
        "audit_result_sha256": sha256_file(stored_audit_result_path),
        "product_ledger_sha256": ledger_hash,
        "selected_exact_head_terminals": len(terminal_results),
        "targets": [target.to_dict() for target in overlay.targets],
    }
