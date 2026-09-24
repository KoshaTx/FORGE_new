"""Audit a second exact-current procurement wave for generated Ugi heads."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_bytes, sha256_file
from forge.core.input_locations import (
    input_location_matches,
    results_match_after_input_relocation,
)
from forge.core.io import stable_json as _stable_json
from forge.synthesis.engine.planner import (
    AvailabilityState,
    EvidenceRecord,
    EvidenceTier,
    ForwardVerificationState,
    KnowledgeDisposition,
    KnowledgeResult,
    RouteKnowledgeSource,
    RouteTarget,
)
from forge.synthesis.terminals.ugi3_high_leverage_head_terminals import (
    _canonical,
    _gzip_csv_bytes,
    _load_json,
    _parse_json_list,
    _parse_utc,
    _read_csv,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi3_second_wave_head_terminal_audit_config.v1"
EVIDENCE_SCHEMA_VERSION = "phase1_ugi3_second_wave_head_terminals.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_second_wave_head_terminal_audit.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi3_second_wave_head_terminal_product_impact.v1"


class Ugi3SecondWaveHeadTerminalError(ValueError):
    """Raised when the second-wave terminal decision is not reproducible."""


class SecondWaveHeadTerminalOverlay:
    """Override authenticated exact head identities and delegate all others."""

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
            _canonical(target.canonical_smiles, label="second-wave route target"),
        )
        result = self._terminal_results.get(key)
        return result if result is not None else self._base_source.lookup(target)


def _validate_inputs(config: dict[str, Any], input_paths: dict[str, Path]) -> None:
    declared = config.get("inputs")
    if not isinstance(declared, dict) or set(declared) != set(input_paths):
        raise Ugi3SecondWaveHeadTerminalError("configured input set is not exact")
    for label, path in input_paths.items():
        record = declared.get(label)
        if not isinstance(record, dict):
            raise Ugi3SecondWaveHeadTerminalError(f"input {label} is malformed")
        asset = record.get("asset")
        if not input_location_matches(Path.cwd(), path, asset, record.get("expected_sha256")):
            raise Ugi3SecondWaveHeadTerminalError(
                f"input {label} location or pinned content invalid: {path}"
            )
        if record.get("expected_sha256") != sha256_file(path):
            raise Ugi3SecondWaveHeadTerminalError(f"input {label} hash changed")


def _validate_supplier_record(
    record: dict[str, Any],
    *,
    readiness_by_id: dict[str, dict[str, str]],
) -> tuple[str, str]:
    component_id = record.get("component_id")
    row = readiness_by_id.get(component_id)
    if row is None:
        raise Ugi3SecondWaveHeadTerminalError("evidence target is absent from registry")
    canonical = _canonical(record.get("canonical_smiles"), label="evidence target")
    required_text = (
        "name",
        "cas",
        "inchi_key",
        "supplier",
        "product_number",
        "source_url",
        "assay",
    )
    if any(not isinstance(record.get(field), str) or not record[field] for field in required_text):
        raise Ugi3SecondWaveHeadTerminalError("supplier identity fields are incomplete")
    if (
        row["role"] != "amine_head"
        or row["canonical_smiles"] != canonical
        or record.get("role") != row["role"]
        or record.get("disposition") != "admit_exact_terminal"
        or record.get("availability")
        not in {
            "available_to_ship_today",
            "available_to_ship_on_dated_snapshot",
        }
        or not isinstance(record.get("available_skus"), list)
        or not record["available_skus"]
        or not record["source_url"].startswith("https://www.sigmaaldrich.com/US/")
    ):
        raise Ugi3SecondWaveHeadTerminalError("head evidence is not exact and current")
    if row["route_complete_component"] != "false":
        raise Ugi3SecondWaveHeadTerminalError("target was already route complete")
    return component_id, f"amine_head\t{canonical}"


def build_second_wave_head_terminal_audit(
    *,
    config_path: Path,
    input_paths: dict[str, Path],
) -> tuple[dict[str, Any], bytes]:
    """Reproduce exact procurement decisions and generated-product impact."""

    config = _load_json(config_path, label="second-wave audit config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3SecondWaveHeadTerminalError("unsupported audit config schema")
    _validate_inputs(config, input_paths)
    evidence = _load_json(input_paths["evidence_pack"], label="second-wave evidence")
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise Ugi3SecondWaveHeadTerminalError("unsupported evidence schema")
    snapshot = evidence.get("snapshot")
    records = evidence.get("records")
    if not isinstance(snapshot, dict) or not isinstance(records, list) or not records:
        raise Ugi3SecondWaveHeadTerminalError("evidence snapshot must contain records")
    accessed = _parse_utc(snapshot.get("accessed_utc"), label="accessed_utc")
    expires = _parse_utc(snapshot.get("expires_utc"), label="expires_utc")
    if expires <= accessed:
        raise Ugi3SecondWaveHeadTerminalError("availability snapshot does not expire later")

    readiness = _read_csv(input_paths["readiness_ledger"], label="readiness ledger")
    readiness_by_id = {row["component_id"]: row for row in readiness}
    admitted_keys: set[str] = set()
    admitted_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise Ugi3SecondWaveHeadTerminalError("evidence record must be an object")
        component_id, key = _validate_supplier_record(
            record,
            readiness_by_id=readiness_by_id,
        )
        if key in admitted_keys or component_id in admitted_ids:
            raise Ugi3SecondWaveHeadTerminalError("duplicate evidence target")
        admitted_keys.add(key)
        admitted_ids.add(component_id)

    prior = _read_csv(input_paths["prior_product_impact"], label="prior product impact")
    prior_result = _load_json(input_paths["prior_result"], label="prior audit result")
    prior_summary = prior_result.get("summary")
    if not isinstance(prior_summary, dict):
        raise Ugi3SecondWaveHeadTerminalError("prior result lacks its summary")
    registry_complete_before = prior_summary.get("registry_complete_components_after")
    if isinstance(registry_complete_before, bool) or not isinstance(registry_complete_before, int):
        raise Ugi3SecondWaveHeadTerminalError("prior registry closure count is invalid")

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
            raise Ugi3SecondWaveHeadTerminalError("component gap key must be text")
        gap_set = set(gaps)
        matched = sorted(gap_set & admitted_keys)
        for key in matched:
            occurrences[key] += 1
        remaining = sorted(gap_set - admitted_keys)
        before = row["complete_after_head_terminals"].lower() == "true"
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
                "complete_before_second_wave": before,
                "complete_after_second_wave": after,
                "newly_complete_from_second_wave": newly,
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
        "complete_before_second_wave",
        "complete_after_second_wave",
        "newly_complete_from_second_wave",
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
        raise Ugi3SecondWaveHeadTerminalError("second-wave summary changed")
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
                "Direct HTTP archival failed; the frozen fields derive from current primary supplier pages rendered through the web index.",
                "Candidate lock requires a fresh item-level availability check.",
                "This bounded audit does not authorize production synthesis guidance.",
            ],
        },
        ledger,
    )


def load_second_wave_head_terminal_overlay(
    *,
    base_source: RouteKnowledgeSource,
    audit_config_path: Path,
    audit_input_paths: dict[str, Path],
    stored_audit_result_path: Path,
    stored_product_ledger_path: Path,
) -> tuple[SecondWaveHeadTerminalOverlay, dict[str, Any]]:
    """Authenticate the second-wave audit and expose exact terminal decisions."""

    fresh_result, fresh_ledger = build_second_wave_head_terminal_audit(
        config_path=audit_config_path,
        input_paths=audit_input_paths,
    )
    stored_result = _load_json(stored_audit_result_path, label="stored second-wave result")
    if not results_match_after_input_relocation(
        stored_result,
        fresh_result,
        repo=Path.cwd(),
        specifications=_load_json(audit_config_path, label="audit config")["inputs"],
        input_paths=audit_input_paths,
    ):
        raise Ugi3SecondWaveHeadTerminalError("stored second-wave result is not reproducible")
    ledger_hash = sha256_bytes(fresh_ledger)
    if sha256_file(stored_product_ledger_path) != ledger_hash:
        raise Ugi3SecondWaveHeadTerminalError("stored second-wave ledger hash mismatch")
    if (
        stored_result.get("artifacts", {}).get("product_impact_ledger.csv.gz", {}).get("sha256")
        != ledger_hash
    ):
        raise Ugi3SecondWaveHeadTerminalError("stored result does not own its ledger")

    evidence_path = audit_input_paths["evidence_pack"]
    evidence = _load_json(evidence_path, label="second-wave evidence")
    records = evidence.get("records")
    if not isinstance(records, list):
        raise Ugi3SecondWaveHeadTerminalError("second-wave evidence records are missing")
    terminal_results: dict[tuple[str, str], KnowledgeResult] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("disposition") != "admit_exact_terminal":
            raise Ugi3SecondWaveHeadTerminalError("second-wave disposition changed")
        key = (
            str(record.get("role")),
            _canonical(record.get("canonical_smiles"), label="second-wave target"),
        )
        terminal_results[key] = KnowledgeResult(
            disposition=KnowledgeDisposition.TERMINAL,
            evidence=(
                EvidenceRecord(
                    evidence_id=f"second-wave-head-terminal:{record.get('component_id')}",
                    tier=EvidenceTier.ACCEPTED_TERMINAL,
                    source_sha256=sha256_file(evidence_path),
                    source_locator=f"{evidence_path}#records[{index}]",
                    exact_substrate=True,
                    forward_verification=ForwardVerificationState.NOT_APPLICABLE,
                    availability=AvailabilityState.CURRENT_CLOSED,
                ),
            ),
            detail="second-wave head has exact current supplier evidence",
        )
    expected_count = fresh_result.get("summary", {}).get("exact_head_terminals_admitted")
    if len(terminal_results) != expected_count:
        raise Ugi3SecondWaveHeadTerminalError("overlay terminal count changed")
    overlay = SecondWaveHeadTerminalOverlay(
        base_source=base_source,
        terminal_results=terminal_results,
    )
    return overlay, {
        "schema_version": "phase1_ugi3_second_wave_head_terminal_overlay.v1",
        "evidence_pack_sha256": sha256_file(evidence_path),
        "audit_result_sha256": sha256_file(stored_audit_result_path),
        "product_ledger_sha256": ledger_hash,
        "selected_exact_head_terminals": len(terminal_results),
        "targets": [target.to_dict() for target in overlay.targets],
    }
