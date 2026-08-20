"""Extend frozen Ugi synthesis values with a versioned exact-terminal wave.

Version 1 remains byte-for-byte immutable. This audit authenticates that frozen
base, applies only the separately qualified second-wave terminal overlay, and
recomputes component and product records without introducing a scalar score or
synthesis-success probability.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_bytes, sha256_file
from forge.core.io import read_json_object
from forge.route.engine.planner import (
    KnowledgeDisposition,
    KnowledgeResult,
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RecursiveRouteAssessor,
    RouteTarget,
)
from forge.route.terminals.ugi3_second_wave_head_terminals import (
    load_second_wave_head_terminal_overlay,
)
from forge.value.synthesis.synthesis import (
    ComponentSynthesisValue,
    ProductSynthesisValue,
    component_synthesis_value_from_assessment,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi3_synthesis_value_audit_config.v2"
RESULT_SCHEMA_VERSION = "phase1_ugi3_synthesis_value_audit.v2"
COMPONENT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_component_synthesis_values.v2"
PRODUCT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_product_synthesis_values.v2"


class Ugi3SynthesisValueAuditV2Error(ValueError):
    """Raised when the versioned value extension is not reproducible."""


class _MissingSource:
    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        return KnowledgeResult(
            disposition=KnowledgeDisposition.MISSING_KNOWLEDGE,
            evidence=(),
            detail="identity is outside the exact second-wave terminal overlay",
        )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=Ugi3SynthesisValueAuditV2Error, label=label)


def _load_gzip_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise Ugi3SynthesisValueAuditV2Error(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3SynthesisValueAuditV2Error(f"{label} must be an object")
    return value


def _read_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise Ugi3SynthesisValueAuditV2Error(f"{label} has no header")
            return list(reader)
    except (OSError, csv.Error) as exc:
        raise Ugi3SynthesisValueAuditV2Error(f"could not read {label}") from exc


def _gzip_json_bytes(value: Any) -> bytes:
    raw = (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(raw)
    return output.getvalue()


def _validate_inputs(config: dict[str, Any], input_paths: dict[str, Path]) -> None:
    declared = config.get("inputs")
    if not isinstance(declared, dict) or set(declared) != set(input_paths):
        raise Ugi3SynthesisValueAuditV2Error("configured input set is not exact")
    for label, path in input_paths.items():
        record = declared.get(label)
        if not isinstance(record, dict):
            raise Ugi3SynthesisValueAuditV2Error(f"input {label} is malformed")
        asset = record.get("asset")
        if not isinstance(asset, str) or Path(asset).resolve() != path.resolve():
            raise Ugi3SynthesisValueAuditV2Error(f"input {label} path changed")
        if record.get("expected_sha256") != sha256_file(path):
            raise Ugi3SynthesisValueAuditV2Error(f"input {label} hash changed")


def _authenticate_base(input_paths: dict[str, Path]) -> tuple[list[dict[str, Any]], int]:
    result = _load_json(input_paths["base_result"], label="base result")
    components = _load_gzip_json(
        input_paths["base_component_values"], label="base component values"
    )
    products = _load_gzip_json(input_paths["base_product_values"], label="base product values")
    component_records = components.get("records")
    product_records = products.get("records")
    if not isinstance(component_records, list) or not isinstance(product_records, list):
        raise Ugi3SynthesisValueAuditV2Error("base ledgers lack records")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        raise Ugi3SynthesisValueAuditV2Error("base result lacks artifacts")
    expected_component_hash = artifacts.get("component_synthesis_values.json.gz", {}).get("sha256")
    expected_product_hash = artifacts.get("product_synthesis_values.json.gz", {}).get("sha256")
    if expected_component_hash != sha256_file(input_paths["base_component_values"]):
        raise Ugi3SynthesisValueAuditV2Error("base result does not own component ledger")
    if expected_product_hash != sha256_file(input_paths["base_product_values"]):
        raise Ugi3SynthesisValueAuditV2Error("base result does not own product ledger")
    summary = result.get("summary")
    if not isinstance(summary, dict) or summary.get("generated_products") != len(product_records):
        raise Ugi3SynthesisValueAuditV2Error("base result denominator changed")
    return component_records, len(product_records)


def _second_wave_component_value(
    *,
    input_paths: dict[str, Path],
    second_wave_input_paths: dict[str, Path],
) -> tuple[tuple[str, str], ComponentSynthesisValue, dict[str, Any]]:
    overlay, metadata = load_second_wave_head_terminal_overlay(
        base_source=_MissingSource(),
        audit_config_path=input_paths["second_wave_config"],
        audit_input_paths=second_wave_input_paths,
        stored_audit_result_path=input_paths["second_wave_result"],
        stored_product_ledger_path=input_paths["second_wave_product_impact"],
    )
    targets = overlay.targets
    if len(targets) != 1:
        raise Ugi3SynthesisValueAuditV2Error("v2 expects one exact second-wave target")
    target = targets[0]
    limits = PlannerBudgetLimits(
        maximum_depth=1,
        maximum_expansions=1,
        maximum_product_candidates=1,
        maximum_verifier_calls=0,
        maximum_elapsed_milliseconds=0,
        maximum_logical_planner_calls=1,
    )
    assessment = RecursiveRouteAssessor(overlay).assess(
        target,
        PlannerBudgetLedger(limits),
    )
    value = component_synthesis_value_from_assessment(assessment)
    if not value.route_complete:
        raise Ugi3SynthesisValueAuditV2Error("qualified second-wave target did not close")
    return (target.role, target.canonical_smiles), value, metadata


def build_ugi3_synthesis_value_audit_v2(
    *,
    config_path: Path,
    input_paths: dict[str, Path],
    second_wave_input_paths: dict[str, Path],
) -> tuple[dict[str, Any], bytes, bytes]:
    """Apply the authenticated exact-terminal delta to frozen v1 values."""

    config = _load_json(config_path, label="synthesis-value v2 config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3SynthesisValueAuditV2Error("unsupported v2 config schema")
    _validate_inputs(config, input_paths)
    base_components, base_product_count = _authenticate_base(input_paths)
    target_key, target_value, overlay_metadata = _second_wave_component_value(
        input_paths=input_paths,
        second_wave_input_paths=second_wave_input_paths,
    )

    component_records: list[dict[str, Any]] = []
    generated_values: dict[tuple[str, str], ComponentSynthesisValue] = {}
    changed = 0
    source_counts: Counter[str] = Counter()
    for row in base_components:
        if not isinstance(row, dict):
            raise Ugi3SynthesisValueAuditV2Error("base component record is malformed")
        key = (str(row.get("role")), str(row.get("canonical_smiles")))
        value = ComponentSynthesisValue.from_dict(row.get("value"))
        output = dict(row)
        if key == target_key:
            if value.route_complete:
                raise Ugi3SynthesisValueAuditV2Error("second-wave target was complete in v1")
            value = target_value
            output["source_class"] = "second_wave_exact_current_terminal"
            output["value"] = value.to_dict()
            changed += 1
        generated_values[key] = value
        source_counts[str(output.get("source_class"))] += 1
        component_records.append(output)
    if changed != 1:
        raise Ugi3SynthesisValueAuditV2Error("second-wave target did not match exactly once")

    generated_components = _read_csv(
        input_paths["generated_component_provenance"],
        label="generated component provenance",
    )
    generated_products = _read_csv(
        input_paths["generated_product_provenance"],
        label="generated product provenance",
    )
    if len(generated_products) != base_product_count:
        raise Ugi3SynthesisValueAuditV2Error("generated product denominator changed")
    components_by_sample: dict[str, dict[str, ComponentSynthesisValue]] = {}
    for row in generated_components:
        key = (row["role"], row["canonical_component_smiles"])
        value = generated_values.get(key)
        if value is None:
            raise Ugi3SynthesisValueAuditV2Error("generated component is absent from v2 values")
        role_values = components_by_sample.setdefault(row["sample_index"], {})
        if row["role"] in role_values:
            raise Ugi3SynthesisValueAuditV2Error("sample contains duplicate component role")
        role_values[row["role"]] = value

    impact = _read_csv(input_paths["second_wave_product_impact"], label="second-wave impact")
    impact_by_sample = {row["sample_index"]: row for row in impact}
    product_records: list[dict[str, Any]] = []
    product_outcomes: Counter[str] = Counter()
    for row in generated_products:
        sample = row["sample_index"]
        role_values = components_by_sample.get(sample)
        if role_values is None or set(role_values) != {
            "amine_head",
            "oxoester_aldehyde_body_tail",
            "isocyanide_tail",
        }:
            raise Ugi3SynthesisValueAuditV2Error("product does not have exactly three roles")
        value = ProductSynthesisValue(
            product_smiles=row["canonical_product_smiles"],
            l1_forward_consistent=True,
            components=tuple(sorted(role_values.items())),
        )
        impact_row = impact_by_sample.get(sample)
        if (
            impact_row is None
            or (impact_row["complete_after_second_wave"].lower() == "true") != value.route_complete
        ):
            raise Ugi3SynthesisValueAuditV2Error(
                "v2 product value disagrees with second-wave impact ledger"
            )
        product_outcomes["complete" if value.route_complete else "noncomplete"] += 1
        product_records.append(
            {
                "sample_index": int(sample),
                "structure_id": row["structure_id"],
                "product_id": row["product_id"],
                "product_structural_provenance_stratum": row[
                    "product_structural_provenance_stratum"
                ],
                "value": value.to_dict(),
            }
        )

    component_payload = {
        "schema_version": COMPONENT_LEDGER_SCHEMA_VERSION,
        "records": component_records,
    }
    product_payload = {
        "schema_version": PRODUCT_LEDGER_SCHEMA_VERSION,
        "records": product_records,
    }
    component_ledger = _gzip_json_bytes(component_payload)
    product_ledger = _gzip_json_bytes(product_payload)
    component_outcomes = Counter(
        value.assessment_outcome.value for value in generated_values.values()
    )
    evidence_support = Counter(value.evidence_support.value for value in generated_values.values())
    summary = {
        "base_version": "v1",
        "second_wave_components_changed": changed,
        "generated_unique_components": len(generated_values),
        "generated_component_value_sources": dict(sorted(source_counts.items())),
        "generated_component_outcomes": dict(sorted(component_outcomes.items())),
        "generated_component_evidence_support": dict(sorted(evidence_support.items())),
        "generated_products": len(product_records),
        "generated_product_outcomes": dict(sorted(product_outcomes.items())),
        "products_with_exact_l1_forward_consistency": sum(
            record["value"]["l1_forward_consistent"] for record in product_records
        ),
        "component_values_with_unknown_protection_burden": sum(
            value.protection_burden.count is None for value in generated_values.values()
        ),
        "component_values_with_unknown_purification_burden": sum(
            value.purification_burden.count is None for value in generated_values.values()
        ),
        "non_null_scalar_values": sum(
            record["value"]["scalar_value"] is not None for record in product_records
        ),
        "non_null_success_probabilities": sum(
            record["value"]["success_probability"] is not None for record in product_records
        ),
    }
    if summary != config.get("expected_summary"):
        raise Ugi3SynthesisValueAuditV2Error("synthesis-value v2 summary changed")
    return (
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "diagnostic_only_not_authorized_for_production_guidance",
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "summary": summary,
            "overlay_metadata": {"second_wave_heads": overlay_metadata},
            "inputs": {
                label: {"path": str(path), "sha256": sha256_file(path)}
                for label, path in sorted(input_paths.items())
            },
            "artifacts": {
                "component_synthesis_values.json.gz": {
                    "schema_version": COMPONENT_LEDGER_SCHEMA_VERSION,
                    "sha256": sha256_bytes(component_ledger),
                },
                "product_synthesis_values.json.gz": {
                    "schema_version": PRODUCT_LEDGER_SCHEMA_VERSION,
                    "sha256": sha256_bytes(product_ledger),
                },
            },
            "nonclaims": [
                "The records are not synthesis-success probabilities.",
                "Unknown protection and purification burdens remain unknown rather than zero.",
                "Missing route knowledge is not chemical incompatibility.",
                "The exact terminal snapshot expires and requires refresh before candidate lock.",
                "The diagnostic does not authorize synthesis-guided candidate generation.",
            ],
        },
        component_ledger,
        product_ledger,
    )
