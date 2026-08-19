"""Extend immutable Ugi synthesis values with the third exact-terminal wave."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_bytes, sha256_file
from forge.route.planner import (
    KnowledgeDisposition,
    KnowledgeResult,
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RecursiveRouteAssessor,
    RouteTarget,
)
from forge.route.ugi3_third_wave_head_terminals import (
    load_third_wave_head_terminal_overlay,
)
from forge.value.synthesis import (
    ComponentSynthesisValue,
    ProductSynthesisValue,
    component_synthesis_value_from_assessment,
)
from forge.value.ugi3_synthesis_value_audit_v2 import (
    _authenticate_base,
    _gzip_json_bytes,
    _load_json,
    _read_csv,
    _validate_inputs,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi3_synthesis_value_audit_config.v3"
RESULT_SCHEMA_VERSION = "phase1_ugi3_synthesis_value_audit.v3"
COMPONENT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_component_synthesis_values.v3"
PRODUCT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_product_synthesis_values.v3"


class Ugi3SynthesisValueAuditV3Error(ValueError):
    """Raised when the versioned v3 value extension is not reproducible."""


class _MissingSource:
    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        return KnowledgeResult(
            disposition=KnowledgeDisposition.MISSING_KNOWLEDGE,
            evidence=(),
            detail="identity is outside the exact third-wave terminal overlay",
        )


def _third_wave_component_value(
    *,
    input_paths: dict[str, Path],
    third_wave_input_paths: dict[str, Path],
) -> tuple[tuple[str, str], ComponentSynthesisValue, dict[str, Any]]:
    overlay, metadata = load_third_wave_head_terminal_overlay(
        base_source=_MissingSource(),
        audit_config_path=input_paths["third_wave_config"],
        audit_input_paths=third_wave_input_paths,
        stored_audit_result_path=input_paths["third_wave_result"],
        stored_product_ledger_path=input_paths["third_wave_product_impact"],
    )
    targets = overlay.targets
    if len(targets) != 1:
        raise Ugi3SynthesisValueAuditV3Error("v3 expects one exact third-wave target")
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
        raise Ugi3SynthesisValueAuditV3Error("qualified third-wave target did not close")
    return (target.role, target.canonical_smiles), value, metadata


def build_ugi3_synthesis_value_audit_v3(
    *,
    config_path: Path,
    input_paths: dict[str, Path],
    third_wave_input_paths: dict[str, Path],
) -> tuple[dict[str, Any], bytes, bytes]:
    """Apply the authenticated third-wave terminal delta to frozen v2 values."""

    config = _load_json(config_path, label="synthesis-value v3 config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3SynthesisValueAuditV3Error("unsupported v3 config schema")
    _validate_inputs(config, input_paths)
    base_components, base_product_count = _authenticate_base(input_paths)
    target_key, target_value, overlay_metadata = _third_wave_component_value(
        input_paths=input_paths,
        third_wave_input_paths=third_wave_input_paths,
    )

    component_records: list[dict[str, Any]] = []
    generated_values: dict[tuple[str, str], ComponentSynthesisValue] = {}
    changed = 0
    source_counts: Counter[str] = Counter()
    for row in base_components:
        if not isinstance(row, dict):
            raise Ugi3SynthesisValueAuditV3Error("base component record is malformed")
        key = (str(row.get("role")), str(row.get("canonical_smiles")))
        value = ComponentSynthesisValue.from_dict(row.get("value"))
        output = dict(row)
        if key == target_key:
            if value.route_complete:
                raise Ugi3SynthesisValueAuditV3Error("third-wave target was complete in v2")
            value = target_value
            output["source_class"] = "third_wave_exact_current_terminal"
            output["value"] = value.to_dict()
            changed += 1
        generated_values[key] = value
        source_counts[str(output.get("source_class"))] += 1
        component_records.append(output)
    if changed != 1:
        raise Ugi3SynthesisValueAuditV3Error("third-wave target did not match exactly once")

    generated_components = _read_csv(
        input_paths["generated_component_provenance"],
        label="generated component provenance",
    )
    generated_products = _read_csv(
        input_paths["generated_product_provenance"],
        label="generated product provenance",
    )
    if len(generated_products) != base_product_count:
        raise Ugi3SynthesisValueAuditV3Error("generated product denominator changed")
    components_by_sample: dict[str, dict[str, ComponentSynthesisValue]] = {}
    for row in generated_components:
        key = (row["role"], row["canonical_component_smiles"])
        value = generated_values.get(key)
        if value is None:
            raise Ugi3SynthesisValueAuditV3Error("generated component is absent from v3 values")
        role_values = components_by_sample.setdefault(row["sample_index"], {})
        if row["role"] in role_values:
            raise Ugi3SynthesisValueAuditV3Error("sample contains duplicate component role")
        role_values[row["role"]] = value

    impact = _read_csv(input_paths["third_wave_product_impact"], label="third-wave impact")
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
            raise Ugi3SynthesisValueAuditV3Error("product does not have exactly three roles")
        value = ProductSynthesisValue(
            product_smiles=row["canonical_product_smiles"],
            l1_forward_consistent=True,
            components=tuple(sorted(role_values.items())),
        )
        impact_row = impact_by_sample.get(sample)
        if (
            impact_row is None
            or (impact_row["complete_after_third_wave"].lower() == "true") != value.route_complete
        ):
            raise Ugi3SynthesisValueAuditV3Error(
                "v3 product value disagrees with third-wave impact ledger"
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

    component_ledger = _gzip_json_bytes(
        {"schema_version": COMPONENT_LEDGER_SCHEMA_VERSION, "records": component_records}
    )
    product_ledger = _gzip_json_bytes(
        {"schema_version": PRODUCT_LEDGER_SCHEMA_VERSION, "records": product_records}
    )
    component_outcomes = Counter(
        value.assessment_outcome.value for value in generated_values.values()
    )
    evidence_support = Counter(value.evidence_support.value for value in generated_values.values())
    summary = {
        "base_version": "v2",
        "third_wave_components_changed": changed,
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
        raise Ugi3SynthesisValueAuditV3Error("synthesis-value v3 summary changed")
    return (
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "diagnostic_only_not_authorized_for_production_guidance",
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "summary": summary,
            "overlay_metadata": {"third_wave_heads": overlay_metadata},
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
