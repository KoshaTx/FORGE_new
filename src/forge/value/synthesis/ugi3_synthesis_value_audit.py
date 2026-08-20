"""Build typed pre-prospective synthesis values for the frozen Ugi sample.

The audit composes only authenticated exact overlays. Generated components
absent from the frozen registry remain ``missing_knowledge`` rather than being
mislabelled incompatible. No scalar route value or synthesis-success
probability is created.
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
from forge.core.io import stable_json as _stable_json
from forge.route.assessment.ugi3_targeted_role_gap_overlay import load_targeted_role_gap_overlay
from forge.route.engine.planner import (
    AssessmentOutcome,
    AssessmentTraceEvent,
    AvailabilityState,
    EvidenceRecord,
    EvidenceTier,
    ForwardVerificationState,
    KnowledgeDisposition,
    KnowledgeResult,
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RecursiveRouteAssessor,
    RouteTarget,
    SynthesisAssessment,
    SynthesisRouteNode,
    TraceAction,
)
from forge.route.evidence.ugi3_targeted_exact_overlay import load_targeted_exact_overlay
from forge.route.terminals.ugi3_high_leverage_head_terminals import (
    load_high_leverage_head_terminal_overlay,
)
from forge.value.synthesis.synthesis import (
    ComponentSynthesisValue,
    ProductSynthesisValue,
    component_synthesis_value_from_assessment,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi3_synthesis_value_audit_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_synthesis_value_audit.v1"
COMPONENT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_component_synthesis_values.v1"
PRODUCT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_product_synthesis_values.v1"


class Ugi3SynthesisValueAuditError(ValueError):
    """Raised when synthesis-value records cannot be reproduced exactly."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=Ugi3SynthesisValueAuditError, label=label)


def _load_gzip_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise Ugi3SynthesisValueAuditError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3SynthesisValueAuditError(f"{label} must be an object")
    return value


def _read_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise Ugi3SynthesisValueAuditError(f"{label} has no header")
            return list(reader)
    except (OSError, csv.Error) as exc:
        raise Ugi3SynthesisValueAuditError(f"could not read {label}") from exc


def _gzip_json_bytes(value: Any) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write((_stable_json(value) + "\n").encode())
    return output.getvalue()


def _validate_inputs(config: dict[str, Any], input_paths: dict[str, Path]) -> None:
    declared = config.get("inputs")
    if not isinstance(declared, dict) or set(declared) != set(input_paths):
        raise Ugi3SynthesisValueAuditError("configured input set is not exact")
    for label, path in input_paths.items():
        record = declared.get(label)
        if not isinstance(record, dict):
            raise Ugi3SynthesisValueAuditError(f"input {label} is malformed")
        asset = record.get("asset")
        if not isinstance(asset, str) or Path(asset).resolve() != path.resolve():
            raise Ugi3SynthesisValueAuditError(f"input {label} path changed")
        if record.get("expected_sha256") != sha256_file(path):
            raise Ugi3SynthesisValueAuditError(f"input {label} hash changed")


_OUTCOME_TO_DISPOSITION = {
    AssessmentOutcome.INCOMPATIBLE: KnowledgeDisposition.INCOMPATIBLE,
    AssessmentOutcome.OUTSIDE_SUPPORT: KnowledgeDisposition.OUTSIDE_SUPPORT,
    AssessmentOutcome.MISSING_KNOWLEDGE: KnowledgeDisposition.MISSING_KNOWLEDGE,
    AssessmentOutcome.INVALID_INPUT: KnowledgeDisposition.INVALID_INPUT,
    AssessmentOutcome.EXECUTION_ERROR: KnowledgeDisposition.EXECUTION_ERROR,
}


class _FrozenAssessmentSource:
    """Replay a qualified assessment ledger as immutable route knowledge."""

    def __init__(self, assessments: tuple[SynthesisAssessment, ...]):
        decisions: dict[tuple[str, str, tuple[str, ...]], KnowledgeResult] = {}

        def visit(node: SynthesisRouteNode) -> None:
            if node.step is not None:
                result = KnowledgeResult(
                    disposition=KnowledgeDisposition.EXPAND,
                    evidence=node.evidence,
                    proposal=node.step,
                    detail=node.detail,
                )
            elif node.outcome is AssessmentOutcome.COMPLETE:
                result = KnowledgeResult(
                    disposition=KnowledgeDisposition.TERMINAL,
                    evidence=node.evidence,
                    detail=node.detail,
                )
            else:
                disposition = _OUTCOME_TO_DISPOSITION.get(node.outcome)
                if disposition is None:
                    raise Ugi3SynthesisValueAuditError(
                        f"cannot replay {node.outcome.value} as frozen route knowledge"
                    )
                result = KnowledgeResult(
                    disposition=disposition,
                    evidence=node.evidence,
                    detail=node.detail,
                )
            prior = decisions.get(node.target.identity)
            if prior is not None and prior != result:
                raise Ugi3SynthesisValueAuditError(
                    "frozen assessment ledger contains conflicting target decisions"
                )
            decisions[node.target.identity] = result
            for child in node.children:
                visit(child)

        for assessment in assessments:
            visit(assessment.route_tree)
        self._decisions = decisions

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        result = self._decisions.get(target.identity)
        if result is None:
            if target.role in {
                "amine_head",
                "oxoester_aldehyde_body_tail",
                "isocyanide_tail",
            }:
                return KnowledgeResult(
                    disposition=KnowledgeDisposition.MISSING_KNOWLEDGE,
                    evidence=(),
                    detail="generated component has no frozen route assessment",
                )
            return KnowledgeResult(
                disposition=KnowledgeDisposition.INVALID_INPUT,
                evidence=(),
                detail="target is absent from frozen assessment replay",
            )
        return result


def _missing_generated_assessment(
    *,
    target: RouteTarget,
    provenance_ledger_path: Path,
) -> SynthesisAssessment:
    evidence = EvidenceRecord(
        evidence_id=(
            "generated-component-missing-route:"
            f"{target.role}:{sha256_bytes(target.canonical_smiles.encode())[:16]}"
        ),
        tier=EvidenceTier.PROVENANCE_ONLY,
        source_sha256=sha256_file(provenance_ledger_path),
        source_locator=(
            f"{provenance_ledger_path}#role={target.role}"
            f"&canonical_component_smiles={target.canonical_smiles}"
        ),
        exact_substrate=True,
        forward_verification=ForwardVerificationState.NOT_RUN,
        availability=AvailabilityState.UNASSESSED,
    )
    detail = "generated component is structurally admitted but lacks an exact route assessment"
    node = SynthesisRouteNode(
        target=target,
        outcome=AssessmentOutcome.MISSING_KNOWLEDGE,
        evidence=(evidence,),
        detail=detail,
    )
    trace = (
        AssessmentTraceEvent(
            sequence=0,
            depth=0,
            action=TraceAction.DECISION,
            target=target,
            outcome=AssessmentOutcome.MISSING_KNOWLEDGE,
            evidence_ids=(evidence.evidence_id,),
            detail=detail,
        ),
    )
    return SynthesisAssessment(
        target=target,
        outcome=AssessmentOutcome.MISSING_KNOWLEDGE,
        route_tree=node,
        trace=trace,
    )


def _assessment_budget(config: dict[str, Any]) -> PlannerBudgetLimits:
    policy = config.get("assessment_policy")
    if not isinstance(policy, dict) or not isinstance(policy.get("per_component_budget"), dict):
        raise Ugi3SynthesisValueAuditError("assessment policy lacks per-component budget")
    return PlannerBudgetLimits.from_dict(policy["per_component_budget"])


def build_ugi3_synthesis_value_audit(
    *,
    config_path: Path,
    input_paths: dict[str, Path],
    targeted_audit_input_paths: dict[str, Path],
    role_gap_input_paths: dict[str, Path],
    head_terminal_input_paths: dict[str, Path],
) -> tuple[dict[str, Any], bytes, bytes]:
    """Compose all qualified evidence into component and product value ledgers."""

    config = _load_json(config_path, label="synthesis-value audit config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3SynthesisValueAuditError("unsupported synthesis-value config schema")
    _validate_inputs(config, input_paths)
    hybrid = _load_gzip_json(input_paths["hybrid_assessment_ledger"], label="hybrid ledger")
    hybrid_records = hybrid.get("records")
    if not isinstance(hybrid_records, list):
        raise Ugi3SynthesisValueAuditError("hybrid ledger lacks assessment records")
    assessments = tuple(
        SynthesisAssessment.from_dict(record.get("assessment"))
        for record in hybrid_records
        if isinstance(record, dict)
    )
    if len(assessments) != len(hybrid_records):
        raise Ugi3SynthesisValueAuditError("hybrid assessment record is malformed")
    roots = tuple(assessment.target for assessment in assessments)
    if len({target.identity for target in roots}) != len(roots):
        raise Ugi3SynthesisValueAuditError("hybrid assessment roots are not unique")

    source: Any = _FrozenAssessmentSource(assessments)
    source, targeted_metadata = load_targeted_exact_overlay(
        base_source=source,
        audit_config_path=input_paths["targeted_audit_config"],
        audit_input_paths=targeted_audit_input_paths,
        stored_audit_result_path=input_paths["targeted_audit_result"],
        stored_route_ledger_path=input_paths["targeted_route_ledger"],
        stored_product_ledger_path=input_paths["targeted_product_ledger"],
    )
    source, role_gap_metadata = load_targeted_role_gap_overlay(
        base_source=source,
        audit_config_path=input_paths["role_gap_config"],
        audit_input_paths=role_gap_input_paths,
        stored_audit_result_path=input_paths["role_gap_result"],
        stored_product_ledger_path=input_paths["role_gap_product_ledger"],
    )
    source, head_metadata = load_high_leverage_head_terminal_overlay(
        base_source=source,
        audit_config_path=input_paths["head_terminal_config"],
        audit_input_paths=head_terminal_input_paths,
        stored_audit_result_path=input_paths["head_terminal_result"],
        stored_product_ledger_path=input_paths["head_terminal_product_ledger"],
    )

    limits = _assessment_budget(config)
    current_assessments = {}
    for target in roots:
        assessment = RecursiveRouteAssessor(source).assess(
            target,
            PlannerBudgetLedger(limits),
        )
        current_assessments[(target.role, target.canonical_smiles)] = assessment
    registry_outcomes = Counter(
        assessment.outcome.value for assessment in current_assessments.values()
    )

    generated_components = _read_csv(
        input_paths["generated_component_provenance"],
        label="generated component provenance",
    )
    unique_component_rows: dict[tuple[str, str], dict[str, str]] = {}
    for row in generated_components:
        key = (row["role"], row["canonical_component_smiles"])
        prior = unique_component_rows.get(key)
        if prior is not None:
            stable_fields = (
                "structural_provenance_stratum",
                "catalog_membership",
                "registry_component_id",
            )
            if any(prior[field] != row[field] for field in stable_fields):
                raise Ugi3SynthesisValueAuditError(
                    "generated component provenance is inconsistent by identity"
                )
        else:
            unique_component_rows[key] = row

    generated_values: dict[tuple[str, str], ComponentSynthesisValue] = {}
    component_records = []
    generated_source_counts: Counter[str] = Counter()
    for key, provenance in sorted(unique_component_rows.items()):
        assessment = current_assessments.get(key)
        source_class = "frozen_registry_assessment"
        if assessment is None:
            source_class = "generated_component_missing_route_knowledge"
            assessment = _missing_generated_assessment(
                target=RouteTarget(role=key[0], canonical_smiles=key[1]),
                provenance_ledger_path=input_paths["generated_component_provenance"],
            )
        value = component_synthesis_value_from_assessment(assessment)
        generated_values[key] = value
        generated_source_counts[source_class] += 1
        component_records.append(
            {
                "role": key[0],
                "canonical_smiles": key[1],
                "source_class": source_class,
                "structural_provenance_stratum": provenance["structural_provenance_stratum"],
                "registry_component_id": provenance["registry_component_id"] or None,
                "value": value.to_dict(),
            }
        )

    generated_products = _read_csv(
        input_paths["generated_product_provenance"],
        label="generated product provenance",
    )
    latest_impact = _read_csv(
        input_paths["latest_product_impact"],
        label="latest product impact",
    )
    impact_by_sample = {row["sample_index"]: row for row in latest_impact}
    components_by_sample: dict[str, dict[str, ComponentSynthesisValue]] = {}
    for row in generated_components:
        sample = row["sample_index"]
        role = row["role"]
        key = (role, row["canonical_component_smiles"])
        role_values = components_by_sample.setdefault(sample, {})
        if role in role_values:
            raise Ugi3SynthesisValueAuditError("sample contains duplicate component role")
        role_values[role] = generated_values[key]

    production = _load_json(input_paths["production_generator"], label="production generator")
    metrics = production.get("fresh_selection_metrics")
    if not isinstance(metrics, dict) or metrics.get("exact_forward_reconstructions") != len(
        generated_products
    ):
        raise Ugi3SynthesisValueAuditError("production L1-forward denominator changed")
    product_records = []
    product_outcomes: Counter[str] = Counter()
    for row in generated_products:
        sample = row["sample_index"]
        role_values = components_by_sample.get(sample)
        if role_values is None or set(role_values) != {
            "amine_head",
            "oxoester_aldehyde_body_tail",
            "isocyanide_tail",
        }:
            raise Ugi3SynthesisValueAuditError("product does not have exactly three Ugi roles")
        value = ProductSynthesisValue(
            product_smiles=row["canonical_product_smiles"],
            l1_forward_consistent=True,
            components=tuple(sorted(role_values.items())),
        )
        impact = impact_by_sample.get(sample)
        if (
            impact is None
            or (impact["complete_after_head_terminals"].lower() == "true") != value.route_complete
        ):
            raise Ugi3SynthesisValueAuditError(
                "structured product value disagrees with deterministic closure ledger"
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
        "registry_components": len(current_assessments),
        "registry_outcomes": dict(sorted(registry_outcomes.items())),
        "generated_unique_components": len(generated_values),
        "generated_component_value_sources": dict(sorted(generated_source_counts.items())),
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
    expected = config.get("expected_summary")
    if expected is not None and summary != expected:
        raise Ugi3SynthesisValueAuditError("synthesis-value summary changed")
    return (
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "diagnostic_only_not_authorized_for_production_guidance",
            "config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "summary": summary,
            "assessment_policy": config.get("assessment_policy"),
            "overlay_metadata": {
                "targeted_aldehydes": targeted_metadata,
                "targeted_role_gap": role_gap_metadata,
                "high_leverage_heads": head_metadata,
            },
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
                "The diagnostic does not authorize synthesis-guided candidate generation.",
            ],
        },
        component_ledger,
        product_ledger,
    )
