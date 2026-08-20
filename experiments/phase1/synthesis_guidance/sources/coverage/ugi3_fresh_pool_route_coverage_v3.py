"""Refresh one already-owned exact terminal in fresh-pool route coverage v2.

Version 2 is immutable.  This audit corrects the remaining stale assessment for
octadecylamine when it is used directly as an Ugi amine head.  The same exact
identity already has current item-level procurement evidence as a precursor in
the frozen virtual-terminal snapshot; no homologue or reaction-family evidence
is transferred.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

from experiments.phase1.synthesis_guidance.sources.coverage.ugi3_fresh_pool_route_coverage import (
    _gzip_json_bytes,
    _load_gzip_json,
    _load_json,
)
from experiments.phase1.synthesis_guidance.sources.coverage.ugi3_fresh_pool_route_coverage_v2 import (
    _terminal_value,
)
from forge.corpus.r1_prime_audit import sha256_bytes, sha256_file
from forge.synthesis.engine.planner import RouteTarget
from forge.synthesis.terminals.ugi3_high_leverage_head_terminals import _parse_utc
from forge.synthesis.value.contracts import ComponentSynthesisValue, ProductSynthesisValue

CONFIG_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_route_coverage_config.v3"
RESULT_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_route_coverage.v3"
COMPONENT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_component_values.v3"
PRODUCT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_product_values.v3"

HEAD_ROLE = "amine_head"
HEAD_SMILES = "CCCCCCCCCCCCCCCCCCN"
HEAD_INCHI_KEY = "REYJJPSVUYRZGE-UHFFFAOYSA-N"


class Ugi3FreshPoolRouteCoverageV3Error(ValueError):
    """Raised when the exact-terminal refresh cannot be reproduced safely."""


def _validate_inputs(config: Mapping[str, Any], repo: Path) -> dict[str, Path]:
    declared = config.get("inputs")
    if not isinstance(declared, dict) or not declared:
        raise Ugi3FreshPoolRouteCoverageV3Error("v3 inputs are missing")
    paths: dict[str, Path] = {}
    for label, record in declared.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise Ugi3FreshPoolRouteCoverageV3Error(f"input {label} is malformed")
        path = repo / str(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise Ugi3FreshPoolRouteCoverageV3Error(f"input hash changed: {label}")
        paths[label] = path
    return paths


def _find_octadecylamine_record(
    payload: Mapping[str, Any], *, assessment_as_of_utc: str
) -> Mapping[str, Any]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise Ugi3FreshPoolRouteCoverageV3Error("virtual terminal records are missing")
    matches = [record for record in records if record.get("canonical_smiles") == HEAD_SMILES]
    if len(matches) != 1:
        raise Ugi3FreshPoolRouteCoverageV3Error(
            "exact octadecylamine procurement record is not unique"
        )
    record = matches[0]
    if (
        record.get("terminal_class") != "primary_amine"
        or record.get("identity", {}).get("title") != "Octadecylamine"
        or record.get("identity", {}).get("inchi_key") != HEAD_INCHI_KEY
        or record.get("identity", {}).get("cas_rn") != "124-30-1"
        or record.get("identity", {}).get("form") != "neutral_free_base"
        or record.get("vendor_evidence", {}).get("vendor") != "MilliporeSigma"
        or record.get("vendor_evidence", {}).get("product_code") != "O1408"
        or record.get("vendor_evidence", {}).get("purity") != "90%"
        or "available to ship today"
        not in str(record.get("vendor_evidence", {}).get("availability_observation", ""))
        or record.get("procurement_status") != "current_item_level_vendor_verified"
        or record.get("current_item_level_procurement_closed") is not True
    ):
        raise Ugi3FreshPoolRouteCoverageV3Error(
            "octadecylamine procurement evidence is not exact/current"
        )
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, dict):
        raise Ugi3FreshPoolRouteCoverageV3Error("virtual terminal snapshot is missing")
    accessed = _parse_utc(snapshot.get("accessed_utc"), label="terminal accessed_utc")
    assessment = _parse_utc(assessment_as_of_utc, label="assessment_as_of_utc")
    expiry_days = snapshot.get("expiry_days")
    if not isinstance(expiry_days, int) or expiry_days <= 0:
        raise Ugi3FreshPoolRouteCoverageV3Error("terminal expiry policy is malformed")
    if assessment < accessed or assessment > accessed + timedelta(days=expiry_days):
        raise Ugi3FreshPoolRouteCoverageV3Error(
            "octadecylamine procurement snapshot is not current at assessment"
        )
    return record


def build_fresh_pool_route_coverage_v3(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes, bytes]:
    """Apply the exact-current octadecylamine head refresh to immutable v2."""

    config = _load_json(config_path, label="fresh-pool route v3 config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3FreshPoolRouteCoverageV3Error("unsupported fresh-pool route v3 config")
    paths = _validate_inputs(config, repo)
    required = {
        "base_component_values",
        "base_product_values",
        "base_result",
        "virtual_terminal_procurement",
        "audit_source",
        "audit_runner",
        "audit_tests",
        "value_source",
    }
    if set(paths) != required:
        raise Ugi3FreshPoolRouteCoverageV3Error("v3 input set changed")
    base_result = _load_json(paths["base_result"], label="base route coverage v2")
    artifacts = base_result.get("artifacts", {})
    if artifacts.get("component_synthesis_values.json.gz", {}).get("sha256") != sha256_file(
        paths["base_component_values"]
    ) or artifacts.get("product_synthesis_values.json.gz", {}).get("sha256") != sha256_file(
        paths["base_product_values"]
    ):
        raise Ugi3FreshPoolRouteCoverageV3Error("base v2 route-coverage ownership failed")

    terminal_payload = _load_json(
        paths["virtual_terminal_procurement"], label="virtual terminal procurement"
    )
    _find_octadecylamine_record(
        terminal_payload,
        assessment_as_of_utc=str(config.get("assessment_as_of_utc")),
    )
    target_key = (HEAD_ROLE, HEAD_SMILES)
    target_value = _terminal_value(
        target=RouteTarget(role=HEAD_ROLE, canonical_smiles=HEAD_SMILES),
        source_path=paths["virtual_terminal_procurement"],
        evidence_id="existing-terminal:octadecylamine:O1408",
        locator=f"{paths['virtual_terminal_procurement']}#records[Octadecylamine]",
    )
    source_class = "existing_exact_current_octadecylamine_terminal_refresh"

    base_components = _load_gzip_json(paths["base_component_values"], label="base components")
    records = base_components.get("records")
    if not isinstance(records, list):
        raise Ugi3FreshPoolRouteCoverageV3Error("base component records are missing")
    updated_values: dict[tuple[str, str], ComponentSynthesisValue] = {}
    source_by_key: dict[tuple[str, str], str] = {}
    component_records: list[dict[str, Any]] = []
    changed = 0
    unique_outcomes: Counter[str] = Counter()
    unique_role_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    unique_sources: Counter[str] = Counter()
    for raw in records:
        if not isinstance(raw, dict):
            raise Ugi3FreshPoolRouteCoverageV3Error("base component record is malformed")
        key = (str(raw.get("role")), str(raw.get("canonical_smiles")))
        value = ComponentSynthesisValue.from_dict(raw.get("value"))
        output = dict(raw)
        if key == target_key:
            if value.route_complete:
                raise Ugi3FreshPoolRouteCoverageV3Error("octadecylamine was already complete")
            value = target_value
            output["source_class"] = source_class
            output["value"] = value.to_dict()
            changed += 1
        updated_values[key] = value
        source_by_key[key] = str(output.get("source_class"))
        component_records.append(output)
        unique_outcomes[value.assessment_outcome.value] += 1
        unique_role_outcomes[key[0]][value.assessment_outcome.value] += 1
        unique_sources[str(output.get("source_class"))] += 1
    if changed != 1:
        raise Ugi3FreshPoolRouteCoverageV3Error("octadecylamine delta did not match once")

    base_products = _load_gzip_json(paths["base_product_values"], label="base products")
    product_rows = base_products.get("records")
    if not isinstance(product_rows, list):
        raise Ugi3FreshPoolRouteCoverageV3Error("base product records are missing")
    product_records: list[dict[str, Any]] = []
    occurrence_outcomes: Counter[str] = Counter()
    role_occurrence_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    source_occurrences: Counter[str] = Counter()
    product_outcomes: Counter[str] = Counter()
    product_outcomes_by_provenance: dict[str, Counter[str]] = defaultdict(Counter)
    complete_component_counts: Counter[int] = Counter()
    target_occurrences = 0
    newly_complete = 0
    base_complete = 0
    for raw in product_rows:
        if not isinstance(raw, dict):
            raise Ugi3FreshPoolRouteCoverageV3Error("base product record is malformed")
        prior = ProductSynthesisValue.from_dict(raw.get("value"))
        base_complete += int(prior.route_complete)
        role_values: dict[str, ComponentSynthesisValue] = {}
        for role, old_value in prior.components:
            key = (role, old_value.target.canonical_smiles)
            value = updated_values.get(key)
            if value is None:
                raise Ugi3FreshPoolRouteCoverageV3Error(
                    "product component is absent from component ledger"
                )
            role_values[role] = value
            if key == target_key:
                target_occurrences += 1
            occurrence_outcomes[value.assessment_outcome.value] += 1
            role_occurrence_outcomes[role][value.assessment_outcome.value] += 1
            source_occurrences[source_class if key == target_key else source_by_key[key]] += 1
        value = ProductSynthesisValue(
            product_smiles=prior.product_smiles,
            l1_forward_consistent=prior.l1_forward_consistent,
            components=tuple(sorted(role_values.items())),
        )
        newly_complete += int(value.route_complete and not prior.route_complete)
        outcome = "complete" if value.route_complete else "noncomplete"
        provenance = str(raw.get("product_structural_provenance_stratum"))
        product_outcomes[outcome] += 1
        product_outcomes_by_provenance[provenance][outcome] += 1
        complete_component_counts[sum(item.route_complete for item in role_values.values())] += 1
        output = dict(raw)
        output["value"] = value.to_dict()
        product_records.append(output)

    component_ledger = _gzip_json_bytes(
        {"schema_version": COMPONENT_LEDGER_SCHEMA_VERSION, "records": component_records}
    )
    product_ledger = _gzip_json_bytes(
        {"schema_version": PRODUCT_LEDGER_SCHEMA_VERSION, "records": product_records}
    )
    summary = {
        "base_version": "fresh_pool_route_coverage_v2",
        "exact_terminal_components_changed": changed,
        "changed_component_keys": [f"{HEAD_ROLE}\t{HEAD_SMILES}"],
        "unique_components": len(component_records),
        "unique_component_outcomes": dict(sorted(unique_outcomes.items())),
        "unique_component_outcomes_by_role": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(unique_role_outcomes.items())
        },
        "unique_component_value_sources": dict(sorted(unique_sources.items())),
        "component_occurrence_outcomes": dict(sorted(occurrence_outcomes.items())),
        "component_occurrence_outcomes_by_role": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(role_occurrence_outcomes.items())
        },
        "component_occurrences_by_value_source": dict(sorted(source_occurrences.items())),
        "exact_l1_products": len(product_records),
        "product_outcomes_before": {
            "complete": base_complete,
            "noncomplete": len(product_records) - base_complete,
        },
        "product_outcomes_after": dict(sorted(product_outcomes.items())),
        "newly_complete_products": newly_complete,
        "target_occurrences": {f"{HEAD_ROLE}\t{HEAD_SMILES}": target_occurrences},
        "products_by_route_complete_component_count": {
            str(key): value for key, value in sorted(complete_component_counts.items())
        },
        "product_outcomes_by_structural_provenance": {
            key: dict(sorted(value.items()))
            for key, value in sorted(product_outcomes_by_provenance.items())
        },
        "non_null_scalar_values": sum(
            record["value"]["scalar_value"] is not None for record in product_records
        ),
        "non_null_success_probabilities": sum(
            record["value"]["success_probability"] is not None for record in product_records
        ),
    }
    if summary != config.get("expected_summary"):
        raise Ugi3FreshPoolRouteCoverageV3Error("fresh-pool route v3 summary changed")
    return (
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "complete_nonselecting_exact_terminal_refresh",
            "summary": summary,
            "inputs": {
                label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
                for label, path in sorted(paths.items())
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
            "policy": config.get("assessment_policy"),
            "adjudication": {
                "route_mining_priority_update_authorized": True,
                "synthesis_guidance_authorized": False,
                "prospective_candidate_selection_changed": False,
            },
            "nonclaims": [
                "Current procurement does not establish Ugi product success.",
                "No neighboring identity, reaction family or homologue is promoted.",
                "Unknown burden remains unknown rather than zero.",
                "This audit does not authorize synthesis-guided generation or selection.",
            ],
        },
        component_ledger,
        product_ledger,
    )
