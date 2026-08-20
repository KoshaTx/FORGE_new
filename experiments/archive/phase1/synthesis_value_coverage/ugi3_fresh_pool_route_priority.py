"""Rank exact one-gap components in the fresh v2 Ugi product pool."""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

from forge.corpus.r1_prime_audit import sha256_bytes, sha256_file
from experiments.phase1.product_l1.evaluation.tail_chemotype import (
    chemotype_signature,
    component_chemotype_metrics,
)
from forge.synthesis.engine.planner import AssessmentOutcome
from forge.synthesis.value.contracts import ComponentSynthesisValue

CONFIG_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_route_priority_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_route_priority.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_route_priority_ledger.v1"

FIELDS = (
    "priority_rank",
    "role",
    "canonical_smiles",
    "assessment_outcome",
    "structural_provenance_stratum",
    "catalog_provenance_substratum",
    "registry_component_id",
    "value_source_class",
    "one_gap_product_count",
    "one_gap_product_fraction",
    "cumulative_one_gap_product_count",
    "cumulative_one_gap_product_fraction",
    "route_mining_lane",
    "chemotype_signature",
    "chemotype_metrics_json",
)


class Ugi3FreshPoolRoutePriorityError(ValueError):
    """Raised when the one-gap priority audit cannot be reproduced."""


def _load_json(path: Path, *, compressed: bool, label: str) -> dict[str, Any]:
    try:
        if compressed:
            with gzip.open(path, "rt") as handle:
                value = json.load(handle)
        else:
            value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Ugi3FreshPoolRoutePriorityError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3FreshPoolRoutePriorityError(f"{label} must be an object")
    return value


def _gzip_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(text.getvalue().encode())
    return output.getvalue()


def _coverage_thresholds(sorted_counts: list[int], denominator: int) -> dict[str, int | None]:
    thresholds = {"25pct": None, "50pct": None, "75pct": None, "90pct": None}
    if denominator <= 0:
        return thresholds
    cumulative = 0
    for index, count in enumerate(sorted_counts, start=1):
        cumulative += count
        fraction = cumulative / denominator
        for label, threshold in (("25pct", 0.25), ("50pct", 0.5), ("75pct", 0.75), ("90pct", 0.9)):
            if thresholds[label] is None and fraction >= threshold:
                thresholds[label] = index
    return thresholds


def build_fresh_pool_route_priority(repo: Path, config_path: Path) -> tuple[dict[str, Any], bytes]:
    """Rank one-gap identities without inferring or planning any route."""

    config = _load_json(config_path, compressed=False, label="priority config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3FreshPoolRoutePriorityError("unsupported priority config")
    specifications = config.get("inputs")
    required = {
        "route_coverage_result",
        "component_values",
        "product_values",
        "priority_source",
        "priority_runner",
        "priority_tests",
    }
    if not isinstance(specifications, dict) or set(specifications) != required:
        raise Ugi3FreshPoolRoutePriorityError("priority input set changed")
    paths = {}
    for label, specification in specifications.items():
        if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
            raise Ugi3FreshPoolRoutePriorityError(f"input {label} is malformed")
        path = (repo / specification["path"]).resolve()
        if sha256_file(path) != specification["sha256"]:
            raise Ugi3FreshPoolRoutePriorityError(f"input hash changed: {label}")
        paths[label] = path

    coverage = _load_json(paths["route_coverage_result"], compressed=False, label="coverage")
    components = _load_json(paths["component_values"], compressed=True, label="components")
    products = _load_json(paths["product_values"], compressed=True, label="products")
    artifacts = coverage.get("artifacts", {})
    if (
        artifacts.get("component_synthesis_values.json.gz", {}).get("sha256")
        != sha256_file(paths["component_values"])
        or artifacts.get("product_synthesis_values.json.gz", {}).get("sha256")
        != sha256_file(paths["product_values"])
        or coverage.get("adjudication", {}).get("route_mining_priority_update_authorized")
        is not True
    ):
        raise Ugi3FreshPoolRoutePriorityError("route-coverage artifact ownership failed")

    component_index = {}
    for record in components.get("records", []):
        value = ComponentSynthesisValue.from_dict(record.get("value"))
        key = (value.target.role, value.target.canonical_smiles)
        if key in component_index:
            raise Ugi3FreshPoolRoutePriorityError("duplicate component value identity")
        component_index[key] = record

    one_gap_counts: Counter[tuple[str, str]] = Counter()
    product_total = 0
    for record in products.get("records", []):
        values = [
            (str(item.get("role")), ComponentSynthesisValue.from_dict(item.get("value")))
            for item in record.get("value", {}).get("components", [])
        ]
        if len(values) != 3:
            raise Ugi3FreshPoolRoutePriorityError("product value lacks three components")
        gaps = [(role, value) for role, value in values if not value.route_complete]
        if len(gaps) != 1:
            continue
        product_total += 1
        role, value = gaps[0]
        one_gap_counts[(role, value.target.canonical_smiles)] += 1

    expected_total = int(
        coverage.get("summary", {})
        .get("products_by_route_complete_component_count", {})
        .get("2", -1)
    )
    if product_total != expected_total:
        raise Ugi3FreshPoolRoutePriorityError("one-gap product denominator changed")

    ordered = sorted(one_gap_counts.items(), key=lambda item: (-item[1], item[0]))
    cumulative = 0
    rows = []
    outcome_counts: Counter[str] = Counter()
    outcome_product_counts: Counter[str] = Counter()
    role_product_counts: Counter[str] = Counter()
    provenance_product_counts: Counter[str] = Counter()
    missing_counts = []
    top = []
    for rank, (key, count) in enumerate(ordered, start=1):
        record = component_index.get(key)
        if record is None:
            raise Ugi3FreshPoolRoutePriorityError("one-gap identity lacks component metadata")
        value = ComponentSynthesisValue.from_dict(record["value"])
        outcome = value.assessment_outcome.value
        if outcome == AssessmentOutcome.MISSING_KNOWLEDGE.value:
            lane = "targeted_exact_evidence_or_qualified_template_search"
            missing_counts.append(count)
        elif outcome == AssessmentOutcome.OUTSIDE_SUPPORT.value:
            lane = "declared_support_review_not_route_mining"
        else:
            lane = "typed_failure_review"
        cumulative += count
        metrics = component_chemotype_metrics(key[1])
        row = {
            "priority_rank": rank,
            "role": key[0],
            "canonical_smiles": key[1],
            "assessment_outcome": outcome,
            "structural_provenance_stratum": record["structural_provenance_stratum"],
            "catalog_provenance_substratum": record["catalog_provenance_substratum"],
            "registry_component_id": record["registry_component_id"] or "",
            "value_source_class": record["source_class"],
            "one_gap_product_count": count,
            "one_gap_product_fraction": count / product_total,
            "cumulative_one_gap_product_count": cumulative,
            "cumulative_one_gap_product_fraction": cumulative / product_total,
            "route_mining_lane": lane,
            "chemotype_signature": chemotype_signature(metrics),
            "chemotype_metrics_json": json.dumps(metrics, separators=(",", ":"), sort_keys=True),
        }
        rows.append(row)
        outcome_counts[outcome] += 1
        outcome_product_counts[outcome] += count
        role_product_counts[key[0]] += count
        provenance_product_counts[record["structural_provenance_stratum"]] += count
        if len(top) < 20:
            top.append(
                {
                    "priority_rank": rank,
                    "role": key[0],
                    "canonical_smiles": key[1],
                    "assessment_outcome": outcome,
                    "one_gap_product_count": count,
                    "route_mining_lane": lane,
                }
            )

    ledger = _gzip_csv_bytes(rows)
    missing_total = sum(missing_counts)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_nonselecting_one_gap_route_priority",
        "config": {"path": str(config_path.relative_to(repo)), "sha256": sha256_file(config_path)},
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "summary": {
            "one_gap_products": product_total,
            "unique_one_gap_components": len(rows),
            "unique_one_gap_components_by_outcome": dict(sorted(outcome_counts.items())),
            "one_gap_products_by_component_outcome": dict(sorted(outcome_product_counts.items())),
            "one_gap_products_by_missing_component_role": dict(sorted(role_product_counts.items())),
            "one_gap_products_by_missing_component_provenance": dict(
                sorted(provenance_product_counts.items())
            ),
            "missing_knowledge_one_gap_products": missing_total,
            "missing_knowledge_unique_components": len(missing_counts),
            "greedy_missing_component_counts_for_coverage": _coverage_thresholds(
                sorted(missing_counts, reverse=True), missing_total
            ),
            "top_20_one_gap_components": top,
        },
        "artifacts": {
            "priority_ledger.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "sha256": sha256_bytes(ledger),
            }
        },
        "adjudication": {
            "route_planning_performed": False,
            "synthesis_guidance_authorized": False,
            "prospective_candidate_selection_changed": False,
            "targeted_evidence_mining_authorized": True,
        },
    }
    return result, ledger
