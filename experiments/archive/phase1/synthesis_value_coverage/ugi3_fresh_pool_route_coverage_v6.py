"""Requalify immutable fresh-pool v5 under the current synthesis-value source."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.corpus.r1_prime_audit import sha256_bytes, sha256_file

CONFIG_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_route_coverage_config.v6"
RESULT_SCHEMA_VERSION = "phase1_ugi3_fresh_pool_route_coverage.v6"


class Ugi3FreshPoolRouteCoverageV6Error(ValueError):
    """Raised when immutable v5 cannot be source-requalified."""


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Ugi3FreshPoolRouteCoverageV6Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise Ugi3FreshPoolRouteCoverageV6Error(f"{label} must be an object")
    return value


def _pin(repo: Path, record: Mapping[str, Any], *, label: str) -> Path:
    if set(record) != {"path", "sha256"}:
        raise Ugi3FreshPoolRouteCoverageV6Error(f"{label} pin is malformed")
    path = (repo / str(record.get("path"))).resolve()
    try:
        relative = path.relative_to(repo.resolve())
    except ValueError as error:
        raise Ugi3FreshPoolRouteCoverageV6Error(f"{label} escapes repository") from error
    lowered = "/".join(relative.parts).lower()
    if "holdout" in lowered or "sealed" in lowered:
        raise Ugi3FreshPoolRouteCoverageV6Error(f"{label} is forbidden")
    if sha256_file(path) != record.get("sha256"):
        raise Ugi3FreshPoolRouteCoverageV6Error(f"{label} hash changed")
    return path


def build_fresh_pool_route_coverage_v6(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes, bytes]:
    repo = repo.resolve()
    config = _load(config_path, label="fresh-pool v6 config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3FreshPoolRouteCoverageV6Error("unsupported fresh-pool v6 config")
    policy = config.get("policy")
    expected_policy = {
        "purpose": "Source-lineage requalification of immutable fresh-pool v5 values.",
        "value_records_changed": 0,
        "route_evidence_scope_changed": False,
        "scalarization_authorized": False,
        "synthesis_guidance_authorized": False,
        "sealed_holdout_access_authorized": False,
    }
    if policy != expected_policy:
        raise Ugi3FreshPoolRouteCoverageV6Error("fresh-pool v6 policy changed")
    inputs = config.get("inputs")
    required = {
        "base_config",
        "base_result",
        "base_component_values",
        "base_product_values",
        "value_source",
        "source_qualification",
        "audit_source",
        "audit_runner",
        "audit_tests",
    }
    if not isinstance(inputs, dict) or set(inputs) != required:
        raise Ugi3FreshPoolRouteCoverageV6Error("fresh-pool v6 inputs changed")
    paths = {
        label: _pin(repo, record, label=label)
        for label, record in inputs.items()
        if isinstance(record, Mapping)
    }
    if set(paths) != required:
        raise Ugi3FreshPoolRouteCoverageV6Error("fresh-pool v6 pin is malformed")
    base_config = _load(paths["base_config"], label="base v5 config")
    base_result = _load(paths["base_result"], label="base v5 result")
    qualification = _load(paths["source_qualification"], label="source qualification")
    if (
        base_config.get("schema_version") != "phase1_ugi3_fresh_pool_route_coverage_config.v5"
        or base_result.get("schema_version") != "phase1_ugi3_fresh_pool_route_coverage.v5"
        or base_result.get("config_sha256") != sha256_file(paths["base_config"])
        or base_result.get("summary") != base_config.get("expected_summary")
        or base_result.get("artifacts", {})
        .get("component_synthesis_values.json.gz", {})
        .get("sha256")
        != sha256_file(paths["base_component_values"])
        or base_result.get("artifacts", {})
        .get("product_synthesis_values.json.gz", {})
        .get("sha256")
        != sha256_file(paths["base_product_values"])
    ):
        raise Ugi3FreshPoolRouteCoverageV6Error("immutable v5 ownership failed")
    if (
        qualification.get("status") != "behavior_preserving_synthesis_value_source_qualified"
        or qualification.get("summary", {}).get("all_behavior_preserving") is not True
        or qualification.get("qualified_source", {}).get("sha256")
        != sha256_file(paths["value_source"])
        or qualification.get("adjudication", {}).get("fresh_pool_vnext_may_advance") is not True
        or qualification.get("adjudication", {}).get("sealed_holdout_accessed") is not False
    ):
        raise Ugi3FreshPoolRouteCoverageV6Error("source qualification does not authorize v6")
    component_bytes = paths["base_component_values"].read_bytes()
    product_bytes = paths["base_product_values"].read_bytes()
    summary = {
        **base_result["summary"],
        "base_version": "fresh_pool_route_coverage_v5",
        "value_records_changed": 0,
        "source_lineage_requalified": True,
    }
    return (
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "immutable_v5_values_requalified_under_current_source",
            "config_sha256": sha256_file(config_path),
            "policy": policy,
            "summary": summary,
            "inputs": {
                label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
                for label, path in sorted(paths.items())
            },
            "artifacts": {
                "component_synthesis_values.json.gz": {
                    "schema_version": "phase1_ugi3_fresh_pool_component_values.v5",
                    "sha256": sha256_bytes(component_bytes),
                    "byte_identical_to_v5": True,
                },
                "product_synthesis_values.json.gz": {
                    "schema_version": "phase1_ugi3_fresh_pool_product_values.v5",
                    "sha256": sha256_bytes(product_bytes),
                    "byte_identical_to_v5": True,
                },
            },
            "adjudication": {
                "route_coverage_changed": False,
                "historical_artifacts_rewritten": False,
                "zero_guidance_requalification_may_advance": True,
                "nonzero_guidance_authorized": False,
                "sealed_holdout_accessed": False,
            },
        },
        component_bytes,
        product_bytes,
    )
