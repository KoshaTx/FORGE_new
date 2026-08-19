"""Read-only contrast gate for source-bounded synthesis readiness."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = "phase1_ugi3_source_bounded_route_value_contrast_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_source_bounded_route_value_contrast.v1"
QUALIFIED_FAMILY = "source_bounded_family_route_all_current"
EXACT = "exact_complete_current"


class Ugi3SourceBoundedContrastError(RuntimeError):
    """Raised when frozen contrast inputs or semantics change."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise Ugi3SourceBoundedContrastError(f"JSON object required: {path}")
    return value


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise Ugi3SourceBoundedContrastError(f"malformed JSONL: {path}")
    return rows


def source_bounded_utility(
    components: list[Mapping[str, Any]], qualified: set[tuple[str, str]]
) -> int:
    if len(components) != 3:
        raise Ugi3SourceBoundedContrastError("terminal readiness requires three components")
    return int(
        all(
            str(component.get("graded_evidence_class")) == EXACT
            or (str(component.get("role")), str(component.get("canonical_smiles"))) in qualified
            for component in components
        )
    )


def build_source_bounded_route_value_contrast(repo: Path, config_path: Path) -> dict[str, Any]:
    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3SourceBoundedContrastError("unsupported contrast config")
    paths = {}
    for label, record in config.get("inputs", {}).items():
        path = repo / str(record["path"])
        if _sha256_file(path) != record["sha256"]:
            raise Ugi3SourceBoundedContrastError(f"input hash changed: {label}")
        paths[str(label)] = path
    if set(paths) != {
        "adjudication_result",
        "adjudication_ledger",
        "frozen_matched_support_records",
    }:
        raise Ugi3SourceBoundedContrastError("contrast input set changed")

    adjudication = _read_json(paths["adjudication_result"])
    adjudication_rows = _read_jsonl_gzip(paths["adjudication_ledger"])
    qualified = {
        (str(row["role"]), str(row["target_smiles"]))
        for row in adjudication_rows
        if row.get("adjudicated_route_state") == QUALIFIED_FAMILY
    }
    if len(qualified) != int(adjudication["summary"]["qualified_component_routes"]):
        raise Ugi3SourceBoundedContrastError("qualified component census changed")

    support = _read_json(paths["frozen_matched_support_records"])
    policy = config["audit_population"]
    checkpoint_rows = [
        row
        for row in support.get("records", [])
        if row.get("arm") == policy["arm"]
        and row.get("assessment_phase") == policy["checkpoint_phase"]
        and int(row.get("checkpoint")) in set(policy["checkpoints"])
    ]
    final_rows = [
        row
        for row in support.get("records", [])
        if row.get("arm") == policy["arm"]
        and row.get("assessment_phase") == policy["terminal_phase"]
    ]
    if not checkpoint_rows or not final_rows:
        raise Ugi3SourceBoundedContrastError("frozen contrast population is empty")

    transitions: Counter[str] = Counter()
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    checkpoint_ready: Counter[int] = Counter()
    checkpoint_total: Counter[int] = Counter()
    for row in checkpoint_rows:
        receipt = row.get("graded_route_readiness")
        if not isinstance(receipt, Mapping):
            raise Ugi3SourceBoundedContrastError("graded checkpoint receipt is missing")
        components = receipt.get("components")
        if not isinstance(components, list):
            raise Ugi3SourceBoundedContrastError("graded component receipt is missing")
        old = int(receipt.get("binary_controller_utility"))
        new = source_bounded_utility(components, qualified)
        transitions[f"{old}->{new}"] += 1
        checkpoint = int(row["checkpoint"])
        groups[(checkpoint, int(row["program_index"]))].append(new)
        checkpoint_ready[checkpoint] += new
        checkpoint_total[checkpoint] += 1

    mixed_by_checkpoint: Counter[int] = Counter()
    mixed_groups = 0
    observed_groups = 0
    for (checkpoint, _), values in groups.items():
        if len(values) < 2:
            continue
        observed_groups += 1
        if len(set(values)) > 1:
            mixed_groups += 1
            mixed_by_checkpoint[checkpoint] += 1

    final_products: set[str] = set()
    for row in final_rows:
        receipt = row.get("graded_route_readiness")
        if not isinstance(receipt, Mapping) or not isinstance(receipt.get("components"), list):
            raise Ugi3SourceBoundedContrastError("graded final receipt is missing")
        if source_bounded_utility(receipt["components"], qualified):
            product = (
                row.get("support_audit", {}).get("support", {}).get("product_smiles")
                if isinstance(row.get("support_audit"), Mapping)
                else None
            )
            final_products.add(str(product or row["terminal_sha256"]))

    thresholds = config["retry_authorization_gates"]
    balanced = int(
        adjudication["summary"]["balanced_unique_products_source_bounded_family_or_better"]
    )
    gate_results = {
        "qualified_component_count": len(qualified)
        >= int(thresholds["minimum_qualified_components"]),
        "balanced_unique_route_ready_products": balanced
        >= int(thresholds["minimum_balanced_unique_route_ready_products"]),
        "mixed_checkpoint_group_count": mixed_groups
        >= int(thresholds["minimum_mixed_checkpoint_groups"]),
        "mixed_checkpoint_group_fraction": (
            observed_groups > 0
            and mixed_groups / observed_groups
            >= float(thresholds["minimum_mixed_checkpoint_group_fraction"])
        ),
        "mixed_group_at_every_checkpoint": all(
            mixed_by_checkpoint[int(checkpoint)] > 0 for checkpoint in policy["checkpoints"]
        ),
        "unique_route_ready_productive_finals": len(final_products)
        >= int(thresholds["minimum_unique_route_ready_productive_finals"]),
    }
    authorized = bool(gate_results) and all(gate_results.values())
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "source_bounded_route_value_contrast_complete",
        "config": {"path": str(config_path.relative_to(repo)), "sha256": _sha256_file(config_path)},
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": _sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "summary": {
            "qualified_components": len(qualified),
            "balanced_unique_products_source_bounded_family_or_better": balanced,
            "checkpoint_assessments": len(checkpoint_rows),
            "observed_checkpoint_groups_with_at_least_two_assessments": observed_groups,
            "mixed_checkpoint_groups": mixed_groups,
            "mixed_checkpoint_group_fraction": (
                None if observed_groups == 0 else mixed_groups / observed_groups
            ),
            "mixed_groups_by_checkpoint": {
                str(checkpoint): mixed_by_checkpoint[int(checkpoint)]
                for checkpoint in policy["checkpoints"]
            },
            "route_ready_by_checkpoint": {
                str(checkpoint): {
                    "ready": checkpoint_ready[int(checkpoint)],
                    "assessed": checkpoint_total[int(checkpoint)],
                }
                for checkpoint in policy["checkpoints"]
            },
            "old_to_source_bounded_utility": dict(sorted(transitions.items())),
            "unique_route_ready_productive_finals": len(final_products),
        },
        "retry_authorization": {
            "gate_results": gate_results,
            "all_gates_pass": authorized,
            "matched_synthesis_tilt_rerun_authorized": authorized,
            "lambda_tuning_authorized": False,
            "synthesis_guidance_promoted": False,
        },
        "decision": {
            "next_step": (
                "run_one frozen matched synthesis-guidance challenger at the previously reviewed lambda"
                if authorized
                else "retain complete generation plus source-neutral post-generation routing; do not rerun synthesis tilting"
            )
        },
        "scientific_authority": {
            "proposal_scores_used": False,
            "biological_scores_used": False,
            "route_value_is_success_probability": False,
            "read_only_reanalysis": True,
        },
    }
    result["result_sha256"] = hashlib.sha256(_stable_json(result).encode()).hexdigest()
    return result


__all__ = [
    "Ugi3SourceBoundedContrastError",
    "build_source_bounded_route_value_contrast",
    "source_bounded_utility",
]
