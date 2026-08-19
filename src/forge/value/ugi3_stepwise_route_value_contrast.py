"""Read-only contrast gate for stepwise terminal-closed route hypotheses."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.value.ugi3_source_bounded_route_value_contrast import (
    EXACT,
    Ugi3SourceBoundedContrastError,
    _read_json,
    _read_jsonl_gzip,
    _sha256_file,
    _stable_json,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi3_stepwise_route_value_contrast_config.v2"
RESULT_SCHEMA_VERSION = "phase1_ugi3_stepwise_route_value_contrast.v2"
FAMILY_ALL = "family_projected_all_current_leaves"


def stepwise_utility(components: list[Mapping[str, Any]], qualified: set[tuple[str, str]]) -> int:
    """Return a monotone product utility over legacy and new route states."""

    if len(components) != 3:
        raise Ugi3SourceBoundedContrastError("terminal readiness requires three components")
    return int(
        all(
            str(component.get("graded_evidence_class")) in {EXACT, FAMILY_ALL}
            or (str(component.get("role")), str(component.get("canonical_smiles"))) in qualified
            for component in components
        )
    )


def build_stepwise_route_value_contrast(repo: Path, config_path: Path) -> dict[str, Any]:
    """Measure whether the chemistry-first route states vary within SMC groups."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3SourceBoundedContrastError("unsupported stepwise contrast config")
    paths: dict[str, Path] = {}
    for label, record in config.get("inputs", {}).items():
        path = repo / str(record["path"])
        if _sha256_file(path) != str(record["sha256"]):
            raise Ugi3SourceBoundedContrastError(f"input hash changed: {label}")
        paths[str(label)] = path
    if set(paths) != {
        "stepwise_result",
        "stepwise_ledger",
        "frozen_matched_support_records",
        "superseded_contrast_config",
        "invalidated_nonmonotone_result",
    }:
        raise Ugi3SourceBoundedContrastError("stepwise contrast input set changed")

    stepwise = _read_json(paths["stepwise_result"])
    rows = _read_jsonl_gzip(paths["stepwise_ledger"])
    qualified = {
        (str(row["role"]), str(row["target_smiles"]))
        for row in rows
        if bool(row.get("terminal_closed_route_hypothesis"))
    }
    expected = int(stepwise["summary"]["terminal_closed_route_hypotheses"])
    if len(qualified) != expected:
        raise Ugi3SourceBoundedContrastError("terminal-closed component census changed")

    support = _read_json(paths["frozen_matched_support_records"])
    policy = config["audit_population"]
    checkpoints = {int(value) for value in policy["checkpoints"]}
    checkpoint_rows = [
        row
        for row in support.get("records", [])
        if row.get("arm") == policy["arm"]
        and row.get("assessment_phase") == policy["checkpoint_phase"]
        and int(row.get("checkpoint")) in checkpoints
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
        if not isinstance(receipt, Mapping) or not isinstance(receipt.get("components"), list):
            raise Ugi3SourceBoundedContrastError("graded checkpoint receipt is missing")
        old = int(receipt.get("binary_controller_utility"))
        new = stepwise_utility(receipt["components"], qualified)
        transitions[f"{old}->{new}"] += 1
        checkpoint = int(row["checkpoint"])
        groups[(checkpoint, int(row["program_index"]))].append(new)
        checkpoint_ready[checkpoint] += new
        checkpoint_total[checkpoint] += 1

    mixed_by_checkpoint: Counter[int] = Counter()
    observed_groups = 0
    mixed_groups = 0
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
        if stepwise_utility(receipt["components"], qualified):
            support_audit = row.get("support_audit")
            product = (
                support_audit.get("support", {}).get("product_smiles")
                if isinstance(support_audit, Mapping)
                else None
            )
            final_products.add(str(product or row["terminal_sha256"]))

    arms = stepwise["summary"]["arms"]
    balanced = min(int(row["stepwise_terminal_closed_or_better"]) for row in arms.values())
    thresholds = config["retry_authorization_gates"]
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
            mixed_by_checkpoint[checkpoint] > 0 for checkpoint in checkpoints
        ),
        "unique_route_ready_productive_finals": len(final_products)
        >= int(thresholds["minimum_unique_route_ready_productive_finals"]),
    }
    authorized = all(gate_results.values())
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "stepwise_route_value_contrast_complete",
        "supersedes": {
            "result": str(paths["invalidated_nonmonotone_result"].relative_to(repo)),
            "reason": (
                "the v1 contrast replaced rather than unioned legacy positive route "
                "tiers, creating impossible 1-to-0 utility regressions"
            ),
            "v1_result_must_not_support_scientific_or_production_decisions": True,
        },
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": _sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": _sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "summary": {
            "qualified_terminal_closed_components": len(qualified),
            "balanced_unique_stepwise_route_ready_products": balanced,
            "checkpoint_assessments": len(checkpoint_rows),
            "observed_checkpoint_groups_with_at_least_two_assessments": observed_groups,
            "mixed_checkpoint_groups": mixed_groups,
            "mixed_checkpoint_group_fraction": (
                None if observed_groups == 0 else mixed_groups / observed_groups
            ),
            "mixed_groups_by_checkpoint": {
                str(checkpoint): mixed_by_checkpoint[checkpoint]
                for checkpoint in sorted(checkpoints)
            },
            "route_ready_by_checkpoint": {
                str(checkpoint): {
                    "ready": checkpoint_ready[checkpoint],
                    "assessed": checkpoint_total[checkpoint],
                }
                for checkpoint in sorted(checkpoints)
            },
            "old_to_stepwise_utility": dict(sorted(transitions.items())),
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
                "run one frozen matched synthesis-guidance challenger at lambda=0.25"
                if authorized
                else "retain complete generation plus post-generation routing"
            )
        },
        "scientific_authority": {
            "proposal_scores_used": False,
            "biological_scores_used": False,
            "stepwise_route_hypothesis_is_experimental_success": False,
            "route_value_is_success_probability": False,
            "read_only_reanalysis": True,
            "legacy_whole_program_family_label_used_as_gate": False,
            "new_route_states_may_only_add_to_legacy_positive_states": True,
            "exact_component_evidence_label": EXACT,
        },
    }
    result["result_sha256"] = hashlib.sha256(_stable_json(result).encode()).hexdigest()
    return result


__all__ = [
    "build_stepwise_route_value_contrast",
    "stepwise_utility",
]
