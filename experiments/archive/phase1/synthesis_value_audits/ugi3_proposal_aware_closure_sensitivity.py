"""Counterfactual L3-closure sensitivity for proposal-aware route contrast.

This module does not promote learned proposals or assume that missing leaves are
available.  It asks a narrower planning question: if independently
forward-verified proposal programs later acquire admissible terminal-material
evidence, how much checkpoint contrast could they create on the already frozen
generation panel?
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.synthesis.sources.ugi3_source_neutral_proposal_adjudication import (
    _content_sha256,
    _gzip_jsonl,
    _sha256_file,
)
from experiments.archive.phase1.synthesis_value_audits.ugi3_proposal_aware_checkpoint_contrast import _product_utility

CONFIG_SCHEMA_VERSION = "forge.ugi3_proposal_aware_closure_sensitivity_config.v1"
RESULT_SCHEMA_VERSION = "forge.ugi3_proposal_aware_closure_sensitivity.v1"
WORKLIST_SCHEMA_VERSION = "forge.ugi3_proposal_aware_leaf_worklist.v1"


class ClosureSensitivityError(RuntimeError):
    """Raised when a frozen closure-sensitivity input is malformed."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ClosureSensitivityError(f"expected JSON object: {path}")
    return value


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ClosureSensitivityError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _forward_verified_program(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("expected_final_proposal_sha256")
        and row.get("all_program_steps_forward_verified")
        and row.get("projected_leaves")
    )


def _qualified_components(
    rows: Sequence[Mapping[str, Any]], scenario: str, leaf: str | None = None
) -> set[tuple[str, str]]:
    qualified: set[tuple[str, str]] = set()
    for row in rows:
        admitted = bool(row.get("terminal_closed_route_hypothesis"))
        if scenario == "all_forward_verified_program_leaves_closed":
            admitted = admitted or _forward_verified_program(row)
        elif scenario == "all_graph_consistent_proposals_accepted":
            admitted = admitted or bool(row.get("proposal_coherent"))
        elif scenario == "single_leaf_closed":
            unresolved = {str(value) for value in row.get("unresolved_projected_leaves", [])}
            admitted = admitted or bool(
                leaf and _forward_verified_program(row) and unresolved and unresolved <= {leaf}
            )
        elif scenario != "current_evidence":
            raise ClosureSensitivityError(f"unknown scenario: {scenario}")
        if admitted:
            qualified.add((str(row["role"]), str(row["target_smiles"])))
    return qualified


def _scenario_metrics(
    support: Mapping[str, Any],
    population: Mapping[str, Any],
    qualified: set[tuple[str, str]],
) -> dict[str, Any]:
    checkpoints = {int(value) for value in population["checkpoints"]}
    checkpoint_rows = [
        row
        for row in support.get("records", [])
        if row.get("arm") == population["arm"]
        and row.get("assessment_phase") == population["checkpoint_phase"]
        and int(row.get("checkpoint")) in checkpoints
    ]
    final_rows = [
        row
        for row in support.get("records", [])
        if row.get("arm") == population["arm"]
        and row.get("assessment_phase") == population["terminal_phase"]
    ]
    if not checkpoint_rows or not final_rows:
        raise ClosureSensitivityError("closure-sensitivity population is empty")

    transitions: Counter[str] = Counter()
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    ready_by_checkpoint: Counter[int] = Counter()
    total_by_checkpoint: Counter[int] = Counter()
    for row in checkpoint_rows:
        receipt = row.get("graded_route_readiness")
        if not isinstance(receipt, Mapping) or not isinstance(receipt.get("components"), list):
            raise ClosureSensitivityError("graded checkpoint receipt is malformed")
        old = int(receipt["binary_controller_utility"])
        new = _product_utility(receipt["components"], qualified)
        transitions[f"{old}->{new}"] += 1
        checkpoint = int(row["checkpoint"])
        groups[(checkpoint, int(row["program_index"]))].append(new)
        ready_by_checkpoint[checkpoint] += new
        total_by_checkpoint[checkpoint] += 1

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
            raise ClosureSensitivityError("graded final receipt is malformed")
        if not _product_utility(receipt["components"], qualified):
            continue
        audit = row.get("support_audit")
        product = (
            audit.get("support", {}).get("product_smiles") if isinstance(audit, Mapping) else None
        )
        final_products.add(str(product or row["terminal_sha256"]))

    return {
        "supplemental_qualified_components": len(qualified),
        "checkpoint_assessments": len(checkpoint_rows),
        "utility_transitions": dict(sorted(transitions.items())),
        "newly_ready_checkpoint_assessments": transitions["0->1"],
        "observed_checkpoint_groups": observed_groups,
        "mixed_checkpoint_groups": mixed_groups,
        "mixed_checkpoint_group_fraction": (
            None if not observed_groups else mixed_groups / observed_groups
        ),
        "mixed_groups_by_checkpoint": {
            str(checkpoint): mixed_by_checkpoint[checkpoint] for checkpoint in sorted(checkpoints)
        },
        "route_ready_by_checkpoint": {
            str(checkpoint): {
                "ready": ready_by_checkpoint[checkpoint],
                "assessed": total_by_checkpoint[checkpoint],
            }
            for checkpoint in sorted(checkpoints)
        },
        "unique_route_ready_productive_finals": len(final_products),
    }


def build_proposal_aware_closure_sensitivity(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Build the frozen non-authoritative closure-sensitivity artifact."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ClosureSensitivityError("closure-sensitivity config schema changed")
    paths: dict[str, Path] = {}
    for label, record in config.get("inputs", {}).items():
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ClosureSensitivityError(f"malformed input: {label}")
        path = repo / str(record["path"])
        if _sha256_file(path) != str(record["sha256"]):
            raise ClosureSensitivityError(f"input hash changed: {label}")
        paths[str(label)] = path
    if set(paths) != {"support_audits", "proposal_adjudication", "component_ledger"}:
        raise ClosureSensitivityError("closure-sensitivity input set changed")

    adjudication = _read_json(paths["proposal_adjudication"])
    if adjudication.get("status") != "proposal_aware_checkpoint_contrast_complete":
        raise ClosureSensitivityError("proposal adjudication is not complete")
    rows = _read_jsonl_gzip(paths["component_ledger"])
    if len(rows) != int(adjudication["summary"]["proposal_evaluated_components"]):
        raise ClosureSensitivityError("component ledger census changed")
    support = _read_json(paths["support_audits"])
    population = config["population"]

    scenarios: dict[str, dict[str, Any]] = {}
    for scenario in (
        "current_evidence",
        "all_forward_verified_program_leaves_closed",
        "all_graph_consistent_proposals_accepted",
    ):
        qualified = _qualified_components(rows, scenario)
        scenarios[scenario] = _scenario_metrics(support, population, qualified)

    leaves = sorted(
        {
            str(leaf)
            for row in rows
            if _forward_verified_program(row)
            for leaf in row.get("unresolved_projected_leaves", [])
        }
    )
    worklist: list[dict[str, Any]] = []
    for leaf in leaves:
        affected = [
            row
            for row in rows
            if _forward_verified_program(row)
            and leaf in {str(value) for value in row.get("unresolved_projected_leaves", [])}
        ]
        qualified = _qualified_components(rows, "single_leaf_closed", leaf=leaf)
        metrics = _scenario_metrics(support, population, qualified)
        worklist.append(
            {
                "schema_version": WORKLIST_SCHEMA_VERSION,
                "leaf_smiles": leaf,
                "affected_component_count": len(affected),
                "affected_roles": sorted({str(row["role"]) for row in affected}),
                "affected_program_annotations": sorted(
                    {str(row["program_annotation"]) for row in affected}
                ),
                "affected_targets": sorted(str(row["target_smiles"]) for row in affected),
                "counterfactual_metrics": metrics,
                "is_current_terminal_claim": False,
                "requires_independent_l3_evidence": True,
            }
        )
    worklist.sort(
        key=lambda row: (
            -int(row["counterfactual_metrics"]["newly_ready_checkpoint_assessments"]),
            -int(row["counterfactual_metrics"]["mixed_checkpoint_groups"]),
            -int(row["counterfactual_metrics"]["unique_route_ready_productive_finals"]),
            str(row["leaf_smiles"]),
        )
    )
    for rank, row in enumerate(worklist, start=1):
        row["priority_rank"] = rank

    worklist_bytes = _gzip_jsonl(worklist)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "proposal_aware_closure_sensitivity_complete",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": _sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": _sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "summary": {
            "component_rows": len(rows),
            "forward_verified_upstream_open_components": sum(
                _forward_verified_program(row)
                and not bool(row.get("terminal_closed_route_hypothesis"))
                for row in rows
            ),
            "unique_missing_projected_leaves": len(leaves),
            "scenarios": scenarios,
            "top_leaf_priorities": [
                {
                    "priority_rank": row["priority_rank"],
                    "leaf_smiles": row["leaf_smiles"],
                    "affected_component_count": row["affected_component_count"],
                    "counterfactual_metrics": row["counterfactual_metrics"],
                }
                for row in worklist[:10]
            ],
        },
        "decision": {
            "production_synthesis_tilt_authorized": False,
            "next_step": "adjudicate the frozen high-impact L3 leaf worklist",
            "retest_condition": (
                "rerun actual checkpoint contrast only after independent evidence changes "
                "the terminal inventory"
            ),
        },
        "scientific_authority": {
            "counterfactual_is_route_evidence": False,
            "counterfactual_is_procurement_evidence": False,
            "proposal_only_ceiling_is_authoritative": False,
            "proposal_scores_used": False,
            "reaction_family_used_as_gate": False,
            "biological_scores_used": False,
            "gates_relaxed": False,
        },
        "artifacts": {
            "leaf_worklist.jsonl.gz": {
                "schema_version": WORKLIST_SCHEMA_VERSION,
                "rows": len(worklist),
                "sha256": hashlib.sha256(worklist_bytes).hexdigest(),
            }
        },
    }
    result["result_sha256"] = _content_sha256(result)
    return result, worklist_bytes


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "ClosureSensitivityError",
    "RESULT_SCHEMA_VERSION",
    "WORKLIST_SCHEMA_VERSION",
    "_qualified_components",
    "_scenario_metrics",
    "build_proposal_aware_closure_sensitivity",
]
