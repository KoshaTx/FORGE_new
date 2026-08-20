"""Diagnostic closure sensitivity for the Graph2Edits--AiZynthFinder cascade."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.route.sources.ugi3_source_neutral_proposal_adjudication import (
    _content_sha256,
    _gzip_jsonl,
    _sha256_file,
)
from forge.value.audit.ugi3_proposal_aware_closure_sensitivity import (
    ClosureSensitivityError,
    _forward_verified_program,
    _qualified_components,
    _read_json,
    _read_jsonl_gzip,
    _scenario_metrics,
)

CONFIG_SCHEMA_VERSION = "forge.ugi3_hybrid_proposal_closure_sensitivity_config.v1"
RESULT_SCHEMA_VERSION = "forge.ugi3_hybrid_proposal_closure_sensitivity.v1"
WORKLIST_SCHEMA_VERSION = "forge.ugi3_hybrid_proposal_evidence_worklist.v1"


def _solved_residual_sets(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[set[tuple[str, str]], set[str]]:
    components: set[tuple[str, str]] = set()
    leaves: set[str] = set()
    for row in rows:
        search = row.get("full_search")
        target = row.get("target")
        if not isinstance(search, Mapping) or not isinstance(target, Mapping):
            raise ClosureSensitivityError("AiZynthFinder residual row is malformed")
        if not bool(search.get("is_solved_to_public_stock")):
            continue
        cohort = str(target["cohort"])
        smiles = str(target["canonical_smiles"])
        if cohort == "graph2edits_residual_component":
            components.add((str(target["role"]), smiles))
        elif cohort == "unresolved_upstream_leaf":
            leaves.add(smiles)
        else:
            raise ClosureSensitivityError(f"unexpected AiZynthFinder cohort: {cohort}")
    return components, leaves


def _hybrid_diagnostic_qualified(
    component_rows: Sequence[Mapping[str, Any]],
    solved_components: set[tuple[str, str]],
    solved_leaves: set[str],
) -> set[tuple[str, str]]:
    qualified = _qualified_components(component_rows, "current_evidence")
    qualified.update(solved_components)
    for row in component_rows:
        unresolved = {str(value) for value in row.get("unresolved_projected_leaves", [])}
        if _forward_verified_program(row) and unresolved and unresolved <= solved_leaves:
            qualified.add((str(row["role"]), str(row["target_smiles"])))
    return qualified


def build_hybrid_proposal_closure_sensitivity(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Build a non-authoritative hybrid proposal ceiling and evidence worklist."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ClosureSensitivityError("hybrid closure-sensitivity config schema changed")
    paths: dict[str, Path] = {}
    for label, record in config.get("inputs", {}).items():
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ClosureSensitivityError(f"malformed input: {label}")
        path = repo / str(record["path"])
        if _sha256_file(path) != str(record["sha256"]):
            raise ClosureSensitivityError(f"input hash changed: {label}")
        paths[str(label)] = path
    if set(paths) != {
        "support_audits",
        "graph2edits_component_ledger",
        "aizynthfinder_result",
        "aizynthfinder_ledger",
    }:
        raise ClosureSensitivityError("hybrid closure-sensitivity input set changed")
    aizynthfinder_result = _read_json(paths["aizynthfinder_result"])
    if aizynthfinder_result.get("status") != "complete_complementary_proposal_only_diagnostic":
        raise ClosureSensitivityError("AiZynthFinder residual diagnostic is incomplete")
    component_rows = _read_jsonl_gzip(paths["graph2edits_component_ledger"])
    aizynthfinder_rows = _read_jsonl_gzip(paths["aizynthfinder_ledger"])
    solved_components, solved_leaves = _solved_residual_sets(aizynthfinder_rows)
    support = _read_json(paths["support_audits"])
    population = config["population"]

    current = _qualified_components(component_rows, "current_evidence")
    graph2edits_ceiling = _qualified_components(
        component_rows, "all_forward_verified_program_leaves_closed"
    )
    hybrid_ceiling = _hybrid_diagnostic_qualified(component_rows, solved_components, solved_leaves)

    worklist: list[dict[str, Any]] = []
    for row in aizynthfinder_rows:
        search = row["full_search"]
        if not bool(search.get("is_solved_to_public_stock")):
            continue
        target = row["target"]
        cohort = str(target["cohort"])
        role = str(target["role"])
        smiles = str(target["canonical_smiles"])
        added: set[tuple[str, str]] = set()
        if cohort == "graph2edits_residual_component":
            added.add((role, smiles))
        else:
            for component in component_rows:
                unresolved = {
                    str(value) for value in component.get("unresolved_projected_leaves", [])
                }
                if _forward_verified_program(component) and unresolved and unresolved <= {smiles}:
                    added.add((str(component["role"]), str(component["target_smiles"])))
        metrics = _scenario_metrics(support, population, current | added)
        worklist.append(
            {
                "schema_version": WORKLIST_SCHEMA_VERSION,
                "cohort": cohort,
                "role": role,
                "target_smiles": smiles,
                "aizynthfinder_target_id": str(target["target_id"]),
                "aizynthfinder_route_steps": int(search["statistics"]["number_of_steps"]),
                "public_stock_precursors": sorted(
                    value.strip()
                    for value in str(search["statistics"]["precursors_in_stock"]).split(",")
                    if value.strip()
                ),
                "potentially_added_components": [
                    {"role": component_role, "smiles": component_smiles}
                    for component_role, component_smiles in sorted(added)
                ],
                "counterfactual_metrics": metrics,
                "proposal_is_route_evidence": False,
                "public_stock_solution_is_current_procurement_evidence": False,
                "requires_independent_forward_scope_operational_and_l3_adjudication": True,
            }
        )
    worklist.sort(
        key=lambda row: (
            -int(row["counterfactual_metrics"]["newly_ready_checkpoint_assessments"]),
            -int(row["counterfactual_metrics"]["mixed_checkpoint_groups"]),
            -int(row["counterfactual_metrics"]["unique_route_ready_productive_finals"]),
            int(row["aizynthfinder_route_steps"]),
            str(row["target_smiles"]),
        )
    )
    for rank, row in enumerate(worklist, start=1):
        row["priority_rank"] = rank

    worklist_bytes = _gzip_jsonl(worklist)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "hybrid_proposal_closure_sensitivity_complete",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": _sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": _sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "summary": {
            "aizynthfinder_public_stock_solved_residual_components": len(solved_components),
            "aizynthfinder_public_stock_solved_upstream_leaves": len(solved_leaves),
            "scenarios": {
                "current_evidence": _scenario_metrics(support, population, current),
                "graph2edits_all_forward_program_leaves_closed": _scenario_metrics(
                    support, population, graph2edits_ceiling
                ),
                "hybrid_public_stock_diagnostic_ceiling": _scenario_metrics(
                    support, population, hybrid_ceiling
                ),
            },
            "top_evidence_priorities": [
                {
                    "priority_rank": row["priority_rank"],
                    "cohort": row["cohort"],
                    "target_smiles": row["target_smiles"],
                    "aizynthfinder_route_steps": row["aizynthfinder_route_steps"],
                    "counterfactual_metrics": row["counterfactual_metrics"],
                }
                for row in worklist[:10]
            ],
        },
        "decision": {
            "production_synthesis_tilt_authorized": False,
            "production_proposal_stack": "exact registry then Graph2Edits",
            "residual_challenger": "AiZynthFinder",
            "residual_challenger_promotion_condition": (
                "material incremental independently verified route closure"
            ),
            "next_step": "adjudicate the frozen high-impact hybrid evidence worklist",
        },
        "scientific_authority": {
            "counterfactual_is_route_evidence": False,
            "public_stock_solution_is_current_procurement_evidence": False,
            "proposal_score_used": False,
            "family_label_used_as_gate": False,
            "gates_relaxed": False,
            "candidate_selection": False,
        },
        "artifacts": {
            "hybrid_evidence_worklist.jsonl.gz": {
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
    "RESULT_SCHEMA_VERSION",
    "WORKLIST_SCHEMA_VERSION",
    "_hybrid_diagnostic_qualified",
    "_solved_residual_sets",
    "build_hybrid_proposal_closure_sensitivity",
]
