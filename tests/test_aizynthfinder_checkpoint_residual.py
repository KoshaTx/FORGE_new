from __future__ import annotations

from forge.route.aizynthfinder_checkpoint_residual import (
    build_residual_targets,
    summarize_residual_records,
)


def test_residual_targets_include_only_graph2edits_misses_and_missing_leaves() -> None:
    components = [
        {
            "role": "amine_head",
            "target_smiles": "CCN",
            "proposal_coherent": False,
            "terminal_closed_route_hypothesis": False,
        },
        {
            "role": "isocyanide_tail",
            "target_smiles": "[C-]#[N+]CC",
            "proposal_coherent": True,
            "terminal_closed_route_hypothesis": False,
        },
    ]
    leaves = [
        {
            "leaf_smiles": "CCCO",
            "priority_rank": 1,
            "affected_targets": ["CCCC=O"],
        }
    ]
    targets = build_residual_targets(components, leaves)
    assert len(targets) == 2
    assert {row["cohort"] for row in targets} == {
        "graph2edits_residual_component",
        "unresolved_upstream_leaf",
    }


def test_residual_summary_separates_proposal_and_stock_search() -> None:
    rows = [
        {
            "target": {"target_id": "one", "cohort": "residual"},
            "single_step_proposals": [{"rank": 1}],
            "full_search": {
                "execution_status": "complete",
                "is_solved_to_public_stock": True,
            },
        },
        {
            "target": {"target_id": "two", "cohort": "residual"},
            "single_step_proposals": [],
            "full_search": {
                "execution_status": "failed",
                "is_solved_to_public_stock": False,
            },
        },
    ]
    summary = summarize_residual_records(rows)["total"]
    assert summary["targets"] == 2
    assert summary["targets_with_single_step_proposals"] == 1
    assert summary["full_search_solved_to_public_stock"] == 1
    assert summary["full_search_execution_failures"] == 1
