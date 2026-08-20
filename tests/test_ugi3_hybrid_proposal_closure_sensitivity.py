from __future__ import annotations

from forge.value.audit.ugi3_hybrid_proposal_closure_sensitivity import (
    _hybrid_diagnostic_qualified,
    _solved_residual_sets,
)


def test_hybrid_ceiling_adds_only_solved_residuals_and_solved_leaves() -> None:
    residual_rows = [
        {
            "target": {
                "cohort": "graph2edits_residual_component",
                "role": "amine_head",
                "canonical_smiles": "CCN",
            },
            "full_search": {"is_solved_to_public_stock": True},
        },
        {
            "target": {
                "cohort": "unresolved_upstream_leaf",
                "role": "upstream_terminal_leaf",
                "canonical_smiles": "CCCO",
            },
            "full_search": {"is_solved_to_public_stock": True},
        },
    ]
    solved_components, solved_leaves = _solved_residual_sets(residual_rows)
    components = [
        {
            "role": "amine_head",
            "target_smiles": "CCN",
            "terminal_closed_route_hypothesis": False,
            "expected_final_proposal_sha256": None,
            "all_program_steps_forward_verified": False,
            "projected_leaves": [],
            "unresolved_projected_leaves": [],
        },
        {
            "role": "isocyanide_tail",
            "target_smiles": "[C-]#[N+]CCC",
            "terminal_closed_route_hypothesis": False,
            "expected_final_proposal_sha256": "proposal",
            "all_program_steps_forward_verified": True,
            "projected_leaves": ["CCCO"],
            "unresolved_projected_leaves": ["CCCO"],
        },
    ]
    assert _hybrid_diagnostic_qualified(components, solved_components, solved_leaves) == {
        ("amine_head", "CCN"),
        ("isocyanide_tail", "[C-]#[N+]CCC"),
    }
