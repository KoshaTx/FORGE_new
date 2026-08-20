from __future__ import annotations

from forge.value.audit.ugi3_proposal_aware_closure_sensitivity import _qualified_components


def test_qualified_components_keep_counterfactuals_separate_from_evidence() -> None:
    rows = [
        {
            "role": "amine_head",
            "target_smiles": "NCC",
            "terminal_closed_route_hypothesis": True,
            "proposal_coherent": True,
            "expected_final_proposal_sha256": "known",
            "all_program_steps_forward_verified": True,
            "projected_leaves": ["CCN"],
            "unresolved_projected_leaves": [],
        },
        {
            "role": "isocyanide_tail",
            "target_smiles": "[C-]#[N+]CCC",
            "terminal_closed_route_hypothesis": False,
            "proposal_coherent": True,
            "expected_final_proposal_sha256": "proposal",
            "all_program_steps_forward_verified": True,
            "projected_leaves": ["CCCN"],
            "unresolved_projected_leaves": ["CCCN"],
        },
        {
            "role": "amine_head",
            "target_smiles": "NCCC",
            "terminal_closed_route_hypothesis": False,
            "proposal_coherent": True,
            "expected_final_proposal_sha256": None,
            "all_program_steps_forward_verified": False,
            "projected_leaves": [],
            "unresolved_projected_leaves": [],
        },
    ]
    current = _qualified_components(rows, "current_evidence")
    assert current == {("amine_head", "NCC")}
    forward = _qualified_components(rows, "all_forward_verified_program_leaves_closed")
    assert forward == {
        ("amine_head", "NCC"),
        ("isocyanide_tail", "[C-]#[N+]CCC"),
    }
    one_leaf = _qualified_components(rows, "single_leaf_closed", leaf="CCCN")
    assert one_leaf == forward
    proposal_ceiling = _qualified_components(rows, "all_graph_consistent_proposals_accepted")
    assert len(proposal_ceiling) == 3
