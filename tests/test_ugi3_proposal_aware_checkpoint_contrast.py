from __future__ import annotations

from forge.value.audit.ugi3_proposal_aware_checkpoint_contrast import (
    _direct_current_trace,
    _product_utility,
)


def test_direct_trace_requires_graph_consistency_and_current_reactants() -> None:
    proposals = [
        {
            "rank": 1,
            "proposal_sha256": "proposal",
            "graph_consistent_discovery_hypothesis": True,
            "raw_discovery_resolution": {
                "traces": [
                    {
                        "reaction_id": "registered-step",
                        "role_ordered_reactants": ["CCO"],
                        "target_reconstructed": True,
                    }
                ]
            },
        }
    ]
    assert _direct_current_trace(proposals, {"CCO"}) == {
        "proposal_sha256": "proposal",
        "proposal_rank": 1,
        "reaction_id": "registered-step",
        "reactants": ["CCO"],
    }
    assert _direct_current_trace(proposals, set()) is None
    proposals[0]["graph_consistent_discovery_hypothesis"] = False
    assert _direct_current_trace(proposals, {"CCO"}) is None


def test_product_utility_only_adds_independently_qualified_components() -> None:
    components = [
        {
            "role": "amine_head",
            "canonical_smiles": "head",
            "graded_evidence_class": "exact_complete_current",
        },
        {
            "role": "oxoester_aldehyde_body_tail",
            "canonical_smiles": "aldehyde",
            "graded_evidence_class": "missing_knowledge",
        },
        {
            "role": "isocyanide_tail",
            "canonical_smiles": "isocyanide",
            "graded_evidence_class": "missing_knowledge",
        },
    ]
    assert _product_utility(components, set()) == 0
    assert (
        _product_utility(
            components,
            {
                ("oxoester_aldehyde_body_tail", "aldehyde"),
                ("isocyanide_tail", "isocyanide"),
            },
        )
        == 1
    )
