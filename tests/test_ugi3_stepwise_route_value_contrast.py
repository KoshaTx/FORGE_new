from __future__ import annotations

from forge.value.audit.ugi3_stepwise_route_value_contrast import stepwise_utility


def test_stepwise_utility_is_component_identity_based_not_family_named() -> None:
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
    qualified = {
        ("oxoester_aldehyde_body_tail", "aldehyde"),
        ("isocyanide_tail", "isocyanide"),
    }
    assert stepwise_utility(components, qualified) == 1


def test_stepwise_utility_preserves_legacy_family_positive_state() -> None:
    components = [
        {
            "role": "amine_head",
            "canonical_smiles": "head",
            "graded_evidence_class": "exact_complete_current",
        },
        {
            "role": "oxoester_aldehyde_body_tail",
            "canonical_smiles": "aldehyde",
            "graded_evidence_class": "family_projected_all_current_leaves",
        },
        {
            "role": "isocyanide_tail",
            "canonical_smiles": "isocyanide",
            "graded_evidence_class": "exact_complete_current",
        },
    ]
    assert stepwise_utility(components, set()) == 1
