from __future__ import annotations

from experiments.archive.phase1.design_audits.ugi_chemistry_bias_attribution import (
    compact_product_motif_summary,
    compact_tail_summary,
    prior_positive_junction_fractions,
    sampled_positive_junction_fractions,
)


def test_compact_tail_summary_separates_occurrence_and_unique_rates() -> None:
    result = compact_tail_summary(["CCCC=O", "CCCC=O", "CC(C)C=O"])
    assert result["component_occurrences"] == 3
    assert result["unique_exact_components"] == 2
    assert result["occurrence_feature_fractions"]["has_carbon_branch"] == 1 / 3
    assert result["unique_component_feature_fractions"]["has_carbon_branch"] == 1 / 2


def test_product_motifs_distinguish_alkyne_and_cumulene() -> None:
    result = compact_product_motif_summary(["CC#CC", "C=C=C", "CCCC"], maximum_examples=2)
    assert result["motifs"]["alkyne"]["count"] == 1
    assert result["motifs"]["allene_or_carbon_cumulene"]["count"] == 1
    assert result["motifs"]["peroxide"]["count"] == 0


def test_program_junction_summaries_use_role_order() -> None:
    prior = {
        "role_priors": {
            role: [
                {"junction_budget": 0, "probability": 0.75},
                {"junction_budget": 1, "probability": 0.25},
            ]
            for role in (
                "amine_head",
                "oxoester_aldehyde_body_tail",
                "isocyanide_tail",
            )
        }
    }
    assert set(prior_positive_junction_fractions(prior).values()) == {0.25}
    rows = [
        {"program": {"junction_budgets": [1, 0, 1]}},
        {"program": {"junction_budgets": [0, 0, 1]}},
    ]
    sampled = sampled_positive_junction_fractions(rows)
    assert sampled["amine_head"] == 0.5
    assert sampled["oxoester_aldehyde_body_tail"] == 0.0
    assert sampled["isocyanide_tail"] == 1.0
