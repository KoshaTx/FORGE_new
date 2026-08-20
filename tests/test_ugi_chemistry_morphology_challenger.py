from __future__ import annotations

from experiments.archive.phase1.design_audits.ugi_chemistry_morphology_challenger import (
    challenger_checks,
)


def test_challenger_checks_apply_frozen_intervals() -> None:
    features = {
        "has_carbon_branch": 0.13,
        "has_carbon_carbon_double_bond": 0.25,
        "has_carbon_carbon_triple_bond": 0.12,
        "has_ester_like_carbonyl": 0.45,
    }
    arm = {
        "valid_fraction": 0.97,
        "exact_l1_fraction_of_valid": 1.0,
        "tail_chemotypes_exact_l1_eligible_only": {
            "oxoester_aldehyde_body_tail": {"feature_occurrence_fractions": features}
        },
    }
    motifs = {
        "motifs": {
            "alkyne": {"fraction": 0.16},
            "allene_or_carbon_cumulene": {"fraction": 0.0},
        }
    }
    gates = {
        "minimum_valid_fraction": 0.95,
        "exact_l1_fraction_of_valid": 1.0,
        "minimum_unique_fraction_of_valid": 0.98,
        "maximum_exact_frozen_product_reproduction_fraction": 0.15,
        "maximum_product_alkyne_fraction": 0.19,
        "maximum_product_allene_or_cumulene_fraction": 0.005,
        "minimum_aldehyde_double_bond_fraction": 0.2,
        "maximum_aldehyde_triple_bond_fraction": 0.18,
        "minimum_aldehyde_ester_fraction": 0.4,
        "aldehyde_branch_fraction_interval": [0.1, 0.18],
    }
    checks = challenger_checks(
        arm,
        motifs,
        unique_fraction=0.99,
        reproduction_fraction=0.1,
        gates=gates,
    )
    assert all(checks.values())
