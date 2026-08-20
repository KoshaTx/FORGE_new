from __future__ import annotations

from typing import Any

from forge.potency.morphology.ugi_morphology_potency_matched_adjudication import (
    EXPECTED_ARMS,
    POTENCY_ARM,
    SUPPORT_ARM,
    _exact_l1,
    _promotion_checks,
    _select_equal_oracle_budget,
)


def _terminal() -> dict[str, Any]:
    return {
        "valid": True,
        "terminal_valid": True,
        "component_reconstruction_valid": True,
        "l1_forward_verification": {
            "exact_product_reconstructed": True,
            "maximum_outcomes_saturated": False,
        },
    }


def test_exact_l1_requires_unsaturated_forward_verification() -> None:
    terminal = _terminal()
    assert _exact_l1(terminal) is True
    terminal["l1_forward_verification"]["maximum_outcomes_saturated"] = True
    assert _exact_l1(terminal) is False


def test_oracle_budget_is_equal_and_deterministic() -> None:
    rows = []
    for arm, count in zip(EXPECTED_ARMS, (3, 2, 4), strict=True):
        for draw in range(count):
            rows.append(
                {
                    "arm_id": arm,
                    "draw_index": draw,
                    "eligible": True,
                    "canonical_product": f"{arm}-{draw}",
                    "oracle_selected": False,
                }
            )
    budget, selected = _select_equal_oracle_budget(rows, EXPECTED_ARMS)
    assert budget == 2
    assert {arm: len(values) for arm, values in selected.items()} == {
        arm: 2 for arm in EXPECTED_ARMS
    }
    assert all(row["oracle_selected"] for values in selected.values() for row in values)


def test_promotion_fails_when_primary_difference_is_zero() -> None:
    arm = {
        "unique_conservative_high_potency_products": 0,
        "exact_l1_fraction": 1.0,
        "unique_exact_l1_products": 10,
        "internal_product_diversity": 0.5,
        "role_effective_component_counts": {
            "amine": 4.0,
            "aldehyde": 4.0,
            "isocyanide": 4.0,
        },
        "role_maximum_component_fractions": {
            "amine": 0.1,
            "aldehyde": 0.1,
            "isocyanide": 0.1,
        },
    }
    metrics = {SUPPORT_ARM: arm, POTENCY_ARM: arm}
    bootstrap = {
        "point_difference_per_generator_call": 0.0,
        "confidence_interval_low": 0.0,
    }
    thresholds = {
        "paired_bootstrap_interval_lower_bound_at_least": 0.0,
        "minimum_unique_high_potency_yield_ratio_vs_applicability": 1.0,
        "minimum_exact_l1_validity_fraction_vs_applicability": 0.99,
        "minimum_unique_product_count_fraction_vs_applicability": 0.8,
        "minimum_product_fingerprint_diversity_fraction_vs_applicability": 0.8,
        "minimum_effective_amine_count_fraction_vs_applicability": 0.8,
        "minimum_effective_aldehyde_count_fraction_vs_applicability": 0.8,
        "minimum_effective_isocyanide_count_fraction_vs_applicability": 0.8,
        "no_single_component_probability_above": 0.2,
    }
    checks = _promotion_checks(metrics, bootstrap, thresholds)
    assert checks["primary_endpoint_positive"] is False
    assert not all(checks.values())
