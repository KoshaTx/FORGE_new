from __future__ import annotations

import math

import pytest

from experiments.phase1.product_l1.evaluation.tail_chemotype import (
    architecture_signature,
    chemotype_signature,
    compare_component_cohorts,
    component_chemotype_metrics,
    summarize_component_cohort,
)


def test_component_chemotype_metrics_distinguish_supported_tail_modes() -> None:
    linear = component_chemotype_metrics("CCCCCCCC=O")
    ester = component_chemotype_metrics("CCCCCC(=O)OCCCC=O")
    unsaturated = component_chemotype_metrics("CCCC/C=C/CCCC=O")
    branched = component_chemotype_metrics("CC(C)CCCC=O")

    assert linear["carbonyl_count"] == 1
    assert linear["ester_like_carbonyl_count"] == 0
    assert ester["carbonyl_count"] == 2
    assert ester["ester_like_carbonyl_count"] == 1
    assert unsaturated["carbon_carbon_double_bonds"] == 1
    assert branched["carbon_branch_atoms"] == 1
    assert chemotype_signature(linear) != chemotype_signature(ester)
    assert architecture_signature(linear) != architecture_signature(ester)


def test_component_summary_separates_exact_and_chemotype_diversity() -> None:
    summary = summarize_component_cohort(["CCCC=O", "CCCC=O", "CCCCC=O", "CCCCC=O", "CCCCC=O"])

    assert summary["component_occurrences"] == 5
    assert summary["unique_exact_components"] == 2
    assert 1.0 < summary["effective_exact_component_count"] < 2.0
    assert summary["unique_chemotype_signatures"] == 2


def test_component_comparison_reports_coverage_recall_and_divergence() -> None:
    generated = summarize_component_cohort(["CCCC=O", "CCCC=O", "CCCCC=O"])
    reference = summarize_component_cohort(["CCCC=O", "CCCCCC=O"])
    comparison = compare_component_cohorts(generated, reference)

    assert comparison["generated_occurrence_exact_identity_coverage_fraction"] == pytest.approx(
        2 / 3
    )
    assert comparison["generated_occurrence_chemotype_coverage_fraction"] == pytest.approx(2 / 3)
    assert comparison["reference_unique_chemotype_recall_fraction"] == pytest.approx(0.5)
    assert math.isfinite(comparison["unique_catalog_signature_jensen_shannon_divergence_bits"])
