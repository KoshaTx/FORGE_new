from __future__ import annotations

import json
from pathlib import Path

from experiments.phase1.product_l1.evaluation.ugi_v0_transformer_assessment import (
    assess_role_morphology,
    native_samples_to_attempts,
    selected_ugi_metrics,
)
from forge.core.io import write_jsonl
from forge.model.local_chemistry_support import LocalChemistrySupport


def test_native_conversion_retains_failed_attempts() -> None:
    attempts = native_samples_to_attempts(
        (
            {"valid": True, "smiles": "CCN"},
            {"valid": False, "smiles": None},
        ),
        method_id="forge_v0_production",
        seed_label=0,
    )

    assert len(attempts) == 2
    assert attempts[0].status == "generated"
    assert attempts[0].product_smiles == "CCN"
    assert attempts[1].status == "invalid"
    assert attempts[1].product_smiles is None
    assert sum(attempt.generator_calls for attempt in attempts) == 2


def test_role_morphology_uses_exact_l1_components(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    support = LocalChemistrySupport.from_mapping(
        json.loads((repo / "results/phase1/local_morphology_support_v2/policy.json").read_text())
    )
    assessed = tmp_path / "assessed.jsonl.gz"
    row = {
        "schema_version": "forge.common_ugi_assessed_attempt.v1",
        "method_id": "forge_ugi_transformer",
        "seed": 0,
        "attempt_index": 0,
        "valid": True,
        "canonical_smiles": "CCCCCCCCNC(=O)C(CCCCCCCC)NCCN(C)C",
        "exact_l1_program": True,
        "exact_l1_traces": [
            {
                "components_by_role": {
                    "amine_head": "NCCN(C)C",
                    "oxoester_aldehyde_body_tail": "CCCCCCCC(=O)OCCCC=O",
                    "isocyanide_tail": "[C-]#[N+]CCCCCCCC",
                }
            }
        ],
    }
    write_jsonl(
        assessed,
        [
            {"schema_version": "forge.common_ugi_assessed_attempts.v1", "rows": 1},
            row,
        ],
    )

    rows, result = assess_role_morphology(
        assessed,
        method_id="forge_ugi_transformer",
        seed_label=0,
        support=support,
    )

    assert len(rows) == 1
    assert result["counts"]["exact_l1_products"] == 1
    assert result["counts"]["exact_l1_products_with_any_supported_tail_trace"] == 1
    assert result["precision_among_exact_l1"]["any_supported_tail_trace"] == 1.0
    assert result["training_fold_only_policy"] is True
    aldehyde = rows[0]["trace_assessments"][0]["components_by_role"]["oxoester_aldehyde_body_tail"]
    assert aldehyde["heteroatoms"] == 3
    assert aldehyde["support_heteroatoms"] == 2
    assert aldehyde["support_count_convention"] == (
        "product_origin_excludes_inverse_transform_restored_aldehyde_oxygen"
    )


def test_selected_metrics_expose_the_frozen_open_ended_novel_primary_per_attempt() -> None:
    common = {
        "common_assessment": {
            "metrics": {
                "valid_fraction": 0.8,
                "exact_l1_yield_per_attempt": 0.5,
                "unique_exact_l1_products_per_attempt": 0.4,
                "unique_open_ended_exact_l1_products_per_attempt": 0.3,
                "unique_whole_product_novel_exact_l1_products_per_1000_attempts": 250.0,
                "unique_open_ended_whole_product_novel_exact_l1_products_per_1000_attempts": 125.0,
                "whole_product_novel_to_train_fraction": 0.6,
                "component_novelty_fraction": 0.4,
                "effective_component_count": 17.0,
                "held_component_exact_l1_products_per_1000_attempts": 75.0,
                "mean_pairwise_ecfp4_distance": 0.7,
            }
        }
    }
    realism = {
        "assessment": {
            "empirical_lipid_manifold": {
                "fingerprint": {"precision_per_requested_attempt": 0.4, "coverage": 0.3},
                "descriptor": {"precision_per_requested_attempt": 0.5, "coverage": 0.2},
                "classifier_two_sample": {"status": "estimated", "auc_mean": 0.65},
            },
            "molecular_output": {"mean_pairwise_ecfp4_distance_among_unique": 0.72},
        }
    }
    local = {"assessment": {"local_support_qualified_exact_l1_yield_per_attempt": 0.45}}
    morphology = {
        "fractions_per_attempt": {
            "exact_l1_products_with_any_fully_supported_trace": 0.42,
            "exact_l1_products_with_any_supported_tail_trace": 0.41,
        }
    }

    metrics = selected_ugi_metrics(common, realism, local, morphology)

    assert metrics["unique_whole_product_novel_exact_l1_products_per_attempt"] == 0.25
    assert metrics["unique_open_ended_whole_product_novel_exact_l1_products_per_attempt"] == 0.125


def test_selected_metrics_preserve_unavailable_small_sample_diagnostics() -> None:
    common = {
        "common_assessment": {
            "metrics": {
                "valid_fraction": 0.0,
                "exact_l1_yield_per_attempt": 0.0,
                "unique_exact_l1_products_per_attempt": 0.0,
                "unique_open_ended_exact_l1_products_per_attempt": 0.0,
                "unique_whole_product_novel_exact_l1_products_per_1000_attempts": 0.0,
                "unique_open_ended_whole_product_novel_exact_l1_products_per_1000_attempts": 0.0,
                "whole_product_novel_to_train_fraction": None,
                "component_novelty_fraction": None,
                "effective_component_count": None,
                "held_component_exact_l1_products_per_1000_attempts": 0.0,
                "mean_pairwise_ecfp4_distance": None,
            }
        }
    }
    realism = {
        "assessment": {
            "empirical_lipid_manifold": {
                "fingerprint": {"precision_per_requested_attempt": 0.0, "coverage": 0.0},
                "descriptor": {"precision_per_requested_attempt": 0.0, "coverage": 0.0},
                "classifier_two_sample": {"status": "not_estimable"},
            },
            "molecular_output": {"mean_pairwise_ecfp4_distance_among_unique": None},
        }
    }
    local = {"assessment": {"local_support_qualified_exact_l1_yield_per_attempt": 0.0}}
    morphology = {
        "fractions_per_attempt": {
            "exact_l1_products_with_any_fully_supported_trace": 0.0,
            "exact_l1_products_with_any_supported_tail_trace": 0.0,
        }
    }

    metrics = selected_ugi_metrics(common, realism, local, morphology)

    assert metrics["whole_product_novel_to_train_fraction"] is None
    assert metrics["component_novelty_fraction"] is None
    assert metrics["effective_component_count"] is None
    assert metrics["mean_pairwise_ecfp4_distance"] is None
    assert metrics["realism_c2st_auc"] is None
    assert metrics["realism_internal_diversity"] is None
