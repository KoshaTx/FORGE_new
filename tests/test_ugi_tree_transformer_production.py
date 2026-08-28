from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest

from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_production import (
    UgiTreeTransformerProductionEvaluationError,
    summarize_component_family_strata,
)
from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_production_aggregate import (
    aggregate_metric_mappings,
)
from experiments.phase1.product_l1.training.ugi_joint_sparse_training import (
    _training_partition,
)
from experiments.phase1.product_l1.training.ugi_tree_transformer_production import (
    UgiTreeTransformerProductionTrainingError,
    build_production_training_config,
)

REPO = Path(__file__).resolve().parents[1]


def _json(relative: str) -> dict:
    return json.loads((REPO / relative).read_text())


def _contracts() -> tuple[dict, dict, dict, dict]:
    return (
        _json("configs/model/phase1_ugi_tree_relational_production_training_v1.json"),
        _json("configs/model/phase1_ugi_tree_transformer_challenger_seed0_v1.json"),
        _json("configs/model/phase1_ugi_tree_transformer_ablation_ladder_v1.json"),
        _json("results/phase1/ugi_tree_transformer_calibration_adjudication_v1/result.json"),
    )


def test_production_training_materializes_three_fresh_fixed_step_seeds() -> None:
    design, base, ablation, adjudication = _contracts()
    values = [
        build_production_training_config(
            design,
            base,
            ablation,
            adjudication,
            profile="full",
            replicate=replicate,
        )
        for replicate in range(3)
    ]
    assert [seed for seed, _ in values] == [20260905, 20260906, 20260907]
    for seed, effective in values:
        assert effective["seed"] == seed
        assert effective["full"]["steps"] == 2700
        assert effective["full"]["checkpoint_steps"] == [2700]
        assert effective["duration_contract"]["maximum_weighted_training_draws"] == 345600
        assert effective["training_partition"]["mode"] == "fixed_train_only"
        assert effective["training_partition"]["training_folds"] == ["train"]
        assert effective["training_partition"]["selection_mode"] == "fixed_final_step"
        assert effective["model"]["backbone"] == "ugi_tree_program_transformer"
        assert effective["model"]["tree_relation_attention"] is True
        assert effective["model"]["role_routed_program_attention"] is True
        assert effective["model"]["role_adapter_dim"] == 0
        assert effective["objective"]["role_block_mask_probability"] == 0.0
        assert (
            effective["promotion_contract"]["heldout_selects_architecture_or_checkpoint"]
            is False
        )
        validated_partition = _training_partition(
            effective,
            ("train", "calibration", "heldout"),
            effective["full"],
        )
        assert validated_partition.mode == "fixed_train_only"


def test_production_training_smoke_uses_cuda_without_changing_full_contract() -> None:
    design, base, ablation, adjudication = _contracts()
    seed, effective = build_production_training_config(
        design, base, ablation, adjudication, profile="smoke", replicate=0
    )
    assert seed == 20260905
    assert effective["smoke"]["device"] == "cuda"
    assert effective["smoke"]["steps"] == 2
    assert effective["smoke"]["checkpoint_steps"] == [2]
    assert effective["full"] == base["full"]


def test_production_training_rejects_rewritten_calibration_decision() -> None:
    design, base, ablation, adjudication = _contracts()
    changed = copy.deepcopy(adjudication)
    changed["selected_model"]["checkpoint_step"] = 600
    with pytest.raises(
        UgiTreeTransformerProductionTrainingError,
        match="does not authorize",
    ):
        build_production_training_config(
            design, base, ablation, changed, profile="full", replicate=0
        )


def _write_assessed(path: Path, rows: list[dict]) -> None:
    header = {"schema_version": "forge.common_ugi_assessed_attempts.v1", "rows": len(rows)}
    payload = "\n".join(json.dumps(value, sort_keys=True) for value in [header, *rows]) + "\n"
    with gzip.open(path, "wt") as handle:
        handle.write(payload)


def test_component_family_strata_preserve_attempt_denominators(tmp_path: Path) -> None:
    native = [
        {"source_stratum": "A", "held_role_class": "amine_head"},
        {"source_stratum": "A", "held_role_class": "isocyanide_tail"},
        {"source_stratum": "B", "held_role_class": "amine_head"},
    ]
    assessed = [
        {
            "attempt_index": 0,
            "valid": True,
            "exact_l1_program": True,
            "method_visible_open_ended_exact_l1": True,
            "held_component_exact_l1": False,
            "canonical_smiles": "CCN",
        },
        {
            "attempt_index": 1,
            "valid": False,
            "exact_l1_program": False,
            "method_visible_open_ended_exact_l1": False,
            "held_component_exact_l1": False,
            "canonical_smiles": None,
        },
        {
            "attempt_index": 2,
            "valid": True,
            "exact_l1_program": True,
            "method_visible_open_ended_exact_l1": False,
            "held_component_exact_l1": True,
            "canonical_smiles": "CCC",
        },
    ]
    path = tmp_path / "assessed.jsonl.gz"
    _write_assessed(path, assessed)
    summary = summarize_component_family_strata(native, path)
    assert summary["source:A"]["attempts"] == 2
    assert summary["source:A"]["valid_fraction_per_attempt"] == 0.5
    assert summary["source:A"]["open_ended_exact_l1_yield_per_attempt"] == 0.5
    assert summary["held_role:amine_head"]["attempts"] == 2
    assert (
        summary["held_role:amine_head"][
            "held_component_exact_l1_products_per_1000_attempts"
        ]
        == 500.0
    )


def test_component_family_strata_reject_order_drift(tmp_path: Path) -> None:
    path = tmp_path / "assessed.jsonl.gz"
    _write_assessed(
        path,
        [
            {
                "attempt_index": 2,
                "valid": False,
                "exact_l1_program": False,
                "method_visible_open_ended_exact_l1": False,
                "held_component_exact_l1": False,
                "canonical_smiles": None,
            }
        ],
    )
    with pytest.raises(
        UgiTreeTransformerProductionEvaluationError,
        match="order changed",
    ):
        summarize_component_family_strata(
            [{"source_stratum": "A", "held_role_class": "amine_head"}], path
        )


def test_metric_aggregate_uses_training_seed_as_replication_unit() -> None:
    result = aggregate_metric_mappings(
        [{"x": 0.8}, {"x": 0.9}, {"x": 1.0}], ["x"]
    )
    assert result["x"]["mean"] == pytest.approx(0.9)
    assert result["x"]["sample_standard_deviation"] == pytest.approx(0.1)
    assert result["x"]["seed_values"] == [0.8, 0.9, 1.0]
