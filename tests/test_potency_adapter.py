from __future__ import annotations

import numpy as np
import pytest

from experiments.phase1.hela_potency.potency_adapter_ordinal import (
    _load_fold_progress,
    _write_fold_progress,
)
from forge.model.defog_feasibility import _model_state_sha256
from forge.model.potency_adapter import (
    PotencyAdapterError,
    apply_potency_adapter_state,
    freeze_base_parameters,
    initialize_potency_adapter_from_base,
    potency_adapter_state_dict,
    potency_parameter_report,
    reconstruct_potency_conditioned_model,
)
from forge.model.potency_conditioning import PotencyAdapterPolicy, PotencyCondition
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.reaction_program_transformer import (
    ReactionProgramGraphTransformer,
    ReactionProgramTransformerError,
)
from forge.potency.adapter_failure_attribution import (
    factorial_dataset_summary,
    label_variance_summary,
)
from forge.potency.adapter_guidance import (
    clustered_auroc_difference_interval,
    clustered_mean_difference_interval,
    deterministic_permutation,
    map_to_training_ecdf,
    monotonicity_gate,
    ordinal_contrastive_loss,
    ordinal_monotonicity_rows,
    promotion_gate,
    quartile_balanced_sample,
    roc_auc,
    training_ecdf_quantiles,
)

torch = pytest.importorskip("torch")


def _model(*, potency: bool) -> ReactionProgramGraphTransformer:
    vocabulary = ReactionProgramVocabulary(
        program_states=("unconditioned", "ugi", "bl"),
        role_states=("unassigned", "head", "tail"),
        core_position_states=("unconditioned", "exterior", "ugi:map_1"),
        maximum_steps=2,
    )
    return ReactionProgramGraphTransformer(
        vocabulary=vocabulary,
        node_classes=5,
        hidden_dim=16,
        layers=2,
        heads=4,
        expert_count=2,
        adapter_dim=4,
        maximum_closures=1,
        maximum_heavy_atoms=8,
        dropout=0.0,
        bond_classes=4,
        potency_adapter_dim=4 if potency else 0,
        potency_condition_dim=8,
    )


def _inputs(*, program: int = 1, t: float = 0.5) -> dict[str, torch.Tensor]:
    node_mask = torch.ones((1, 4), dtype=torch.bool)
    child_mask = node_mask.clone()
    child_mask[:, 0] = False
    return {
        "nodes": torch.tensor([[1, 2, 3, 1]]),
        "parents": torch.tensor([[0, 0, 1, 2]]),
        "parent_bonds": torch.zeros((1, 4), dtype=torch.long),
        "closure_left": torch.zeros((1, 1), dtype=torch.long),
        "closure_right": torch.zeros((1, 1), dtype=torch.long),
        "closure_bonds": torch.zeros((1, 1), dtype=torch.long),
        "t": torch.tensor([t]),
        "node_mask": node_mask,
        "child_mask": child_mask,
        "closure_mask": torch.zeros((1, 1), dtype=torch.bool),
        "program_states": torch.tensor([program]),
        "role_states": torch.tensor([[1, 1, 2, 2]]),
        "core_position_states": torch.tensor([[2, 1, 1, 1]]),
        "program_depths": torch.tensor([1]),
        "adapter_mask": node_mask,
    }


def _policy() -> PotencyAdapterPolicy:
    return PotencyAdapterPolicy(
        policy_id="test-policy",
        endpoint_id="expt_Hela",
        program_id="ugi",
        minimum_quantile=0.0,
        maximum_quantile=1.0,
        active_time_intervals=((0.4, 0.8),),
    )


def _assert_outputs_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_null_condition_is_exact_before_and_after_adapter_updates() -> None:
    torch.manual_seed(17)
    base = _model(potency=False).eval()
    adapted = _model(potency=True).eval()
    initialize_potency_adapter_from_base(adapted, base.state_dict())
    adapted.configure_potency_policy(_policy())
    with torch.no_grad():
        reference = base(**_inputs())
        initial = adapted(
            **_inputs(),
            potency_condition=PotencyCondition("expt_Hela", 0.9, "test-policy"),
        )
        _assert_outputs_equal(reference, initial)
        adapted.blocks[0].potency_adapter.output.bias.fill_(0.5)
        null = adapted(**_inputs())
        inactive = adapted(
            **_inputs(t=0.2),
            potency_condition=PotencyCondition("expt_Hela", 0.9, "test-policy"),
        )
        inactive_reference = base(**_inputs(t=0.2))
        active = adapted(
            **_inputs(),
            potency_condition=PotencyCondition("expt_Hela", 0.9, "test-policy"),
        )
    _assert_outputs_equal(reference, null)
    _assert_outputs_equal(inactive_reference, inactive)
    assert any(not torch.equal(reference[key], active[key]) for key in reference)


def test_adapter_is_ugi_only_and_overlay_round_trips() -> None:
    torch.manual_seed(23)
    base = _model(potency=False)
    first = _model(potency=True)
    initialize_potency_adapter_from_base(first, base.state_dict())
    first.configure_potency_policy(_policy())
    trainable = freeze_base_parameters(first)
    report = potency_parameter_report(first)
    assert trainable
    assert report["adapter_fraction"] < 0.25
    assert all(
        parameter.requires_grad == (".potency_adapter." in name)
        for name, parameter in first.named_parameters()
    )
    with pytest.raises(ReactionProgramTransformerError, match="declared reaction program"):
        first(
            **_inputs(program=2),
            potency_condition=PotencyCondition("expt_Hela", 0.9, "test-policy"),
        )
    with torch.no_grad():
        first.blocks[0].potency_adapter.output.bias.fill_(0.25)
    delta = potency_adapter_state_dict(first)
    second = _model(potency=True)
    initialize_potency_adapter_from_base(second, base.state_dict())
    second.configure_potency_policy(_policy())
    apply_potency_adapter_state(second, delta)
    assert all(torch.equal(second.state_dict()[key], value) for key, value in delta.items())
    with pytest.raises(PotencyAdapterError, match="keys changed"):
        apply_potency_adapter_state(second, dict(list(delta.items())[1:]))

    model_config = {
        "architecture": "reaction_program_graph_transformer",
        "hidden_dim": 16,
        "layers": 2,
        "attention_heads": 4,
        "expert_count": 2,
        "adapter_dim": 4,
        "maximum_closures": 1,
        "maximum_heavy_atoms": 8,
        "dropout": 0.0,
        "bond_classes": 4,
    }
    rebuilt, policy = reconstruct_potency_conditioned_model(
        base_package={
            "model_config": model_config,
            "model_state": base.state_dict(),
            "model_state_sha256": _model_state_sha256(base),
        },
        overlay={
            "schema_version": "test.overlay.v1",
            "trusted_local_checkpoint": True,
            "base": {"model_state_sha256": _model_state_sha256(base)},
            "model_config": {
                **model_config,
                "potency_adapter_dim": 4,
                "potency_condition_dim": 8,
            },
            "potency_policy": _policy().to_mapping(),
            "adapter_state": delta,
            "combined_model_state_sha256": _model_state_sha256(second),
        },
        vocabulary=base.vocabulary,
        node_classes=5,
        device=torch.device("cpu"),
        checkpoint_schema="test.overlay.v1",
    )
    assert policy == _policy()
    assert _model_state_sha256(rebuilt) == _model_state_sha256(second)


def test_quantiles_sampling_controls_and_bootstrap_are_deterministic() -> None:
    quantiles = training_ecdf_quantiles([1.0, 1.0, 2.0, 3.0])
    assert quantiles.tolist() == pytest.approx([0.25, 0.25, 0.625, 0.875])
    assert map_to_training_ecdf([1.0, 2.0, 3.0], [0.0, 2.0, 4.0]).tolist() == [0.0, 2 / 3, 1.0]
    source = np.linspace(0.01, 0.99, 100)
    first = quartile_balanced_sample(source, size=16, rng=np.random.default_rng(11))
    second = quartile_balanced_sample(source, size=16, rng=np.random.default_rng(11))
    assert np.array_equal(first, second)
    assert np.array_equal(
        deterministic_permutation(20, seed=7, namespace="x"),
        deterministic_permutation(20, seed=7, namespace="x"),
    )
    labels = [0, 0, 1, 1, 0, 0, 1, 1]
    real = [0.0, 0.1, 0.9, 1.0, 0.2, 0.3, 0.7, 0.8]
    shuffled = [0.8, 0.2, 0.4, 0.1, 0.7, 0.3, 0.6, 0.5]
    clusters = ["a", "a", "b", "b", "c", "c", "d", "d"]
    assert roc_auc(labels, real) == 1.0
    interval = clustered_auroc_difference_interval(
        labels,
        real,
        shuffled,
        clusters,
        replicates=200,
        seed=31,
    )
    assert interval == clustered_auroc_difference_interval(
        labels,
        real,
        shuffled,
        clusters,
        replicates=200,
        seed=31,
    )


def test_promotion_gate_requires_gain_and_generator_noninferiority() -> None:
    metrics = {
        "null_unique_conservative_high_exact_l1_per_1000": 20.0,
        "guided_unique_conservative_high_exact_l1_per_1000": 30.0,
        "paired_program_bootstrap_lower": 1.0,
        "guided_valid_fraction": 0.91,
        "null_valid_fraction": 0.93,
        "guided_exact_l1_fraction": 0.80,
        "null_exact_l1_fraction": 0.82,
        "guided_distinct_fraction": 0.95,
        "guided_effective_count_fraction": 0.85,
        "fixed_state_failures": 0,
        "support_overflows": 0,
    }
    thresholds = {
        "minimum_relative_gain": 0.2,
        "minimum_absolute_gain_per_1000": 5.0,
        "maximum_validity_drop": 0.05,
        "maximum_exact_l1_drop": 0.05,
        "minimum_distinct_fraction": 0.9,
        "minimum_effective_count_fraction": 0.8,
    }
    assert promotion_gate(metrics, thresholds)["passes"]
    metrics["guided_exact_l1_fraction"] = 0.70
    assert not promotion_gate(metrics, thresholds)["passes"]


def test_failure_attribution_counts_independent_factorial_support() -> None:
    rows = [
        {"source_lipid_name": "A1_B1_C1", "model_smiles": "C", "label_value": "0.0"},
        {"source_lipid_name": "A1_B1_C2", "model_smiles": "CC", "label_value": "1.0"},
        {"source_lipid_name": "A2_B1_C1", "model_smiles": "CCC", "label_value": "2.0"},
        {"source_lipid_name": "A2_B1_C2", "model_smiles": "CCCC", "label_value": "3.0"},
    ]
    geometry = factorial_dataset_summary(rows)
    assert geometry["rows"] == 4
    assert geometry["unique_heads"] == 2
    assert geometry["unique_aldehyde_isocyanide_pairs"] == 2
    assert geometry["complete_cartesian_library"]
    variance = label_variance_summary(rows)
    assert variance["additive_head_plus_pair_r2"] == pytest.approx(1.0)
    assert variance["unreplicated_interaction_or_noise_fraction"] == pytest.approx(0.0)


def test_ordinal_contrast_and_monotonicity_use_counterfactual_requests() -> None:
    nll = torch.tensor(
        [
            [0.1, 0.4, 0.9],
            [0.8, 0.3, 0.1],
            [0.5, 0.1, 0.6],
        ],
        requires_grad=True,
    )
    targets = torch.tensor([0.05, 0.95, 0.5])
    primary, contrast = ordinal_contrastive_loss(
        nll,
        targets,
        anchors=(0.1, 0.5, 0.9),
        margin=0.1,
    )
    assert primary.item() == pytest.approx(0.1)
    assert contrast.item() == pytest.approx(0.0)
    (primary + contrast).backward()
    assert torch.isfinite(nll.grad).all()
    rows = ordinal_monotonicity_rows(
        nll.detach().numpy(),
        targets.numpy(),
        anchors=(0.1, 0.5, 0.9),
    )
    assert rows["anchor_correct"].tolist() == [True, True, True]
    assert rows["outer_monotonic"].tolist() == [True, True, False]


def test_monotonicity_gate_requires_paired_shuffled_separation() -> None:
    clusters = ["a", "a", "b", "b", "c", "c", "d", "d"]
    interval = clustered_mean_difference_interval(
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        clusters,
        replicates=200,
        seed=9,
    )
    assert interval["lower_95"] == pytest.approx(1.0)
    thresholds = {
        "minimum_outer_monotonic_fraction": 0.3,
        "minimum_outer_monotonic_gap": 0.05,
        "minimum_anchor_accuracy": 0.45,
        "minimum_anchor_accuracy_gap": 0.05,
    }
    assert monotonicity_gate(
        real_outer_fraction=0.6,
        shuffled_outer_fraction=0.2,
        outer_difference_lower=0.1,
        real_anchor_accuracy=0.7,
        shuffled_anchor_accuracy=0.4,
        anchor_difference_lower=0.1,
        thresholds=thresholds,
    )["passes"]
    assert not monotonicity_gate(
        real_outer_fraction=0.6,
        shuffled_outer_fraction=0.58,
        outer_difference_lower=-0.01,
        real_anchor_accuracy=0.7,
        shuffled_anchor_accuracy=0.69,
        anchor_difference_lower=-0.02,
        thresholds=thresholds,
    )["passes"]


def test_ordinal_fold_progress_is_restartable_and_fingerprint_bound(tmp_path) -> None:
    path = tmp_path / "held_head_5fold__fold_0.json"
    record = {"fold": 0, "scheme": "held_head_5fold", "test_rows": 2}
    rows = [{"label": "A1B1C1"}, {"label": "A2B1C1"}]
    _write_fold_progress(
        path,
        fold_signature="b" * 64,
        fold_record=record,
        evaluation_rows=rows,
    )
    restored_record, restored_rows = _load_fold_progress(
        path,
        expected_signature="b" * 64,
        expected_test_rows=2,
        expected_evaluation_rows=2,
    )
    assert restored_record == record
    assert restored_rows == rows
    with pytest.raises(ValueError, match="fold progress is invalid"):
        _load_fold_progress(
            path,
            expected_signature="c" * 64,
            expected_test_rows=2,
            expected_evaluation_rows=2,
        )
