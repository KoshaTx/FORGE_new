from __future__ import annotations

import hashlib
import json
import math
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_calibration import (
    select_calibration_programs,
)
from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_checkpoint_calibration import (
    UgiTreeTransformerCheckpointCalibrationError,
    extract_authenticated_checkpoint_archive,
)
from experiments.phase1.product_l1.sampling.ugi_joint_sparse_sampling import (
    sample_restartable_terminals,
)
from experiments.phase1.product_l1.stages import _tree_ablation_effective_config
from experiments.phase1.product_l1.training.ugi_joint_sparse_training import (
    UgiJointSparseTrainingError,
    _learning_rate_at_step,
    _training_objective,
)
from forge.model.ugi_joint_sparse_flow import (
    TREE_CHILD,
    TREE_DIFFERENT_ROLE,
    TREE_PARENT,
    TREE_SELF,
    TREE_SIBLING,
    UgiJointSparseFlow,
    UgiJointSparseFlowError,
    apply_component_role_mask,
    noisy_preorder_relation_states,
    ugi_joint_sparse_loss,
)
from forge.model.ugi_morphology_program import UgiMorphologyProgram

torch = pytest.importorskip("torch")


def _batch() -> dict[str, object]:
    node_mask = torch.ones((2, 6), dtype=torch.bool)
    closure_mask = torch.zeros((2, 1), dtype=torch.bool)
    decoration_anchors = torch.tensor([[1, 0], [2, 5]], dtype=torch.long)
    return {
        "offspring": torch.tensor([[1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0]]),
        "nodes": torch.tensor([[0, 1, 2, 0, 1, 2], [1, 2, 0, 1, 2, 0]]),
        "parent_bonds": torch.ones((2, 6), dtype=torch.long),
        "role_states": torch.tensor([[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]]),
        "within_role_positions": torch.tensor([[0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1]]),
        "programs": torch.tensor(
            [
                [2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            ]
        ),
        "node_mask": node_mask,
        "closure_mask": closure_mask,
        "closure_left": torch.zeros((2, 1), dtype=torch.long),
        "closure_right": torch.zeros((2, 1), dtype=torch.long),
        "closure_bonds": torch.zeros((2, 1), dtype=torch.long),
        "decoration_anchors": decoration_anchors,
        "decoration_atoms": torch.tensor([[1, 0], [2, 3]], dtype=torch.long),
        "decoration_bonds": torch.tensor([[1, 0], [1, 1]], dtype=torch.long),
        "decoration_present_mask": decoration_anchors > 0,
    }


def _model(**overrides: object) -> UgiJointSparseFlow:
    config = {
        "maximum_children": 3,
        "atom_classes": 4,
        "bond_classes": 4,
        "maximum_component_atoms": 8,
        "maximum_total_atoms": 24,
        "maximum_junction_budget": 3,
        "maximum_cycle_rank": 2,
        "maximum_attachment_count": 2,
        "maximum_decorations": 2,
        "hidden_dim": 32,
        "layers": 2,
        "dropout": 0.0,
        "decoration_state_conditioning": "bidirectional_anchor_local",
        "backbone": "ugi_tree_program_transformer",
        "attention_heads": 4,
        "role_adapter_dim": 8,
    }
    config.update(overrides)
    return UgiJointSparseFlow(**config)


def _predict(model: UgiJointSparseFlow, batch: dict[str, object]) -> dict[str, object]:
    return model(
        offspring=batch["offspring"],
        nodes=batch["nodes"],
        parent_bonds=batch["parent_bonds"],
        role_states=batch["role_states"],
        within_role_positions=batch["within_role_positions"],
        programs=batch["programs"],
        node_mask=batch["node_mask"],
        t=torch.full((2,), 0.5),
        closure_left=batch["closure_left"],
        closure_right=batch["closure_right"],
        decoration_anchors=batch["decoration_anchors"],
        decoration_atoms=batch["decoration_atoms"],
        decoration_bonds=batch["decoration_bonds"],
    )


def test_noisy_preorder_relations_are_exact_for_valid_role_forests() -> None:
    offspring = torch.tensor([[2, 0, 0, 1, 0, 0]])
    roles = torch.tensor([[0, 0, 0, 1, 1, 2]])
    mask = torch.ones_like(offspring, dtype=torch.bool)

    relations = noisy_preorder_relation_states(offspring, roles, mask)[0]

    assert int(relations[0, 0]) == TREE_SELF
    assert int(relations[1, 0]) == TREE_CHILD
    assert int(relations[0, 1]) == TREE_PARENT
    assert int(relations[1, 2]) == TREE_SIBLING
    assert int(relations[4, 3]) == TREE_CHILD
    assert int(relations[0, 3]) == TREE_DIFFERENT_ROLE


def test_noisy_preorder_relations_fail_soft_without_clean_tree_access() -> None:
    offspring = torch.tensor([[3, 3, 3, 3, 3, 3]])
    roles = torch.tensor([[0, 0, 1, 1, 2, 2]])
    mask = torch.tensor([[True, True, True, True, True, False]])

    first = noisy_preorder_relation_states(offspring, roles, mask)
    second = noisy_preorder_relation_states(offspring, roles, mask)

    assert torch.equal(first, second)
    assert first.shape == (1, 6, 6)
    assert int(first.min()) >= 0
    assert int(first.max()) <= TREE_DIFFERENT_ROLE


def test_tree_transformer_routes_programs_and_trains_structural_parameters() -> None:
    batch = _batch()
    model = _model()
    predictions = _predict(model, batch)
    loss, metrics = ugi_joint_sparse_loss(
        predictions,
        batch,
        consistency_weights={
            "attachment_count_consistency": 0.5,
            "junction_budget_consistency": 0.5,
        },
    )
    loss.backward()

    block = model.transformer_blocks[0]
    routed_bias = block._program_attention_bias(
        batch["role_states"], model._program_tokens(batch["programs"], torch.full((2,), 0.5))
    )
    assert routed_bias.shape == (2, 1, 6, 4)
    assert torch.isfinite(routed_bias[:, :, :, 0]).all()
    assert torch.isneginf(routed_bias[0, 0, 0, 2:]).all()
    assert block.relation_bias.weight.grad is not None
    assert block.relative_position_bias.weight.grad is not None
    assert block.role_adapters[0][0].weight.grad is not None
    assert model.core_port_embedding.weight.grad is not None
    assert metrics["attachment_count_consistency"] >= 0
    assert metrics["junction_budget_consistency"] >= 0


def test_program_consistency_loss_rewards_exact_offspring_program() -> None:
    batch = _batch()
    model = _model(layers=1)
    predictions = _predict(model, batch)
    random_metrics = ugi_joint_sparse_loss(predictions, batch)[1]
    exact_logits = torch.full_like(predictions["offspring"], -20.0)
    exact_logits.scatter_(2, batch["offspring"][:, :, None], 20.0)
    exact_predictions = {**predictions, "offspring": exact_logits}
    exact_metrics = ugi_joint_sparse_loss(exact_predictions, batch)[1]

    assert exact_metrics["attachment_count_consistency"] < 1e-8
    assert exact_metrics["junction_budget_consistency"] < 1e-8
    assert (
        exact_metrics["attachment_count_consistency"]
        < random_metrics["attachment_count_consistency"]
    )


def test_component_role_mask_is_deterministic_and_uses_only_source_states() -> None:
    batch = _batch()
    noisy = {
        key: batch[key].clone()
        for key in (
            "offspring",
            "nodes",
            "parent_bonds",
            "decoration_anchors",
            "decoration_atoms",
            "decoration_bonds",
        )
    }
    sources = {
        "offspring": torch.nn.functional.one_hot(torch.tensor([3, 3, 3]), 4).float(),
        "atoms": torch.nn.functional.one_hot(torch.tensor([3, 3, 3]), 4).float(),
        "bonds": torch.nn.functional.one_hot(torch.tensor([2, 2, 2]), 4).float(),
        "decoration": torch.tensor([1.0, 0.0]),
        "decoration_atoms": torch.tensor([0.0, 0.0, 0.0, 1.0]),
        "decoration_bonds": torch.tensor([0.0, 0.0, 1.0, 0.0]),
    }

    first, first_roles = apply_component_role_mask(
        noisy,
        batch,
        sources,
        probability=1.0,
        generator=torch.Generator().manual_seed(73),
    )
    second, second_roles = apply_component_role_mask(
        noisy,
        batch,
        sources,
        probability=1.0,
        generator=torch.Generator().manual_seed(73),
    )

    assert torch.equal(first_roles, second_roles)
    assert all(torch.equal(first[key], second[key]) for key in first)
    for index, role in enumerate(first_roles.tolist()):
        selected = batch["role_states"][index] == role
        assert torch.all(first["offspring"][index, selected] == 3)
        assert torch.all(first["nodes"][index, selected] == 3)
        assert torch.all(first["parent_bonds"][index, selected] == 2)
    assert torch.all(first["decoration_anchors"] == 0)
    assert torch.all(first["decoration_atoms"] == 3)
    assert torch.all(first["decoration_bonds"] == 2)


def test_warmup_cosine_schedule_and_objective_validation() -> None:
    runtime = {
        "steps": 100,
        "learning_rate": 1e-3,
        "learning_rate_schedule": {
            "mode": "warmup_cosine",
            "warmup_steps": 10,
            "minimum_learning_rate_ratio": 0.1,
        },
    }
    assert _learning_rate_at_step(runtime, 1) == pytest.approx(1e-4)
    assert _learning_rate_at_step(runtime, 10) == pytest.approx(1e-3)
    assert _learning_rate_at_step(runtime, 100) == pytest.approx(1e-4)
    assert _learning_rate_at_step(runtime, 50) < 1e-3
    assert math.isfinite(_learning_rate_at_step(runtime, 50))

    objective = _training_objective(
        {
            "objective": {
                "loss_weights": {"closure_bond_ce": 0.25},
                "program_consistency_weights": {"junction_budget_consistency": 0.5},
                "role_block_mask_probability": 0.2,
            }
        }
    )
    assert objective["role_block_mask_probability"] == 0.2
    with pytest.raises(UgiJointSparseTrainingError, match="mask probability"):
        _training_objective({"objective": {"role_block_mask_probability": 1.1}})


def test_tree_transformer_rejects_invalid_role_adapter_support() -> None:
    with pytest.raises(UgiJointSparseFlowError, match="invalid joint sparse architecture"):
        _model(role_adapter_dim=-1)


def test_tree_transformer_uses_the_unchanged_restartable_sampler() -> None:
    model = _model(layers=1)
    sources = {
        "offspring": np.full((3, 4), 0.25),
        "atoms": np.full((3, 4), 0.25),
        "bonds": np.full((3, 4), 0.25),
        "closure_bonds": np.full(4, 0.25),
        "decoration": np.full(2, 0.5),
        "decoration_atoms": np.full(4, 0.25),
        "decoration_bonds": np.full(4, 0.25),
    }
    program = UgiMorphologyProgram(
        node_counts=(2, 2, 2),
        junction_budgets=(0, 0, 0),
        cycle_ranks=(0, 0, 0),
        attachment_counts=(1, 1, 1),
    )

    terminals, metadata = sample_restartable_terminals(
        model,
        (program,),
        sources,
        sample_steps=2,
        batch_size=1,
        seed=20260830,
        device="cpu",
        maximum_adjacent_branch_runs=(2, 1, 1),
    )

    assert len(terminals) == 1
    assert terminals[0].program == program
    assert metadata["terminal_tree_repairs"] == 0


def test_tree_transformer_development_contract_is_pinned_and_nonselecting_on_holdout() -> None:
    repo = Path(__file__).resolve().parents[1]
    config_path = repo / "configs/model/phase1_ugi_tree_transformer_challenger_seed0_v1.json"
    design_path = repo / "configs/model/phase1_ugi_tree_transformer_ablation_ladder_v1.json"
    descriptor_path = (
        repo / "experiments/phase1/product_l1/ugi_tree_transformer_challenger_smoke_v1.json"
    )
    config = json.loads(config_path.read_text())
    design = json.loads(design_path.read_text())
    descriptor = json.loads(descriptor_path.read_text())
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()

    assert config["training_partition"]["training_folds"] == ["train"]
    assert config["training_partition"]["diagnostic_folds"] == ["calibration"]
    assert config["duration_contract"]["maximum_weighted_training_draws"] == 384000
    assert config["promotion_contract"]["heldout_selects_architecture_or_checkpoint"] is False
    assert (
        "held_component_exact_l1_per_1000"
        not in config["promotion_contract"]["required_external_calibration_metrics"]
    )
    assert config["promotion_contract"]["required_nonselecting_heldout_metrics"] == [
        "held_component_exact_l1_per_1000"
    ]
    assert (
        config["promotion_contract"]["development_training_authorized_after_implementation_review"]
        is True
    )
    assert config["promotion_contract"]["production_training_authorized_by_this_config"] is False
    assert design["base_config"]["sha256"] == config_hash
    assert set(design["arms"]) == {
        "dense_corrected_schedule",
        "tree_relations_and_routing",
        "tree_plus_balanced_consistency",
        "full_component_masking",
    }
    assert design["matched_contract"]["heldout_selects_nothing"] is True
    assert design["molecular_calibration"]["nonselecting_heldout_metrics"] == [
        "held_component_exact_l1_per_1000"
    ]
    calibration = design["molecular_calibration"]
    assert calibration["programs_per_checkpoint"] == 512
    assert calibration["inputs"]["program_draw"]["sha256"] == (
        "ad51b852116e03ca6ab9ed3e44fefd899795f435337a1f615564ec697aa01054"
    )
    assert calibration["sampling"]["flow_seed"] + 1 == calibration["sampling"]["closure_seed"]
    assert calibration["sampling"]["candidate_selection"] is False
    joint = next(stage for stage in descriptor["stages"] if stage["id"] == "joint")
    assert joint["config"]["sha256"] == config_hash


def test_h100_descriptor_materializes_one_matched_arm_per_replicate() -> None:
    from experiments._runtime.spec import ExperimentSpec

    repo = Path(__file__).resolve().parents[1]
    design_path = repo / "configs/model/phase1_ugi_tree_transformer_ablation_ladder_v1.json"
    descriptor_path = (
        repo / "experiments/phase1/product_l1/ugi_tree_transformer_development_h100_v1.json"
    )
    design = json.loads(design_path.read_text())
    descriptor = json.loads(descriptor_path.read_text())
    spec = ExperimentSpec.load(descriptor_path)
    joint = next(stage for stage in spec.stages if stage.stage_id == "joint")

    assert descriptor["replicates"] == {"smoke": 1, "full": 4}
    assert joint.resources.device == "cuda"
    assert joint.resources.gpu_type == "H100!"
    assert sum(stage.resources.timeout_seconds for stage in spec.stages) == 86400
    assert joint.config.sha256 == hashlib.sha256(design_path.read_bytes()).hexdigest()

    def materialize(profile: str, replicate: int) -> tuple[str, dict[str, object]]:
        context = SimpleNamespace(
            profile=profile,
            replicate=replicate,
            stage=joint,
            input=lambda label: joint.inputs[label].resolve(repo),
        )
        return _tree_ablation_effective_config(context, design)

    smoke_arm, smoke = materialize("smoke", 0)
    assert smoke_arm == "full_component_masking"
    assert smoke["smoke"]["device"] == "cuda"

    observed = []
    for replicate in range(4):
        arm_id, effective = materialize("full", replicate)
        observed.append(arm_id)
        assert effective["full"]["steps"] == 3000
        assert effective["full"]["batch_size"] == 128
        assert effective["training_partition"]["training_folds"] == ["train"]
        assert (
            effective["promotion_contract"]["heldout_selects_architecture_or_checkpoint"] is False
        )
        assert effective["experiment_arm"]["replicate"] == replicate
    assert observed == design["profile_arm_order"]["full"]
    assert materialize("full", 0)[1]["model"]["backbone"] == "ugi_program_transformer"
    assert materialize("full", 3)[1]["objective"]["role_block_mask_probability"] == 0.2


def test_calibration_program_draw_is_paired_balanced_and_order_invariant() -> None:
    role_classes = (
        ("source_a", ("amine_head",)),
        ("source_a", ("isocyanide_tail",)),
        ("source_b", ("oxoester_aldehyde_body_tail",)),
        ("source_b", ("amine_head", "isocyanide_tail")),
    )
    assignments = []
    records = []
    for group, (source, calibration_roles) in enumerate(role_classes):
        for index in range(6):
            product_id = f"P-{group}-{index}"
            assignment = {
                "product_id": product_id,
                "primary_product_fold": "calibration",
                "source_stratum": source,
                "component_novelty_class": "calibration_component",
            }
            for role in ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail"):
                assignment[f"{role}_family_fold"] = (
                    "calibration" if role in calibration_roles else "train"
                )
            assignments.append(assignment)
            records.append(
                SimpleNamespace(
                    product_id=product_id,
                    program=UgiMorphologyProgram(
                        node_counts=(3 + group, 4, 5),
                        junction_budgets=(0, 0, 0),
                        cycle_ranks=(0, 0, 0),
                        attachment_counts=(1, 1, 1),
                    ),
                )
            )

    samples, selection = select_calibration_programs(
        assignments,
        records,
        count=8,
        seed=20260831,
        source_mass={"source_a": 0.5, "source_b": 0.5},
    )
    repeated, repeated_selection = select_calibration_programs(
        list(reversed(assignments)),
        list(reversed(records)),
        count=8,
        seed=20260831,
        source_mass={"source_a": 0.5, "source_b": 0.5},
    )

    assert samples == repeated
    assert selection == repeated_selection
    assert selection["source_quotas"] == {"source_a": 4, "source_b": 4}
    assert set(selection["stratum_quotas"].values()) == {2}
    assert len({row["product_id"] for row in samples}) == 8
    assert {row["evaluation_fold"] for row in samples} == {"calibration"}
    assert all("component_id" not in row for row in samples)


def test_calibration_program_draw_descriptor_is_hash_pinned_and_nonheldout() -> None:
    from experiments._runtime.spec import ExperimentSpec

    repo = Path(__file__).resolve().parents[1]
    config_path = (
        repo / "configs/model/phase1_ugi_tree_transformer_calibration_program_draw_v1.json"
    )
    descriptor_path = (
        repo / "experiments/phase1/product_l1/ugi_tree_transformer_calibration_program_draw_v1.json"
    )
    config = json.loads(config_path.read_text())
    descriptor = json.loads(descriptor_path.read_text())
    spec = ExperimentSpec.load(descriptor_path)

    assert config["fold"] == "calibration"
    assert config["count"] == 512
    assert config["selection_policy"]["heldout_rows_allowed"] is False
    assert descriptor["metadata"]["heldout_rows_used"] is False
    assert spec.stages[0].config.sha256 == hashlib.sha256(config_path.read_bytes()).hexdigest()


def test_checkpoint_archive_extraction_authenticates_every_declared_member(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint_step_300.pt"
    source.write_bytes(b"checkpoint-300")
    archive_path = tmp_path / "checkpoints.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(source, arcname=source.name)
    expected = [
        {
            "member": source.name,
            "step": 300,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    ]

    extracted = extract_authenticated_checkpoint_archive(
        archive_path, tmp_path / "extracted", expected
    )

    assert extracted[300].read_bytes() == b"checkpoint-300"


def test_checkpoint_archive_rejects_undeclared_members(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint_step_300.pt"
    extra = tmp_path / "unexpected.pt"
    source.write_bytes(b"checkpoint-300")
    extra.write_bytes(b"unexpected")
    archive_path = tmp_path / "checkpoints.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(source, arcname=source.name)
        archive.add(extra, arcname=extra.name)
    expected = [
        {
            "member": source.name,
            "step": 300,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    ]

    with pytest.raises(
        UgiTreeTransformerCheckpointCalibrationError,
        match="undeclared member",
    ):
        extract_authenticated_checkpoint_archive(archive_path, tmp_path / "extracted", expected)
