from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.phase1.product_l1.sampling.ugi_joint_end_to_end_sampling import (
    UgiJointEndToEndSamplingError,
    _resolve_prepared_cache_path,
)
from experiments.phase1.product_l1.sampling.ugi_joint_sparse_sampling import (
    sample_restartable_terminals,
)
from experiments.phase1.product_l1.training.ugi_joint_sparse_training import (
    UgiJointSparseTrainingError,
    _training_partition,
)
from forge.model.ugi_joint_sparse_flow import (
    UgiJointSparseFlow,
    UgiJointSparseFlowError,
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
        "backbone": "ugi_program_transformer",
        "attention_heads": 4,
    }
    config.update(overrides)
    return UgiJointSparseFlow(**config)


def test_ugi_program_transformer_preserves_v0_role_and_decoration_contract() -> None:
    batch = _batch()
    model = _model()
    predictions = model(
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
    loss, metrics = ugi_joint_sparse_loss(predictions, batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["offspring_ce"] > 0
    assert model.sequence is None
    assert model.transformer_blocks is not None
    assert len(model.transformer_blocks) == 2
    assert model.transformer_blocks[0].program_attention.in_proj_weight.grad is not None
    assert model.decoration_atom_embedding.weight.grad is not None


def test_program_tokens_change_node_predictions_at_every_transformer_layer() -> None:
    batch = _batch()
    model = _model()
    arguments = {
        "offspring": batch["offspring"],
        "nodes": batch["nodes"],
        "parent_bonds": batch["parent_bonds"],
        "role_states": batch["role_states"],
        "within_role_positions": batch["within_role_positions"],
        "node_mask": batch["node_mask"],
        "t": torch.full((2,), 0.5),
        "decoration_anchors": batch["decoration_anchors"],
        "decoration_atoms": batch["decoration_atoms"],
        "decoration_bonds": batch["decoration_bonds"],
    }
    baseline = model(**arguments, programs=batch["programs"])["nodes"]
    changed_programs = batch["programs"].clone()
    changed_programs[:, 0] += 1
    changed = model(**arguments, programs=changed_programs)["nodes"]

    assert not torch.equal(baseline, changed)
    assert all(block.program_attention is not None for block in model.transformer_blocks)


def test_ugi_program_transformer_rejects_flat_or_size_only_semantics() -> None:
    with pytest.raises(UgiJointSparseFlowError, match="role-structured full morphology"):
        _model(semantic_organization="flat_true_role")
    with pytest.raises(UgiJointSparseFlowError, match="role-structured full morphology"):
        _model(conditioning_mode="size_only")


def test_fixed_train_only_partition_is_disjoint_and_nonselecting() -> None:
    config = {
        "training_partition": {
            "mode": "fixed_train_only",
            "training_folds": ["train"],
            "diagnostic_folds": ["calibration"],
            "selection_mode": "fixed_final_step",
        }
    }
    partition = _training_partition(
        config,
        ("train", "calibration", "heldout"),
        {"early_stopping": {"patience": 0}},
    )

    assert partition.training_folds == ("train",)
    assert partition.diagnostic_folds == ("calibration",)
    assert partition.overlapping_folds == ()
    assert partition.selection_mode == "fixed_final_step"


def test_fixed_train_only_partition_rejects_holdout_or_early_stopping() -> None:
    config = {
        "training_partition": {
            "mode": "fixed_train_only",
            "training_folds": ["train", "heldout"],
            "diagnostic_folds": ["calibration"],
            "selection_mode": "fixed_final_step",
        }
    }
    with pytest.raises(UgiJointSparseTrainingError, match="exactly the frozen train fold"):
        _training_partition(
            config,
            ("train", "calibration", "heldout"),
            {"early_stopping": {"patience": 0}},
        )

    config["training_partition"]["training_folds"] = ["train"]
    with pytest.raises(UgiJointSparseTrainingError, match="data-dependent early stopping"):
        _training_partition(
            config,
            ("train", "calibration", "heldout"),
            {"early_stopping": {"patience": 1}},
        )


def test_ugi_program_transformer_can_overfit_one_tensor_batch() -> None:
    torch.manual_seed(20260828)
    batch = _batch()
    model = _model(layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)

    def objective() -> object:
        predictions = model(
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
        return ugi_joint_sparse_loss(predictions, batch)[0]

    initial = float(objective().detach())
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        optimizer.step()
    final = float(objective().detach())

    assert final < 0.25 * initial


def test_pinned_ugi_v0_port_contract_excludes_multireaction_and_holdout_fitting() -> None:
    repo = Path(__file__).resolve().parents[1]
    config_path = repo / "configs/model/phase1_ugi_v0_port_transformer_seed0_v1.json"
    config = json.loads(config_path.read_text())

    assert config["duration_contract"] == {
        "reference": "qualified v0 fixed-duration production exposure",
        "optimizer_steps": 5100,
        "batch_size": 128,
        "weighted_training_draws": 652800,
        "data_dependent_early_stopping": False,
    }
    assert config["training_partition"]["training_folds"] == ["train"]
    assert config["training_partition"]["diagnostic_folds"] == ["calibration"]
    assert config["training_partition"]["selection_mode"] == "fixed_final_step"
    assert config["model"]["backbone"] == "ugi_program_transformer"
    assert config["model"]["conditioning_mode"] == "full_morphology"
    assert config["model"]["semantic_organization"] == "role_structured"
    assert config["model"]["decoration_state_conditioning"] == "bidirectional_anchor_local"
    assert config["port_contract"]["reaction_program"] == "ugi_3cr_agile"
    assert config["port_contract"]["reverse_star_sampling_steps"] == 8
    assert config["port_contract"]["bl_or_lx_training_rows"] == 0
    assert config["port_contract"]["component_ids_enter_neural_tensors"] is False
    assert config["port_contract"]["fragment_tokens_enter_neural_tensors"] is False
    assert config["port_contract"]["repairs_or_retries"] is False

    expected_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    for relative_path in (
        "experiments/phase1/product_l1/ugi_v0_port_transformer_smoke_v1.json",
        "experiments/phase1/product_l1/ugi_v0_port_transformer_seed0_h100_v1.json",
    ):
        descriptor = json.loads((repo / relative_path).read_text())
        joint = next(stage for stage in descriptor["stages"] if stage["id"] == "joint")
        assert joint["config"]["sha256"] == expected_hash


def test_ugi_program_transformer_uses_the_qualified_restartable_v0_sampler() -> None:
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
        seed=20260828,
        device="cpu",
        maximum_adjacent_branch_runs=(2, 1, 1),
    )

    assert len(terminals) == 1
    assert terminals[0].program == program
    assert metadata["sample_steps"] == 2
    assert metadata["terminal_tree_repairs"] == 0
    assert metadata["program_fields_supplied"] == [
        "node_counts",
        "junction_budgets",
        "cycle_ranks",
        "attachment_counts",
    ]


def test_sampling_cache_override_must_match_the_checkpoint_pin(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    cache_path = tmp_path / "ugi_training_cache.pt"
    cache_path.write_bytes(b"pinned training cache")
    digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    checkpoint = {
        "inputs": {
            "prepared_cache": {
                "path": "/__modal/volumes/retired/job/repo/cache.pt",
                "sha256": digest,
            }
        }
    }

    assert _resolve_prepared_cache_path(repo, checkpoint, cache_path) == cache_path

    cache_path.write_bytes(b"changed")
    with pytest.raises(UgiJointEndToEndSamplingError, match="differs from the checkpoint pin"):
        _resolve_prepared_cache_path(repo, checkpoint, cache_path)
