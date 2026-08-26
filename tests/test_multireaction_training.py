from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.phase1.multireaction.training import (
    MultiReactionTrainingError,
    _fixed_noise_reconstruction,
    _train_arm,
)
from forge.assembly import ReactionProgramSpec
from forge.model.defog_feasibility import AtomState
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.reaction_program_graph import tensorize_reaction_program_product


def test_fixed_noise_reconstruction_excludes_impossible_parent_logits() -> None:
    clean = {
        "nodes": torch.tensor([[0, 1, 0]]),
        "parents": torch.tensor([[0, 0, 1]]),
        "parent_bonds": torch.tensor([[0, 0, 0]]),
        "closure_left": torch.zeros((1, 0), dtype=torch.long),
        "closure_right": torch.zeros((1, 0), dtype=torch.long),
        "closure_bonds": torch.zeros((1, 0), dtype=torch.long),
        "node_mask": torch.tensor([[True, True, True]]),
        "child_mask": torch.tensor([[False, True, True]]),
        "closure_mask": torch.zeros((1, 0), dtype=torch.bool),
    }
    predictions = {
        "nodes": torch.tensor([[[9.0, 0.0], [0.0, 9.0], [9.0, 0.0]]]),
        # Invalid self/future positions dominate the raw logits. The correct preceding parent is
        # nevertheless the maximum over each child's admissible candidate set.
        "parents": torch.tensor([[[0.0, 99.0, 99.0], [9.0, 99.0, 99.0], [0.0, 9.0, 99.0]]]),
        "parent_bonds": torch.tensor([[[9.0], [9.0], [9.0]]]),
        "closure_left": torch.zeros((1, 0, 3)),
        "closure_right": torch.zeros((1, 0, 3)),
        "closure_bonds": torch.zeros((1, 0, 1)),
    }

    result = _fixed_noise_reconstruction(predictions, clean)

    assert result["exact_tensor_records"] == 1
    assert result["field_accuracy"]["parents"] == 1.0


def _restart_fixture() -> tuple[object, dict[str, object], tuple[object, ...], np.ndarray]:
    vocabulary = ReactionProgramVocabulary.from_specs(
        (ReactionProgramSpec("aza", "aza", "head", "tail", 1, 1),)
    )
    atom_vocabulary = (
        AtomState("C", 0, False, 0),
        AtomState("N", 0, False, 0),
    )
    record = tensorize_reaction_program_product(
        record_id="restart-example",
        program_id="aza",
        canonical_product_smiles="CCN",
        atom_roles=("tail", "tail", "head"),
        program_depth=1,
        accumulator_role="head",
        repeat_role="tail",
        vocabulary=vocabulary,
        atom_vocabulary=atom_vocabulary,
    )
    corpus = SimpleNamespace(vocabulary=vocabulary, atom_vocabulary=atom_vocabulary)
    config: dict[str, object] = {
        "seed": 17,
        "model": {
            "hidden_dim": 16,
            "layers": 1,
            "maximum_closures": 1,
            "maximum_heavy_atoms": 8,
            "dropout": 0.1,
            "bond_classes": 3,
        },
        "training": {
            "steps": 4,
            "batch_size": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
        },
        "execution": {"cpu_threads": 1, "checkpoint_interval_steps": 2},
    }
    return corpus, config, (record,), np.asarray([1.0])


def test_training_restart_is_identical_to_uninterrupted_training(tmp_path: Path) -> None:
    corpus, config, records, weights = _restart_fixture()
    node_p0 = torch.tensor([0.5, 0.5])
    bond_p0 = torch.tensor([1.0, 0.0, 0.0])
    common = {
        "records": records,
        "weights": weights,
        "device": torch.device("cpu"),
        "node_p0": node_p0,
        "bond_p0": bond_p0,
        "restart_identity": "fixed-contract",
    }
    _, uninterrupted = _train_arm(
        corpus,
        config,
        "program",
        restart_path=tmp_path / "uninterrupted.pt",
        resume=False,
        **common,
    )
    interrupted_path = tmp_path / "interrupted.pt"
    with pytest.raises(MultiReactionTrainingError, match="test interruption"):
        _train_arm(
            corpus,
            config,
            "program",
            restart_path=interrupted_path,
            resume=False,
            interrupt_after_step=2,
            **common,
        )
    _, resumed = _train_arm(
        corpus,
        config,
        "program",
        restart_path=interrupted_path,
        resume=True,
        **common,
    )

    assert resumed["model_state_sha256"] == uninterrupted["model_state_sha256"]
    assert resumed["initial_total_loss"] == uninterrupted["initial_total_loss"]
    assert resumed["final_total_loss"] == uninterrupted["final_total_loss"]
    assert resumed["minimum_total_loss"] == uninterrupted["minimum_total_loss"]
    assert resumed["sampled_programs"] == uninterrupted["sampled_programs"]
    assert resumed == uninterrupted
