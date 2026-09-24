"""Family-block minibatches retain the admitted graph measure and restart exactly."""

# ruff: noqa: F811

import copy
import json

import numpy as np
import pytest
import torch

from forge.core.io import write_json
from forge.corpus.compose_lipid_training_data import ComposeLipidTrainingData
from tests.test_compose_lipid_restoration import OBJECTIVE
from tests.test_compose_lipid_run import assert_exact, execute, run_args, saved  # noqa: F401
from tests.test_compose_lipid_training_data import prepared  # noqa: F401


def test_family_blocks_preserve_graph_probability_including_unequal_family_sizes(prepared):
    args, _, _ = prepared
    with ComposeLipidTrainingData(**args) as data:
        rng = np.random.default_rng(981)
        identity = list(
            data._database.execute(
                "SELECT record_index,family,probability FROM weights ORDER BY record_index"
            )
        )
        names = np.array([r[1] for r in identity])
        expected = np.array([r[2] for r in identity])
        counts = np.zeros(len(data))
        for _ in range(6000):
            selected = data.sample_indices(6, rng, families_per_batch=3)
            _, per_family = np.unique(names[selected], return_counts=True)
            assert per_family.tolist() == [2, 2, 2]
            counts += np.bincount(selected, minlength=len(data))
        np.testing.assert_allclose(counts / counts.sum(), expected, atol=0.006, rtol=0)
        assert (counts > 0).all()
        state = copy.deepcopy(rng.bit_generator.state)
        expected_draw = data.sample_indices(66, rng, families_per_batch=3)
        rng.bit_generator.state = state
        np.testing.assert_array_equal(
            data.sample_indices(66, rng, families_per_batch=3), expected_draw
        )


@pytest.mark.parametrize("families", [True, 0, 24, 4])
def test_invalid_family_block_requests_fail(prepared, families):
    with ComposeLipidTrainingData(**prepared[0]) as data:
        with pytest.raises(ValueError):
            data.sample_indices(6, np.random.default_rng(0), families_per_batch=families)


@pytest.mark.parametrize("schedule", ["independent", "balanced_cycles"])
@pytest.mark.parametrize("padding", ["batch", "family"])
def test_block_sampler_restored_training_restart_is_exact(run_args, schedule, padding):
    config = json.loads(run_args["config_path"].read_text())
    config["runtime"].update(
        families_per_batch=2,
        batch_size=6,
        core_conditioning="qualified_core",
        repeat_supervision="exact_fragment",
        node_padding=padding,
        family_schedule=schedule,
    )
    config["model"].update(
        maximum_children=127, role_morphology_conditioning=True, program_routed_output_heads=True
    )
    config["objective"] = OBJECTIVE
    write_json(run_args["config_path"], config)
    execute(run_args, "block-continuous")

    def disconnect():
        p = run_args["repo"] / "block-interrupted/latest.json"
        if json.loads(p.read_text())["completed_steps"] == 2:
            raise ConnectionError("after durable checkpoint")

    with pytest.raises(ConnectionError):
        execute(run_args, "block-interrupted", commit=disconnect)
    execute(run_args, "block-interrupted", resume=True)
    for name in ("model", "optimizer", "random", "last_metrics"):
        assert_exact(
            saved(run_args, "block-continuous")[name], saved(run_args, "block-interrupted")[name]
        )
    if schedule == "balanced_cycles":
        continuous = saved(run_args, "block-continuous")
        assert (
            continuous["family_presentations"]
            == saved(run_args, "block-interrupted")["family_presentations"]
        )
        assert sum(continuous["family_presentations"].values()) == continuous["examples_seen"]


def test_every_complete_cycle_has_equal_exposure_and_restart_needs_no_cursor():
    from forge.model.family_exposure import balanced_families, exposure_budget

    counts = np.zeros(22, dtype=np.int64)
    for step in range(66):
        ids = balanced_families(families=22, per_batch=3, step=step, seed=5)
        assert len(set(ids)) == 3
        counts[ids] += 22
        np.testing.assert_array_equal(
            ids, balanced_families(families=22, per_batch=3, step=step, seed=5)
        )
        if (step + 1) % 22 == 0:
            assert len(set(counts)) == 1
    budget = exposure_budget(families=22, batch_size=66, minimum_per_family=400000)
    assert budget["presentations_per_family"] == 400026
    assert budget["optimizer_steps"] == 133342
    assert budget["total_presentations"] == 8800572


def test_family_padding_preserves_closures_when_axis_widths_coincide():
    from forge.model.compose_lipid_training import trim_family_padding

    # Two graphs padded to 12 nodes and 12 closure slots; only 5 nodes are occupied.
    clean = {
        "nodes": torch.arange(24).reshape(2, 12),
        "node_mask": torch.arange(12)[None, :] < torch.tensor([3, 5])[:, None],
        "role_morphology_states": torch.arange(96).reshape(2, 12, 4),
        "repeat_atom_groups": torch.arange(24).reshape(2, 12),
        "closure_left": torch.ones(2, 12, dtype=torch.long),
        "closure_right": torch.full((2, 12), 2, dtype=torch.long),
        "closure_mask": torch.arange(12)[None, :].expand(2, -1) < 1,
        "fixed_closure_endpoint_mask": torch.zeros(2, 12, dtype=torch.bool),
        "family_states": torch.tensor([0, 0]),
    }
    before = copy.deepcopy(clean)
    result = trim_family_padding(clean)
    assert_exact(clean, before)
    assert result["nodes"].shape == (2, 5)
    assert_exact(result["node_mask"].sum(1), torch.tensor([3, 5]))
    for key in ("nodes", "role_morphology_states", "repeat_atom_groups"):
        assert_exact(result[key], clean[key][:, :5])
    for key in (
        "closure_left",
        "closure_right",
        "closure_mask",
        "fixed_closure_endpoint_mask",
        "family_states",
    ):
        assert result[key] is clean[key]


def test_family_padding_rejects_pooled_gradients(run_args):
    from experiments.phase1.multireaction.compose_lipid_run import read_admitted_config
    from forge.model.training_restart import TrainingRestartError

    config = json.loads(run_args["config_path"].read_text())
    config["runtime"]["node_padding"] = "family"
    write_json(run_args["config_path"], config)
    with pytest.raises(TrainingRestartError, match="family-balanced"):
        read_admitted_config(run_args["repo"], run_args["config_path"])
