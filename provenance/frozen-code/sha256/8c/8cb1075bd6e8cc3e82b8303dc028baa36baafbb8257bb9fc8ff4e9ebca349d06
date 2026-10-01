"""Check unequal shards against the independent admitted full-batch objective."""

import json

import numpy as np
import pytest
import torch

from forge.corpus.compose_lipid_training_data import ComposeLipidTrainingData
from forge.model.compose_lipid_sharded_loss import FamilyLoss
from forge.model.compose_lipid_workers import add_results, split_gradient
from forge.model.reaction_program_transformer import reaction_program_transformer_loss
from forge.model.repeat_supervision import aligned_repeat_consistency
from forge.model.synthesis_program_training import (
    build_synthesis_program_flow,
    synthesis_program_paired_topology_forward,
)
from tests.test_compose_lipid_restoration import OBJECTIVE
from tests.test_compose_lipid_run import run_args  # noqa: F401
from tests.test_compose_lipid_training_data import prepared  # noqa: F401


def fixture(run_args):  # noqa: F811
    torch.set_num_threads(1)
    torch.manual_seed(602)
    config = json.loads(run_args["config_path"].read_text())
    config["objective"] = OBJECTIVE
    config["model"].update(
        maximum_children=127, role_morphology_conditioning=True, program_routed_output_heads=True
    )
    with ComposeLipidTrainingData(
        run_args["repo"],
        **{
            key: config["inputs"][key]
            for key in ("population", "verification", "measure", "admission")
        },
    ) as data:
        model = build_synthesis_program_flow(
            vocabulary=data.vocabulary,
            node_classes=len(data.atom_vocabulary),
            model_config=config["model"],
            device="cpu",
        )
        clean = data.batch(
            data.sample_indices(12, np.random.default_rng(59), families_per_batch=3),
            maximum_nodes=128,
            maximum_closures=2,
            node_padding="batch",
            repeat_supervision="exact_fragment",
            core_conditioning="qualified_core",
        )
        node = torch.ones(len(data.atom_vocabulary)) / len(data.atom_vocabulary)
    return model, clean, config, node, torch.ones(3) / 3


@pytest.mark.parametrize("empty", [False, True])
def test_unequal_shards_preserve_global_loss_and_prediction_gradients(
    run_args, empty  # noqa: F811
):  # noqa: F811
    model, clean, config, node, bond = fixture(run_args)
    if empty:
        for key in (
            "atom_variable_mask",
            "parent_variable_mask",
            "parent_bond_variable_mask",
            "closure_endpoint_variable_mask",
            "closure_bond_variable_mask",
            "repeat_atom_groups",
            "repeat_bond_groups",
        ):
            clean[key] = torch.zeros_like(clean[key])
    generator = torch.Generator().manual_seed(719)
    prediction, _, topology = synthesis_program_paired_topology_forward(
        model, clean, node, bond, torch.rand(12, generator=generator), generator
    )
    prediction = {k: v.detach().requires_grad_() for k, v in prediction.items()}
    topology = {k: v.detach().requires_grad_() for k, v in topology.items()}
    settings = {
        k: v for k, v in OBJECTIVE.items() if k not in ("gradient_balancing", "pcgrad_backend")
    }
    weights = dict(config["semantic_weights"], repeat_consistency_weight=0.0)
    expected, metrics = reaction_program_transformer_loss(
        prediction,
        clean,
        **weights,
        **settings,
        topology_conditioned_predictions=topology,
        materialize_metrics=False,
    )
    repeat, atoms, bonds = aligned_repeat_consistency(
        prediction, clean["repeat_atom_groups"], clean["repeat_bond_groups"]
    )
    expected = expected + config["semantic_weights"]["repeat_consistency_weight"] * repeat
    metrics.update(
        repeat_consistency_mse=repeat.detach(),
        repeat_consistency_pairs=atoms,
        repeat_bond_pairs=bonds,
        semantic_total=expected.detach(),
    )
    normalizer = FamilyLoss(clean, 127)
    contributions = []
    observed_metrics = {}
    # Deliberately unequal, with potentially absent roles and zero pairs in a shard.
    for region in (slice(0, 1), slice(1, 5), slice(5, 12)):
        loss, values = normalizer.loss(
            {k: v[region] for k, v in prediction.items()},
            {k: v[region] for k, v in topology.items()},
            region,
            config,
        )
        contributions.append(loss)
        for key, value in values.items():
            observed_metrics[key] = observed_metrics.get(key, 0) + value
    actual = sum(contributions)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert set(observed_metrics) == set(metrics)
    for key in metrics:
        torch.testing.assert_close(observed_metrics[key], metrics[key], atol=2e-6, rtol=2e-5)
    leaves = tuple(prediction.values()) + tuple(topology.values())
    reference_grad = torch.autograd.grad(expected, leaves, allow_unused=True)
    actual_grad = torch.autograd.grad(actual, leaves, allow_unused=True)
    for actual, expected in zip(actual_grad, reference_grad, strict=True):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)


def test_serial_and_independent_shards_preserve_gradients_and_metrics(run_args):  # noqa: F811
    model, clean, config, node, bond = fixture(run_args)
    kwargs = dict(seed=810, device="cpu")
    serial = split_gradient(model, clean, config, node, bond, parts=(0, 1), **kwargs)
    separate = add_results(
        [
            split_gradient(model, clean, config, node, bond, parts=(part,), **kwargs)
            for part in (0, 1)
        ]
    )
    for index in (0, 2):
        torch.testing.assert_close(serial[index], separate[index], atol=0, rtol=0)
    assert serial[1] == separate[1]
    for key in serial[3]:
        torch.testing.assert_close(serial[3][key], separate[3][key], atol=0, rtol=0)
