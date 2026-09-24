"""Independent projection equivalence and seeded family-gradient restart contract."""

import copy
import json

import numpy as np
import pytest
import torch

from forge.corpus.compose_lipid_training_data import ComposeLipidTrainingData
from forge.model.compose_lipid_parallel import (
    FamilyGradientTrainer,
    copy_replica_parameters,
    family_seed,
)
from forge.model.reaction_program_transformer import (
    balanced_pcgrad_backward,
    project_family_gradients,
)
from forge.model.synthesis_program_training import build_synthesis_program_flow
from tests.test_compose_lipid_restoration import OBJECTIVE
from tests.test_compose_lipid_run import assert_exact, run_args  # noqa: F401
from tests.test_compose_lipid_training_data import prepared  # noqa: F401


def test_replica_broadcast_preserves_storage_optimizer_ownership_and_values():
    model = torch.nn.Sequential(torch.nn.Linear(3, 7), torch.nn.Linear(7, 2))
    optimizer = torch.optim.AdamW(model.parameters())
    parameters = list(model.parameters())
    pointers = [p.data_ptr() for p in parameters]
    weights = torch.arange(sum(p.numel() for p in parameters), dtype=torch.float32)
    expected = weights.clone()
    copy_replica_parameters(model, weights)
    weights.fill_(-999)
    assert [p.data_ptr() for p in model.parameters()] == pointers
    assert all(p is q for p, q in zip(parameters, optimizer.param_groups[0]["params"], strict=True))
    torch.testing.assert_close(
        torch.nn.utils.parameters_to_vector(parameters), expected, rtol=0, atol=0
    )
    for invalid in (torch.zeros(1), expected.reshape(1, -1)):
        with pytest.raises(ValueError, match="wrong size"):
            copy_replica_parameters(model, invalid)
        torch.testing.assert_close(
            torch.nn.utils.parameters_to_vector(parameters), expected, rtol=0, atol=0
        )


def test_external_gradients_preserve_projection_order_and_unused_parameters():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
            self.b = torch.nn.Parameter(torch.tensor([3.0]))

    model = Model()
    losses = {
        2: model.a.square().sum(),
        4: -model.a.sum() + model.b.square().sum(),
        9: (model.a * torch.tensor([-2.0, 5.0])).sum(),
    }
    params = list(model.parameters())
    gradients = [
        torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
        for loss in losses.values()
    ]
    flat = torch.stack(
        [
            torch.cat(
                [
                    (torch.zeros_like(p) if g is None else g).reshape(-1)
                    for p, g in zip(params, row, strict=True)
                ]
            )
            for row in gradients
        ]
    )
    availability = [[g is not None for g in row] for row in gradients]
    expected = balanced_pcgrad_backward(losses, model)
    reference = [p.grad.clone() for p in params]
    model.zero_grad(set_to_none=True)
    actual = project_family_gradients(flat, availability, params, program_states=list(losses))
    assert actual["projected_conflicts"] == expected["projected_conflicts"]
    for p, value in zip(params, reference, strict=True):
        torch.testing.assert_close(p.grad, value, atol=0, rtol=0)


def test_family_stream_restart_reproduces_optimizer_and_metrics(run_args):  # noqa: F811
    torch.set_num_threads(1)
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

        def model():
            return build_synthesis_program_flow(
                vocabulary=data.vocabulary,
                node_classes=len(data.atom_vocabulary),
                model_config=config["model"],
                device="cpu",
            )

        first = model()
        initial = copy.deepcopy(first.state_dict())
        noise = torch.full((len(data.atom_vocabulary),), 1 / len(data.atom_vocabulary))
        bond = torch.ones(3) / 3
        rng = np.random.default_rng(50)
        batches = [
            data.batch(
                data.sample_indices(6, rng, families_per_batch=3),
                maximum_nodes=128,
                maximum_closures=2,
                node_padding="batch",
                repeat_supervision="exact_fragment",
                core_conditioning="qualified_core",
            )
            for _ in range(4)
        ]
        optimizer = torch.optim.AdamW(first.parameters(), **config["optimizer"])
        trainer = FamilyGradientTrainer(
            first, optimizer, config, noise, bond, seed=50, devices=["cpu"]
        )
        metrics = []
        for step, batch in enumerate(batches):
            metrics.append(trainer.step(batch, step))
            if step == 1:
                checkpoint = copy.deepcopy((first.state_dict(), optimizer.state_dict()))
        restored = model()
        restored.load_state_dict(checkpoint[0])
        opt = torch.optim.AdamW(restored.parameters(), **config["optimizer"])
        opt.load_state_dict(checkpoint[1])
        resumed = FamilyGradientTrainer(
            restored, opt, config, noise, bond, seed=50, devices=["cpu"]
        )
        for step in (2, 3):
            assert resumed.step(batches[step], step) == metrics[step]
        assert_exact(first.state_dict(), restored.state_dict())
        assert_exact(optimizer.state_dict(), opt.state_dict())
        assert any(not torch.equal(v, first.state_dict()[k]) for k, v in initial.items())
        with pytest.raises(ValueError, match="distinct CUDA"):
            FamilyGradientTrainer(
                first, optimizer, config, noise, bond, seed=50, devices=["cpu", "cpu"]
            )
    assert family_seed(50, 0, 1) != family_seed(50, 0, 2)
