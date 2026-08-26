from __future__ import annotations

import pytest

from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.reaction_program_transformer import (
    ReactionProgramGraphTransformer,
    balanced_pcgrad_backward,
    per_program_transformer_losses,
    reaction_program_transformer_loss,
)

torch = pytest.importorskip("torch")


def _batch() -> dict[str, torch.Tensor]:
    batch, nodes, closures = 3, 6, 1
    node_mask = torch.ones((batch, nodes), dtype=torch.bool)
    child_mask = node_mask.clone()
    child_mask[:, 0] = False
    closure_mask = torch.zeros((batch, closures), dtype=torch.bool)
    program_states = torch.tensor([1, 2, 3])
    return {
        "nodes": torch.tensor([[1, 2, 3, 1, 2, 3]] * batch),
        "parents": torch.tensor([[0, 0, 1, 1, 2, 3]] * batch),
        "parent_bonds": torch.tensor([[0, 1, 1, 2, 1, 1]] * batch),
        "closure_left": torch.zeros((batch, closures), dtype=torch.long),
        "closure_right": torch.zeros((batch, closures), dtype=torch.long),
        "closure_bonds": torch.zeros((batch, closures), dtype=torch.long),
        "t": torch.tensor([0.2, 0.5, 0.8]),
        "node_mask": node_mask,
        "child_mask": child_mask,
        "closure_mask": closure_mask,
        "program_states": program_states,
        "source_program_states": program_states.clone(),
        "role_states": torch.tensor([[1, 1, 2, 2, 3, 3], [1, 2, 2, 3, 3, 1], [2, 2, 1, 1, 3, 3]]),
        "core_position_states": torch.tensor(
            [[2, 1, 1, 1, 1, 1], [3, 1, 1, 1, 1, 1], [4, 1, 1, 1, 1, 1]]
        ),
        "program_depths": torch.tensor([1, 2, 3]),
        "component_instance_states": torch.tensor([[1, 1, 2, 2, 3, 3]] * batch),
        "component_position_states": torch.tensor([[1, 2, 1, 2, 1, 2]] * batch),
        "repeat_group_states": torch.tensor([[2, 2, 2, 2, 0, 0]] * batch),
        "adapter_mask": node_mask.clone(),
        "atom_variable_mask": node_mask.clone(),
        "parent_variable_mask": child_mask.clone(),
        "parent_bond_variable_mask": child_mask.clone(),
        "closure_endpoint_variable_mask": closure_mask.clone(),
        "closure_bond_variable_mask": closure_mask.clone(),
    }


def _model(**overrides: object) -> ReactionProgramGraphTransformer:
    vocabulary = ReactionProgramVocabulary(
        program_states=("unconditioned", "ugi", "bl", "lx"),
        role_states=("unassigned", "head", "tail_a", "tail_b"),
        core_position_states=("unconditioned", "exterior", "ugi:c1", "bl:c1", "lx:c1"),
        maximum_steps=4,
    )
    arguments = {
        "vocabulary": vocabulary,
        "node_classes": 5,
        "hidden_dim": 32,
        "layers": 3,
        "heads": 4,
        "expert_count": 3,
        "adapter_dim": 8,
        "maximum_closures": 2,
        "maximum_heavy_atoms": 16,
        "dropout": 0.0,
        "bond_classes": 4,
        **overrides,
    }
    return ReactionProgramGraphTransformer(**arguments)


def test_transformer_cross_attends_in_every_layer_and_balances_family_gradients() -> None:
    torch.manual_seed(17)
    model = _model()
    batch = _batch()
    calls = [0, 0, 0]
    hooks = [
        block.cross_attention.register_forward_hook(
            lambda _module, _inputs, _output, index=index: calls.__setitem__(
                index, calls[index] + 1
            )
        )
        for index, block in enumerate(model.blocks)
    ]
    predictions = model(
        **{
            key: value
            for key, value in batch.items()
            if key
            not in {
                "source_program_states",
                "atom_variable_mask",
                "parent_variable_mask",
                "parent_bond_variable_mask",
                "closure_endpoint_variable_mask",
                "closure_bond_variable_mask",
            }
        }
    )
    for hook in hooks:
        hook.remove()

    assert calls == [1, 1, 1]
    assert predictions["nodes"].shape == (3, 6, 5)
    assert predictions["role_states"].shape == (3, 6, 4)
    assert predictions["core_position_states"].shape == (3, 6, 5)
    assert predictions["expert_weights"].shape == (3, 3, 3)
    assert torch.allclose(predictions["expert_weights"].sum(dim=-1), torch.ones((3, 3)))

    family_losses, metrics = per_program_transformer_losses(
        predictions, batch, role_weight=0.25, core_weight=0.25
    )
    diagnostic = balanced_pcgrad_backward(family_losses, model)
    assert sorted(family_losses) == [1, 2, 3]
    assert all(torch.isfinite(loss) for loss in family_losses.values())
    assert all(value > 0.0 for value in diagnostic["raw_gradient_norms"])
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert "program_1_role_consistency_ce" in metrics


def test_transformer_is_deterministic_and_program_conditioning_changes_output() -> None:
    batch = _batch()
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key
        not in {
            "source_program_states",
            "atom_variable_mask",
            "parent_variable_mask",
            "parent_bond_variable_mask",
            "closure_endpoint_variable_mask",
            "closure_bond_variable_mask",
        }
    }
    torch.manual_seed(23)
    first = _model().eval()
    torch.manual_seed(23)
    second = _model().eval()
    with torch.no_grad():
        first_output = first(**model_inputs)["nodes"]
        second_output = second(**model_inputs)["nodes"]
        changed = dict(model_inputs)
        changed["program_states"] = torch.tensor([2, 3, 1])
        changed_output = first(**changed)["nodes"]
    assert torch.equal(first_output, second_output)
    assert not torch.equal(first_output, changed_output)


def test_program_memory_preserves_node_alignment() -> None:
    torch.manual_seed(31)
    model = _model().eval()
    batch = _batch()
    with torch.no_grad():
        tokens, mask, _ = model.program_encoder(
            program_states=batch["program_states"],
            role_states=torch.ones_like(batch["role_states"]),
            core_position_states=torch.ones_like(batch["core_position_states"]),
            program_depths=batch["program_depths"],
            adapter_mask=batch["adapter_mask"],
        )
    assert torch.all(mask)
    # Program and depth occupy slots 0/1. Identical role/core states at node slots still differ
    # because the memory retains the structural slot to which each semantic coordinate applies.
    assert not torch.equal(tokens[:, 2], tokens[:, 3])


def test_repeat_group_conditioning_uses_position_but_not_component_identity() -> None:
    torch.manual_seed(33)
    model = _model(repeat_group_conditioning=True).eval()
    batch = _batch()
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key
        not in {
            "source_program_states",
            "atom_variable_mask",
            "parent_variable_mask",
            "parent_bond_variable_mask",
            "closure_endpoint_variable_mask",
            "closure_bond_variable_mask",
        }
    }
    changed_instances = dict(model_inputs)
    changed_instances["component_instance_states"] = torch.flip(
        model_inputs["component_instance_states"], dims=(1,)
    )
    removed_repeats = dict(model_inputs)
    removed_repeats["repeat_group_states"] = torch.zeros_like(model_inputs["repeat_group_states"])
    with torch.no_grad():
        reference = model(**model_inputs)["nodes"]
        instance_changed = model(**changed_instances)["nodes"]
        repeat_removed = model(**removed_repeats)["nodes"]
    assert torch.equal(reference, instance_changed)
    assert not torch.equal(reference, repeat_removed)


def test_repeat_consistency_loss_aligns_matched_exterior_positions() -> None:
    torch.manual_seed(35)
    batch = _batch()
    model = _model().eval()
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key
        not in {
            "source_program_states",
            "atom_variable_mask",
            "parent_variable_mask",
            "parent_bond_variable_mask",
            "closure_endpoint_variable_mask",
            "closure_bond_variable_mask",
        }
    }
    with torch.no_grad():
        predictions = model(**model_inputs)
    aligned = {key: value.clone() for key, value in predictions.items()}
    aligned["nodes"][:, 2] = aligned["nodes"][:, 0]
    aligned["nodes"][:, 3] = aligned["nodes"][:, 1]
    aligned["parent_bonds"][:, 2] = aligned["parent_bonds"][:, 0]
    aligned["parent_bonds"][:, 3] = aligned["parent_bonds"][:, 1]
    _, aligned_metrics = reaction_program_transformer_loss(
        aligned,
        batch,
        role_weight=0.0,
        core_weight=0.0,
        repeat_consistency_weight=1.0,
    )
    misaligned = {key: value.clone() for key, value in aligned.items()}
    misaligned["nodes"][:, 3].zero_()
    misaligned["nodes"][:, 3, 0] = 20.0
    _, misaligned_metrics = reaction_program_transformer_loss(
        misaligned,
        batch,
        role_weight=0.0,
        core_weight=0.0,
        repeat_consistency_weight=1.0,
    )
    assert aligned_metrics["repeat_consistency_mse"] == pytest.approx(0.0)
    assert aligned_metrics["repeat_consistency_pairs"] > 0
    assert misaligned_metrics["repeat_consistency_mse"] > 0.0


def test_factorized_control_prevents_cross_role_identity_messages() -> None:
    torch.manual_seed(37)
    model = _model(role_isolated_attention=True).eval()
    batch = _batch()
    model_inputs = {
        key: value.clone()
        for key, value in batch.items()
        if key
        not in {
            "source_program_states",
            "atom_variable_mask",
            "parent_variable_mask",
            "parent_bond_variable_mask",
            "closure_endpoint_variable_mask",
            "closure_bond_variable_mask",
        }
    }
    changed = {key: value.clone() for key, value in model_inputs.items()}
    target_role = 1
    other_roles = changed["role_states"] != target_role
    changed["nodes"][other_roles] = (changed["nodes"][other_roles] + 1) % 5
    changed["core_position_states"][other_roles] = 1
    with torch.no_grad():
        reference = model(**model_inputs)
        intervention = model(**changed)
    target_nodes = model_inputs["role_states"] == target_role
    assert torch.equal(reference["nodes"][target_nodes], intervention["nodes"][target_nodes])
    assert torch.equal(
        reference["role_states"][target_nodes], intervention["role_states"][target_nodes]
    )
    assert torch.equal(
        reference["core_position_states"][target_nodes],
        intervention["core_position_states"][target_nodes],
    )


def test_family_balancing_does_not_amplify_a_converged_zero_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    model = torch.nn.ParameterList([parameter])
    diagnostic = balanced_pcgrad_backward({1: parameter.square(), 2: parameter.sum() * 0.0}, model)
    assert parameter.grad is not None
    assert parameter.grad.item() == pytest.approx(1.0)
    assert diagnostic["raw_gradient_norms"] == pytest.approx([2.0, 0.0])
    assert diagnostic["family_weighting"] == "equal_loss_mass_without_norm_amplification"


def test_family_balancing_preserves_unused_parameter_semantics_and_accumulates() -> None:
    shared = torch.nn.Parameter(torch.tensor(0.0))
    first_only = torch.nn.Parameter(torch.tensor(0.0))
    model = torch.nn.ParameterList([shared, first_only])
    losses = {1: shared + first_only, 2: -shared}

    diagnostic = balanced_pcgrad_backward(losses, model, scale=0.5, materialize_diagnostics=False)
    assert shared.grad is not None
    assert first_only.grad is not None
    assert shared.grad.item() == pytest.approx(-0.125)
    assert first_only.grad.item() == pytest.approx(0.25)
    assert diagnostic["projected_conflicts"].item() == 2

    losses = {1: shared + first_only, 2: -shared}
    balanced_pcgrad_backward(losses, model, scale=0.5, materialize_diagnostics=False)
    assert shared.grad.item() == pytest.approx(-0.25)
    assert first_only.grad.item() == pytest.approx(0.5)
