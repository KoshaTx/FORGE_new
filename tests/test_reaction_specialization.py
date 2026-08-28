from __future__ import annotations

import pytest

from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.reaction_program_transformer import ReactionProgramGraphTransformer
from forge.model.reaction_specialization import (
    ReactionSpecializationError,
    apply_specialist_state,
    exact_exposure_schedule,
    freeze_shared_parameters,
    initialize_specialist_from_shared_state,
    specialist_parameter_report,
    specialist_state_dict,
)

torch = pytest.importorskip("torch")


def _model(
    *, specialist: bool, topology_head: bool = False
) -> ReactionProgramGraphTransformer:
    vocabulary = ReactionProgramVocabulary(
        program_states=("unconditioned", "ugi", "bl", "lx"),
        role_states=("unassigned", "head", "tail"),
        core_position_states=("unconditioned", "exterior", "ugi:map_1"),
        maximum_steps=4,
    )
    return ReactionProgramGraphTransformer(
        vocabulary=vocabulary,
        node_classes=5,
        hidden_dim=16,
        layers=2,
        heads=4,
        expert_count=3,
        adapter_dim=4,
        maximum_closures=1,
        maximum_heavy_atoms=8,
        dropout=0.0,
        bond_classes=4,
        specialist_adapter_dim=4 if specialist else 0,
        maximum_children=3 if topology_head else 0,
    )


def _inputs() -> dict[str, torch.Tensor]:
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
        "t": torch.tensor([0.5]),
        "node_mask": node_mask,
        "child_mask": child_mask,
        "closure_mask": torch.zeros((1, 1), dtype=torch.bool),
        "program_states": torch.tensor([1]),
        "role_states": torch.tensor([[1, 1, 2, 2]]),
        "core_position_states": torch.tensor([[2, 1, 1, 1]]),
        "program_depths": torch.tensor([1]),
        "adapter_mask": node_mask,
    }


def test_specialist_is_an_exact_zero_delta_before_fine_tuning() -> None:
    torch.manual_seed(17)
    shared = _model(specialist=False).eval()
    specialist = _model(specialist=True).eval()
    missing = initialize_specialist_from_shared_state(specialist, shared.state_dict())

    assert missing
    assert all(".specialist_adapter." in name for name in missing)
    with torch.no_grad():
        reference = shared(**_inputs())
        observed = specialist(**_inputs())
    assert reference.keys() == observed.keys()
    for key in reference:
        assert torch.equal(reference[key], observed[key]), key


def test_specialist_delta_is_small_frozen_and_round_trips() -> None:
    torch.manual_seed(23)
    shared = _model(specialist=False)
    first = _model(specialist=True)
    initialize_specialist_from_shared_state(first, shared.state_dict())
    trainable = freeze_shared_parameters(first)
    report = specialist_parameter_report(first)

    assert trainable
    assert report["specialist_fraction"] < 0.25
    assert all(
        parameter.requires_grad == (".specialist_adapter." in name)
        for name, parameter in first.named_parameters()
    )
    with torch.no_grad():
        first.blocks[0].specialist_adapter.adapter[-1].bias.fill_(0.25)
    delta = specialist_state_dict(first)

    second = _model(specialist=True)
    initialize_specialist_from_shared_state(second, shared.state_dict())
    apply_specialist_state(second, delta)
    for name, value in delta.items():
        assert torch.equal(second.state_dict()[name], value)

    with pytest.raises(ReactionSpecializationError, match="keys changed"):
        apply_specialist_state(second, {key: value for key, value in delta.items() if key != name})


def test_topology_specialist_declares_child_head_without_unfreezing_shared_model() -> None:
    torch.manual_seed(29)
    shared = _model(specialist=False)
    first = _model(specialist=True, topology_head=True)
    missing = initialize_specialist_from_shared_state(
        first,
        shared.state_dict(),
        include_topology_head=True,
    )
    trainable = freeze_shared_parameters(first, include_topology_head=True)
    report = specialist_parameter_report(first, include_topology_head=True)

    assert set(missing) == set(trainable)
    assert {name for name in trainable if name.startswith("offspring_output.")} == {
        "offspring_output.bias",
        "offspring_output.weight",
    }
    assert all(
        parameter.requires_grad == (name in set(trainable))
        for name, parameter in first.named_parameters()
    )
    assert report["shared_frozen_parameters"] > report["specialist_trainable_parameters"]

    with torch.no_grad():
        first.offspring_output.bias.fill_(0.125)
    delta = specialist_state_dict(first, include_topology_head=True)
    second = _model(specialist=True, topology_head=True)
    initialize_specialist_from_shared_state(
        second,
        shared.state_dict(),
        include_topology_head=True,
    )
    apply_specialist_state(second, delta, include_topology_head=True)
    assert torch.equal(second.offspring_output.bias, first.offspring_output.bias)


def test_exact_exposure_schedule_matches_v0_without_padding() -> None:
    ugi = exact_exposure_schedule(
        existing_examples=68_000,
        target_examples=384_000,
        effective_batch_size=128,
        micro_batch_size=32,
    )
    bl = exact_exposure_schedule(
        existing_examples=74_800,
        target_examples=384_000,
        effective_batch_size=128,
        micro_batch_size=32,
    )

    assert ugi.additional_examples == 316_000
    assert len(ugi.optimizer_steps) == 2_469
    assert ugi.final_microbatches == (32, 32, 32)
    assert sum(map(sum, ugi.optimizer_steps)) == 316_000
    assert bl.additional_examples == 309_200
    assert len(bl.optimizer_steps) == 2_416
    assert bl.final_microbatches == (32, 32, 16)
    assert sum(map(sum, bl.optimizer_steps)) == 309_200


def test_exposure_schedule_rejects_implicit_padding_geometry() -> None:
    with pytest.raises(ReactionSpecializationError, match="geometry"):
        exact_exposure_schedule(
            existing_examples=68_000,
            target_examples=384_000,
            effective_batch_size=126,
            micro_batch_size=32,
        )
