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
    *,
    specialist: bool,
    topology_head: bool = False,
    chemistry_specialist: bool = False,
    contextual_chemistry_specialist: bool = False,
    role_local_decoder: bool = False,
    structured_topology: bool = False,
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
        chemistry_specialist_adapter_dim=4 if chemistry_specialist else 0,
        chemistry_specialist_program_index=1 if chemistry_specialist else None,
        contextual_chemistry_layers=2 if contextual_chemistry_specialist else 0,
        contextual_chemistry_adapter_dim=8 if contextual_chemistry_specialist else 0,
        contextual_chemistry_program_index=(1 if contextual_chemistry_specialist else None),
        contextual_chemistry_degree_buckets=6,
        role_local_decoder_adapter_dim=8 if role_local_decoder else 0,
        role_local_decoder_program_index=1 if role_local_decoder else None,
        structured_topology_adapter_dim=8 if structured_topology else 0,
        structured_topology_program_index=1 if structured_topology else None,
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


def test_structured_topology_head_is_zero_residual_and_round_trips_as_only_delta() -> None:
    torch.manual_seed(30)
    shared = _model(specialist=False, topology_head=True).eval()
    specialist = _model(
        specialist=False,
        topology_head=True,
        structured_topology=True,
    ).eval()
    missing = initialize_specialist_from_shared_state(
        specialist,
        shared.state_dict(),
        structured_topology_only=True,
    )
    trainable = freeze_shared_parameters(specialist, structured_topology_only=True)
    report = specialist_parameter_report(specialist, structured_topology_only=True)

    assert set(missing) == set(trainable)
    assert trainable
    assert all(name.startswith("structured_topology_head.") for name in trainable)
    assert report["specialist_fraction"] < 0.25
    with torch.no_grad():
        reference = shared(**_inputs())
        identity = specialist(**_inputs())
    for key, value in reference.items():
        assert torch.equal(value, identity[key]), key
    assert torch.equal(identity["structured_offspring"], reference["offspring"])
    assert torch.equal(identity["structured_closure_left"], reference["closure_left"])
    assert torch.equal(identity["structured_closure_right"], reference["closure_right"])

    with torch.no_grad():
        specialist.structured_topology_head.offspring.bias[2] = 0.5
        changed = specialist(**_inputs())
    assert torch.equal(changed["offspring"], reference["offspring"])
    assert not torch.equal(changed["structured_offspring"], reference["offspring"])

    delta = specialist_state_dict(specialist, structured_topology_only=True)
    reloaded = _model(
        specialist=False,
        topology_head=True,
        structured_topology=True,
    ).eval()
    initialize_specialist_from_shared_state(
        reloaded,
        shared.state_dict(),
        structured_topology_only=True,
    )
    apply_specialist_state(reloaded, delta, structured_topology_only=True)
    with torch.no_grad():
        round_trip = reloaded(**_inputs())
    assert torch.equal(round_trip["structured_offspring"], changed["structured_offspring"])


def test_chemistry_specialist_is_identity_then_cannot_change_topology_logits() -> None:
    torch.manual_seed(31)
    shared = _model(specialist=False).eval()
    specialist = _model(specialist=False, chemistry_specialist=True).eval()
    missing = initialize_specialist_from_shared_state(
        specialist,
        shared.state_dict(),
        chemistry_only=True,
    )
    trainable = freeze_shared_parameters(specialist, chemistry_only=True)

    assert set(missing) == set(trainable)
    assert trainable
    assert all("chemistry_specialist_adapter" in name for name in trainable)
    with torch.no_grad():
        reference = shared(**_inputs())
        identity = specialist(**_inputs())
    for key in reference:
        assert torch.equal(reference[key], identity[key]), key

    with torch.no_grad():
        specialist.chemistry_specialist_adapter.adapter[-1].bias.fill_(0.25)
        specialist.closure_chemistry_specialist_adapter.adapter[-1].bias.fill_(-0.2)
        changed = specialist(**_inputs())
    topology_fields = (
        "parents",
        "closure_left",
        "closure_right",
        "node_count",
        "closure_count",
        "role_states",
        "core_position_states",
        "expert_weights",
    )
    for key in topology_fields:
        assert torch.equal(reference[key], changed[key]), key
    assert not torch.equal(reference["nodes"], changed["nodes"])
    assert not torch.equal(reference["parent_bonds"], changed["parent_bonds"])

    non_target_inputs = _inputs()
    non_target_inputs["program_states"] = torch.tensor([2])
    with torch.no_grad():
        non_target_reference = shared(**non_target_inputs)
        non_target_specialist = specialist(**non_target_inputs)
    for key in non_target_reference:
        assert torch.equal(non_target_reference[key], non_target_specialist[key]), key

    delta = specialist_state_dict(specialist, chemistry_only=True)
    reloaded = _model(specialist=False, chemistry_specialist=True).eval()
    initialize_specialist_from_shared_state(
        reloaded,
        shared.state_dict(),
        chemistry_only=True,
    )
    apply_specialist_state(reloaded, delta, chemistry_only=True)
    with torch.no_grad():
        round_trip = reloaded(**_inputs())
    for key in changed:
        assert torch.equal(changed[key], round_trip[key]), key


def test_contextual_chemistry_refiner_is_identity_and_chemistry_only() -> None:
    torch.manual_seed(37)
    shared = _model(specialist=False).eval()
    specialist = _model(
        specialist=False,
        contextual_chemistry_specialist=True,
    ).eval()
    missing = initialize_specialist_from_shared_state(
        specialist,
        shared.state_dict(),
        chemistry_only=True,
    )
    trainable = freeze_shared_parameters(specialist, chemistry_only=True)

    assert set(missing) == set(trainable)
    assert trainable
    assert all(
        "contextual_chemistry_layers" in name or "contextual_closure_chemistry_layers" in name
        for name in trainable
    )
    with torch.no_grad():
        reference = shared(**_inputs())
        identity = specialist(**_inputs())
    for key in reference:
        assert torch.equal(reference[key], identity[key]), key

    with torch.no_grad():
        specialist.contextual_chemistry_layers[0].output.bias.fill_(0.2)
        specialist.contextual_closure_chemistry_layers[0].output.bias.fill_(-0.1)
        changed = specialist(**_inputs())
    topology_fields = (
        "parents",
        "closure_left",
        "closure_right",
        "node_count",
        "closure_count",
        "role_states",
        "core_position_states",
        "expert_weights",
    )
    for key in topology_fields:
        assert torch.equal(reference[key], changed[key]), key
    assert not torch.equal(reference["nodes"], changed["nodes"])
    assert not torch.equal(reference["parent_bonds"], changed["parent_bonds"])

    non_target_inputs = _inputs()
    non_target_inputs["program_states"] = torch.tensor([2])
    with torch.no_grad():
        non_target_reference = shared(**non_target_inputs)
        non_target_specialist = specialist(**non_target_inputs)
    for key in non_target_reference:
        assert torch.equal(non_target_reference[key], non_target_specialist[key]), key

    delta = specialist_state_dict(specialist, chemistry_only=True)
    reloaded = _model(
        specialist=False,
        contextual_chemistry_specialist=True,
    ).eval()
    initialize_specialist_from_shared_state(
        reloaded,
        shared.state_dict(),
        chemistry_only=True,
    )
    apply_specialist_state(reloaded, delta, chemistry_only=True)
    with torch.no_grad():
        round_trip = reloaded(**_inputs())
    for key in changed:
        assert torch.equal(changed[key], round_trip[key]), key


def test_role_local_tree_decoder_is_identity_then_changes_only_target_program() -> None:
    torch.manual_seed(41)
    shared = _model(specialist=False).eval()
    specialist = _model(specialist=False, role_local_decoder=True).eval()
    missing = initialize_specialist_from_shared_state(
        specialist,
        shared.state_dict(),
        role_local_decoder_only=True,
    )
    trainable = freeze_shared_parameters(
        specialist,
        role_local_decoder_only=True,
    )

    assert set(missing) == set(trainable)
    assert trainable
    assert all("role_local_tree_decoder" in name for name in trainable)
    with torch.no_grad():
        reference = shared(**_inputs())
        identity = specialist(**_inputs())
    for key in reference:
        assert torch.equal(reference[key], identity[key]), key

    with torch.no_grad():
        specialist.role_local_tree_decoder.role_decoders[1][-1].bias.fill_(0.2)
        specialist.role_local_tree_decoder.role_decoders[2][-1].bias.fill_(-0.1)
        changed = specialist(**_inputs())
    assert not torch.equal(reference["nodes"], changed["nodes"])
    assert not torch.equal(reference["parents"], changed["parents"])
    # The fixed reaction-core node is excluded from the role-local exterior update.
    assert torch.equal(reference["nodes"][:, 0], changed["nodes"][:, 0])

    non_target_inputs = _inputs()
    non_target_inputs["program_states"] = torch.tensor([2])
    with torch.no_grad():
        non_target_reference = shared(**non_target_inputs)
        non_target_specialist = specialist(**non_target_inputs)
    for key in non_target_reference:
        assert torch.equal(non_target_reference[key], non_target_specialist[key]), key

    delta = specialist_state_dict(specialist, role_local_decoder_only=True)
    reloaded = _model(specialist=False, role_local_decoder=True).eval()
    initialize_specialist_from_shared_state(
        reloaded,
        shared.state_dict(),
        role_local_decoder_only=True,
    )
    apply_specialist_state(
        reloaded,
        delta,
        role_local_decoder_only=True,
    )
    with torch.no_grad():
        round_trip = reloaded(**_inputs())
    for key in changed:
        assert torch.equal(changed[key], round_trip[key]), key


def test_full_model_finetune_scope_round_trips_every_parameter() -> None:
    torch.manual_seed(43)
    shared = _model(specialist=False)
    finetuned = _model(specialist=False)
    missing = initialize_specialist_from_shared_state(
        finetuned,
        shared.state_dict(),
        full_model=True,
    )
    trainable = freeze_shared_parameters(finetuned, full_model=True)
    report = specialist_parameter_report(finetuned, full_model=True)

    assert missing == ()
    assert trainable == tuple(name for name, _ in finetuned.named_parameters())
    assert all(parameter.requires_grad for parameter in finetuned.parameters())
    assert report["shared_frozen_parameters"] == 0
    assert report["specialist_fraction"] == 1.0

    with torch.no_grad():
        finetuned.role_output.bias.add_(0.25)
        finetuned.core_output.bias.sub_(0.1)
    state = specialist_state_dict(finetuned, full_model=True)
    reloaded = _model(specialist=False)
    apply_specialist_state(reloaded, state, full_model=True)
    assert set(state) == set(reloaded.state_dict())
    for name, value in state.items():
        assert torch.equal(reloaded.state_dict()[name], value), name

    with pytest.raises(ReactionSpecializationError, match="keys changed"):
        apply_specialist_state(
            reloaded,
            {key: value for index, (key, value) in enumerate(state.items()) if index},
            full_model=True,
        )


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
