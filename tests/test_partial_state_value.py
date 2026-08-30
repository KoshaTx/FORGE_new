from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.phase1.hela_potency.partial_state_attention_value import (
    UgiPartialStateAttentionValueError,
    _load_progress,
    _validate_config,
    _write_progress,
)
from experiments.phase1.hela_potency.partial_state_value import (
    UgiPartialStateValueError,
    _load_fold_progress,
    _write_fold_progress,
)
from forge.model.partial_state_value import (
    LayerwiseRoleAttentionValueHead,
    PartialStateValueError,
    RoleAwarePartialStateValueHead,
    direct_value_loss,
    role_aware_graph_summary,
)
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.reaction_program_transformer import ReactionProgramGraphTransformer

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]


def _model() -> ReactionProgramGraphTransformer:
    vocabulary = ReactionProgramVocabulary(
        program_states=("unconditioned", "ugi"),
        role_states=("unassigned", "amine", "aldehyde", "isocyanide"),
        core_position_states=("unconditioned", "exterior", "ugi:core"),
        maximum_steps=1,
    )
    return ReactionProgramGraphTransformer(
        vocabulary=vocabulary,
        node_classes=5,
        hidden_dim=16,
        layers=4,
        heads=4,
        expert_count=2,
        adapter_dim=4,
        maximum_closures=1,
        maximum_heavy_atoms=8,
        dropout=0.0,
        bond_classes=4,
    ).eval()


def _inputs() -> dict[str, torch.Tensor]:
    node_mask = torch.tensor([[True, True, True, True, False]])
    child_mask = torch.tensor([[False, True, True, True, False]])
    return {
        "nodes": torch.tensor([[1, 2, 3, 1, 0]]),
        "parents": torch.tensor([[0, 0, 1, 2, 0]]),
        "parent_bonds": torch.zeros((1, 5), dtype=torch.long),
        "closure_left": torch.zeros((1, 1), dtype=torch.long),
        "closure_right": torch.zeros((1, 1), dtype=torch.long),
        "closure_bonds": torch.zeros((1, 1), dtype=torch.long),
        "t": torch.tensor([0.5]),
        "node_mask": node_mask,
        "child_mask": child_mask,
        "closure_mask": torch.zeros((1, 1), dtype=torch.bool),
        "program_states": torch.tensor([1]),
        "role_states": torch.tensor([[1, 1, 2, 3, 0]]),
        "core_position_states": torch.tensor([[2, 1, 1, 1, 0]]),
        "program_depths": torch.tensor([1]),
        "adapter_mask": node_mask,
    }


def test_transformer_hidden_state_is_explicit_and_non_mutating() -> None:
    torch.manual_seed(13)
    model = _model()
    with torch.inference_mode():
        ordinary = model(**_inputs())
        exposed = model(**_inputs(), return_hidden_state=True, return_hidden_layers=4)
    assert "hidden_state" not in ordinary
    assert "hidden_layers" not in ordinary
    assert exposed["hidden_state"].shape == (1, 5, 16)
    assert exposed["hidden_layers"].shape == (1, 4, 5, 16)
    assert torch.equal(exposed["hidden_layers"][:, -1], exposed["hidden_state"])
    assert ordinary.keys() == exposed.keys() - {"hidden_state", "hidden_layers"}
    for key, value in ordinary.items():
        assert torch.equal(value, exposed[key]), key
    with pytest.raises(ValueError, match="return_hidden_layers"):
        model(**_inputs(), return_hidden_layers=5)
    with pytest.raises(ValueError, match="return_hidden_layers"):
        model(**_inputs(), return_hidden_layers=True)


def test_role_aware_summary_ignores_padding_and_separates_roles() -> None:
    torch.manual_seed(17)
    hidden = torch.randn(2, 5, 8)
    node_mask = torch.tensor([[True, True, True, False, False], [True, True, True, False, False]])
    roles = torch.tensor([[1, 1, 2, 0, 0], [1, 2, 2, 0, 0]])
    time = torch.tensor([0.25, 0.75])
    reference = role_aware_graph_summary(hidden, node_mask, roles, time, role_count=4)
    hidden[:, 3:] = 10_000.0
    changed_padding = role_aware_graph_summary(hidden, node_mask, roles, time, role_count=4)
    assert torch.equal(reference, changed_padding)
    assert reference.shape == (2, 8 * 5 + 3 + 3)
    assert not torch.equal(reference[0], reference[1])
    with pytest.raises(PartialStateValueError, match="outside the vocabulary"):
        role_aware_graph_summary(hidden, node_mask, roles + 10, time, role_count=4)


def test_direct_value_head_and_loss_propagate_only_through_the_head() -> None:
    torch.manual_seed(19)
    hidden = torch.randn(4, 6, 8, requires_grad=False)
    mask = torch.ones((4, 6), dtype=torch.bool)
    roles = torch.tensor([[1, 1, 2, 2, 3, 3]]).expand(4, -1)
    time = torch.linspace(0.2, 0.8, 4)
    head = RoleAwarePartialStateValueHead(hidden_dim=8, role_count=4, value_hidden_dim=16)
    prediction = head(hidden, mask, roles, time)
    loss, parts = direct_value_loss(
        prediction,
        torch.tensor([-1.0, -0.5, 0.5, 1.0]),
        rank_weight=0.25,
        huber_delta=1.0,
    )
    loss.backward()
    assert prediction.shape == (4,)
    assert float(parts["ranking"].detach()) > 0.0
    assert hidden.grad is None
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_layerwise_attention_head_preserves_nodes_roles_and_clean_probabilities() -> None:
    torch.manual_seed(23)
    hidden = torch.randn(3, 4, 7, 8, requires_grad=False)
    atom_probabilities = torch.softmax(torch.randn(3, 7, 5), dim=-1)
    parent_bond_probabilities = torch.softmax(torch.randn(3, 7, 4), dim=-1)
    closure_bond_probabilities = torch.softmax(torch.randn(3, 2, 4), dim=-1)
    node_mask = torch.tensor(
        [
            [True, True, True, True, True, False, False],
            [True, True, True, True, True, False, False],
            [True, True, True, True, True, False, False],
        ]
    )
    role_states = torch.tensor(
        [
            [1, 1, 2, 2, 3, 0, 0],
            [1, 2, 2, 3, 3, 0, 0],
            [1, 1, 2, 3, 3, 0, 0],
        ]
    )
    closure_mask = torch.tensor([[True, False], [False, False], [True, True]])
    flow_time = torch.tensor([0.2, 0.5, 0.8])
    head = LayerwiseRoleAttentionValueHead(
        hidden_dim=8,
        layer_count=4,
        atom_classes=5,
        bond_classes=4,
        role_count=4,
        attention_role_indices=(1, 2, 3),
        value_hidden_dim=16,
    )
    inputs = {
        "hidden_layers": hidden,
        "atom_probabilities": atom_probabilities,
        "parent_bond_probabilities": parent_bond_probabilities,
        "closure_bond_probabilities": closure_bond_probabilities,
        "node_mask": node_mask,
        "closure_mask": closure_mask,
        "role_states": role_states,
        "flow_time": flow_time,
    }
    reference_summary = head.summarize(**inputs)
    prediction = head(**inputs)
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["hidden_layers"][:, :, 5:] = 10_000.0
    changed["atom_probabilities"][:, 5:] = 10_000.0
    changed["parent_bond_probabilities"][:, 5:] = 10_000.0
    changed_summary = head.summarize(**changed)
    assert torch.equal(reference_summary, changed_summary)
    assert reference_summary.shape == (3, 8 * 8 + 3)
    assert prediction.shape == (3,)
    prediction.sum().backward()
    assert hidden.grad is None
    assert head.layer_logits.grad is not None
    assert all(parameter.grad is not None for parameter in head.parameters())

    changed_roles = dict(inputs)
    changed_roles["role_states"] = role_states.roll(1, dims=1)
    assert not torch.equal(reference_summary, head.summarize(**changed_roles))


def test_partial_state_fold_progress_is_signature_bound(tmp_path) -> None:
    path = tmp_path / "fold.json"
    record = {"scheme": "held_head_5fold", "fold": 0, "test_rows": 2}
    rows = [{"label": "A"}, {"label": "B"}]
    _write_fold_progress(
        path,
        fold_signature="abc",
        fold_record=record,
        evaluation_rows=rows,
    )
    restored_record, restored_rows = _load_fold_progress(
        path,
        expected_signature="abc",
        expected_test_rows=2,
        expected_evaluation_rows=2,
    )
    assert restored_record == record
    assert restored_rows == rows
    with pytest.raises(UgiPartialStateValueError, match="fold progress is invalid"):
        _load_fold_progress(
            path,
            expected_signature="changed",
            expected_test_rows=2,
            expected_evaluation_rows=2,
        )


def test_attention_value_progress_is_representation_signature_bound(tmp_path) -> None:
    path = tmp_path / "attention-fold.json"
    record = {"scheme": "held_head_5fold", "fold": 0, "test_rows": 2}
    rows = [{"label": "A"}, {"label": "B"}]
    _write_progress(
        path,
        signature="representation-v2",
        fold_record=record,
        evaluation_rows=rows,
    )
    restored_record, restored_rows = _load_progress(
        path,
        signature="representation-v2",
        test_rows=2,
        evaluation_rows=2,
    )
    assert restored_record == record
    assert restored_rows == rows
    with pytest.raises(UgiPartialStateAttentionValueError, match="progress is invalid"):
        _load_progress(
            path,
            signature="changed",
            test_rows=2,
            evaluation_rows=2,
        )


def test_attention_value_full_fit_is_authorized_while_generation_remains_fail_closed() -> None:
    config = json.loads(
        (REPO / "configs/bio/phase1_ugi_hela_partial_state_attention_value_v2.json").read_text()
    )
    _validate_config(config, profile="smoke", allocated_device="cpu")
    _validate_config(config, profile="full", allocated_device="cuda")
    disabled = dict(config)
    disabled["authorization"] = dict(config["authorization"])
    disabled["authorization"]["full_value_fit_execution_authorized"] = False
    with pytest.raises(UgiPartialStateAttentionValueError, match="full attention-value fit"):
        _validate_config(disabled, profile="full", allocated_device="cuda")
    changed = dict(config)
    changed["authorization"] = dict(config["authorization"])
    changed["authorization"]["nonzero_generation_authorized"] = True
    with pytest.raises(UgiPartialStateAttentionValueError, match="must not authorize generation"):
        _validate_config(changed, profile="smoke", allocated_device="cpu")
