from __future__ import annotations

import copy
import math

import pytest
import torch

from forge.model.ugi_candidate_graph_residual import (
    CandidateGraphResidual,
    CandidateGraphResidualError,
    candidate_class_loss,
)


def _head(*, use_adjacency: bool = True) -> CandidateGraphResidual:
    return CandidateGraphResidual(4, 2, initialization_seed=902, use_adjacency=use_adjacency)


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "hidden": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10,
        "adjacency": torch.tensor(
            [[[0, 1, 0], [1, 0, 1], [0, 1, 0]], [[0, 1, 1], [1, 0, 0], [1, 0, 0]]],
            dtype=torch.float32,
        ),
        "node_colors": torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]]
        ),
        "node_mask": torch.ones(2, 3, dtype=torch.bool),
        "flow_time": torch.tensor([0.2, 0.8]),
        "role_index": torch.tensor([0, 1]),
    }


def _nonzero_output(head: CandidateGraphResidual) -> None:
    with torch.no_grad():
        head.output.weight.copy_(torch.linspace(-0.2, 0.3, head.width)[None])
        head.output.bias.fill_(0.05)


def test_zero_initialization_exactly_preserves_every_baseline_score() -> None:
    for use_adjacency in (True, False):
        head = _head(use_adjacency=use_adjacency)
        inputs = _inputs()
        before = {key: value.clone() for key, value in inputs.items()}
        assert torch.count_nonzero(head.output.weight) == 0
        assert torch.count_nonzero(head.output.bias) == 0
        residual = head(**inputs)
        baseline = torch.tensor([1.1234567, -4.5])
        assert residual.shape == baseline.shape
        assert torch.equal(residual, torch.zeros_like(baseline))
        assert torch.equal(baseline + residual, baseline)
        assert all(torch.equal(inputs[key], value) for key, value in before.items())


def test_local_initialization_matches_arms_and_preserves_global_rng() -> None:
    inputs = _inputs()
    before = torch.random.get_rng_state().clone()
    treatment, control = _head(), _head(use_adjacency=False)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert treatment.state_dict().keys() == control.state_dict().keys()
    for name, value in treatment.state_dict().items():
        assert torch.equal(value, control.state_dict()[name])
    treatment(**inputs)
    control(**inputs)
    assert torch.equal(torch.random.get_rng_state(), before)
    other = CandidateGraphResidual(4, 2, initialization_seed=903)
    assert not torch.equal(treatment.projection.weight, other.projection.weight)
    assert all(
        parameter.dtype == torch.float32 and parameter.device.type == "cpu"
        for parameter in treatment.parameters()
    )


@pytest.mark.parametrize("use_adjacency", [True, False])
def test_node_permutation_invariance_after_nonzero_learning(use_adjacency: bool) -> None:
    head = _head(use_adjacency=use_adjacency)
    _nonzero_output(head)
    inputs = _inputs()
    permutation = torch.tensor([2, 0, 1])
    permuted = dict(inputs)
    for name in ("hidden", "node_colors", "node_mask"):
        permuted[name] = inputs[name][:, permutation]
    permuted["adjacency"] = inputs["adjacency"][:, permutation][:, :, permutation]
    torch.testing.assert_close(head(**inputs), head(**permuted), atol=1e-7, rtol=1e-6)
    assert torch.equal(head.train()(**inputs), head.eval()(**inputs))


@pytest.mark.parametrize("use_adjacency", [True, False])
def test_padding_and_incident_padded_edges_do_not_change_scores(use_adjacency: bool) -> None:
    head = _head(use_adjacency=use_adjacency)
    _nonzero_output(head)
    original = _inputs()
    padded = dict(original)
    padded["hidden"] = torch.cat((original["hidden"], torch.full((2, 2, 4), 1000.0)), dim=1)
    padded["node_colors"] = torch.cat(
        (original["node_colors"], torch.full((2, 2, 2), -1000.0)), dim=1
    )
    padded["node_mask"] = torch.cat(
        (original["node_mask"], torch.zeros(2, 2, dtype=torch.bool)), dim=1
    )
    padded["adjacency"] = torch.zeros(2, 5, 5)
    padded["adjacency"][:, :3, :3] = original["adjacency"]
    padded["adjacency"][:, 0, 3] = 1
    padded["adjacency"][:, 3, 0] = 1
    torch.testing.assert_close(head(**original), head(**padded), atol=1e-7, rtol=1e-6)


def test_complete_graph_interaction_is_not_an_additive_node_score() -> None:
    head = CandidateGraphResidual(1, 1, width=2, initialization_seed=1)
    # Encode max(color_i + color_neighbor - 1, 0), followed by identity and mean.
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.projection.weight[0, 4] = 1
        head.self_updates[0].weight[0, 0] = 1
        head.neighbor_updates[0].weight[0, 0] = 1
        head.self_updates[0].bias[0] = -1
        head.self_updates[1].weight[0, 0] = 1
        head.output.weight[0, 0] = 1
    inputs = {
        "hidden": torch.zeros(4, 2, 1),
        "node_colors": torch.tensor(
            [[[0.0], [0.0]], [[0.0], [1.0]], [[1.0], [0.0]], [[1.0], [1.0]]]
        ),
        "adjacency": torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]).expand(4, -1, -1),
        "node_mask": torch.ones(4, 2, dtype=torch.bool),
        "flow_time": torch.full((4,), 0.5),
        "role_index": torch.zeros(4, dtype=torch.long),
    }
    values = head(**inputs)
    assert torch.equal(values, torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert values[3] - values[2] - values[1] + values[0] != 0


def test_degree_control_distinguishes_local_degree_assignments() -> None:
    head = CandidateGraphResidual(1, 1, width=2, initialization_seed=1, use_adjacency=False)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.projection.weight[0, 0] = -1
        head.projection.weight[0, -1] = 1
        for layer in head.self_updates:
            layer.weight[0, 0] = 1
        head.output.weight[0, 0] = 1
    inputs = {
        "hidden": torch.tensor([[[1.5], [0.0], [0.0]]]).expand(2, -1, -1),
        "node_colors": torch.zeros(2, 3, 1),
        "adjacency": _inputs()["adjacency"],
        "node_mask": torch.ones(2, 3, dtype=torch.bool),
        "flow_time": torch.zeros(2),
        "role_index": torch.zeros(2, dtype=torch.long),
    }
    values = head(**inputs)
    torch.testing.assert_close(values, torch.tensor([1.0, 2.5 / 3]))
    assert values[0] != values[1]


def test_degree_control_cannot_distinguish_same_degree_rewiring_but_keeps_colors() -> None:
    head = _head(use_adjacency=False)
    _nonzero_output(head)
    inputs = {
        "hidden": torch.arange(16, dtype=torch.float32).reshape(1, 4, 4).expand(2, -1, -1),
        "node_colors": torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]]).expand(
            2, -1, -1
        ),
        "adjacency": torch.tensor(
            [
                [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
                [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]],
            ],
            dtype=torch.float32,
        ),
        "node_mask": torch.ones(2, 4, dtype=torch.bool),
        "flow_time": torch.zeros(2),
        "role_index": torch.zeros(2, dtype=torch.long),
    }
    assert torch.equal(inputs["adjacency"].sum(-1)[0], inputs["adjacency"].sum(-1)[1])
    values = head(**inputs)
    assert values[0] == values[1]
    capture = []
    handle = head.projection.register_forward_pre_hook(lambda module, args: capture.append(args[0]))
    try:
        head(**inputs)
    finally:
        handle.remove()
    assert torch.equal(capture[0][..., 7:9], inputs["node_colors"])
    assert torch.equal(capture[0][..., -1], inputs["adjacency"].sum(-1))


@pytest.mark.parametrize("use_adjacency", [True, False])
def test_only_model_parameters_receive_gradients(use_adjacency: bool) -> None:
    head = _head(use_adjacency=use_adjacency)
    _nonzero_output(head)
    inputs = _inputs()
    for name in ("hidden", "adjacency", "node_colors", "flow_time"):
        inputs[name].requires_grad_()
    head(**inputs).sum().backward()
    assert all(
        inputs[name].grad is None for name in ("hidden", "adjacency", "node_colors", "flow_time")
    )
    assert all(parameter.grad is not None for parameter in head.parameters())
    assert any(bool(parameter.grad.count_nonzero()) for parameter in head.parameters())
    if not use_adjacency:
        assert all(not bool(layer.weight.grad.count_nonzero()) for layer in head.neighbor_updates)


def test_parameter_receipt_rejects_unexpected_trainable_parameters() -> None:
    head = _head()
    receipt = head.trainable_parameter_receipt()
    assert receipt["owns_backbone"] is False
    assert receipt["trainable_parameter_count"] == sum(p.numel() for p in head.parameters())
    assert receipt["message_passing_rounds"] == 2
    head.register_parameter("extra", torch.nn.Parameter(torch.zeros(1)))
    with pytest.raises(CandidateGraphResidualError, match="boundary"):
        head.trainable_parameter_receipt()
    head = _head()
    head.output.bias.requires_grad_(False)
    with pytest.raises(CandidateGraphResidualError, match="boundary"):
        head.trainable_parameter_receipt()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("hidden", torch.zeros(2, 3, 5), "hidden"),
        ("hidden", torch.zeros(2, 3, 4, dtype=torch.long), "hidden"),
        ("hidden", torch.zeros(2, 3, 4, dtype=torch.float64), "dtype"),
        ("hidden", torch.full((2, 3, 4), float("nan")), "finite"),
        ("node_colors", torch.zeros(2, 3, 3), "node_colors"),
        ("node_colors", torch.full((2, 3, 2), float("inf")), "finite"),
        ("node_mask", torch.ones(2, 3), "boolean"),
        ("node_mask", torch.zeros(2, 3, dtype=torch.bool), "real node"),
        ("adjacency", torch.zeros(2, 3, 2), "adjacency"),
        ("adjacency", torch.full((2, 3, 3), 0.5), "binary"),
        ("adjacency", torch.full((2, 3, 3), float("nan")), "finite"),
        ("adjacency", torch.eye(3).expand(2, -1, -1), "self edges"),
        ("adjacency", torch.triu(torch.ones(2, 3, 3), diagonal=1), "symmetric"),
        ("flow_time", torch.zeros(2, 1), "flow_time"),
        ("flow_time", torch.tensor([-0.1, 0.5]), "\\[0, 1\\]"),
        ("flow_time", torch.tensor([0.1, 1.1]), "\\[0, 1\\]"),
        ("role_index", torch.tensor([0.0, 1.0]), "integer"),
        ("role_index", torch.tensor([0, 2]), "\\[0, 2\\)"),
        ("role_index", torch.tensor([-1, 1]), "\\[0, 2\\)"),
    ],
)
def test_malformed_forward_inputs_fail(name: str, value: torch.Tensor, message: str) -> None:
    inputs = _inputs()
    inputs[name] = value
    with pytest.raises(CandidateGraphResidualError, match=message):
        _head()(**inputs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_dim": 0},
        {"node_color_dim": 0},
        {"width": -1},
        {"width": True},
        {"initialization_seed": -1},
        {"initialization_seed": True},
        {"initialization_seed": 2**63},
        {"use_adjacency": 1},
    ],
)
def test_invalid_constructor_values_fail(kwargs: dict) -> None:
    args = {"hidden_dim": 4, "node_color_dim": 2, "initialization_seed": 902, **kwargs}
    with pytest.raises(CandidateGraphResidualError):
        CandidateGraphResidual(**args)


def test_grouped_class_loss_preserves_positive_multiplicity_and_all_row_accounting() -> None:
    # Rows: mixed with duplicate positives; empty; absent target; singleton positive; all positive.
    scores = torch.zeros(8, requires_grad=True)
    offsets = torch.tensor([0, 3, 3, 5, 6, 8])
    positives = torch.tensor([True, True, False, False, False, True, True, True])
    result = candidate_class_loss(scores, offsets, positives)
    assert result.counts == {
        "rows": 5,
        "candidates": 8,
        "positive_candidates": 5,
        "empty_rows": 1,
        "absent_target_rows": 1,
        "singleton_rows": 1,
        "all_positive_rows": 2,
        "optimizable_rows": 3,
        "unlearnable_rows": 2,
        "zero_signal_rows": 4,
        "informative_rows": 1,
    }
    torch.testing.assert_close(result.per_row_loss, torch.tensor([math.log(3 / 2), 0, 0, 0, 0]))
    torch.testing.assert_close(result.loss, torch.tensor(math.log(3 / 2) / 3))
    torch.testing.assert_close(result.per_row_loss.mean(), torch.tensor(math.log(3 / 2) / 5))
    result.loss.backward()
    torch.testing.assert_close(scores.grad[:3], torch.tensor([-1 / 18, -1 / 18, 1 / 9]))
    assert torch.equal(scores.grad[3:], torch.zeros(5))
    # Deduplicating either positive representative would incorrectly change log(3/2) to log(2).
    reduced = candidate_class_loss(
        torch.zeros(2), torch.tensor([0, 2]), torch.tensor([True, False])
    )
    assert not torch.isclose(result.per_row_loss[0], reduced.loss)


def test_grouped_loss_matches_direct_logsumexp_and_gradients_with_large_logits() -> None:
    scores = torch.tensor(
        [1000.0, 1001.0, 999.0, -1000.0, -999.0], dtype=torch.float64, requires_grad=True
    )
    positive = torch.tensor([False, True, True, True, False])
    result = candidate_class_loss(scores, torch.tensor([0, 3, 5]), positive)
    expected = torch.stack(
        (
            torch.logsumexp(scores[:3], 0) - torch.logsumexp(scores[1:3], 0),
            torch.logsumexp(scores[3:], 0) - scores[3],
        )
    )
    torch.testing.assert_close(result.per_row_loss, expected)
    grad = torch.autograd.grad(result.loss, scores, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.mean(), scores)[0]
    torch.testing.assert_close(grad, expected_grad)


def test_loss_preserves_negative_multiplicity_and_candidate_order_invariance() -> None:
    scores = torch.tensor([0.0, 1.0, 1.0], requires_grad=True)
    mask = torch.tensor([True, False, False])
    result = candidate_class_loss(scores, torch.tensor([0, 3]), mask)
    permutation = torch.tensor([2, 0, 1])
    permuted = candidate_class_loss(scores[permutation], torch.tensor([0, 3]), mask[permutation])
    torch.testing.assert_close(result.loss, permuted.loss)
    assert result.loss > candidate_class_loss(scores[:2], torch.tensor([0, 2]), mask[:2]).loss


@pytest.mark.parametrize(
    ("scores", "offsets", "positive"),
    [
        ([], [0, 0, 0], []),
        ([2.0, -1.0], [0, 1, 2], [False, False]),
        ([2.0, -1.0], [0, 1, 2], [True, True]),
        ([2.0, -1.0], [0, 2], [True, True]),
    ],
)
def test_zero_signal_rows_have_exact_zero_loss_and_gradients(scores, offsets, positive) -> None:
    values = torch.tensor(scores, dtype=torch.float32, requires_grad=True)
    result = candidate_class_loss(
        values, torch.tensor(offsets), torch.tensor(positive, dtype=torch.bool)
    )
    assert result.loss.item() == 0
    assert not bool(result.per_row_loss.count_nonzero())
    result.loss.backward()
    assert torch.equal(values.grad, torch.zeros_like(values))


@pytest.mark.parametrize(
    ("scores", "offsets", "positive", "message"),
    [
        (torch.zeros(2, 1), torch.tensor([0, 2]), torch.ones(2, dtype=torch.bool), "vector"),
        (
            torch.ones(2, dtype=torch.long),
            torch.tensor([0, 2]),
            torch.ones(2, dtype=torch.bool),
            "floating",
        ),
        (
            torch.tensor([0.0, float("nan")]),
            torch.tensor([0, 2]),
            torch.ones(2, dtype=torch.bool),
            "finite",
        ),
        (torch.zeros(2), torch.tensor([0.0, 2.0]), torch.ones(2, dtype=torch.bool), "integer"),
        (torch.zeros(2), torch.tensor([0]), torch.ones(2, dtype=torch.bool), "length"),
        (torch.zeros(2), torch.tensor([1, 2]), torch.ones(2, dtype=torch.bool), "span"),
        (torch.zeros(2), torch.tensor([0, 3]), torch.ones(2, dtype=torch.bool), "span"),
        (
            torch.zeros(2),
            torch.tensor([0, 2, 1, 2]),
            torch.ones(2, dtype=torch.bool),
            "nondecreasing",
        ),
        (torch.zeros(2), torch.tensor([0, 2]), torch.ones(1, dtype=torch.bool), "shape"),
        (torch.zeros(2), torch.tensor([0, 2]), torch.ones(2), "boolean"),
    ],
)
def test_malformed_candidate_grouping_fails(scores, offsets, positive, message) -> None:
    with pytest.raises(CandidateGraphResidualError, match=message):
        candidate_class_loss(scores, offsets, positive)


def test_loss_trains_candidate_head_without_backbone_gradient() -> None:
    head = _head()
    inputs = _inputs()
    inputs["hidden"].requires_grad_()
    baseline = torch.tensor([0.0, 0.0])
    scores = baseline + head(**inputs)
    loss = candidate_class_loss(scores, torch.tensor([0, 2]), torch.tensor([True, False]))
    initial = copy.deepcopy(head.state_dict())
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    loss.loss.backward()
    optimizer.step()
    assert inputs["hidden"].grad is None
    assert not torch.equal(initial["output.weight"], head.output.weight)
