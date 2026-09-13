from __future__ import annotations

import copy

import pytest
import torch

from forge.model.ugi_semantic_residual import (
    TRAINABLE_PARAMETER_ALLOWLIST,
    UgiSemanticResidualError,
    UgiSemanticResidualHead,
    apply_semantic_residual,
)


def _head(*, use_semantics: bool = True) -> UgiSemanticResidualHead:
    return UgiSemanticResidualHead(
        3,
        2,
        semantic_scales=(2.0, 3.0, 4.0, 5.0),
        initialization_seed=51,
        use_semantics=use_semantics,
    )


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "hidden": torch.arange(18, dtype=torch.float32).reshape(2, 3, 3) / 10,
        "flow_time": torch.tensor([0.25, 0.75]),
        "amine_semantics": torch.tensor([[2.0, 3.0, 4.0, 5.0], [4.0, 6.0, 0.0, 0.0]]),
        "variable_amine_mask": torch.tensor([[True, False, True], [False, True, False]]),
    }


def _predictions() -> dict[str, torch.Tensor]:
    return {
        "nodes": torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 10,
        "bond": torch.ones(2, 3, 4),
        "offspring": torch.zeros(2, 3, 5),
    }


def test_zero_initialization_preserves_exact_predictions_and_other_tensor_identity() -> None:
    head = _head()
    predictions = _predictions()
    before = {name: value.clone() for name, value in predictions.items()}
    result = apply_semantic_residual(predictions, head, **_inputs())

    assert result is not predictions
    assert result.keys() == predictions.keys()
    for name, value in predictions.items():
        assert torch.equal(result[name], before[name])
        assert torch.equal(value, before[name])
        if name != "nodes":
            assert result[name] is value
    assert torch.count_nonzero(head.output.weight) == 0
    assert torch.count_nonzero(head.output.bias) == 0


def test_nonzero_correction_only_changes_supplied_variable_amine_nodes() -> None:
    head = _head()
    with torch.no_grad():
        head.output.bias.copy_(torch.tensor([1.0, -2.0]))
    inputs = _inputs()
    predictions = _predictions()
    result = apply_semantic_residual(predictions, head, **inputs)
    mask = inputs["variable_amine_mask"]

    assert torch.equal(result["nodes"][~mask], predictions["nodes"][~mask])
    assert torch.equal(
        result["nodes"][mask], predictions["nodes"][mask] + torch.tensor([1.0, -2.0])
    )
    assert result["bond"] is predictions["bond"]
    assert result["offspring"] is predictions["offspring"]


def test_arms_have_identical_initialization_and_forward_preserves_rng() -> None:
    before = torch.random.get_rng_state().clone()
    informative = _head()
    zero = _head(use_semantics=False)
    assert torch.equal(torch.random.get_rng_state(), before)
    for name, parameter in informative.state_dict().items():
        assert torch.equal(parameter, zero.state_dict()[name])
    informative(**_inputs())
    zero(**_inputs())
    assert torch.equal(torch.random.get_rng_state(), before)
    assert informative.trainable_parameter_receipt()["parameter_shapes"] == (
        zero.trainable_parameter_receipt()["parameter_shapes"]
    )


def test_fixed_scales_and_zero_arm_keep_four_coordinates_in_the_same_interface() -> None:
    informative = _head()
    zero = _head(use_semantics=False)
    captured: list[torch.Tensor] = []

    def capture(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        del module
        captured.append(args[0].clone())

    handles = [head.hidden.register_forward_pre_hook(capture) for head in (informative, zero)]
    try:
        informative(**_inputs())
        zero(**_inputs())
    finally:
        for handle in handles:
            handle.remove()
    inputs = _inputs()
    assert captured[0].shape == captured[1].shape == (2, 3, 8)
    assert torch.equal(captured[0][..., :3], inputs["hidden"])
    assert torch.equal(captured[0][..., 3], inputs["flow_time"][:, None].expand(-1, 3))
    assert torch.equal(captured[0][0, :, -4:], torch.ones(3, 4))
    assert torch.equal(captured[0][1, :, -4:], torch.tensor([[2.0, 2.0, 0.0, 0.0]]).expand(3, -1))
    assert torch.count_nonzero(captured[1][..., -4:]) == 0


def test_nonlinear_semantics_can_reverse_fixed_composition_positional_ranking() -> None:
    head = UgiSemanticResidualHead(1, 2, semantic_scales=(1.0,) * 4, initialization_seed=9)
    # Atom-class-1 correction is |node_feature - first_semantic_coordinate|. The two
    # assignments each contain one class-0 and one class-1 atom; only their positions differ.
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.hidden.weight[0, 0] = 1
        head.hidden.weight[0, 2] = -1
        head.hidden.weight[1, 0] = -1
        head.hidden.weight[1, 2] = 1
        head.output.weight[1, :2] = 1
    common = {
        "hidden": torch.tensor([[[0.0], [2.0]]]),
        "flow_time": torch.tensor([0.5]),
        "variable_amine_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    low = head(amine_semantics=torch.zeros(1, 4), **common).log_softmax(-1)
    high = head(amine_semantics=torch.tensor([[2.0, 0.0, 0.0, 0.0]]), **common).log_softmax(-1)

    def assignment_margin(log_prob: torch.Tensor) -> torch.Tensor:
        return (log_prob[0, 0, 0] + log_prob[0, 1, 1]) - (log_prob[0, 0, 1] + log_prob[0, 1, 0])

    assert assignment_margin(low) > 0
    assert assignment_margin(high) < 0
    zero = copy.deepcopy(head)
    zero.use_semantics = False
    assert torch.equal(
        zero(amine_semantics=torch.zeros(1, 4), **common),
        zero(amine_semantics=torch.tensor([[2.0, 0.0, 0.0, 0.0]]), **common),
    )


def test_only_residual_parameters_receive_gradients() -> None:
    head = _head()
    with torch.no_grad():
        head.output.weight.fill_(0.1)
    inputs = _inputs()
    for name in ("hidden", "flow_time", "amine_semantics"):
        inputs[name].requires_grad_()
    predictions = _predictions()
    predictions["nodes"].requires_grad_()
    result = apply_semantic_residual(predictions, head, **inputs)
    result["nodes"].sum().backward()

    assert predictions["nodes"].grad is None
    assert all(inputs[name].grad is None for name in ("hidden", "flow_time", "amine_semantics"))
    assert all(parameter.grad is not None for parameter in head.parameters())
    assert any(bool(torch.count_nonzero(parameter.grad)) for parameter in head.parameters())
    assert head.semantic_scales.requires_grad is False


def test_allowlist_receipt_rejects_unexpected_or_frozen_parameters() -> None:
    head = _head()
    receipt = head.trainable_parameter_receipt()
    assert receipt["trainable_parameter_allowlist"] == list(TRAINABLE_PARAMETER_ALLOWLIST)
    assert receipt["trainable_parameter_count"] == sum(p.numel() for p in head.parameters())
    assert receipt["owns_backbone"] is False
    head.register_parameter("unexpected", torch.nn.Parameter(torch.zeros(1)))
    with pytest.raises(UgiSemanticResidualError, match="allowlist"):
        head.trainable_parameter_receipt()
    head = _head()
    head.hidden.weight.requires_grad_(False)
    with pytest.raises(UgiSemanticResidualError, match="allowlist"):
        head.trainable_parameter_receipt()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("hidden", torch.zeros(2, 3, 4), "hidden"),
        ("hidden", torch.zeros(2, 3, 3, dtype=torch.int64), "hidden"),
        ("hidden", torch.full((2, 3, 3), float("nan")), "finite"),
        ("hidden", torch.zeros(2, 3, 3, dtype=torch.float64), "dtype"),
        ("flow_time", torch.tensor([0.0, 1.1]), "\\[0, 1\\]"),
        ("flow_time", torch.tensor([-0.1, 1.0]), "\\[0, 1\\]"),
        ("flow_time", torch.tensor([float("inf"), 1.0]), "finite"),
        ("flow_time", torch.zeros(2, 1), "shape"),
        ("amine_semantics", torch.zeros(2, 3), "shape"),
        ("amine_semantics", -torch.ones(2, 4), "nonnegative"),
        ("amine_semantics", torch.full((2, 4), float("nan")), "finite"),
        ("variable_amine_mask", torch.ones(2, 3), "boolean"),
        ("variable_amine_mask", torch.ones(2, 2, dtype=torch.bool), "boolean"),
    ],
)
def test_malformed_forward_inputs_fail(name: str, value: torch.Tensor, message: str) -> None:
    inputs = _inputs()
    inputs[name] = value
    with pytest.raises(UgiSemanticResidualError, match=message):
        _head()(**inputs)


@pytest.mark.parametrize(
    "scales", [(1, 1, 1), (1, 1, 0, 1), (1, 1, -1, 1), (1, 1, float("inf"), 1)]
)
def test_invalid_normalization_scales_fail(scales: tuple[float, ...]) -> None:
    with pytest.raises(UgiSemanticResidualError, match="semantic_scales"):
        UgiSemanticResidualHead(3, 2, semantic_scales=scales, initialization_seed=51)


@pytest.mark.parametrize("baseline", [torch.zeros(2, 3, 3), torch.full((2, 3, 2), float("nan"))])
def test_invalid_baseline_predictions_fail(baseline: torch.Tensor) -> None:
    with pytest.raises(UgiSemanticResidualError):
        apply_semantic_residual({"nodes": baseline}, _head(), **_inputs())


def test_no_selected_nodes_leaves_predictions_unchanged() -> None:
    head = _head()
    with torch.no_grad():
        head.output.bias.fill_(1)
    inputs = _inputs()
    inputs["variable_amine_mask"].zero_()
    predictions = _predictions()
    assert torch.equal(
        apply_semantic_residual(predictions, head, **inputs)["nodes"], predictions["nodes"]
    )
