"""Scientific invariants for the finite joint-denoising qualification."""

import copy
import math

import numpy as np
import pytest
import torch
from scipy.optimize import brentq

from experiments.phase1.multireaction.ugi_joint_denoising_verify import certify_projection
from forge.model.joint_categorical import (
    independent_log_joint,
    marginals,
    project_joint,
    support_dimension,
)
from forge.model.ugi_joint_denoising import EdgeBlockLaw, collate, make_queries, train_loss


def test_information_projection_matches_independent_analytic_solution():
    reference = torch.tensor([[[0.45, 0.05], [0.05, 0.45]]], dtype=torch.float64)
    targets = (
        torch.tensor([[0.6, 0.4]], dtype=torch.float64),
        torch.tensor([[0.3, 0.7]], dtype=torch.float64),
    )
    result = project_joint(reference.log(), targets)
    x = brentq(lambda z: z * (0.1 + z) - 81 * (0.6 - z) * (0.3 - z), 0, 0.3)
    expected = torch.tensor([[[x, 0.6 - x], [0.3 - x, 0.1 + x]]], dtype=torch.float64)
    torch.testing.assert_close(result.log_probabilities.exp(), expected, atol=1e-9, rtol=0)


def test_destroying_dependence_recovers_product_law():
    generator = torch.Generator().manual_seed(32)
    reference = torch.randn(3, 4, 4, 2, generator=generator, dtype=torch.float64).softmax(-1)
    reference /= reference.sum((1, 2, 3), keepdim=True)
    targets = tuple(
        torch.randn(3, n, generator=generator, dtype=torch.float64).softmax(-1) for n in (4, 4, 2)
    )
    result = project_joint(independent_log_joint(marginals(reference)), targets)
    torch.testing.assert_close(
        result.log_probabilities.exp(), independent_log_joint(targets).exp(), atol=1e-10, rtol=0
    )


def test_optimality_certificate_rejects_correct_marginals_with_wrong_dependence():
    raw = np.log(np.array([[[[0.45], [0.05]], [[0.05], [0.45]]]]))
    target = [np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]]), np.ones((1, 1))]
    q = np.full_like(raw, np.log(0.25))
    with pytest.raises(ValueError, match="optimality"):
        certify_projection(raw, target, q)
    _, error, residual = certify_projection(raw, target, raw)
    assert error < 1e-14 and residual < 1e-14


def test_deterministic_marginals_have_no_coupling_freedom():
    targets = (
        torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
    )
    result = project_joint(torch.zeros(1, 2, 2), targets)
    assert result.log_probabilities.exp().tolist() == [[[0.0, 1.0], [0.0, 0.0]]]
    assert support_dimension(np.array([[False, True], [False, False]])) == 0
    assert support_dimension(np.ones((8, 8, 4), dtype=bool)) == 238


def test_infeasible_structural_zero_support_is_rejected():
    reference = torch.tensor([[[-torch.inf, 0.0], [0.0, -torch.inf]]])
    targets = (torch.tensor([[0.9, 0.1]], dtype=torch.float64),) * 2
    with pytest.raises(ValueError, match="infeasible"):
        project_joint(reference, targets)


def test_feasible_structural_zeros_remain_zero():
    reference = torch.tensor([[[0.0, -torch.inf], [-torch.inf, 0.0]]])
    targets = (torch.tensor([[0.7, 0.3]], dtype=torch.float64),) * 2
    result = project_joint(reference, targets)
    torch.testing.assert_close(
        result.log_probabilities.exp(),
        torch.tensor([[[0.7, 0.0], [0.0, 0.3]]], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    "bad",
    [torch.tensor([[0.4, 0.4]]), torch.tensor([[1.1, -0.1]]), torch.tensor([[math.nan, 0.5]])],
)
def test_bad_marginals_fail_loudly(bad):
    with pytest.raises(ValueError, match="marginal"):
        project_joint(torch.zeros(1, 2, 2), (bad, torch.tensor([[0.5, 0.5]])))


def test_nonconvergence_does_not_return_an_approximate_pass():
    reference = torch.tensor([[[10.0, -10.0], [-10.0, 10.0]]])
    targets = (
        torch.tensor([[0.7, 0.3]], dtype=torch.float64),
        torch.tensor([[0.3, 0.7]], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="did not converge"):
        project_joint(reference, targets, max_iterations=1)


def test_newton_acceleration_preserves_the_same_information_projection():
    reference = torch.tensor([[[10.0, -10.0], [-10.0, 10.0]]], dtype=torch.float64)
    targets = (
        torch.tensor([[0.7, 0.3]], dtype=torch.float64),
        torch.tensor([[0.3, 0.7]], dtype=torch.float64),
    )
    result = project_joint(reference, targets, max_iterations=200, newton_after=16)
    for observed, target in zip(marginals(result.log_probabilities.exp()), targets, strict=True):
        torch.testing.assert_close(observed, target, atol=1e-10, rtol=0)
    q = result.log_probabilities
    assert float(q[0, 0, 0] + q[0, 1, 1] - q[0, 0, 1] - q[0, 1, 0]) == pytest.approx(40.0)


def fixture():
    graph = {
        "tokens": [0, 1, 2, 0],
        "bond_edges": [[0, 1, 0], [1, 2, 1], [2, 3, 0]],
        "role_index": 0,
        "role": "head",
        "fold": 0,
        "measured": True,
    }
    query = {
        "component_index": 0,
        "edge": 1,
        "mask_probability": 0.5,
        "hidden": [False, True, True, False, False, True, False],
    }
    return [graph], [query]


def test_focal_labels_do_not_leak_into_any_input():
    graphs, queries = fixture()
    original, targets = collate(graphs, queries, 3, 2)
    changed = copy.deepcopy(graphs)
    changed[0]["tokens"][1:3] = [0, 0]
    changed[0]["bond_edges"][1][2] = 0
    altered, new_targets = collate(changed, queries, 3, 2)
    assert not torch.equal(targets, new_targets)
    for key in original:
        assert torch.equal(original[key], altered[key])


def test_joint_table_is_equivariant_to_endpoint_exchange_and_node_permutation():
    torch.manual_seed(82)
    graphs, queries = fixture()
    inputs, _ = collate(graphs, queries, 3, 2)
    model = EdgeBlockLaw(3, 2, joint=True)
    torch.nn.init.normal_(model.output.weight)
    expected = model(**inputs)
    swapped = dict(inputs, left=inputs["right"], right=inputs["left"])
    torch.testing.assert_close(model(**swapped), expected.transpose(1, 2), atol=1e-6, rtol=1e-6)
    order = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(order)
    permuted = dict(
        inputs,
        tokens=inputs["tokens"][:, order],
        node_mask=inputs["node_mask"][:, order],
        adjacency=inputs["adjacency"][:, :, order][:, :, :, order],
        left=inverse[inputs["left"]],
        right=inverse[inputs["right"]],
    )
    torch.testing.assert_close(model(**permuted), expected, atol=1e-6, rtol=1e-6)


def test_padding_cannot_change_predictions():
    graphs, queries = fixture()
    inputs, _ = collate(graphs, queries, 3, 2)
    model = EdgeBlockLaw(3, 2, joint=True)
    torch.nn.init.normal_(model.output.weight)
    padded = dict(
        inputs,
        tokens=torch.nn.functional.pad(inputs["tokens"], (0, 2), value=3),
        node_mask=torch.nn.functional.pad(inputs["node_mask"], (0, 2)),
        adjacency=torch.nn.functional.pad(inputs["adjacency"], (0, 2, 0, 2)),
    )
    torch.testing.assert_close(model(**inputs), model(**padded), atol=1e-6, rtol=1e-6)


def test_models_share_initial_encoder_and_normalized_uniform_predictions():
    graphs, queries = fixture()
    inputs, targets = collate(graphs, queries, 3, 2)
    models = []
    for joint in (False, True):
        torch.manual_seed(17)
        model = EdgeBlockLaw(3, 2, joint=joint)
        output = model(**inputs)
        loss = train_loss(output, targets, joint=joint)
        assert float(loss.detach()) == pytest.approx(math.log(18))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        models.append(model)
    for name, value in models[0].state_dict().items():
        if name in models[1].state_dict():
            assert torch.equal(value, models[1].state_dict()[name])


def test_query_masks_are_label_independent_and_all_evaluation_edges_are_retained():
    graphs, _ = fixture()
    changed = copy.deepcopy(graphs)
    changed[0]["tokens"] = [2, 2, 2, 2]
    changed[0]["bond_edges"] = [[a, b, 1 - label] for a, b, label in changed[0]["bond_edges"]]
    a = make_queries(graphs, 0, training=False, seed=10)
    b = make_queries(changed, 0, training=False, seed=10)
    assert a == b
    assert len(a) == 9
