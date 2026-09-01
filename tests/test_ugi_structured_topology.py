from __future__ import annotations

import math

import pytest

from forge.model.ugi_morphology_program import (
    UgiMorphologyProgramError,
    attached_forest_structured_nll,
)

torch = pytest.importorskip("torch")


def test_structured_tree_loss_normalizes_over_complete_valid_words() -> None:
    logits = torch.zeros((2, 4, 4), requires_grad=True)
    targets = torch.tensor(
        [
            [2, 0, 1, 0],
            [2, 1, 0, 0],
        ]
    )

    loss = attached_forest_structured_nll(
        logits,
        targets,
        junction_budget=1,
        attachment_count=1,
        maximum_adjacent_branch_run=1,
    )

    assert loss.shape == (2,)
    assert torch.allclose(loss[0], loss[1])
    assert float(loss[0].detach()) > 0.0
    loss.mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_structured_tree_loss_is_zero_when_the_grammar_has_one_path() -> None:
    logits = torch.randn((3, 4), requires_grad=True)
    target = torch.tensor([1, 1, 0])

    loss = attached_forest_structured_nll(
        logits,
        target,
        junction_budget=0,
        attachment_count=1,
        maximum_adjacent_branch_run=0,
    )

    assert math.isclose(float(loss.detach()), 0.0, abs_tol=1e-6)


def test_structured_tree_loss_rejects_a_target_outside_the_declared_grammar() -> None:
    with pytest.raises(UgiMorphologyProgramError, match="outside"):
        attached_forest_structured_nll(
            torch.zeros((4, 4)),
            torch.tensor([1, 1, 1, 0]),
            junction_budget=1,
            attachment_count=1,
            maximum_adjacent_branch_run=1,
        )
