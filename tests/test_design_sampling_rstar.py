"""Tests for forge.design.sampling.rstar.

The load-bearing tests are the equivalence ones. `rstar_step` was transcribed out of a frozen
module that ten samplers depend on, so the only thing that licenses repointing any caller is proof
that the copy behaves identically -- not a reading of the diff. A divergence here would not raise;
it would quietly change the molecules the model generates.

Equality is asserted exactly, not approximately. Both implementations are driven from separately
seeded generators primed to the same state, so identical inputs must yield identical draws.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from forge.design.sampling.rstar import (  # noqa: E402
    SamplingError,
    rstar_step,
    sample_categorical,
)

legacy = pytest.importorskip("forge.design.flow.defog_feasibility")


def _generator(seed: int = 20260819):
    return torch.Generator(device="cpu").manual_seed(seed)


def _case(rows: int = 6, classes: int = 4, seed: int = 11):
    """A batch of positions, some masked out, with a normalised clean-state prediction."""
    torch.manual_seed(seed)
    current = torch.randint(0, classes, (rows,))
    clean = torch.softmax(torch.randn(rows, classes), dim=1)
    marginal = torch.softmax(torch.randn(classes), dim=0)
    valid = torch.zeros(rows, dtype=torch.bool)
    valid[: rows - 2] = True
    return current, clean, marginal, valid


# ------------------------------------------------------------------ equivalence with the frozen copy


@pytest.mark.parametrize("t,dt", [(0.1, 0.05), (0.5, 0.1), (0.9, 0.02), (0.99, 0.001)])
def test_rstar_step_matches_the_frozen_original(t: float, dt: float) -> None:
    """The proof that licenses migrating a caller."""
    current, clean, marginal, valid = _case()
    mine = rstar_step(current.clone(), clean.clone(), marginal.clone(), t, dt, valid, _generator())
    theirs = legacy._rstar_step(
        current.clone(), clean.clone(), marginal.clone(), t, dt, valid, _generator()
    )
    assert torch.equal(mine, theirs)


def test_matches_with_a_per_position_marginal() -> None:
    """The second supported marginal shape: one distribution per position, not one shared."""
    current, clean, _, valid = _case()
    marginal = torch.softmax(torch.randn(current.shape[0], clean.shape[1]), dim=1)
    mine = rstar_step(
        current.clone(), clean.clone(), marginal.clone(), 0.4, 0.05, valid, _generator()
    )
    theirs = legacy._rstar_step(
        current.clone(), clean.clone(), marginal.clone(), 0.4, 0.05, valid, _generator()
    )
    assert torch.equal(mine, theirs)


def test_matches_when_nothing_is_selected() -> None:
    """An all-false mask must short-circuit identically rather than sampling into nothing."""
    current, clean, marginal, _ = _case()
    valid = torch.zeros(current.shape[0], dtype=torch.bool)
    mine = rstar_step(
        current.clone(), clean.clone(), marginal.clone(), 0.5, 0.1, valid, _generator()
    )
    theirs = legacy._rstar_step(
        current.clone(), clean.clone(), marginal.clone(), 0.5, 0.1, valid, _generator()
    )
    assert torch.equal(mine, theirs)
    assert torch.equal(mine, current)


def test_sample_categorical_matches_the_frozen_original() -> None:
    probabilities = torch.softmax(torch.randn(8, 5), dim=1)
    assert torch.equal(
        sample_categorical(probabilities.clone(), _generator()),
        legacy._sample_categorical(probabilities.clone(), _generator()),
    )


def test_equivalence_holds_across_many_random_cases() -> None:
    """One case can agree by luck; forty across varied shapes and times cannot."""
    for seed in range(40):
        current, clean, marginal, valid = _case(rows=4 + seed % 7, classes=3 + seed % 4, seed=seed)
        t = 0.05 + (seed % 19) * 0.05
        mine = rstar_step(
            current.clone(), clean.clone(), marginal.clone(), t, 0.03, valid, _generator(seed)
        )
        theirs = legacy._rstar_step(
            current.clone(), clean.clone(), marginal.clone(), t, 0.03, valid, _generator(seed)
        )
        assert torch.equal(mine, theirs), f"diverged at seed {seed}"


# ------------------------------------------------------------------ determinism


def test_same_seed_gives_the_same_draw() -> None:
    """A run has to be reproducible from its recorded seed; the artifact contract depends on it."""
    current, clean, marginal, valid = _case()
    args = (current.clone(), clean.clone(), marginal.clone(), 0.3, 0.05, valid)
    assert torch.equal(rstar_step(*args, _generator(7)), rstar_step(*args, _generator(7)))


def test_different_seeds_give_different_draws() -> None:
    """Guards against the generator being ignored, which would make the seed a lie."""
    current = torch.zeros(64, dtype=torch.long)
    clean = torch.softmax(torch.randn(64, 5), dim=1)
    marginal = torch.full((5,), 0.2)
    valid = torch.ones(64, dtype=torch.bool)
    args = (current.clone(), clean.clone(), marginal.clone(), 0.5, 0.5, valid)
    assert not torch.equal(rstar_step(*args, _generator(1)), rstar_step(*args, _generator(2)))


# ------------------------------------------------------------------ contract


def test_positions_outside_the_mask_are_untouched() -> None:
    """What lets one call advance a ragged batch of differently sized molecules."""
    current, clean, marginal, valid = _case()
    output = rstar_step(current.clone(), clean, marginal, 0.5, 0.1, valid, _generator())
    assert torch.equal(output[~valid], current[~valid])


def test_the_input_tensor_is_not_mutated() -> None:
    current, clean, marginal, valid = _case()
    before = current.clone()
    rstar_step(current, clean, marginal, 0.5, 0.1, valid, _generator())
    assert torch.equal(current, before)


def test_a_misshaped_marginal_raises_rather_than_broadcasting() -> None:
    """Silent broadcasting would sample plausibly from the wrong prior."""
    current, clean, _, valid = _case()
    marginal = torch.softmax(torch.randn(3, 3, 4), dim=-1)
    with pytest.raises(SamplingError, match="incompatible support"):
        rstar_step(current, clean, marginal, 0.5, 0.1, valid, _generator())


def test_output_stays_within_the_class_vocabulary() -> None:
    current, clean, marginal, valid = _case()
    output = rstar_step(current, clean, marginal, 0.5, 0.1, valid, _generator())
    assert int(output.min()) >= 0
    assert int(output.max()) < clean.shape[1]
