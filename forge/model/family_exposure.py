"""Deterministic family schedules with exact exposure at complete cycle boundaries."""

import math

import numpy as np


def balanced_families(*, families: int, per_batch: int, step: int, seed: int) -> np.ndarray:
    """Select distinct families; each family appears k times in each F-step cycle.

    A fresh uniform permutation per cycle preserves the uniform marginal family law.
    The schedule is a pure function of the run seed and zero-based update, so restart
    needs no mutable scheduler cursor or additional random state.
    """
    if any(type(v) is not int for v in (families, per_batch, step, seed)) or not (
        1 <= per_batch <= families and step >= 0 and 0 <= seed < 2**32
    ):
        raise ValueError("Invalid balanced family schedule")
    cycle, position = divmod(step, families)
    rng = np.random.default_rng(np.random.SeedSequence([seed, cycle, 0x46414D]))
    permutation = rng.permutation(families)
    return permutation[(position * per_batch + np.arange(per_batch)) % families]


def exposure_budget(*, families: int, batch_size: int, minimum_per_family: int) -> dict:
    """Round up to complete cycles, preserving equal family exposure."""
    if any(type(v) is not int or v < 1 for v in (families, batch_size, minimum_per_family)):
        raise ValueError("Exposure budget values must be positive integers")
    cycles = math.ceil(minimum_per_family / batch_size)
    return dict(
        cycles=cycles,
        optimizer_steps=cycles * families,
        presentations_per_family=cycles * batch_size,
        total_presentations=cycles * families * batch_size,
    )


def expected_exposure(*, families: int, per_batch: int, batch_size: int, steps: int, seed: int):
    """Exact counters for any prefix, including a checkpoint inside a cycle."""
    balanced_families(families=families, per_batch=per_batch, step=steps, seed=seed)
    if type(batch_size) is not int or batch_size < 1 or batch_size % per_batch:
        raise ValueError("Exposure batches must divide evenly across families")
    cycles, remainder = divmod(steps, families)
    counts = np.full(families, cycles * batch_size, dtype=np.int64)
    for step in range(cycles * families, cycles * families + remainder):
        indices = balanced_families(families=families, per_batch=per_batch, step=step, seed=seed)
        counts[indices] += batch_size // per_batch
    return counts
