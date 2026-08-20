"""Controller mechanics for future Ugi synthesis-guided sampling.

The functions in this module are qualified only with diagnostic fake values.
They do not define a synthesis score and do not authorize production guidance.
At guidance strength zero, ancestry is always the identity map so the controller
cannot perturb the frozen generator. At nonzero guidance, an exactly uniform
effective ancestry law also preserves identity rather than adding sampling noise.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from forge.core.seeds import keyed_seed


class UgiSynthesisGuidanceError(RuntimeError):
    """Raised when a controller request violates the frozen SMC contract."""


@dataclass(frozen=True)
class SMCAncestry:
    """One deterministic resampling decision and its normalized probabilities."""

    probabilities: np.ndarray
    ancestors: np.ndarray
    resampled: bool
    keyed_seed: int | None


class RolloutDisposition(str, Enum):
    """Terminal-rollout state retained without chemistry reinterpretation."""

    ASSESSED = "assessed"
    INVALID_TERMINAL = "invalid_terminal"
    NONEXACT_L1 = "nonexact_l1"
    COMPLETION_ERROR = "completion_error"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class LockedRolloutTerminal:
    """One sealed completion passed from the generator to a value evaluator."""

    terminal_id: str
    terminal_locked: bool
    terminal_valid: bool
    exact_l1: bool
    payload: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.terminal_id, str) or not self.terminal_id:
            raise UgiSynthesisGuidanceError("terminal_id must be a nonempty string")
        if not all(
            isinstance(value, bool)
            for value in (self.terminal_locked, self.terminal_valid, self.exact_l1)
        ):
            raise UgiSynthesisGuidanceError("terminal admission fields must be boolean")
        if self.exact_l1 and not self.terminal_valid:
            raise UgiSynthesisGuidanceError("an invalid terminal cannot be exact L1")


@dataclass(frozen=True)
class TerminalValueEvaluation:
    """Diagnostic scalar plus the realized logical route-compute accounting."""

    value: float
    value_policy_id: str
    logical_planner_calls: int
    verifier_calls: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.value):
            raise UgiSynthesisGuidanceError("terminal diagnostic value must be finite")
        if not isinstance(self.value_policy_id, str) or not self.value_policy_id:
            raise UgiSynthesisGuidanceError("value_policy_id must be a nonempty string")
        for name in ("logical_planner_calls", "verifier_calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise UgiSynthesisGuidanceError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class RolloutBudgetLimits:
    """Hard ceilings for one diagnostic rollout batch."""

    terminal_completions: int
    logical_planner_calls: int
    verifier_calls: int

    def __post_init__(self) -> None:
        for name in ("terminal_completions", "logical_planner_calls", "verifier_calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise UgiSynthesisGuidanceError(f"{name} budget must be nonnegative")


@dataclass(frozen=True)
class RolloutRecord:
    """One auditable fixed-order completion/assessment attempt."""

    particle_index: int
    rollout_index: int
    keyed_seed: int
    terminal_id: str | None
    disposition: RolloutDisposition
    terminal_completed: bool
    route_evaluated: bool
    value: float | None
    value_policy_id: str | None
    logical_planner_calls: int
    verifier_calls: int
    detail: str | None


@dataclass(frozen=True)
class FixedBudgetRolloutBatch:
    """Ordered terminal rollouts and realized productive-compute totals."""

    records: tuple[RolloutRecord, ...]
    particle_count: int
    rollouts_per_particle: int
    terminal_completions: int
    logical_planner_calls: int
    verifier_calls: int
    budget_exhausted: bool

    @property
    def value_matrix(self) -> np.ndarray:
        """Return diagnostic values without inventing values for censored rows."""

        matrix = np.full((self.particle_count, self.rollouts_per_particle), np.nan)
        for record in self.records:
            if record.value is not None:
                matrix[record.particle_index, record.rollout_index] = record.value
        return matrix


def run_fixed_budget_terminal_rollouts(
    particle_count: int,
    *,
    rollouts_per_particle: int,
    base_seed: int,
    arm: str,
    program_index: int,
    checkpoint_index: int,
    budget_limits: RolloutBudgetLimits,
    complete_terminal: Callable[[int, int, int], LockedRolloutTerminal],
    evaluate_terminal: Callable[[LockedRolloutTerminal, int, int], TerminalValueEvaluation],
) -> FixedBudgetRolloutBatch:
    """Complete and assess particles without ever scoring a partial graph.

    The evaluator receives only a locked, valid and exact-L1 terminal plus its
    remaining planner and verifier budgets. It must honor those ceilings. This
    function does not aggregate values or select ancestry because no scalar
    synthesis policy has been frozen for production use.
    """

    integer_inputs = {
        "particle_count": particle_count,
        "rollouts_per_particle": rollouts_per_particle,
        "base_seed": base_seed,
        "program_index": program_index,
        "checkpoint_index": checkpoint_index,
    }
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in integer_inputs.values()
        )
        or particle_count == 0
        or rollouts_per_particle == 0
    ):
        raise UgiSynthesisGuidanceError("rollout coordinates and counts are invalid")
    if not isinstance(arm, str) or not arm:
        raise UgiSynthesisGuidanceError("rollout arm must be a nonempty string")
    records = []
    completions = 0
    planner_calls = 0
    verifier_calls = 0
    exhausted = False
    for particle_index in range(particle_count):
        for rollout_index in range(rollouts_per_particle):
            seed = keyed_random_seed(
                base_seed,
                arm=arm,
                program_index=program_index,
                particle_index=particle_index,
                checkpoint_index=checkpoint_index,
                rollout_index=rollout_index,
            )
            if completions >= budget_limits.terminal_completions:
                exhausted = True
                records.append(
                    RolloutRecord(
                        particle_index=particle_index,
                        rollout_index=rollout_index,
                        keyed_seed=seed,
                        terminal_id=None,
                        disposition=RolloutDisposition.BUDGET_EXHAUSTED,
                        terminal_completed=False,
                        route_evaluated=False,
                        value=None,
                        value_policy_id=None,
                        logical_planner_calls=0,
                        verifier_calls=0,
                        detail="terminal completion budget exhausted",
                    )
                )
                continue
            try:
                terminal = complete_terminal(particle_index, rollout_index, seed)
            except Exception as error:  # completion failures are retained, not hidden
                completions += 1
                records.append(
                    RolloutRecord(
                        particle_index=particle_index,
                        rollout_index=rollout_index,
                        keyed_seed=seed,
                        terminal_id=None,
                        disposition=RolloutDisposition.COMPLETION_ERROR,
                        terminal_completed=True,
                        route_evaluated=False,
                        value=None,
                        value_policy_id=None,
                        logical_planner_calls=0,
                        verifier_calls=0,
                        detail=f"{type(error).__name__}: {error}",
                    )
                )
                continue
            completions += 1
            if not terminal.terminal_locked:
                raise UgiSynthesisGuidanceError(
                    "route evaluation requires a sealed terminal candidate"
                )
            if not terminal.terminal_valid or not terminal.exact_l1:
                disposition = (
                    RolloutDisposition.INVALID_TERMINAL
                    if not terminal.terminal_valid
                    else RolloutDisposition.NONEXACT_L1
                )
                records.append(
                    RolloutRecord(
                        particle_index=particle_index,
                        rollout_index=rollout_index,
                        keyed_seed=seed,
                        terminal_id=terminal.terminal_id,
                        disposition=disposition,
                        terminal_completed=True,
                        route_evaluated=False,
                        value=None,
                        value_policy_id=None,
                        logical_planner_calls=0,
                        verifier_calls=0,
                        detail="route evaluator not called",
                    )
                )
                continue
            remaining_planner = budget_limits.logical_planner_calls - planner_calls
            remaining_verifier = budget_limits.verifier_calls - verifier_calls
            if remaining_planner <= 0:
                exhausted = True
                records.append(
                    RolloutRecord(
                        particle_index=particle_index,
                        rollout_index=rollout_index,
                        keyed_seed=seed,
                        terminal_id=terminal.terminal_id,
                        disposition=RolloutDisposition.BUDGET_EXHAUSTED,
                        terminal_completed=True,
                        route_evaluated=False,
                        value=None,
                        value_policy_id=None,
                        logical_planner_calls=0,
                        verifier_calls=0,
                        detail="logical planner-call budget exhausted",
                    )
                )
                continue
            evaluation = evaluate_terminal(
                terminal,
                remaining_planner,
                remaining_verifier,
            )
            if (
                evaluation.logical_planner_calls > remaining_planner
                or evaluation.verifier_calls > remaining_verifier
            ):
                raise UgiSynthesisGuidanceError(
                    "terminal evaluator exceeded its declared remaining budget"
                )
            planner_calls += evaluation.logical_planner_calls
            verifier_calls += evaluation.verifier_calls
            records.append(
                RolloutRecord(
                    particle_index=particle_index,
                    rollout_index=rollout_index,
                    keyed_seed=seed,
                    terminal_id=terminal.terminal_id,
                    disposition=RolloutDisposition.ASSESSED,
                    terminal_completed=True,
                    route_evaluated=True,
                    value=float(evaluation.value),
                    value_policy_id=evaluation.value_policy_id,
                    logical_planner_calls=evaluation.logical_planner_calls,
                    verifier_calls=evaluation.verifier_calls,
                    detail=None,
                )
            )
    return FixedBudgetRolloutBatch(
        records=tuple(records),
        particle_count=particle_count,
        rollouts_per_particle=rollouts_per_particle,
        terminal_completions=completions,
        logical_planner_calls=planner_calls,
        verifier_calls=verifier_calls,
        budget_exhausted=exhausted,
    )


def keyed_random_seed(base_seed: int, **coordinates: Any) -> int:
    """Derive an order-independent 63-bit seed from declared coordinates."""

    if not isinstance(base_seed, int) or base_seed < 0:
        raise UgiSynthesisGuidanceError("base seed must be a nonnegative integer")
    return keyed_seed(base_seed, **coordinates)


def annealed_ancestry_probabilities(
    log_weights: np.ndarray,
    current_values: np.ndarray,
    previous_values: np.ndarray,
    *,
    current_beta: float,
    previous_beta: float,
    guidance_strength: float,
) -> np.ndarray:
    """Return the normalized Feynman--Kac ancestry law for fake or frozen values."""

    log_weights = np.asarray(log_weights, dtype=np.float64)
    current_values = np.asarray(current_values, dtype=np.float64)
    previous_values = np.asarray(previous_values, dtype=np.float64)
    if (
        log_weights.ndim != 1
        or not log_weights.size
        or current_values.shape != log_weights.shape
        or previous_values.shape != log_weights.shape
        or not np.all(np.isfinite(log_weights))
        or not np.all(np.isfinite(current_values))
        or not np.all(np.isfinite(previous_values))
    ):
        raise UgiSynthesisGuidanceError("SMC inputs must be aligned finite vectors")
    if (
        not 0 <= previous_beta <= current_beta <= 1
        or not np.isfinite(guidance_strength)
        or guidance_strength < 0
    ):
        raise UgiSynthesisGuidanceError("invalid annealing or guidance strength")
    logits = log_weights + guidance_strength * (
        current_beta * current_values - previous_beta * previous_values
    )
    logits -= logits.max()
    weights = np.exp(logits)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise UgiSynthesisGuidanceError("SMC weights cannot be normalized")
    return weights / total


def select_smc_ancestry(
    log_weights: np.ndarray,
    current_values: np.ndarray,
    previous_values: np.ndarray,
    *,
    current_beta: float,
    previous_beta: float,
    guidance_strength: float,
    base_seed: int,
    arm: str,
    program_index: int,
    checkpoint_index: int,
    rollout_index: int,
) -> SMCAncestry:
    """Select ancestry with zero-guidance and uniform-effective-law bypasses.

    Zero guidance remains an unconditional identity operation. For nonzero
    guidance, the bypass is determined from the final normalized ancestry
    probabilities, not from the guidance potential alone. Consequently, equal
    incremental potentials preserve legitimate nonuniform base weights, while
    exact cancellation to a uniform effective law consumes no random seed.
    """

    probabilities = annealed_ancestry_probabilities(
        log_weights,
        current_values,
        previous_values,
        current_beta=current_beta,
        previous_beta=previous_beta,
        guidance_strength=guidance_strength,
    )
    particle_count = probabilities.size
    if guidance_strength == 0 or np.all(probabilities == probabilities[0]):
        return SMCAncestry(
            probabilities=probabilities,
            ancestors=np.arange(particle_count, dtype=np.int64),
            resampled=False,
            keyed_seed=None,
        )
    if not arm or min(program_index, checkpoint_index, rollout_index) < 0:
        raise UgiSynthesisGuidanceError("SMC key coordinates are invalid")
    seed = keyed_random_seed(
        base_seed,
        arm=arm,
        program_index=program_index,
        checkpoint_index=checkpoint_index,
        rollout_index=rollout_index,
    )
    generator = np.random.default_rng(seed)
    return SMCAncestry(
        probabilities=probabilities,
        ancestors=generator.choice(
            particle_count,
            size=particle_count,
            replace=True,
            p=probabilities,
        ).astype(np.int64),
        resampled=True,
        keyed_seed=seed,
    )
