from __future__ import annotations

import numpy as np
import pytest

from experiments.phase1.synthesis_guidance.guidance.ugi_synthesis_guidance import (
    LockedRolloutTerminal,
    RolloutBudgetLimits,
    RolloutDisposition,
    TerminalValueEvaluation,
    UgiSynthesisGuidanceError,
    annealed_ancestry_probabilities,
    keyed_random_seed,
    run_fixed_budget_terminal_rollouts,
    select_smc_ancestry,
)


def test_two_particle_fake_value_has_analytic_ancestry_probabilities() -> None:
    probabilities = annealed_ancestry_probabilities(
        np.zeros(2),
        np.asarray([0.0, np.log(3.0)]),
        np.zeros(2),
        current_beta=1.0,
        previous_beta=0.0,
        guidance_strength=1.0,
    )

    assert np.allclose(probabilities, [0.25, 0.75])


def test_zero_guidance_is_identity_and_consumes_no_resampling_seed() -> None:
    ancestry = select_smc_ancestry(
        np.asarray([-1.0, 0.0, 1.0]),
        np.asarray([100.0, -100.0, 4.0]),
        np.asarray([-5.0, 7.0, 0.0]),
        current_beta=0.7,
        previous_beta=0.2,
        guidance_strength=0.0,
        base_seed=11,
        arm="guided",
        program_index=0,
        checkpoint_index=0,
        rollout_index=0,
    )

    assert np.array_equal(ancestry.ancestors, np.arange(3))
    assert not ancestry.resampled
    assert ancestry.keyed_seed is None


def test_nonzero_guidance_with_uniform_effective_law_is_identity() -> None:
    arguments = {
        "log_weights": np.zeros(4),
        "current_values": np.full(4, 2.0),
        "previous_values": np.full(4, 1.0),
        "current_beta": 0.75,
        "previous_beta": 0.25,
        "guidance_strength": 3.0,
        "base_seed": 13,
        "arm": "guided",
        "program_index": 2,
        "checkpoint_index": 1,
        "rollout_index": 0,
    }

    first = select_smc_ancestry(**arguments)
    repeated = select_smc_ancestry(**arguments)

    assert np.array_equal(first.probabilities, np.full(4, 0.25))
    assert np.array_equal(first.ancestors, np.arange(4))
    assert not first.resampled
    assert first.keyed_seed is None
    assert np.array_equal(first.ancestors, repeated.ancestors)
    assert repeated.keyed_seed is None


def test_equal_guidance_increments_do_not_erase_nonuniform_base_weights() -> None:
    arguments = {
        "log_weights": np.log(np.asarray([0.1, 0.2, 0.7])),
        "current_values": np.full(3, 4.0),
        "previous_values": np.full(3, 1.0),
        "current_beta": 0.8,
        "previous_beta": 0.3,
        "guidance_strength": 2.0,
        "base_seed": 17,
        "arm": "guided",
        "program_index": 4,
        "checkpoint_index": 2,
        "rollout_index": 1,
    }

    first = select_smc_ancestry(**arguments)
    repeated = select_smc_ancestry(**arguments)

    assert np.allclose(first.probabilities, [0.1, 0.2, 0.7])
    assert first.resampled
    assert first.keyed_seed is not None
    assert first.keyed_seed == repeated.keyed_seed
    assert np.array_equal(first.ancestors, repeated.ancestors)


def test_slightly_nonuniform_effective_law_is_not_bypassed() -> None:
    ancestry = select_smc_ancestry(
        np.zeros(3),
        np.asarray([0.0, 0.0, 1e-6]),
        np.zeros(3),
        current_beta=1.0,
        previous_beta=0.0,
        guidance_strength=1.0,
        base_seed=29,
        arm="guided",
        program_index=1,
        checkpoint_index=2,
        rollout_index=3,
    )

    assert not np.array_equal(ancestry.probabilities, np.full(3, 1.0 / 3.0))
    assert ancestry.resampled
    assert ancestry.keyed_seed is not None


def test_exact_cancellation_to_uniform_effective_law_is_identity() -> None:
    ancestry = select_smc_ancestry(
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, -1.0]),
        np.zeros(2),
        current_beta=1.0,
        previous_beta=0.0,
        guidance_strength=1.0,
        base_seed=31,
        arm="guided",
        program_index=1,
        checkpoint_index=2,
        rollout_index=3,
    )

    assert np.array_equal(ancestry.probabilities, [0.5, 0.5])
    assert np.array_equal(ancestry.ancestors, np.arange(2))
    assert not ancestry.resampled
    assert ancestry.keyed_seed is None


def test_keyed_substreams_are_reproducible_and_order_independent() -> None:
    first = keyed_random_seed(
        19,
        arm="guided",
        program_index=7,
        checkpoint_index=2,
        rollout_index=1,
    )
    repeated = keyed_random_seed(
        19,
        rollout_index=1,
        checkpoint_index=2,
        program_index=7,
        arm="guided",
    )
    changed = keyed_random_seed(
        19,
        arm="post_hoc",
        program_index=7,
        checkpoint_index=2,
        rollout_index=1,
    )

    assert first == repeated
    assert first != changed


def test_invalid_controller_inputs_fail_closed() -> None:
    with pytest.raises(UgiSynthesisGuidanceError):
        annealed_ancestry_probabilities(
            np.zeros(2),
            np.zeros(3),
            np.zeros(2),
            current_beta=1.0,
            previous_beta=0.0,
            guidance_strength=1.0,
        )
    with pytest.raises(UgiSynthesisGuidanceError):
        annealed_ancestry_probabilities(
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            current_beta=0.2,
            previous_beta=0.7,
            guidance_strength=1.0,
        )


def test_fixed_budget_rollouts_only_assess_locked_valid_exact_l1_terminals() -> None:
    evaluated = []

    def complete(particle: int, rollout: int, seed: int) -> LockedRolloutTerminal:
        index = particle * 2 + rollout
        return LockedRolloutTerminal(
            terminal_id=f"terminal-{index}-{seed}",
            terminal_locked=True,
            terminal_valid=index != 1,
            exact_l1=index not in {1, 2},
            payload={"index": index},
        )

    def evaluate(
        terminal: LockedRolloutTerminal,
        remaining_planner: int,
        remaining_verifier: int,
    ) -> TerminalValueEvaluation:
        assert terminal.terminal_locked and terminal.terminal_valid and terminal.exact_l1
        assert remaining_planner > 0 and remaining_verifier > 0
        evaluated.append(terminal.payload["index"])
        return TerminalValueEvaluation(
            value=float(terminal.payload["index"]),
            value_policy_id="fake-diagnostic-v1",
            logical_planner_calls=1,
            verifier_calls=1,
        )

    result = run_fixed_budget_terminal_rollouts(
        2,
        rollouts_per_particle=2,
        base_seed=19,
        arm="diagnostic-guided",
        program_index=3,
        checkpoint_index=1,
        budget_limits=RolloutBudgetLimits(
            terminal_completions=4,
            logical_planner_calls=4,
            verifier_calls=4,
        ),
        complete_terminal=complete,
        evaluate_terminal=evaluate,
    )

    assert evaluated == [0, 3]
    assert [record.disposition for record in result.records] == [
        RolloutDisposition.ASSESSED,
        RolloutDisposition.INVALID_TERMINAL,
        RolloutDisposition.NONEXACT_L1,
        RolloutDisposition.ASSESSED,
    ]
    assert result.terminal_completions == 4
    assert result.logical_planner_calls == 2
    assert result.verifier_calls == 2
    assert np.allclose(result.value_matrix[[0, 1], [0, 1]], [0.0, 3.0])
    assert np.isnan(result.value_matrix[0, 1])
    assert np.isnan(result.value_matrix[1, 0])


def test_rollout_budget_exhaustion_is_explicit_and_deterministic() -> None:
    completion_calls = []

    def complete(particle: int, rollout: int, seed: int) -> LockedRolloutTerminal:
        completion_calls.append((particle, rollout, seed))
        return LockedRolloutTerminal(
            terminal_id=f"terminal-{particle}-{rollout}",
            terminal_locked=True,
            terminal_valid=True,
            exact_l1=True,
        )

    def evaluate(
        terminal: LockedRolloutTerminal,
        remaining_planner: int,
        remaining_verifier: int,
    ) -> TerminalValueEvaluation:
        return TerminalValueEvaluation(
            value=1.0,
            value_policy_id="fake-diagnostic-v1",
            logical_planner_calls=1,
            verifier_calls=0,
        )

    arguments = {
        "particle_count": 2,
        "rollouts_per_particle": 2,
        "base_seed": 23,
        "arm": "diagnostic-guided",
        "program_index": 4,
        "checkpoint_index": 2,
        "budget_limits": RolloutBudgetLimits(
            terminal_completions=3,
            logical_planner_calls=1,
            verifier_calls=0,
        ),
        "complete_terminal": complete,
        "evaluate_terminal": evaluate,
    }
    first = run_fixed_budget_terminal_rollouts(**arguments)
    first_calls = list(completion_calls)
    completion_calls.clear()
    repeated = run_fixed_budget_terminal_rollouts(**arguments)

    assert first.records == repeated.records
    assert completion_calls == first_calls
    assert first.terminal_completions == 3
    assert first.logical_planner_calls == 1
    assert first.budget_exhausted
    assert [record.disposition for record in first.records] == [
        RolloutDisposition.ASSESSED,
        RolloutDisposition.BUDGET_EXHAUSTED,
        RolloutDisposition.BUDGET_EXHAUSTED,
        RolloutDisposition.BUDGET_EXHAUSTED,
    ]


def test_unsealed_rollout_terminal_is_rejected_before_value_evaluation() -> None:
    def complete(particle: int, rollout: int, seed: int) -> LockedRolloutTerminal:
        return LockedRolloutTerminal(
            terminal_id="unsealed",
            terminal_locked=False,
            terminal_valid=True,
            exact_l1=True,
        )

    with pytest.raises(UgiSynthesisGuidanceError, match="sealed terminal"):
        run_fixed_budget_terminal_rollouts(
            1,
            rollouts_per_particle=1,
            base_seed=3,
            arm="diagnostic",
            program_index=0,
            checkpoint_index=0,
            budget_limits=RolloutBudgetLimits(1, 1, 1),
            complete_terminal=complete,
            evaluate_terminal=lambda terminal, planner, verifier: TerminalValueEvaluation(
                value=0.0,
                value_policy_id="must-not-run",
                logical_planner_calls=0,
                verifier_calls=0,
            ),
        )
