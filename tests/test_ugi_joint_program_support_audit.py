from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.phase1.synthesis_guidance.adapters.terminal_support import (
    canonical_morphology_program_bytes,
)
from experiments.phase1.synthesis_guidance.schedule.joint_program_support import (
    build_empirical_joint_program_prior,
    build_joint_program_kernel,
    evaluate_program_mode,
    joint_candidate_pool_probabilities,
    joint_program_neighborhood,
    sample_empirical_joint_program_prior,
)
from forge.model.ugi_morphology_program import UgiMorphologyProgram


def _program(
    node_counts: tuple[int, int, int],
    junction_budgets: tuple[int, int, int],
) -> UgiMorphologyProgram:
    return UgiMorphologyProgram(
        node_counts=node_counts,
        junction_budgets=junction_budgets,
        cycle_ranks=(0, 0, 0),
        attachment_counts=(1, 1, 1),
    )


def test_empirical_prior_preserves_complete_tuple_frequencies() -> None:
    first = _program((4, 18, 12), (0, 0, 0))
    second = _program((8, 24, 18), (1, 0, 0))
    prior = build_empirical_joint_program_prior((first, second, first))

    by_payload = {
        canonical_morphology_program_bytes(program): count
        for program, count in zip(prior.programs, prior.counts, strict=True)
    }
    assert by_payload == {
        canonical_morphology_program_bytes(first): 2,
        canonical_morphology_program_bytes(second): 1,
    }
    assert np.allclose(sorted(prior.probabilities), [1 / 3, 2 / 3])


def test_joint_sampling_never_recombines_precursor_role_states() -> None:
    first = _program((4, 18, 12), (0, 0, 0))
    second = _program((8, 24, 18), (1, 0, 0))
    prior = build_empirical_joint_program_prior((first, second))
    sampled = sample_empirical_joint_program_prior(
        prior,
        count=512,
        rng=np.random.default_rng(20260803),
    )
    admitted = {
        canonical_morphology_program_bytes(first),
        canonical_morphology_program_bytes(second),
    }

    assert {canonical_morphology_program_bytes(program) for program in sampled} <= admitted
    recombined = _program((4, 24, 18), (0, 0, 0))
    assert canonical_morphology_program_bytes(recombined) not in admitted


def test_joint_kernel_preserves_nearby_unobserved_complete_programs() -> None:
    inside = _program((4, 18, 12), (0, 0, 0))
    second_anchor = _program((6, 18, 12), (0, 0, 0))
    local_unobserved = _program((5, 18, 12), (0, 0, 0))
    outside = _program((12, 30, 24), (2, 1, 1))
    kernel = build_joint_program_kernel((inside, second_anchor))

    open_result = evaluate_program_mode(kernel, outside, mode="open")
    bounded_inside = evaluate_program_mode(kernel, inside, mode="bounded_joint_program")
    bounded_local = evaluate_program_mode(kernel, local_unobserved, mode="bounded_joint_program")
    bounded_outside = evaluate_program_mode(kernel, outside, mode="bounded_joint_program")

    assert open_result["schedule_admitted"] is True
    assert open_result["identity_multiplier"] == 1.0
    assert open_result["incremental_potential"] == 0.0
    assert bounded_inside["schedule_admitted"] is True
    assert bounded_local["schedule_admitted"] is True
    assert bounded_outside["schedule_admitted"] is False
    assert joint_program_neighborhood(kernel, local_unobserved) == ("local_smoothed_neighborhood")
    probabilities = joint_candidate_pool_probabilities(
        kernel, (inside, local_unobserved, second_anchor)
    )
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities > 0.0)
    assert bounded_inside["biological_guidance_active"] is False
    assert bounded_inside["synthesis_guidance_active"] is False


def test_frozen_production_result_preserves_measured_rows_and_abstention() -> None:
    path = Path("results/phase1/ugi_joint_program_support_audit_v2/result.json")
    if not path.is_file():
        return
    result = json.loads(path.read_text())

    assert result["measured_joint_prior"]["measured_rows"] == 1_100
    assert result["measured_joint_prior"]["measured_rows_inside_support"] == 1_100
    assert result["measured_joint_prior"]["measured_rows_outside_support"] == 0
    assert result["program_definition"]["role_marginal_recombination"] is False
    assert result["program_definition"]["exact_tuple_membership_role"] == (
        "strict_diagnostic_ablation"
    )
    assert result["valid_terminal_cross_tab"]["active_contrast_exact_plus_local"] == 32
    assert result["valid_terminal_cross_tab"]["all_three_new_action"] == "abstain"
    assert result["scope"]["nonzero_guidance_execution"] is False
    assert result["adjudication"]["nonzero_guidance_authorized"] is False
