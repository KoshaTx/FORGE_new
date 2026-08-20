from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from forge.design.flow.ugi_matched_synthesis_experiment import (
    ROLE_NAMES,
    ArmSeedOutcome,
    ComputeReceipt,
    ExperimentArm,
    NoSignalReceipt,
    UgiMatchedSynthesisExperimentError,
    adjudicate_evaluation,
    load_matched_synthesis_experiment_plan,
    select_calibration_guidance_strength,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_matched_synthesis_guidance_preregistration_v1.json"


def _plan():
    return load_matched_synthesis_experiment_plan(REPO, CONFIG)


def _no_signal(**changes: bool) -> NoSignalReceipt:
    values = {
        "zero_guidance_bitwise_identity": True,
        "uniform_effective_law_identity_without_rng": True,
        "nonuniform_effective_law_resamples": True,
        "equal_guidance_preserves_nonuniform_base_weights": True,
        "morphology_program_fields_identical": True,
    }
    values.update(changes)
    return NoSignalReceipt(**values)


def _outcome(
    arm: ExperimentArm,
    seed: int,
    *,
    strength: float = 0.0,
    primary: int = 2,
    validity: float = 0.95,
    abstain: bool = True,
) -> ArmSeedOutcome:
    budget = _plan().budget_for(arm)
    return ArmSeedOutcome(
        arm=arm,
        seed=seed,
        guidance_strength=strength,
        selected_candidates=budget.final_candidates,
        primary_route_closed_beyond_catalog_count=primary,
        validity_fraction=validity,
        exact_l1_fraction=0.9,
        uniqueness_fraction=0.95,
        internal_diversity=0.8,
        broad_distribution_coverage=0.8,
        beyond_catalog_fraction=0.5,
        role_effective_component_counts=tuple((role, 20.0) for role in ROLE_NAMES),
        role_max_component_fractions=tuple((role, 0.1) for role in ROLE_NAMES),
        oracle_assessed_fraction=1.0,
        oracle_all_actions_abstain=abstain,
        oracle_guidance_scores_all_null=True,
        compute=ComputeReceipt(**budget.__dict__),
        sealed_ledger_sha256=hashlib.sha256(f"{arm.value}:{seed}:{strength}".encode()).hexdigest(),
    )


def _changed_config(tmp_path: Path, mutate) -> Path:
    value = json.loads(CONFIG.read_text())
    mutate(value)
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def test_frozen_plan_authenticates_and_derives_matched_schedule_budget() -> None:
    first = _plan()
    second = _plan()

    assert first.to_dict() == second.to_dict()
    assert first.plan_sha256 == second.plan_sha256
    assert first.production_execution_authorized is False
    assert first.prerequisite_gates_passed is False
    assert first.blocker_expected_sha256 == (
        "dba5b11067513e4f431f78f3b9a316dfe0d58c5f046210c170c55b9a0e241e3c"
    )
    assert first.blocker_observed_sha256 == (
        "6e77063f45012144912883f16a12698eda809446635c94d59d1ab6211993f189"
    )
    assert first.budget_for(ExperimentArm.IN_TRAJECTORY_SMC).product_transition_calls == (
        64 * 8 + 64 * ((8 - 2) + (8 - 4) + (8 - 6))
    )
    assert first.budget_for(ExperimentArm.IN_TRAJECTORY_SMC).terminal_completions == 256
    assert first.budget_for(ExperimentArm.FINITE_LIBRARY).product_transition_calls == 0
    assert len(first.evaluation_seeds) == 5
    assert first.exact_sign_test_probability == 0.03125


def test_artifact_or_blocker_drift_fails_closed(tmp_path: Path) -> None:
    artifact = _changed_config(
        tmp_path,
        lambda value: value["artifacts"]["generator_checkpoint"].update({"sha256": "0" * 64}),
    )
    with pytest.raises(UgiMatchedSynthesisExperimentError, match="artifact generator"):
        load_matched_synthesis_experiment_plan(REPO, artifact)

    blocker = _changed_config(
        tmp_path,
        lambda value: value["prerequisites"]["blocker"].update({"observed_sha256": "0" * 64}),
    )
    with pytest.raises(UgiMatchedSynthesisExperimentError, match="not reproducible"):
        load_matched_synthesis_experiment_plan(REPO, blocker)


def test_budget_or_primary_endpoint_redefinition_fails_closed(tmp_path: Path) -> None:
    budget = _changed_config(
        tmp_path,
        lambda value: value["matched_compute"]["post_hoc_route_filtering"].update(
            {"terminal_completions": 255}
        ),
    )
    with pytest.raises(UgiMatchedSynthesisExperimentError, match="budgets must match"):
        load_matched_synthesis_experiment_plan(REPO, budget)

    endpoint = _changed_config(
        tmp_path,
        lambda value: value["primary_endpoint"].update(
            {"requires_all_three_roles_l2_l3_closed": False}
        ),
    )
    with pytest.raises(UgiMatchedSynthesisExperimentError, match="endpoint semantics"):
        load_matched_synthesis_experiment_plan(REPO, endpoint)


def _calibration_matrix() -> tuple[ArmSeedOutcome, ...]:
    plan = _plan()
    rows = []
    for seed in plan.calibration_seeds:
        rows.extend(
            (
                _outcome(ExperimentArm.UNGUIDED, seed, primary=2),
                _outcome(ExperimentArm.POST_HOC, seed, primary=3),
            )
        )
        for strength, primary in ((0.25, 4), (0.5, 5), (1.0, 5), (2.0, 4)):
            rows.append(
                _outcome(
                    ExperimentArm.IN_TRAJECTORY_SMC,
                    seed,
                    strength=strength,
                    primary=primary,
                )
            )
    return tuple(rows)


def test_calibration_selects_smallest_passing_maximizer() -> None:
    assert (
        select_calibration_guidance_strength(_plan(), _calibration_matrix(), no_signal=_no_signal())
        == 0.5
    )


def test_calibration_returns_none_for_no_signal_or_safeguard_failure() -> None:
    assert (
        select_calibration_guidance_strength(
            _plan(),
            _calibration_matrix(),
            no_signal=_no_signal(zero_guidance_bitwise_identity=False),
        )
        is None
    )
    failed = tuple(
        replace(row, validity_fraction=0.5) if row.arm is ExperimentArm.IN_TRAJECTORY_SMC else row
        for row in _calibration_matrix()
    )
    assert select_calibration_guidance_strength(_plan(), failed, no_signal=_no_signal()) is None


def _evaluation_matrix(*, tie_seed: int | None = None) -> tuple[ArmSeedOutcome, ...]:
    plan = _plan()
    rows = []
    for seed in plan.evaluation_seeds:
        post_hoc_primary = 3
        guided_primary = post_hoc_primary if seed == tie_seed else 4
        rows.extend(
            (
                _outcome(ExperimentArm.UNGUIDED, seed, primary=2),
                _outcome(ExperimentArm.POST_HOC, seed, primary=post_hoc_primary),
                _outcome(
                    ExperimentArm.IN_TRAJECTORY_SMC,
                    seed,
                    strength=0.5,
                    primary=guided_primary,
                ),
                _outcome(ExperimentArm.FINITE_LIBRARY, seed, primary=0),
            )
        )
    return tuple(rows)


def test_evaluation_promotes_only_when_every_paired_seed_wins() -> None:
    plan = _plan()
    promoted = adjudicate_evaluation(
        plan,
        _evaluation_matrix(),
        selected_guidance_strength=0.5,
        no_signal=_no_signal(),
        prerequisites_passed=True,
    )
    assert promoted.promoted is True
    assert [difference for _, difference in promoted.paired_primary_differences] == [1] * 5

    rejected = adjudicate_evaluation(
        plan,
        _evaluation_matrix(tie_seed=plan.evaluation_seeds[0]),
        selected_guidance_strength=0.5,
        no_signal=_no_signal(),
        prerequisites_passed=True,
    )
    assert rejected.promoted is False


def test_evaluation_rejects_incomplete_matrix_compute_drift_and_oracle_escape() -> None:
    plan = _plan()
    with pytest.raises(UgiMatchedSynthesisExperimentError, match="incomplete"):
        adjudicate_evaluation(
            plan,
            _evaluation_matrix()[:-1],
            selected_guidance_strength=0.5,
            no_signal=_no_signal(),
            prerequisites_passed=True,
        )

    rows = list(_evaluation_matrix())
    rows[0] = replace(
        rows[0],
        compute=replace(rows[0].compute, product_transition_calls=1279),
    )
    with pytest.raises(UgiMatchedSynthesisExperimentError, match="was not exact"):
        adjudicate_evaluation(
            plan,
            tuple(rows),
            selected_guidance_strength=0.5,
            no_signal=_no_signal(),
            prerequisites_passed=True,
        )

    rows = list(_evaluation_matrix())
    rows[0] = replace(rows[0], oracle_all_actions_abstain=False)
    with pytest.raises(UgiMatchedSynthesisExperimentError, match="escaped abstention"):
        adjudicate_evaluation(
            plan,
            tuple(rows),
            selected_guidance_strength=0.5,
            no_signal=_no_signal(),
            prerequisites_passed=True,
        )


def test_finite_library_cannot_claim_beyond_catalog_primary_endpoint() -> None:
    plan = _plan()
    rows = list(_evaluation_matrix())
    finite_index = next(
        index for index, row in enumerate(rows) if row.arm is ExperimentArm.FINITE_LIBRARY
    )
    rows[finite_index] = replace(rows[finite_index], primary_route_closed_beyond_catalog_count=1)
    with pytest.raises(UgiMatchedSynthesisExperimentError, match="cannot satisfy"):
        adjudicate_evaluation(
            plan,
            tuple(rows),
            selected_guidance_strength=0.5,
            no_signal=_no_signal(),
            prerequisites_passed=True,
        )
