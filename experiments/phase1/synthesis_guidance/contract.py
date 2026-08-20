"""Preregistered policy for the matched FORGE synthesis-guidance experiment.

This module authenticates a nonexecuting four-arm experiment plan and adjudicates
already-sealed aggregate outcomes.  It does not run the generator, planner,
oracle, selection procedure or private holdout.  In particular, it cannot turn
structured route assessments into an unstated scalar synthesis-success score.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from experiments._runtime.historical import HistoricalPinArchiveError, resolve_pinned_input
from forge.core.hashing import sha256_json as _sha256_payload

CONFIG_SCHEMA_VERSION = "phase1_ugi_matched_synthesis_guidance_preregistration.v1"
PLAN_SCHEMA_VERSION = "forge.ugi_matched_synthesis_guidance_plan.v1"
EXPECTED_SCOPE = "nonexecuting_preregistered_matched_synthesis_guidance_experiment"
ROLE_NAMES = (
    "amine_head",
    "oxoester_aldehyde_body_tail",
    "isocyanide_tail",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class UgiMatchedSynthesisExperimentError(RuntimeError):
    """Raised when the preregistration or a result set violates its contract."""


class ExperimentArm(str, Enum):
    """The four frozen experiment arms."""

    UNGUIDED = "unguided_product_prior"
    POST_HOC = "post_hoc_route_filtering"
    IN_TRAJECTORY_SMC = "in_trajectory_smc"
    FINITE_LIBRARY = "finite_library_baseline"


@dataclass(frozen=True)
class ArtifactPin:
    """One authenticated repository artifact."""

    name: str
    path: str
    sha256: str


@dataclass(frozen=True)
class ComputeBudget:
    """Per-arm, per-seed productive-compute ceilings."""

    product_transition_calls: int
    terminal_completions: int
    logical_planner_calls: int
    physical_planner_calls: int
    logical_verifier_calls: int
    physical_verifier_calls: int
    final_candidates: int
    wall_seconds: int
    gpu_device_seconds: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise UgiMatchedSynthesisExperimentError(
                    f"compute budget {name} must be a nonnegative integer"
                )
        if self.physical_planner_calls > self.logical_planner_calls:
            raise UgiMatchedSynthesisExperimentError(
                "physical planner ceiling cannot exceed logical planner ceiling"
            )
        if self.physical_verifier_calls > self.logical_verifier_calls:
            raise UgiMatchedSynthesisExperimentError(
                "physical verifier ceiling cannot exceed logical verifier ceiling"
            )


@dataclass(frozen=True)
class ComputeReceipt:
    """Realized per-arm, per-seed compute retained for adjudication."""

    product_transition_calls: int
    terminal_completions: int
    logical_planner_calls: int
    physical_planner_calls: int
    logical_verifier_calls: int
    physical_verifier_calls: int
    final_candidates: int
    wall_seconds: int
    gpu_device_seconds: int

    def __post_init__(self) -> None:
        ComputeBudget(**self.__dict__)

    def fits(self, budget: ComputeBudget) -> bool:
        return all(
            getattr(self, name) <= getattr(budget, name) for name in self.__dataclass_fields__
        )


@dataclass(frozen=True)
class SafeguardPolicy:
    """Predeclared floors preventing route-closure-by-collapse."""

    validity_absolute_drop: float
    exact_l1_absolute_drop: float
    uniqueness_ratio: float
    internal_diversity_ratio: float
    broad_coverage_ratio: float
    beyond_catalog_ratio: float
    role_effective_count_ratio: float
    role_max_concentration_absolute_increase: float
    oracle_assessed_fraction_floor: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise UgiMatchedSynthesisExperimentError(f"safeguard {name} must be numeric")
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise UgiMatchedSynthesisExperimentError(f"safeguard {name} must lie in [0, 1]")


@dataclass(frozen=True)
class FrozenMatchedSynthesisPlan:
    """Authenticated nonexecuting preregistration."""

    config_path: str
    config_sha256: str
    artifacts: tuple[ArtifactPin, ...]
    arms: tuple[ExperimentArm, ...]
    particles_per_seed: int
    sample_steps: int
    synthesis_checkpoints: tuple[int, ...]
    terminal_rollouts_per_particle_checkpoint: int
    guidance_strengths: tuple[float, ...]
    calibration_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    budgets: tuple[tuple[ExperimentArm, ComputeBudget], ...]
    safeguards: SafeguardPolicy
    all_evaluation_seed_wins_required: bool
    exact_sign_test_probability: float
    production_execution_authorized: bool
    prerequisite_gates_passed: bool
    blocker_id: str
    blocker_path: str
    blocker_expected_sha256: str
    blocker_observed_sha256: str

    def budget_for(self, arm: ExperimentArm) -> ComputeBudget:
        return dict(self.budgets)[arm]

    @property
    def plan_sha256(self) -> str:
        return _sha256_payload(self.to_dict(include_plan_sha256=False))

    def to_dict(self, *, include_plan_sha256: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "scope": EXPECTED_SCOPE,
            "config": {"path": self.config_path, "sha256": self.config_sha256},
            "artifacts": [artifact.__dict__ for artifact in self.artifacts],
            "arms": [arm.value for arm in self.arms],
            "design": {
                "particles_per_seed": self.particles_per_seed,
                "sample_steps": self.sample_steps,
                "synthesis_checkpoints": list(self.synthesis_checkpoints),
                "terminal_rollouts_per_particle_checkpoint": (
                    self.terminal_rollouts_per_particle_checkpoint
                ),
                "guidance_strengths": list(self.guidance_strengths),
                "calibration_seeds": list(self.calibration_seeds),
                "evaluation_seeds": list(self.evaluation_seeds),
                "no_retries": True,
                "no_seed_search": True,
            },
            "budgets": {arm.value: budget.__dict__ for arm, budget in self.budgets},
            "safeguards": self.safeguards.__dict__,
            "promotion": {
                "all_evaluation_seed_wins_required": (self.all_evaluation_seed_wins_required),
                "exact_sign_test_probability": self.exact_sign_test_probability,
                "ties_or_losses_fail_promotion": True,
                "aggregate_primary_difference_must_be_positive": True,
            },
            "prerequisites": {
                "all_passed": self.prerequisite_gates_passed,
                "blocker": {
                    "id": self.blocker_id,
                    "path": self.blocker_path,
                    "expected_sha256": self.blocker_expected_sha256,
                    "observed_sha256": self.blocker_observed_sha256,
                },
            },
            "scope_guards": {
                "production_execution_authorized_by_this_config": (
                    self.production_execution_authorized
                ),
                "biological_guidance": False,
                "private_holdout_access": False,
                "synthesis_success_probability_defined": False,
                "route_likelihood_used_as_success_probability": False,
            },
        }
        if include_plan_sha256:
            value["plan_sha256"] = self.plan_sha256
        return value


@dataclass(frozen=True)
class NoSignalReceipt:
    """Required no-signal identity and sensitivity checks."""

    zero_guidance_bitwise_identity: bool
    uniform_effective_law_identity_without_rng: bool
    nonuniform_effective_law_resamples: bool
    equal_guidance_preserves_nonuniform_base_weights: bool
    morphology_program_fields_identical: bool

    @property
    def passed(self) -> bool:
        return all(getattr(self, name) is True for name in self.__dataclass_fields__)


@dataclass(frozen=True)
class ArmSeedOutcome:
    """One sealed aggregate outcome; no candidate structure is exposed here."""

    arm: ExperimentArm
    seed: int
    guidance_strength: float
    selected_candidates: int
    primary_route_closed_beyond_catalog_count: int
    validity_fraction: float
    exact_l1_fraction: float
    uniqueness_fraction: float
    internal_diversity: float
    broad_distribution_coverage: float
    beyond_catalog_fraction: float
    role_effective_component_counts: tuple[tuple[str, float], ...]
    role_max_component_fractions: tuple[tuple[str, float], ...]
    oracle_assessed_fraction: float
    oracle_all_actions_abstain: bool
    oracle_guidance_scores_all_null: bool
    compute: ComputeReceipt
    sealed_ledger_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.arm, ExperimentArm):
            raise UgiMatchedSynthesisExperimentError("outcome arm is unsupported")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise UgiMatchedSynthesisExperimentError("outcome seed is invalid")
        if not math.isfinite(self.guidance_strength) or self.guidance_strength < 0:
            raise UgiMatchedSynthesisExperimentError("guidance strength is invalid")
        for name in (
            "selected_candidates",
            "primary_route_closed_beyond_catalog_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise UgiMatchedSynthesisExperimentError(f"{name} is invalid")
        if self.primary_route_closed_beyond_catalog_count > self.selected_candidates:
            raise UgiMatchedSynthesisExperimentError(
                "primary endpoint cannot exceed selected candidates"
            )
        for name in (
            "validity_fraction",
            "exact_l1_fraction",
            "uniqueness_fraction",
            "internal_diversity",
            "broad_distribution_coverage",
            "beyond_catalog_fraction",
            "oracle_assessed_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise UgiMatchedSynthesisExperimentError(f"{name} must lie in [0, 1]")
        _validate_role_metrics(
            self.role_effective_component_counts,
            label="role effective component counts",
            unit_interval=False,
        )
        _validate_role_metrics(
            self.role_max_component_fractions,
            label="role maximum component fractions",
            unit_interval=True,
        )
        if not isinstance(self.oracle_all_actions_abstain, bool) or not isinstance(
            self.oracle_guidance_scores_all_null, bool
        ):
            raise UgiMatchedSynthesisExperimentError("oracle policy fields must be boolean")
        _require_sha256(self.sealed_ledger_sha256, label="sealed outcome ledger")


@dataclass(frozen=True)
class ExperimentDecision:
    """Fail-closed preregistered promotion decision."""

    promoted: bool
    selected_guidance_strength: float | None
    reason: str
    paired_primary_differences: tuple[tuple[int, int], ...]


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise UgiMatchedSynthesisExperimentError(f"{label} is not a lowercase SHA-256")
    return value


def _require_exact_fields(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise UgiMatchedSynthesisExperimentError(f"{label} has an unsupported field set")
    return value


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise UgiMatchedSynthesisExperimentError(
            f"{label} must be an integer no smaller than {minimum}"
        )
    return value


def _require_float(value: Any, *, label: str, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UgiMatchedSynthesisExperimentError(f"{label} must be numeric")
    output = float(value)
    if not math.isfinite(output) or output < minimum:
        raise UgiMatchedSynthesisExperimentError(f"{label} is outside support")
    return output


def _validate_role_metrics(
    value: tuple[tuple[str, float], ...], *, label: str, unit_interval: bool
) -> None:
    if not isinstance(value, tuple) or tuple(role for role, _ in value) != ROLE_NAMES:
        raise UgiMatchedSynthesisExperimentError(f"{label} must retain all roles in order")
    for _, metric in value:
        if not isinstance(metric, (int, float)) or isinstance(metric, bool):
            raise UgiMatchedSynthesisExperimentError(f"{label} must be numeric")
        if not math.isfinite(float(metric)) or metric < 0 or (unit_interval and metric > 1):
            raise UgiMatchedSynthesisExperimentError(f"{label} is outside support")


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise UgiMatchedSynthesisExperimentError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiMatchedSynthesisExperimentError(f"cannot load preregistration: {path}") from error
    if not isinstance(value, dict):
        raise UgiMatchedSynthesisExperimentError("preregistration must contain an object")
    return value


def _read_repo_artifact(
    repo_root: Path,
    relative: Any,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[str, bytes]:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise UgiMatchedSynthesisExperimentError(f"{label} path is invalid")
    try:
        resolved = resolve_pinned_input(repo_root, relative, expected_sha256)
        raw = resolved.read_bytes()
    except (OSError, HistoricalPinArchiveError) as error:
        raise UgiMatchedSynthesisExperimentError(
            f"{label} is missing or outside the repository"
        ) from error
    return relative, raw


def _parse_compute(value: Any, *, label: str) -> ComputeBudget:
    fields = set(ComputeBudget.__dataclass_fields__)
    record = _require_exact_fields(value, fields, label=label)
    return ComputeBudget(
        **{name: _require_int(record[name], label=f"{label}.{name}") for name in fields}
    )


def load_matched_synthesis_experiment_plan(
    repo_root: Path,
    config_path: Path,
) -> FrozenMatchedSynthesisPlan:
    """Authenticate and load the frozen nonexecuting experiment plan."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    value = _load_json(config_path)
    _require_exact_fields(
        value,
        {
            "schema_version",
            "scope",
            "arms",
            "design",
            "matched_compute",
            "primary_endpoint",
            "safeguards",
            "oracle_policy",
            "promotion",
            "prerequisites",
            "artifacts",
            "scope_guards",
        },
        label="preregistration",
    )
    if value["schema_version"] != CONFIG_SCHEMA_VERSION or value["scope"] != EXPECTED_SCOPE:
        raise UgiMatchedSynthesisExperimentError("unsupported preregistration identity")

    expected_arms = tuple(ExperimentArm)
    if tuple(value["arms"]) != tuple(arm.value for arm in expected_arms):
        raise UgiMatchedSynthesisExperimentError("the four-arm order changed")

    design = _require_exact_fields(
        value["design"],
        {
            "particles_per_seed",
            "sample_steps",
            "synthesis_checkpoints",
            "terminal_rollouts_per_particle_checkpoint",
            "guidance_strengths",
            "calibration_seeds",
            "evaluation_seeds",
            "no_retries",
            "no_seed_search",
        },
        label="design",
    )
    if design["no_retries"] is not True or design["no_seed_search"] is not True:
        raise UgiMatchedSynthesisExperimentError("retry or seed-search policy changed")
    particles = _require_int(design["particles_per_seed"], label="particles", minimum=1)
    sample_steps = _require_int(design["sample_steps"], label="sample steps", minimum=1)
    checkpoints = tuple(design["synthesis_checkpoints"])
    if (
        not checkpoints
        or any(isinstance(item, bool) or not isinstance(item, int) for item in checkpoints)
        or tuple(sorted(set(checkpoints))) != checkpoints
        or checkpoints[0] <= 0
        or checkpoints[-1] >= sample_steps
    ):
        raise UgiMatchedSynthesisExperimentError("synthesis checkpoints are invalid")
    rollouts = _require_int(
        design["terminal_rollouts_per_particle_checkpoint"],
        label="terminal rollouts",
        minimum=1,
    )
    strengths = tuple(
        _require_float(item, label="guidance strength") for item in design["guidance_strengths"]
    )
    if strengths != (0.0, 0.25, 0.5, 1.0, 2.0):
        raise UgiMatchedSynthesisExperimentError("guidance-strength sweep changed")
    calibration_seeds = _parse_seeds(design["calibration_seeds"], label="calibration seeds")
    evaluation_seeds = _parse_seeds(design["evaluation_seeds"], label="evaluation seeds")
    if set(calibration_seeds) & set(evaluation_seeds):
        raise UgiMatchedSynthesisExperimentError("calibration and evaluation seeds overlap")

    matched_compute = _require_exact_fields(
        value["matched_compute"],
        {arm.value for arm in expected_arms} | {"matching_policy"},
        label="matched compute",
    )
    budgets = tuple(
        (arm, _parse_compute(matched_compute[arm.value], label=f"budget {arm.value}"))
        for arm in expected_arms
    )
    main_budget = dict(budgets)[ExperimentArm.IN_TRAJECTORY_SMC]
    for arm in (ExperimentArm.UNGUIDED, ExperimentArm.POST_HOC):
        if dict(budgets)[arm] != main_budget:
            raise UgiMatchedSynthesisExperimentError(
                "unguided, post-hoc and SMC budgets must match exactly"
            )
    expected_completions = particles * (len(checkpoints) * rollouts + 1)
    expected_transitions = particles * sample_steps + particles * rollouts * sum(
        sample_steps - checkpoint for checkpoint in checkpoints
    )
    if (
        main_budget.terminal_completions != expected_completions
        or main_budget.product_transition_calls != expected_transitions
    ):
        raise UgiMatchedSynthesisExperimentError("budget does not derive from the frozen schedule")
    policy = matched_compute["matching_policy"]
    if policy != {
        "generator_and_logical_route_ceilings_exact_for_three_generator_arms": True,
        "physical_calls_wall_and_gpu_reported_and_capped_separately": True,
        "finite_library_excluded_from_causal_compute_match": True,
    }:
        raise UgiMatchedSynthesisExperimentError("matched-compute policy changed")

    primary = _require_exact_fields(
        value["primary_endpoint"],
        {
            "name",
            "deduplication",
            "requires_exact_l1",
            "requires_all_three_roles_l2_l3_closed",
            "requires_current_terminal_evidence",
            "requires_component_outside_frozen_catalog",
            "finite_library_endpoint_is_zero_by_definition",
            "nonclosing_evidence_classes",
        },
        label="primary endpoint",
    )
    if primary != {
        "name": "unique_fully_route_closed_beyond_catalog_candidates",
        "deduplication": "canonical_constitutional_product_identity",
        "requires_exact_l1": True,
        "requires_all_three_roles_l2_l3_closed": True,
        "requires_current_terminal_evidence": True,
        "requires_component_outside_frozen_catalog": True,
        "finite_library_endpoint_is_zero_by_definition": True,
        "nonclosing_evidence_classes": [
            "handle_qualified",
            "family_projected",
            "analogue_only",
            "route_model_likelihood",
        ],
    }:
        raise UgiMatchedSynthesisExperimentError("primary endpoint semantics changed")

    safeguards_raw = _require_exact_fields(
        value["safeguards"], set(SafeguardPolicy.__dataclass_fields__), label="safeguards"
    )
    safeguards = SafeguardPolicy(
        **{name: _require_float(safeguards_raw[name], label=name) for name in safeguards_raw}
    )
    oracle_policy = value["oracle_policy"]
    if oracle_policy != {
        "descriptive_only": True,
        "all_actions_must_abstain": True,
        "all_guidance_scores_must_be_null": True,
        "biological_ranking_or_promotion_forbidden": True,
    }:
        raise UgiMatchedSynthesisExperimentError("oracle abstention policy changed")

    promotion = _require_exact_fields(
        value["promotion"],
        {
            "calibration_selects_smallest_passing_maximizer",
            "all_evaluation_seed_wins_required",
            "exact_sign_test_probability",
            "ties_or_losses_fail_promotion",
            "aggregate_primary_difference_must_be_positive",
            "no_retry_after_failure",
        },
        label="promotion",
    )
    expected_sign_probability = 0.5 ** len(evaluation_seeds)
    if (
        promotion["calibration_selects_smallest_passing_maximizer"] is not True
        or promotion["all_evaluation_seed_wins_required"] is not True
        or promotion["ties_or_losses_fail_promotion"] is not True
        or promotion["aggregate_primary_difference_must_be_positive"] is not True
        or promotion["no_retry_after_failure"] is not True
        or float(promotion["exact_sign_test_probability"]) != expected_sign_probability
    ):
        raise UgiMatchedSynthesisExperimentError("promotion rule changed")

    prerequisites = _require_exact_fields(
        value["prerequisites"],
        {
            "restartable_v2_passed",
            "production_zero_guidance_v2_passed",
            "scalar_value_policy_frozen",
            "current_l3_snapshot_frozen",
            "all_passed",
            "blocker",
        },
        label="prerequisites",
    )
    blocker = _require_exact_fields(
        prerequisites["blocker"],
        {"id", "path", "expected_sha256", "observed_sha256", "cause_status"},
        label="prerequisite blocker",
    )
    if prerequisites != {
        "restartable_v2_passed": True,
        "production_zero_guidance_v2_passed": False,
        "scalar_value_policy_frozen": False,
        "current_l3_snapshot_frozen": False,
        "all_passed": False,
        "blocker": blocker,
    }:
        raise UgiMatchedSynthesisExperimentError("prerequisite state is unsupported")
    blocker_expected = _require_sha256(blocker["expected_sha256"], label="blocker expected")
    blocker_observed = _require_sha256(blocker["observed_sha256"], label="blocker observed")
    try:
        blocker_relative, blocker_bytes = _read_repo_artifact(
            repo_root,
            blocker["path"],
            expected_sha256=blocker_observed,
            label="prerequisite blocker",
        )
    except UgiMatchedSynthesisExperimentError as error:
        raise UgiMatchedSynthesisExperimentError(
            "recorded prerequisite drift is not reproducible"
        ) from error
    observed_blocker_sha = hashlib.sha256(blocker_bytes).hexdigest()
    if observed_blocker_sha != blocker_observed or blocker_expected == blocker_observed:
        raise UgiMatchedSynthesisExperimentError("recorded prerequisite drift is not reproducible")
    if blocker["cause_status"] != ("structured_pareto_value_source_supersession_pending_replay"):
        raise UgiMatchedSynthesisExperimentError("blocker cause cannot be overstated")

    artifacts_raw = value["artifacts"]
    if not isinstance(artifacts_raw, dict) or not artifacts_raw:
        raise UgiMatchedSynthesisExperimentError("artifact pins are missing")
    artifacts = []
    for name in sorted(artifacts_raw):
        record = _require_exact_fields(
            artifacts_raw[name], {"path", "sha256"}, label=f"artifact {name}"
        )
        digest = _require_sha256(record["sha256"], label=f"artifact {name}")
        relative, raw = _read_repo_artifact(
            repo_root,
            record["path"],
            expected_sha256=digest,
            label=f"artifact {name}",
        )
        if hashlib.sha256(raw).hexdigest() != digest:
            raise UgiMatchedSynthesisExperimentError(f"artifact {name} SHA-256 mismatch")
        artifacts.append(ArtifactPin(name=name, path=relative, sha256=digest))

    guards = value["scope_guards"]
    expected_guards = {
        "production_execution_authorized_by_this_config": False,
        "biological_guidance": False,
        "private_holdout_access": False,
        "synthesis_success_probability_defined": False,
        "route_likelihood_used_as_success_probability": False,
    }
    if guards != expected_guards:
        raise UgiMatchedSynthesisExperimentError("scope guards changed")

    return FrozenMatchedSynthesisPlan(
        config_path=str(config_path.relative_to(repo_root)),
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        artifacts=tuple(artifacts),
        arms=expected_arms,
        particles_per_seed=particles,
        sample_steps=sample_steps,
        synthesis_checkpoints=checkpoints,
        terminal_rollouts_per_particle_checkpoint=rollouts,
        guidance_strengths=strengths,
        calibration_seeds=calibration_seeds,
        evaluation_seeds=evaluation_seeds,
        budgets=budgets,
        safeguards=safeguards,
        all_evaluation_seed_wins_required=True,
        exact_sign_test_probability=expected_sign_probability,
        production_execution_authorized=False,
        prerequisite_gates_passed=False,
        blocker_id=str(blocker["id"]),
        blocker_path=blocker_relative,
        blocker_expected_sha256=blocker_expected,
        blocker_observed_sha256=blocker_observed,
    )


def _parse_seeds(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise UgiMatchedSynthesisExperimentError(f"{label} must be nonempty")
    seeds = tuple(_require_int(seed, label=label) for seed in value)
    if len(seeds) != len(set(seeds)):
        raise UgiMatchedSynthesisExperimentError(f"{label} contain duplicates")
    return seeds


def _safeguards_pass(
    guided: ArmSeedOutcome,
    unguided: ArmSeedOutcome,
    policy: SafeguardPolicy,
) -> bool:
    if guided.validity_fraction < unguided.validity_fraction - policy.validity_absolute_drop:
        return False
    if guided.exact_l1_fraction < unguided.exact_l1_fraction - policy.exact_l1_absolute_drop:
        return False
    for field, ratio in (
        ("uniqueness_fraction", policy.uniqueness_ratio),
        ("internal_diversity", policy.internal_diversity_ratio),
        ("broad_distribution_coverage", policy.broad_coverage_ratio),
        ("beyond_catalog_fraction", policy.beyond_catalog_ratio),
    ):
        if getattr(guided, field) < ratio * getattr(unguided, field):
            return False
    guided_effective = dict(guided.role_effective_component_counts)
    unguided_effective = dict(unguided.role_effective_component_counts)
    guided_max = dict(guided.role_max_component_fractions)
    unguided_max = dict(unguided.role_max_component_fractions)
    for role in ROLE_NAMES:
        if guided_effective[role] < policy.role_effective_count_ratio * unguided_effective[role]:
            return False
        if guided_max[role] > (
            unguided_max[role] + policy.role_max_concentration_absolute_increase
        ):
            return False
    return (
        guided.oracle_assessed_fraction >= policy.oracle_assessed_fraction_floor
        and guided.oracle_all_actions_abstain
        and guided.oracle_guidance_scores_all_null
    )


def _validate_outcome(
    outcome: ArmSeedOutcome,
    *,
    plan: FrozenMatchedSynthesisPlan,
) -> None:
    if outcome.selected_candidates != plan.budget_for(outcome.arm).final_candidates:
        raise UgiMatchedSynthesisExperimentError("final candidate count drifted")
    if not outcome.compute.fits(plan.budget_for(outcome.arm)):
        raise UgiMatchedSynthesisExperimentError("realized compute exceeded its frozen ceiling")
    if (
        outcome.compute.product_transition_calls
        != plan.budget_for(outcome.arm).product_transition_calls
        or outcome.compute.terminal_completions != plan.budget_for(outcome.arm).terminal_completions
        or outcome.compute.final_candidates != plan.budget_for(outcome.arm).final_candidates
    ):
        raise UgiMatchedSynthesisExperimentError("productive or candidate compute was not exact")
    if outcome.arm is ExperimentArm.FINITE_LIBRARY:
        if outcome.primary_route_closed_beyond_catalog_count != 0:
            raise UgiMatchedSynthesisExperimentError(
                "finite-library baseline cannot satisfy the beyond-catalog endpoint"
            )
    elif outcome.arm is not ExperimentArm.IN_TRAJECTORY_SMC and outcome.guidance_strength != 0:
        raise UgiMatchedSynthesisExperimentError("only SMC can have nonzero guidance")
    if not outcome.oracle_all_actions_abstain or not outcome.oracle_guidance_scores_all_null:
        raise UgiMatchedSynthesisExperimentError("biological oracle escaped abstention")


def select_calibration_guidance_strength(
    plan: FrozenMatchedSynthesisPlan,
    outcomes: tuple[ArmSeedOutcome, ...],
    *,
    no_signal: NoSignalReceipt,
) -> float | None:
    """Select the smallest passing maximizer without using evaluation seeds."""

    if not no_signal.passed:
        return None
    by_key: dict[tuple[ExperimentArm, int, float], ArmSeedOutcome] = {}
    for outcome in outcomes:
        _validate_outcome(outcome, plan=plan)
        key = (outcome.arm, outcome.seed, outcome.guidance_strength)
        if key in by_key:
            raise UgiMatchedSynthesisExperimentError("duplicate calibration outcome")
        if outcome.seed not in plan.calibration_seeds:
            raise UgiMatchedSynthesisExperimentError(
                "evaluation or unknown seed used in calibration"
            )
        by_key[key] = outcome
    candidates = []
    for strength in plan.guidance_strengths[1:]:
        aggregate = 0
        passing = True
        for seed in plan.calibration_seeds:
            guided = by_key.get((ExperimentArm.IN_TRAJECTORY_SMC, seed, strength))
            post_hoc = by_key.get((ExperimentArm.POST_HOC, seed, 0.0))
            unguided = by_key.get((ExperimentArm.UNGUIDED, seed, 0.0))
            if guided is None or post_hoc is None or unguided is None:
                raise UgiMatchedSynthesisExperimentError("calibration matrix is incomplete")
            if not _safeguards_pass(guided, unguided, plan.safeguards):
                passing = False
            aggregate += (
                guided.primary_route_closed_beyond_catalog_count
                - post_hoc.primary_route_closed_beyond_catalog_count
            )
        if passing and aggregate > 0:
            candidates.append((aggregate, strength))
    if not candidates:
        return None
    maximum = max(aggregate for aggregate, _ in candidates)
    return min(strength for aggregate, strength in candidates if aggregate == maximum)


def adjudicate_evaluation(
    plan: FrozenMatchedSynthesisPlan,
    outcomes: tuple[ArmSeedOutcome, ...],
    *,
    selected_guidance_strength: float | None,
    no_signal: NoSignalReceipt,
    prerequisites_passed: bool,
) -> ExperimentDecision:
    """Adjudicate sealed evaluation outcomes under the frozen all-seed rule."""

    if selected_guidance_strength not in plan.guidance_strengths[1:]:
        return ExperimentDecision(False, selected_guidance_strength, "no calibrated strength", ())
    if not prerequisites_passed or not no_signal.passed:
        return ExperimentDecision(False, selected_guidance_strength, "preflight gate failed", ())
    by_key: dict[tuple[ExperimentArm, int], ArmSeedOutcome] = {}
    for outcome in outcomes:
        _validate_outcome(outcome, plan=plan)
        if outcome.seed not in plan.evaluation_seeds:
            raise UgiMatchedSynthesisExperimentError(
                "calibration or unknown seed used in evaluation"
            )
        if outcome.arm is ExperimentArm.IN_TRAJECTORY_SMC:
            if outcome.guidance_strength != selected_guidance_strength:
                raise UgiMatchedSynthesisExperimentError("evaluation guidance strength drifted")
        elif outcome.guidance_strength != 0:
            raise UgiMatchedSynthesisExperimentError("non-SMC evaluation guidance is nonzero")
        key = (outcome.arm, outcome.seed)
        if key in by_key:
            raise UgiMatchedSynthesisExperimentError("duplicate evaluation outcome")
        by_key[key] = outcome
    expected = {(arm, seed) for arm in ExperimentArm for seed in plan.evaluation_seeds}
    if set(by_key) != expected:
        raise UgiMatchedSynthesisExperimentError("four-arm evaluation matrix is incomplete")

    differences = []
    safeguards_ok = True
    for seed in plan.evaluation_seeds:
        guided = by_key[(ExperimentArm.IN_TRAJECTORY_SMC, seed)]
        post_hoc = by_key[(ExperimentArm.POST_HOC, seed)]
        unguided = by_key[(ExperimentArm.UNGUIDED, seed)]
        safeguards_ok &= _safeguards_pass(guided, unguided, plan.safeguards)
        differences.append(
            (
                seed,
                guided.primary_route_closed_beyond_catalog_count
                - post_hoc.primary_route_closed_beyond_catalog_count,
            )
        )
    every_seed_wins = all(difference > 0 for _, difference in differences)
    aggregate_positive = sum(difference for _, difference in differences) > 0
    promoted = safeguards_ok and every_seed_wins and aggregate_positive
    reason = (
        "promoted: all paired evaluation seeds won and safeguards passed"
        if promoted
        else "not promoted: tie/loss or safeguard failure"
    )
    return ExperimentDecision(
        promoted,
        selected_guidance_strength,
        reason,
        tuple(differences),
    )
