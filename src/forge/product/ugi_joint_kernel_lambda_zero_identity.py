"""All-seed lambda-zero qualification for bounded joint Ugi programs.

The module executes the already-frozen selected-v3 restartable generator under
the eight bounded whole-program schedules.  It delegates paired identity to
the existing lambda-zero state machine and observes one arm only after that
gate returns successfully.  Observed terminals are then classified by the
frozen version-3 four-view structural applicability thresholds and exact role
policy.  No potency model, synthesis model, route planner, proposal model, or
candidate selector is constructed or called.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from forge.bio import ugi_distributional_applicability as applicability_v1
from forge.bio import ugi_distributional_applicability_v2 as applicability_v2
from forge.bio.ugi_generated_role_applicability_census import (
    EXPECTED_SUPPORTED_PATTERNS,
    ROLE_TO_SAMPLE_ROLE,
    ROLES,
    VIEWS,
)
from forge.core.hashing import sha256_json as _sha256_payload
from forge.data.r1_prime_audit import sha256_bytes, sha256_file
from forge.product.ugi_hela_potency_guidance_seam_v1 import (
    CALIBRATION_SEEDS,
    EVALUATION_SEEDS,
    EXPECTED_SEEDS,
    run_lambda_zero_identity_gate,
)
from forge.product.ugi_nonzero_guidance_runner import (
    FrozenSeedProgramAssignment,
    GroupedSMCScheduleQualification,
    GuidanceSchedule,
    GuidanceStateReceipt,
    GuidanceTerminalCompletionReceipt,
    ParticleGroupDesign,
    RestartableGuidanceLane,
    load_grouped_smc_schedule_qualification,
)
from forge.product.ugi_restartable_terminal_support_adapter import (
    native_completion_record_from_locked_terminal,
)
from forge.product.ugi_selected_guidance_adapter_v3 import (
    build_selected_model_restartable_guidance_lane_v3,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi_joint_kernel_lambda_zero_identity_config.v1"
SEED_RESULT_SCHEMA_VERSION = "phase1_ugi_joint_kernel_lambda_zero_seed.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_joint_kernel_lambda_zero_identity.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_joint_kernel_lambda_zero_terminal_ledger.v1"

EXPECTED_STATUS = "frozen_before_all_eight_lambda_zero_identity"
EXPECTED_SCHEDULE_STATUS = "grouped_smc_schedule_qualified_nonexecuting"
EXPECTED_SCHEDULE_RESULT_SCHEMA = "phase1_ugi_grouped_smc_schedule_qualification.v1"
EXPECTED_APPLICABILITY_STATUS = "complete_component_shift_calibrated_audit_guidance_still_abstained"
EXPECTED_APPLICABILITY_SCHEMA = "phase1_ugi_distributional_applicability.v3"
EXPECTED_ROLE_POLICY_SCHEMA = "phase1_ugi_generated_role_applicability_census_policy.v1"

EXPECTED_SCOPE = {
    "all_three_component_graphs_generated": True,
    "applicability_gate_only": True,
    "biological_guidance": False,
    "biological_target_values_used": False,
    "candidate_selection": False,
    "generator_trajectories_advanced": True,
    "lambda_zero_only": True,
    "nonzero_guidance": False,
    "oracle_calls": 0,
    "potency_predictions_consumed": False,
    "proposal_calls": 0,
    "prospective_candidate_lock": False,
    "route_evaluation": False,
    "sealed_holdout_access": False,
    "synthesis_calls": 0,
}

EXPECTED_POLICY = {
    "assignment_seeds": list(EXPECTED_SEEDS),
    "calibration_seeds": list(CALIBRATION_SEEDS),
    "evaluation_seeds": list(EVALUATION_SEEDS),
    "programs_per_seed": 16,
    "particles_per_program": 4,
    "sample_steps": 8,
    "checkpoints": [2, 4, 6],
    "checkpoint_betas": [0.25, 0.5, 0.75],
    "rollouts_per_particle_checkpoint": 1,
    "device": "cpu",
    "historical_old_schedule_manifest_comparison": False,
    "seed_search_retry_or_exclusion_allowed": False,
    "all_eight_identity_required_before_nonzero": True,
    "all_four_views_must_be_interpolative": True,
    "exact_measured_combination_action": "neutral",
    "unsupported_role_pattern_action": "abstain",
    "all_three_new_action": "abstain",
    "supported_no_score_patterns": {
        "amine": "amine_only",
        "aldehyde+isocyanide": "aldehyde_isocyanide_pair",
    },
}

EXPECTED_INPUTS = {
    "applicability_v3_result",
    "bio_policy",
    "curated_agile_structures",
    "lambda_zero_seam_source",
    "role_policy",
    "runner",
    "schedule_config",
    "schedule_result",
    "schedule_source",
    "selected_v3_adapter_source",
    "selected_v3_equivalence_result",
    "selected_v3_equivalence_rows",
    "source",
    "tests",
}

ROLE_FIELDS = {
    "amine": "A_smiles",
    "aldehyde": "B_smiles",
    "isocyanide": "C_smiles",
}
GENERATED_FIELDS = {
    "amine": "amine_smiles",
    "aldehyde": "aldehyde_smiles",
    "isocyanide": "isocyanide_smiles",
}

LEDGER_FIELDS = (
    "seed",
    "seed_partition",
    "checkpoint",
    "checkpoint_kind",
    "particle_index",
    "program_index",
    "within_program_particle_index",
    "invocation_seed",
    "schedule_sha256",
    "morphology_program_sha256",
    "bounded_neighborhood",
    "branch_class",
    "terminal_present",
    "terminal_valid",
    "exact_l1",
    "terminal_id",
    "terminal_sha256",
    "generation_trace_sha256",
    "completion_error",
    "exact_measured_combination",
    "exact_unseen_roles_json",
    "policy_pattern_id",
    "applicability_action",
    "applicability_reason",
    "overall_distribution_bin",
    "noninterpolative_views_json",
    "extrapolative_views_json",
    *(
        field
        for view in VIEWS
        for field in (
            f"{view}_distribution_bin",
            f"{view}_fingerprint_distance",
            f"{view}_descriptor_distance",
            f"{view}_interpolative_radius_ratio",
            f"{view}_boundary_radius_ratio",
        )
    ),
)


class UgiJointKernelLambdaZeroError(RuntimeError):
    """Raised when bounded-schedule identity or structural gating fails closed."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiJointKernelLambdaZeroError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiJointKernelLambdaZeroError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiJointKernelLambdaZeroError(f"{label} pin is malformed")
    path = (repo / str(record["path"])).resolve()
    try:
        relative = path.relative_to(repo)
    except ValueError as error:
        raise UgiJointKernelLambdaZeroError(f"{label} path escapes repository") from error
    lowered = "/".join(relative.parts).lower()
    if "holdout" in lowered or "sealed" in lowered:
        raise UgiJointKernelLambdaZeroError(f"{label} path is forbidden")
    if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
        raise UgiJointKernelLambdaZeroError(f"{label} hash changed")
    return path


def _logical_result_hash(value: Mapping[str, Any], *, label: str) -> str:
    claimed = value.get("result_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise UgiJointKernelLambdaZeroError(f"{label} result hash is malformed")
    content = {key: item for key, item in value.items() if key != "result_sha256"}
    if _sha256_payload(content) != claimed:
        raise UgiJointKernelLambdaZeroError(f"{label} result hash changed")
    return claimed


def _read_structural_rows(path: Path) -> list[dict[str, str]]:
    required = {"label", "model_smiles", *ROLE_FIELDS.values()}
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise UgiJointKernelLambdaZeroError("curated AGILE structural fields changed")
            rows = [{field: str(row[field]) for field in required} for row in reader]
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise UgiJointKernelLambdaZeroError("curated AGILE structures are invalid") from error
    if len(rows) != 1100 or len({row["label"] for row in rows}) != len(rows):
        raise UgiJointKernelLambdaZeroError("curated AGILE structural population changed")
    return [{**row, "product_smiles": row["model_smiles"]} for row in rows]


def _ledger_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=LEDGER_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in LEDGER_FIELDS})
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", filename="", mtime=0) as handle:
        handle.write(buffer.getvalue().encode())
    return compressed.getvalue()


def _read_ledger(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise UgiJointKernelLambdaZeroError(f"invalid terminal ledger: {path}") from error
    if not rows or set(rows[0]) != set(LEDGER_FIELDS):
        raise UgiJointKernelLambdaZeroError("terminal ledger schema changed")
    return rows


@dataclass(frozen=True)
class _ObservedState:
    inner: Any
    arm: str
    seed: int


@dataclass(frozen=True)
class TerminalObservation:
    arm: str
    seed: int
    checkpoint: int
    particle_index: int
    invocation_seed: int
    receipt: GuidanceTerminalCompletionReceipt


class RecordingGuidanceLane:
    """Narrow observer around an unmodified restartable guidance lane."""

    def __init__(self, lane: RestartableGuidanceLane) -> None:
        self.lane = lane
        self._initializations: Counter[int] = Counter()
        self.observations: list[TerminalObservation] = []

    @staticmethod
    def _state(value: Any) -> _ObservedState:
        if not isinstance(value, _ObservedState):
            raise UgiJointKernelLambdaZeroError("recording lane received an unwrapped state")
        return value

    @staticmethod
    def _wrap(receipt: GuidanceStateReceipt, state: _ObservedState) -> GuidanceStateReceipt:
        if not isinstance(receipt, GuidanceStateReceipt):
            raise UgiJointKernelLambdaZeroError("base lane returned an untyped state receipt")
        return GuidanceStateReceipt(
            state=_ObservedState(receipt.state, state.arm, state.seed),
            product_transition_calls=receipt.product_transition_calls,
            gpu_device_seconds=receipt.gpu_device_seconds,
            consumed_particle_seed_manifest_sha256=(receipt.consumed_particle_seed_manifest_sha256),
        )

    def initialize(
        self,
        programs: tuple[bytes, ...],
        *,
        seed: int,
        particle_seeds: tuple[int, ...],
        device: str,
    ) -> GuidanceStateReceipt:
        index = self._initializations[seed]
        if index >= 2:
            raise UgiJointKernelLambdaZeroError("one seed initialized more than two arms")
        arm = ("left", "right")[index]
        self._initializations[seed] += 1
        receipt = self.lane.initialize(
            programs,
            seed=seed,
            particle_seeds=particle_seeds,
            device=device,
        )
        return self._wrap(receipt, _ObservedState(receipt.state, arm, seed))

    def advance(self, state: Any, *, target_step: int) -> GuidanceStateReceipt:
        wrapped = self._state(state)
        receipt = self.lane.advance(wrapped.inner, target_step=target_step)
        return self._wrap(receipt, wrapped)

    def snapshot(self, state: Any) -> GuidanceStateReceipt:
        wrapped = self._state(state)
        receipt = self.lane.snapshot(wrapped.inner)
        return self._wrap(receipt, wrapped)

    def complete_terminal(
        self,
        state: Any,
        *,
        particle_index: int,
        seed: int,
        checkpoint_index: int,
    ) -> GuidanceTerminalCompletionReceipt:
        wrapped = self._state(state)
        receipt = self.lane.complete_terminal(
            wrapped.inner,
            particle_index=particle_index,
            seed=seed,
            checkpoint_index=checkpoint_index,
        )
        if not isinstance(receipt, GuidanceTerminalCompletionReceipt):
            raise UgiJointKernelLambdaZeroError("base lane returned an untyped terminal receipt")
        self.observations.append(
            TerminalObservation(
                arm=wrapped.arm,
                seed=wrapped.seed,
                checkpoint=checkpoint_index,
                particle_index=particle_index,
                invocation_seed=seed,
                receipt=receipt,
            )
        )
        return receipt

    def apply_ancestry(
        self,
        state: Any,
        ancestors: Sequence[int],
    ) -> GuidanceStateReceipt:
        wrapped = self._state(state)
        receipt = self.lane.apply_ancestry(wrapped.inner, ancestors)
        return self._wrap(receipt, wrapped)

    def arm_observations(self, seed: int, arm: str) -> tuple[TerminalObservation, ...]:
        if self._initializations[seed] != 2 or arm not in {"left", "right"}:
            raise UgiJointKernelLambdaZeroError("paired arm initialization is incomplete")
        return tuple(row for row in self.observations if row.seed == seed and row.arm == arm)


@dataclass(frozen=True)
class StructuralApplicabilityContext:
    thresholds: Mapping[str, Mapping[str, Mapping[str, float]]]
    references: Mapping[str, Any]
    component_sets: Mapping[str, frozenset[str]]
    measured_products_by_triple: Mapping[tuple[str, str, str], frozenset[str]]
    supported_patterns: Mapping[tuple[str, ...], str]
    thresholds_sha256: str

    def classify_candidate(self, candidate: Mapping[str, str]) -> dict[str, Any]:
        required = {
            "label",
            "product_smiles",
            "amine_smiles",
            "aldehyde_smiles",
            "isocyanide_smiles",
        }
        if set(candidate) != required:
            raise UgiJointKernelLambdaZeroError("candidate applicability fields changed")
        canonical = {
            "product": applicability_v1._canonical(candidate["product_smiles"]),
            **{role: applicability_v1._canonical(candidate[f"{role}_smiles"]) for role in ROLES},
        }
        triple = tuple(canonical[role] for role in ROLES)
        measured_products = self.measured_products_by_triple.get(triple, frozenset())
        exact_measured = bool(measured_products)
        if exact_measured and canonical["product"] not in measured_products:
            raise UgiJointKernelLambdaZeroError(
                "exact measured precursor triple reconstructed an unrecognized product"
            )
        unseen = tuple(role for role in ROLES if canonical[role] not in self.component_sets[role])
        distances = applicability_v1._distance_fields(
            self.references,
            candidate,
            generated=True,
        )
        bins, overall = applicability_v1._bins(distances, self.thresholds)
        pattern_id = self.supported_patterns.get(unseen)
        if exact_measured:
            action = "neutral"
            reason = "exact_measured_combination"
            pattern_id = None
        elif unseen == ROLES:
            action = "abstain"
            reason = "unsupported_all_three_new"
            pattern_id = None
        elif pattern_id is None:
            action = "abstain"
            reason = "unsupported_exact_unseen_role_pattern"
        elif any(bins[view] != "interpolative" for view in VIEWS):
            action = "abstain"
            reason = "one_or_more_views_not_interpolative"
        else:
            action = "active_eligible_no_score"
            reason = "supported_pattern_all_views_interpolative"
        return {
            "canonical": canonical,
            "exact_measured_combination": exact_measured,
            "unseen_roles": unseen,
            "pattern_id": pattern_id,
            "action": action,
            "reason": reason,
            "distances": distances,
            "bins": bins,
            "overall_bin": overall,
        }


@dataclass(frozen=True)
class LambdaZeroExecutionContract:
    repo: Path
    config_path: Path
    config: Mapping[str, Any]
    paths: Mapping[str, Path]
    schedule: GroupedSMCScheduleQualification
    schedule_metadata: Mapping[str, Mapping[str, Any]]
    applicability: StructuralApplicabilityContext


def _supported_patterns(policy: Mapping[str, Any]) -> dict[tuple[str, ...], str]:
    restrictions = policy.get("role_pattern_policy")
    if not isinstance(restrictions, Mapping):
        raise UgiJointKernelLambdaZeroError("role-pattern policy is missing")
    if (
        restrictions.get("all_three_new_action") != "abstain"
        or restrictions.get("unsupported_pattern_action") != "abstain"
        or restrictions.get("exact_measured_combination_action") != "neutral"
    ):
        raise UgiJointKernelLambdaZeroError("role-pattern actions changed")
    raw = restrictions.get("supported_exact_unseen_role_patterns")
    if not isinstance(raw, list):
        raise UgiJointKernelLambdaZeroError("supported role patterns are missing")
    output: dict[tuple[str, ...], str] = {}
    for record in raw:
        if not isinstance(record, Mapping):
            raise UgiJointKernelLambdaZeroError("supported role pattern is malformed")
        roles = tuple(str(value) for value in record.get("unseen_roles", ()))
        pattern_id = str(record.get("pattern_id", ""))
        if not roles or not pattern_id or roles in output:
            raise UgiJointKernelLambdaZeroError("supported role pattern is ambiguous")
        output[roles] = pattern_id
    if output != EXPECTED_SUPPORTED_PATTERNS:
        raise UgiJointKernelLambdaZeroError("supported role patterns changed")
    return output


def _schedule_metadata(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    schedules = value.get("seed_schedules")
    if not isinstance(schedules, list) or len(schedules) != len(EXPECTED_SEEDS):
        raise UgiJointKernelLambdaZeroError("bounded schedule records changed")
    output: dict[str, Mapping[str, Any]] = {}
    for schedule in schedules:
        programs = schedule.get("programs") if isinstance(schedule, Mapping) else None
        if not isinstance(programs, list) or len(programs) != 16:
            raise UgiJointKernelLambdaZeroError("bounded schedule program metadata changed")
        for record in programs:
            if not isinstance(record, Mapping):
                raise UgiJointKernelLambdaZeroError("bounded schedule program is malformed")
            digest = str(record.get("morphology_program_sha256", ""))
            if digest in output:
                raise UgiJointKernelLambdaZeroError("bounded program metadata is duplicated")
            if record.get("bounded_neighborhood") not in {
                "exact_anchor",
                "local_smoothed_neighborhood",
            }:
                raise UgiJointKernelLambdaZeroError("bounded program neighborhood changed")
            output[digest] = record
    if len(output) != 128:
        raise UgiJointKernelLambdaZeroError("bounded schedule no longer has 128 programs")
    return output


def load_execution_contract(repo: Path, config_path: Path) -> LambdaZeroExecutionContract:
    """Authenticate the execution-only schedule and no-score structural gate."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="lambda-zero execution config")
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("status") != EXPECTED_STATUS
        or config.get("scope") != EXPECTED_SCOPE
        or config.get("policy") != EXPECTED_POLICY
    ):
        raise UgiJointKernelLambdaZeroError("lambda-zero execution contract changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != EXPECTED_INPUTS:
        raise UgiJointKernelLambdaZeroError("lambda-zero input set changed")
    paths = {name: _pin(repo, record, label=name) for name, record in inputs.items()}

    schedule_value = _load_json(paths["schedule_result"], label="bounded schedule result")
    if (
        schedule_value.get("schema_version") != EXPECTED_SCHEDULE_RESULT_SCHEMA
        or schedule_value.get("status") != EXPECTED_SCHEDULE_STATUS
    ):
        raise UgiJointKernelLambdaZeroError("bounded schedule is not qualified")
    _logical_result_hash(schedule_value, label="bounded schedule")
    bounded = schedule_value.get("bounded_joint_program_contract")
    schedule_scope = schedule_value.get("scope")
    if not isinstance(bounded, Mapping) or any(
        bounded.get(key) != expected
        for key, expected in {
            "candidate_pool_programs": 485,
            "program_overlap_across_seeds": 0,
            "role_marginal_recombination": False,
            "schedule_builder_mutation_or_role_composition": False,
            "all_three_component_graphs_still_generated": True,
            "biological_labels_or_oracle_outputs_read": False,
        }.items()
    ):
        raise UgiJointKernelLambdaZeroError("bounded whole-program contract changed")
    if not isinstance(schedule_scope, Mapping) or any(
        schedule_scope.get(key) != expected
        for key, expected in {
            "biology_used": False,
            "oracle_calls": 0,
            "synthesis_calls": 0,
            "proposal_calls": 0,
            "nonzero_guidance": False,
        }.items()
    ):
        raise UgiJointKernelLambdaZeroError("bounded schedule scope changed")
    schedule = load_grouped_smc_schedule_qualification(paths["schedule_result"])
    if tuple(assignment.seed for assignment in schedule.assignments) != EXPECTED_SEEDS:
        raise UgiJointKernelLambdaZeroError("bounded assignment seeds changed")

    applicability_result = _load_json(
        paths["applicability_v3_result"], label="applicability v3 result"
    )
    if (
        applicability_result.get("schema_version") != EXPECTED_APPLICABILITY_SCHEMA
        or applicability_result.get("status") != EXPECTED_APPLICABILITY_STATUS
    ):
        raise UgiJointKernelLambdaZeroError("applicability v3 result changed")
    _logical_result_hash(applicability_result, label="applicability v3")
    thresholds = applicability_result.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(VIEWS):
        raise UgiJointKernelLambdaZeroError("applicability v3 thresholds are incomplete")

    role_policy = _load_json(paths["role_policy"], label="role policy")
    if role_policy.get("schema_version") != EXPECTED_ROLE_POLICY_SCHEMA:
        raise UgiJointKernelLambdaZeroError("role policy schema changed")
    source_policy = role_policy.get("source_policy")
    expected_source_policy = {
        "path": str(paths["bio_policy"].relative_to(repo)),
        "sha256": sha256_file(paths["bio_policy"]),
    }
    if source_policy != expected_source_policy:
        raise UgiJointKernelLambdaZeroError("role policy source binding changed")
    threshold_identity = role_policy.get("threshold_identity")
    threshold_sha256 = _sha256_payload(thresholds)
    if not isinstance(threshold_identity, Mapping) or any(
        threshold_identity.get(key) != expected
        for key, expected in {
            "applicability_version": 3,
            "applicability_result_sha256": applicability_result.get("result_sha256"),
            "thresholds_sha256": threshold_sha256,
            "fingerprint_and_descriptor_must_both_pass": True,
            "worst_view_defines_overall_bin": True,
        }.items()
    ):
        raise UgiJointKernelLambdaZeroError("role policy threshold binding changed")
    supported = _supported_patterns(role_policy)

    structural_rows = _read_structural_rows(paths["curated_agile_structures"])
    references = applicability_v2._references(structural_rows)
    component_sets = {
        role: frozenset(
            applicability_v1._canonical(row[ROLE_FIELDS[role]]) for row in structural_rows
        )
        for role in ROLES
    }
    products_by_triple: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in structural_rows:
        triple = tuple(applicability_v1._canonical(row[ROLE_FIELDS[role]]) for role in ROLES)
        products_by_triple[triple].add(applicability_v1._canonical(row["product_smiles"]))
    applicability = StructuralApplicabilityContext(
        thresholds=thresholds,
        references=references,
        component_sets=component_sets,
        measured_products_by_triple={
            key: frozenset(value) for key, value in products_by_triple.items()
        },
        supported_patterns=supported,
        thresholds_sha256=threshold_sha256,
    )
    return LambdaZeroExecutionContract(
        repo=repo,
        config_path=config_path,
        config=config,
        paths=paths,
        schedule=schedule,
        schedule_metadata=_schedule_metadata(schedule_value),
        applicability=applicability,
    )


def _radius_ratio(
    distances: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    view: str,
    radius: str,
) -> float:
    ratios = []
    for kind in ("fingerprint", "descriptor"):
        threshold = float(thresholds[view][kind][radius])
        distance = float(getattr(distances[view], kind))
        if not (math.isfinite(threshold) and threshold > 0.0):
            raise UgiJointKernelLambdaZeroError("applicability threshold is invalid")
        ratios.append(distance / threshold)
    return max(ratios)


def _blank_applicability() -> dict[str, Any]:
    output: dict[str, Any] = {
        "exact_measured_combination": "",
        "exact_unseen_roles_json": "[]",
        "policy_pattern_id": "",
        "overall_distribution_bin": "",
        "noninterpolative_views_json": "[]",
        "extrapolative_views_json": "[]",
    }
    for view in VIEWS:
        output[f"{view}_distribution_bin"] = ""
        output[f"{view}_fingerprint_distance"] = ""
        output[f"{view}_descriptor_distance"] = ""
        output[f"{view}_interpolative_radius_ratio"] = ""
        output[f"{view}_boundary_radius_ratio"] = ""
    return output


def _terminal_row(
    observation: TerminalObservation,
    *,
    assignment: FrozenSeedProgramAssignment,
    contract: LambdaZeroExecutionContract,
) -> dict[str, Any]:
    receipt = observation.receipt
    terminal = receipt.terminal
    particle = observation.particle_index
    program_index = particle // 4
    program_sha256 = assignment.program_sha256s[program_index]
    metadata = contract.schedule_metadata[program_sha256]
    base: dict[str, Any] = {
        "seed": observation.seed,
        "seed_partition": (
            "calibration" if observation.seed in CALIBRATION_SEEDS else "evaluation"
        ),
        "checkpoint": observation.checkpoint,
        "checkpoint_kind": (
            "productive_final" if observation.checkpoint == 8 else "shadow_completion"
        ),
        "particle_index": particle,
        "program_index": program_index,
        "within_program_particle_index": particle % 4,
        "invocation_seed": observation.invocation_seed,
        "schedule_sha256": assignment.schedule_sha256,
        "morphology_program_sha256": program_sha256,
        "bounded_neighborhood": metadata["bounded_neighborhood"],
        "branch_class": metadata["branch_class"],
        "terminal_present": terminal is not None,
        "terminal_valid": bool(terminal is not None and terminal.terminal_valid),
        "exact_l1": bool(terminal is not None and terminal.exact_l1),
        "terminal_id": "" if terminal is None else terminal.terminal_id,
        "terminal_sha256": "" if terminal is None else terminal.terminal_sha256,
        "generation_trace_sha256": ("" if terminal is None else terminal.generation_trace_sha256),
        "completion_error": receipt.error_detail or "",
        **_blank_applicability(),
    }
    if terminal is None:
        return {
            **base,
            "applicability_action": "abstain",
            "applicability_reason": "terminal_completion_failed",
        }
    if not terminal.terminal_valid:
        return {
            **base,
            "applicability_action": "abstain",
            "applicability_reason": "invalid_terminal",
        }
    if not terminal.exact_l1:
        return {
            **base,
            "applicability_action": "abstain",
            "applicability_reason": "nonexact_l1_terminal",
        }
    native = native_completion_record_from_locked_terminal(terminal)
    components = native.get("component_smiles_by_role")
    if (
        native.get("valid") is not True
        or native.get("component_reconstruction_valid") is not True
        or not isinstance(components, Mapping)
        or set(components) != set(ROLE_TO_SAMPLE_ROLE.values())
    ):
        raise UgiJointKernelLambdaZeroError("exact terminal lacks exact component graphs")
    candidate = {
        "label": terminal.terminal_sha256,
        "product_smiles": str(native["smiles"]),
        **{GENERATED_FIELDS[role]: str(components[ROLE_TO_SAMPLE_ROLE[role]]) for role in ROLES},
    }
    classification = contract.applicability.classify_candidate(candidate)
    distances = classification["distances"]
    bins = classification["bins"]
    output = {
        **base,
        "exact_measured_combination": str(classification["exact_measured_combination"]).lower(),
        "exact_unseen_roles_json": json.dumps(
            classification["unseen_roles"], separators=(",", ":")
        ),
        "policy_pattern_id": classification["pattern_id"] or "",
        "applicability_action": classification["action"],
        "applicability_reason": classification["reason"],
        "overall_distribution_bin": classification["overall_bin"],
        "noninterpolative_views_json": json.dumps(
            [view for view in VIEWS if bins[view] != "interpolative"],
            separators=(",", ":"),
        ),
        "extrapolative_views_json": json.dumps(
            [view for view in VIEWS if bins[view] == "extrapolative"],
            separators=(",", ":"),
        ),
    }
    for view in VIEWS:
        output[f"{view}_distribution_bin"] = bins[view]
        output[f"{view}_fingerprint_distance"] = distances[view].fingerprint
        output[f"{view}_descriptor_distance"] = distances[view].descriptor
        output[f"{view}_interpolative_radius_ratio"] = _radius_ratio(
            distances,
            contract.applicability.thresholds,
            view,
            "interpolative_max",
        )
        output[f"{view}_boundary_radius_ratio"] = _radius_ratio(
            distances,
            contract.applicability.thresholds,
            view,
            "boundary_max",
        )
    return output


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise UgiJointKernelLambdaZeroError("terminal applicability census is empty")
    actions = Counter(str(row["applicability_action"]) for row in rows)
    reasons = Counter(str(row["applicability_reason"]) for row in rows)
    checkpoints: defaultdict[str, Counter[str]] = defaultdict(Counter)
    unseen: Counter[str] = Counter()
    overall: Counter[str] = Counter()
    valid_exact = 0
    for row in rows:
        checkpoints[str(row["checkpoint"])][str(row["applicability_action"])] += 1
        unseen[str(row["exact_unseen_roles_json"])] += 1
        if row["overall_distribution_bin"]:
            overall[str(row["overall_distribution_bin"])] += 1
        valid_exact += int(str(row["terminal_valid"]).lower() == "true") * int(
            str(row["exact_l1"]).lower() == "true"
        )
    eligible = actions["active_eligible_no_score"]
    return {
        "scheduled_terminal_attempts": len(rows),
        "terminal_present": sum(str(row["terminal_present"]).lower() == "true" for row in rows),
        "valid_exact_l1": valid_exact,
        "active_eligible_no_score": eligible,
        "active_eligible_fraction_of_all_attempts": eligible / len(rows),
        "active_eligible_fraction_of_valid_exact_l1": (
            eligible / valid_exact if valid_exact else 0.0
        ),
        "actions": dict(sorted(actions.items())),
        "reasons": dict(sorted(reasons.items())),
        "exact_unseen_role_patterns": dict(sorted(unseen.items())),
        "overall_distribution_bins_among_classified": dict(sorted(overall.items())),
        "actions_by_checkpoint": {
            checkpoint: dict(sorted(values.items()))
            for checkpoint, values in sorted(checkpoints.items(), key=lambda item: int(item[0]))
        },
    }


def run_seed_qualification(
    repo: Path,
    config_path: Path,
    seed: int,
    *,
    lane: RestartableGuidanceLane | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Execute one prespecified seed and return a deterministic result and ledger."""

    contract = load_execution_contract(repo, config_path)
    if seed not in EXPECTED_SEEDS:
        raise UgiJointKernelLambdaZeroError("seed is not one of the eight frozen assignments")
    assignment = contract.schedule.by_seed()[seed]
    design = ParticleGroupDesign(morphology_program_count=16, particles_per_program=4)
    schedule = GuidanceSchedule(
        sample_steps=8,
        checkpoints=(2, 4, 6),
        checkpoint_betas=(0.25, 0.5, 0.75),
        rollouts_per_particle_checkpoint=1,
        final_selection_count=64,
    )
    base_lane = lane or build_selected_model_restartable_guidance_lane_v3(contract.repo)
    recording = RecordingGuidanceLane(base_lane)
    gate = run_lambda_zero_identity_gate(
        assignment,
        lane=recording,
        design=design,
        schedule=schedule,
        device="cpu",
        historical_productive_manifest=None,
    )
    if (
        not gate.bitwise_identity_passed
        or gate.historical_hard_reference_passed is not None
        or gate.product_transition_calls_per_arm != 1280
        or gate.terminal_completions_per_arm != 256
    ):
        raise UgiJointKernelLambdaZeroError("lambda-zero seed gate changed")
    left = recording.arm_observations(seed, "left")
    right = recording.arm_observations(seed, "right")
    if len(left) != 256 or len(right) != 256:
        raise UgiJointKernelLambdaZeroError("recorded terminal-attempt count changed")
    expected_checkpoints = Counter({2: 64, 4: 64, 6: 64, 8: 64})
    if (
        Counter(row.checkpoint for row in left) != expected_checkpoints
        or Counter(row.checkpoint for row in right) != expected_checkpoints
    ):
        raise UgiJointKernelLambdaZeroError("recorded checkpoint partition changed")
    rows = [_terminal_row(row, assignment=assignment, contract=contract) for row in left]
    ledger = _ledger_bytes(rows)
    summary = _summarize_rows(rows)
    content: dict[str, Any] = {
        "schema_version": SEED_RESULT_SCHEMA_VERSION,
        "status": "joint_kernel_lambda_zero_seed_qualified_no_oracle",
        "decision": "seed_identity_passed_applicability_hard_gate_observed_no_score",
        "config": {
            "path": str(contract.config_path.relative_to(contract.repo)),
            "sha256": sha256_file(contract.config_path),
        },
        "seed": seed,
        "seed_partition": "calibration" if seed in CALIBRATION_SEEDS else "evaluation",
        "schedule_binding": {
            "qualification_file_sha256": contract.schedule.file_sha256,
            "qualification_result_sha256": contract.schedule.result_sha256,
            "assignment_sha256": assignment.assignment_sha256,
            "schedule_sha256": assignment.schedule_sha256,
            "program_count": 16,
            "particles_per_program": 4,
            "particle_count": 64,
        },
        "lambda_zero_identity": asdict(gate),
        "terminal_applicability_census": summary,
        "applicability_contract": {
            "version": 3,
            "thresholds_sha256": contract.applicability.thresholds_sha256,
            "views": list(VIEWS),
            "fingerprint_and_descriptor_must_both_pass": True,
            "all_four_views_must_be_interpolative": True,
            "supported_patterns": {
                "+".join(roles): pattern
                for roles, pattern in sorted(contract.applicability.supported_patterns.items())
            },
            "exact_measured_combination_action": "neutral",
            "all_three_new_action": "abstain",
            "applicability_is_a_hard_gate_not_a_potency_score": True,
        },
        "artifacts": {
            "ledger.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "records": len(rows),
                "sha256": sha256_bytes(ledger),
                "contains_oracle_predictions": False,
                "contains_potency_scores": False,
            }
        },
        "inputs": {
            name: {"path": str(path.relative_to(contract.repo)), "sha256": sha256_file(path)}
            for name, path in sorted(contract.paths.items())
        },
        "scope": EXPECTED_SCOPE,
        "nonclaims": [
            "This seed gate does not execute or authorize nonzero guidance.",
            "This seed gate does not call or score a biological oracle.",
            "This seed gate does not evaluate synthesis routes or synthesis values.",
            "This seed gate does not select or lock candidates.",
        ],
    }
    return {**content, "result_sha256": _sha256_payload(content)}, ledger


def _seed_result_paths(seed_root: Path, seed: int) -> tuple[Path, Path]:
    directory = seed_root / str(seed)
    return directory / "result.json", directory / "ledger.csv.gz"


def finalize_all_seed_qualification(
    repo: Path,
    config_path: Path,
    seed_root: Path,
) -> tuple[dict[str, Any], bytes]:
    """Combine eight immutable seed receipts after revalidating every byte."""

    contract = load_execution_contract(repo, config_path)
    seed_root = seed_root.resolve()
    results = []
    all_rows: list[dict[str, str]] = []
    for seed in EXPECTED_SEEDS:
        result_path, ledger_path = _seed_result_paths(seed_root, seed)
        result = _load_json(result_path, label=f"seed {seed} result")
        if (
            result.get("schema_version") != SEED_RESULT_SCHEMA_VERSION
            or result.get("status") != "joint_kernel_lambda_zero_seed_qualified_no_oracle"
            or result.get("seed") != seed
            or result.get("scope") != EXPECTED_SCOPE
        ):
            raise UgiJointKernelLambdaZeroError(f"seed {seed} qualification changed")
        _logical_result_hash(result, label=f"seed {seed}")
        expected_config = {
            "path": str(contract.config_path.relative_to(contract.repo)),
            "sha256": sha256_file(contract.config_path),
        }
        if result.get("config") != expected_config:
            raise UgiJointKernelLambdaZeroError(f"seed {seed} config binding changed")
        artifact = result.get("artifacts", {}).get("ledger.csv.gz", {})
        if (
            artifact.get("schema_version") != LEDGER_SCHEMA_VERSION
            or artifact.get("records") != 256
            or artifact.get("sha256") != sha256_file(ledger_path)
        ):
            raise UgiJointKernelLambdaZeroError(f"seed {seed} ledger binding changed")
        rows = _read_ledger(ledger_path)
        if len(rows) != 256 or {int(row["seed"]) for row in rows} != {seed}:
            raise UgiJointKernelLambdaZeroError(f"seed {seed} ledger rows changed")
        results.append(result)
        all_rows.extend(rows)
    if len(all_rows) != 2048:
        raise UgiJointKernelLambdaZeroError("all-seed attempt accounting changed")
    all_ledger = _ledger_bytes(all_rows)
    census = _summarize_rows(all_rows)
    aggregate_transitions = sum(
        int(row["lambda_zero_identity"]["product_transition_calls_per_arm"]) for row in results
    )
    aggregate_completions = sum(
        int(row["lambda_zero_identity"]["terminal_completions_per_arm"]) for row in results
    )
    per_seed_census = {str(row["seed"]): row["terminal_applicability_census"] for row in results}
    content: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "all_eight_joint_kernel_lambda_zero_identity_qualified_no_oracle",
        "decision": (
            "bounded_schedule_identity_passed_and_v3_applicability_hard_gate_bound; "
            "nonzero_guidance_remains_blocked"
        ),
        "config": {
            "path": str(contract.config_path.relative_to(contract.repo)),
            "sha256": sha256_file(contract.config_path),
        },
        "schedule_binding": {
            "qualification_file_sha256": contract.schedule.file_sha256,
            "qualification_result_sha256": contract.schedule.result_sha256,
            "assignment_seeds": list(EXPECTED_SEEDS),
            "calibration_seeds": list(CALIBRATION_SEEDS),
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "programs_per_seed": 16,
            "particles_per_program": 4,
            "unique_programs": 128,
            "program_overlap_across_seeds": 0,
        },
        "lambda_zero_identity": {
            "all_eight_passed": all(
                row["lambda_zero_identity"]["bitwise_identity_passed"] is True for row in results
            ),
            "historical_old_schedule_manifest_comparison_performed": False,
            "product_transition_calls_per_arm_across_eight": aggregate_transitions,
            "terminal_completions_per_arm_across_eight": aggregate_completions,
            "expected_product_transition_calls_per_arm_across_eight": 10240,
            "expected_terminal_completions_per_arm_across_eight": 2048,
            "seed_receipts": [
                {
                    "seed": row["seed"],
                    "result_sha256": row["result_sha256"],
                    "identity_receipt_sha256": row["lambda_zero_identity"]["receipt_sha256"],
                    "productive_manifest_sha256": row["lambda_zero_identity"][
                        "productive_manifest_sha256"
                    ],
                }
                for row in results
            ],
        },
        "terminal_applicability_census": {
            **census,
            "per_seed": per_seed_census,
        },
        "applicability_contract": {
            "version": 3,
            "thresholds_sha256": contract.applicability.thresholds_sha256,
            "views": list(VIEWS),
            "all_four_views_and_both_distance_types_required": True,
            "supported_patterns": {
                "+".join(roles): pattern
                for roles, pattern in sorted(contract.applicability.supported_patterns.items())
            },
            "exact_measured_combination_action": "neutral",
            "unsupported_pattern_action": "abstain",
            "all_three_new_action": "abstain",
            "future_oracle_evaluation_may_only_follow_active_eligible_no_score": True,
            "no_oracle_was_invoked_by_this_qualification": True,
        },
        "artifacts": {
            "ledger.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "records": len(all_rows),
                "sha256": sha256_bytes(all_ledger),
                "contains_oracle_predictions": False,
                "contains_potency_scores": False,
            }
        },
        "scope": EXPECTED_SCOPE,
        "adjudication": {
            "lambda_zero_identity_ready": True,
            "terminal_v3_applicability_gate_bound": True,
            "biological_oracle_scoring_executed": False,
            "nonzero_guidance_authorized": False,
            "synthesis_guidance_authorized": False,
            "candidate_selection_changed": False,
            "next_gate": (
                "review active-eligible census and freeze an explicit nonzero execution "
                "authorization before any potency or synthesis guidance"
            ),
        },
        "inputs": {
            name: {"path": str(path.relative_to(contract.repo)), "sha256": sha256_file(path)}
            for name, path in sorted(contract.paths.items())
        },
    }
    if aggregate_transitions != 10240 or aggregate_completions != 2048:
        raise UgiJointKernelLambdaZeroError("all-seed lambda-zero compute changed")
    return {**content, "result_sha256": _sha256_payload(content)}, all_ledger


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "LEDGER_FIELDS",
    "LEDGER_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "SEED_RESULT_SCHEMA_VERSION",
    "RecordingGuidanceLane",
    "StructuralApplicabilityContext",
    "TerminalObservation",
    "UgiJointKernelLambdaZeroError",
    "finalize_all_seed_qualification",
    "load_execution_contract",
    "run_seed_qualification",
]
