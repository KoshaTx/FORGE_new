"""Typed lambda-zero qualification seam for the selected Ugi v3 lane.

This additive qualification composes the equivalence-bound selected-v3
restartable generator, the frozen matched-arm runner, isolated planner-cache
overlays and the binary exact-dossier route utility.  Guidance strength is
exactly zero.  Every invoked route assessment is retained as a complete
support, potential and assessment-receipt record; invalid, nonexact,
duplicate and planner-censored outcomes remain distinct.

Passing this seam is an integration qualification only.  It neither executes
nor authorizes nonzero guidance, biology, candidate selection, prospective
locking or sealed-holdout access.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.core.io import stable_json as _stable_json
from forge.data.r1_prime_audit import sha256_file
from forge.design.ugi_matched_budget_orchestration import (
    MatchedArm,
    MatchedAssessmentContext,
)
from forge.design.ugi_matched_planner_cache_binding import (
    LazyMatchedPlannerCacheBinding,
    preflight_lazy_matched_planner_cache_binding,
)
from forge.design.ugi_nonzero_guidance_runner import (
    GuidanceAssessmentContext,
    GuidanceCacheIsolationContract,
    GuidanceComputeBudget,
    GuidanceRouteEvaluation,
    GuidanceSchedule,
    ParticleGroupDesign,
    finalize_lazy_matched_cache_binding,
    load_grouped_smc_schedule_qualification,
    run_development_matched_guidance,
)
from forge.design.ugi_production_terminal_route_evaluator import (
    ProductionUgiTerminalAwarePlannerFactory,
    build_production_ugi_terminal_aware_planner_factory,
)
from forge.design.ugi_restartable_terminal_support_adapter import (
    native_completion_record_from_locked_terminal,
)
from forge.design.ugi_selected_guidance_adapter_v3 import (
    EQUIVALENCE_RESULT_FILE_SHA256,
    EQUIVALENCE_RESULT_LOGICAL_SHA256,
    EQUIVALENCE_RESULT_PATH,
    EQUIVALENCE_ROWS_FILE_SHA256,
    EQUIVALENCE_ROWS_LOGICAL_SHA256,
    EQUIVALENCE_ROWS_PATH,
    SELECTED_GUIDANCE_ADAPTER_V3_SCHEMA_VERSION,
    SelectedModelRestartableGuidanceLaneV3,
    build_selected_model_restartable_guidance_lane_v3,
)
from forge.design.ugi_selected_restartable_generator_v2 import (
    GENERATOR_CHECKPOINT_SHA256,
    MAXIMUM_ADJACENT_BRANCH_RUNS,
    PRODUCTION_GENERATOR_MANIFEST_SHA256,
    TERMINAL_DECODER_ID,
)
from forge.design.ugi_zero_guidance_typed_audit import (
    build_support_audit_artifact,
    build_zero_guidance_run_artifact,
    derive_typed_counts,
)
from forge.route.engine.planner_cache import FilePlannerCache
from forge.route.terminals.terminal_assessment import (
    ValidatedUgiTerminalPayload,
    assess_locked_ugi_terminal_routes,
)
from forge.value.guidance.ugi_exact_closure_guidance import (
    UGI_EXACT_CLOSURE_GUIDANCE_POLICY_SHA256,
    exact_closure_potential_from_product_value,
    smc_utility_bridge_from_exact_closure,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi_production_zero_guidance_seam_v3_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_production_zero_guidance_seam_v3.v1"
EXPECTED_INPUT_KEYS = {
    "binary_utility_source",
    "current_source_requalification",
    "grouped_schedule",
    "matched_cache_binding",
    "nonzero_runner",
    "production_generator_manifest",
    "production_terminal_route_evaluator",
    "route_completion_utility",
    "seam_runner",
    "seam_source",
    "seam_tests",
    "selected_guidance_adapter_v2",
    "selected_guidance_adapter_v3",
    "selected_restartable_generator_v2",
    "selected_v3_equivalence_config",
    "selected_v3_equivalence_result",
    "selected_v3_equivalence_rows",
    "zero_guidance_typed_audit",
}
EXPECTED_SCOPE = {
    "guidance_strength": 0.0,
    "qualification_only": True,
    "production_execution": False,
    "nonzero_guidance": False,
    "biological_guidance": False,
    "candidate_selection": False,
    "prospective_candidate_lock": False,
    "sealed_holdout_access": False,
}
EXPECTED_DESIGN = {
    "assignment_seed": 20260821,
    "morphology_program_count": 16,
    "particles_per_program": 4,
    "sample_steps": 8,
    "checkpoints": [2, 4, 6],
    "checkpoint_betas": [0.25, 0.5, 0.75],
    "rollouts_per_particle_checkpoint": 1,
    "final_selection_count": 32,
    "product_transition_calls": 1280,
    "terminal_completions": 256,
    "logical_planner_calls": 768,
    "logical_verifier_calls": 3328,
}

_PRODUCTIVE_TERMINAL_ID = re.compile(
    r"^selected-guidance:"
    r"p(?P<particle_index>[0-9]+):"
    r"c(?P<checkpoint_index>[0-9]+):"
    r"s(?P<invocation_seed>[0-9]+):"
    r"pool_state=(?P<pool_state>[0-9a-f]{64}):"
    r"pool_provenance=(?P<pool_provenance>[0-9a-f]{64}):"
    r"pool_lineage=(?P<pool_lineage>[0-9a-f]{64}):"
    r"particle_state=(?P<particle_state>[0-9a-f]{64}):"
    r"particle_provenance=(?P<particle_provenance>[0-9a-f]{64}):"
    r"particle_lineage=(?P<particle_lineage>[0-9a-f]{64}):"
    r"adapter=(?P<adapter_identity>[0-9a-f]{64}):"
    r"(?P<productive_seed>[0-9]+)$"
)


class UgiProductionZeroGuidanceSeamV3Error(RuntimeError):
    """Raised when the typed selected-v3 lambda-zero seam fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return (_stable_json(value) + "\n").encode()


def _canonical_file_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiProductionZeroGuidanceSeamV3Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiProductionZeroGuidanceSeamV3Error(f"{label} must be a JSON object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise UgiProductionZeroGuidanceSeamV3Error(f"{label} pin is malformed")
    path = (repo / record["path"]).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiProductionZeroGuidanceSeamV3Error(f"{label} path escapes repository") from error
    if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
        raise UgiProductionZeroGuidanceSeamV3Error(f"{label} hash changed")
    return path


def _canonical_identity(terminal_bytes: bytes) -> str:
    return ValidatedUgiTerminalPayload.from_bytes(terminal_bytes).product_smiles


@dataclass
class ProductionGuidanceRouteEvaluatorV3:
    """Retain complete evidence for every planner invocation."""

    binding: LazyMatchedPlannerCacheBinding
    factory: ProductionUgiTerminalAwarePlannerFactory
    support_records: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self,
        terminal: Any,
        context: GuidanceAssessmentContext,
    ) -> GuidanceRouteEvaluation:
        treatment = MatchedArm.GUIDED if context.treatment_arm == "guided" else MatchedArm.POST_HOC
        assessment_context = MatchedAssessmentContext(
            arm=treatment,
            route_seed=context.route_seed,
            remaining_budget=context.reservation,
            unit_reservation=context.reservation,
            cache_snapshot_sha256=context.base_snapshot_sha256,
            cache_clone_id=context.cache_clone_id,
            post_hoc_lock_manifest_sha256=context.post_hoc_productive_lock_sha256,
        )
        overlay = self.binding.bind(assessment_context)
        planner = self.factory.build_planner(
            overlay,
            self.factory.planner_context,
            assessment_context,
            terminal,
        )
        receipt = assess_locked_ugi_terminal_routes(
            terminal,
            l1_reverifier=self.factory.l1_reverifier,
            planner=planner,
            planner_context=self.factory.planner_context,
            assessment_context=assessment_context,
            assessment_at_utc=self.factory.assessment_as_of_utc,
        )
        potential = exact_closure_potential_from_product_value(receipt.product_value)
        bridge = smc_utility_bridge_from_exact_closure(potential)
        potential_record = {
            **potential.to_dict(),
            "true_planner_censor": bridge.censored,
        }
        support_record = {
            "ordinal": len(self.support_records),
            "arm": context.treatment_arm,
            "assessment_phase": context.assessment_phase,
            "checkpoint": context.checkpoint,
            "terminal_id": receipt.terminal_id,
            "terminal_sha256": receipt.terminal_sha256,
            "generation_trace_sha256": receipt.generation_trace_sha256,
            "morphology_program_sha256": receipt.morphology_program_sha256,
            "route_seed": context.route_seed,
            "reservation": asdict(context.reservation),
            "base_snapshot_sha256": context.base_snapshot_sha256,
            "planner_context_sha256": context.planner_context_sha256,
            "cache_preflight_sha256": context.cache_preflight_sha256,
            "cache_clone_id": context.cache_clone_id,
            "post_hoc_productive_lock_sha256": context.post_hoc_productive_lock_sha256,
            "support_audit": planner.support_audit.to_dict(),
            "potential": potential_record,
            "utility_bridge": bridge.to_dict(),
            "assessment_receipt": receipt.to_dict(),
            "assessment_receipt_sha256": receipt.assessment_sha256,
            "route_dossier_sha256": receipt.assessment_sha256 if bridge.support_bonus else None,
            "realized_route_usage": asdict(receipt.realized_route_usage),
            "scalar_value": None,
            "success_probability": None,
        }
        self.support_records.append(support_record)
        return GuidanceRouteEvaluation(
            route_completion_utility=bridge.route_completion_utility,
            value_policy_id=UGI_EXACT_CLOSURE_GUIDANCE_POLICY_SHA256,
            usage=receipt.realized_route_usage,
            assessment_receipt_sha256=receipt.assessment_sha256,
            route_dossier_sha256=(receipt.assessment_sha256 if bridge.support_bonus else None),
        )


def _validate_config(repo: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = _load(config_path, label="production zero-guidance seam-v3 config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiProductionZeroGuidanceSeamV3Error("unsupported seam-v3 config schema")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiProductionZeroGuidanceSeamV3Error("seam-v3 scope changed")
    if config.get("design") != EXPECTED_DESIGN:
        raise UgiProductionZeroGuidanceSeamV3Error("seam-v3 design changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != EXPECTED_INPUT_KEYS:
        raise UgiProductionZeroGuidanceSeamV3Error("seam-v3 input set changed")
    paths = {label: _pin(repo, record, label=label) for label, record in inputs.items()}

    current_source = _load(paths["current_source_requalification"], label="current source")
    source = current_source.get("current_cumulative_source")
    if (
        current_source.get("status")
        != "current_source_l3_and_zero_guidance_route_values_requalified"
        or not isinstance(source, dict)
        or source.get("all_l3_windows_cover_decision_horizon") is not True
        or source.get("assessment_at_utc") != config.get("assessment_as_of_utc")
        or source.get("inputs_sha256") != config.get("cumulative_source_inputs_sha256")
    ):
        raise UgiProductionZeroGuidanceSeamV3Error(
            "current cumulative route source is not authenticated for seam v3"
        )
    utility = _load(paths["route_completion_utility"], label="route utility")
    if (
        utility.get("status") != "binary_exact_dossier_route_completion_utility_qualified"
        or utility.get("implementation_policy", {}).get("policy_sha256")
        != UGI_EXACT_CLOSURE_GUIDANCE_POLICY_SHA256
    ):
        raise UgiProductionZeroGuidanceSeamV3Error("binary route utility is not qualified")
    manifest = _load(paths["production_generator_manifest"], label="production generator")
    if (
        sha256_file(paths["production_generator_manifest"]) != PRODUCTION_GENERATOR_MANIFEST_SHA256
        or manifest.get("identity", {}).get("checkpoint_step") != 2000
        or manifest.get("identity", {}).get("terminal_decoder") != "bond_stochastic"
    ):
        raise UgiProductionZeroGuidanceSeamV3Error("selected production generator changed")
    if (
        str(paths["selected_v3_equivalence_result"].relative_to(repo)) != EQUIVALENCE_RESULT_PATH
        or sha256_file(paths["selected_v3_equivalence_result"]) != EQUIVALENCE_RESULT_FILE_SHA256
        or str(paths["selected_v3_equivalence_rows"].relative_to(repo)) != EQUIVALENCE_ROWS_PATH
        or sha256_file(paths["selected_v3_equivalence_rows"]) != EQUIVALENCE_ROWS_FILE_SHA256
    ):
        raise UgiProductionZeroGuidanceSeamV3Error("selected-v3 equivalence pins changed")
    identities = config.get("selected_v3_identities")
    expected_identities = {
        "adapter_schema_version": SELECTED_GUIDANCE_ADAPTER_V3_SCHEMA_VERSION,
        "equivalence_result_file_sha256": EQUIVALENCE_RESULT_FILE_SHA256,
        "equivalence_result_logical_sha256": EQUIVALENCE_RESULT_LOGICAL_SHA256,
        "equivalence_rows_file_sha256": EQUIVALENCE_ROWS_FILE_SHA256,
        "equivalence_rows_logical_sha256": EQUIVALENCE_ROWS_LOGICAL_SHA256,
    }
    if not isinstance(identities, dict) or any(
        identities.get(key) != value for key, value in expected_identities.items()
    ):
        raise UgiProductionZeroGuidanceSeamV3Error("selected-v3 identity declaration changed")
    for key in ("adapter_identity_sha256", "bound_closure_identity_sha256"):
        value = identities.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise UgiProductionZeroGuidanceSeamV3Error(f"selected-v3 {key} is malformed")
    if set(identities) != set(expected_identities) | {
        "adapter_identity_sha256",
        "bound_closure_identity_sha256",
    }:
        raise UgiProductionZeroGuidanceSeamV3Error("selected-v3 identity fields changed")
    return config, paths


def _selected_v3_reference_rows(
    lane: SelectedModelRestartableGuidanceLaneV3,
) -> list[dict[str, Any]]:
    try:
        rows = json.loads(lane.equivalence_binding.rows_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiProductionZeroGuidanceSeamV3Error(
            "selected-v3 equivalence rows cannot be loaded"
        ) from error
    if not isinstance(rows, list):
        raise UgiProductionZeroGuidanceSeamV3Error("selected-v3 equivalence rows are malformed")
    direct = [row.get("direct") for row in rows if row.get("comparison") == "productive_completion"]
    if len(direct) != 64 or any(not isinstance(item, dict) for item in direct):
        raise UgiProductionZeroGuidanceSeamV3Error(
            "selected-v3 productive reference coverage changed"
        )
    return direct


def _parse_productive_terminal_id(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, str):
        raise UgiProductionZeroGuidanceSeamV3Error(f"{label} terminal ID is malformed")
    match = _PRODUCTIVE_TERMINAL_ID.fullmatch(value)
    if match is None:
        raise UgiProductionZeroGuidanceSeamV3Error(f"{label} terminal ID is malformed")
    return match.groupdict()


def _annotate_support_record_coordinates(
    run_payload: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    """Join runner-owned particle coordinates onto every evaluator invocation."""

    record_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (
            record.get("arm"),
            record.get("assessment_phase"),
            record.get("assessment_receipt_sha256"),
        )
        if not all(isinstance(value, str) and value for value in key) or key in record_by_key:
            raise UgiProductionZeroGuidanceSeamV3Error(
                "support ledger contains a missing or duplicate assessment join key"
            )
        record_by_key[key] = record

    claimed: set[tuple[str, str, str]] = set()

    def assign(
        *,
        arm: str,
        phase: str,
        receipt_sha256: Any,
        checkpoint: int,
        program_index: int,
        particle_index: int,
        productive_index: int | None,
    ) -> None:
        if not isinstance(receipt_sha256, str) or not receipt_sha256:
            raise UgiProductionZeroGuidanceSeamV3Error(
                "invoked route assessment is missing its source receipt hash"
            )
        key = (arm, phase, receipt_sha256)
        if key not in record_by_key or key in claimed:
            raise UgiProductionZeroGuidanceSeamV3Error(
                "runner assessment cannot be joined exactly once to the support ledger"
            )
        record = record_by_key[key]
        if record.get("checkpoint") != checkpoint:
            raise UgiProductionZeroGuidanceSeamV3Error(
                "joined support record disagrees with the runner checkpoint"
            )
        record.update(
            {
                "program_index": program_index,
                "particle_index": particle_index,
                "productive_index": productive_index,
            }
        )
        claimed.add(key)

    for arm_name in ("guided", "post_hoc"):
        arm = run_payload.get(arm_name)
        if not isinstance(arm, dict):
            raise UgiProductionZeroGuidanceSeamV3Error(f"{arm_name} run is malformed")
        checkpoint_groups = arm.get("checkpoint_groups")
        productive_assessments = arm.get("productive_assessments")
        if not isinstance(checkpoint_groups, (list, tuple)) or not isinstance(
            productive_assessments, (list, tuple)
        ):
            raise UgiProductionZeroGuidanceSeamV3Error(
                f"{arm_name} runner assessment records are malformed"
            )
        for group in checkpoint_groups:
            if not isinstance(group, dict):
                raise UgiProductionZeroGuidanceSeamV3Error(
                    f"{arm_name} checkpoint group is malformed"
                )
            particle_indices = group.get("global_particle_indices")
            receipts = group.get("source_assessment_receipt_sha256s")
            if (
                not isinstance(particle_indices, (list, tuple))
                or not isinstance(receipts, (list, tuple))
                or len(particle_indices) != len(receipts)
            ):
                raise UgiProductionZeroGuidanceSeamV3Error(
                    f"{arm_name} checkpoint join columns are malformed"
                )
            for particle_index, receipt_sha256 in zip(
                particle_indices,
                receipts,
                strict=True,
            ):
                if receipt_sha256 is None:
                    continue
                assign(
                    arm=arm_name,
                    phase="checkpoint_shadow",
                    receipt_sha256=receipt_sha256,
                    checkpoint=group["checkpoint"],
                    program_index=group["program_index"],
                    particle_index=particle_index,
                    productive_index=None,
                )
        for productive_index, assessment in enumerate(productive_assessments):
            if not isinstance(assessment, dict):
                raise UgiProductionZeroGuidanceSeamV3Error(
                    f"{arm_name} productive assessment is malformed"
                )
            if assessment.get("disposition") != "assessed_representative":
                continue
            assign(
                arm=arm_name,
                phase="productive_final",
                receipt_sha256=assessment.get("source_assessment_receipt_sha256"),
                checkpoint=EXPECTED_DESIGN["sample_steps"],
                program_index=productive_index // EXPECTED_DESIGN["particles_per_program"],
                particle_index=productive_index,
                productive_index=productive_index,
            )
    if claimed != set(record_by_key):
        raise UgiProductionZeroGuidanceSeamV3Error(
            "one or more support records have no runner-owned particle coordinate"
        )


def _require_selected_v3_productive_identity(
    run_payload: dict[str, Any],
    lane: SelectedModelRestartableGuidanceLaneV3,
) -> str:
    """Require reference terminal bytes with deliberately rebound v3 provenance.

    The frozen direct-reference rows were produced by the selected-v2 adapter.
    Binding the repaired equivalence receipt creates the selected-v3 adapter
    identity, which is intentionally embedded in the completion unit ID and
    therefore changes the canonical generation trace.  Molecular terminal
    bytes must remain identical to the independently replayed direct
    reference; the trace must instead be a well-formed, changed digest whose
    terminal ID contains the selected-v3 adapter identity.  Guided/post-hoc
    trace equality is enforced separately by the lambda-zero arm check.
    """

    direct = _selected_v3_reference_rows(lane)
    manifest_rows: list[dict[str, Any]] = []
    guided_bindings: list[tuple[str, str, str]] = []
    for arm_name in ("guided", "post_hoc"):
        arm = run_payload.get(arm_name)
        if not isinstance(arm, dict):
            raise UgiProductionZeroGuidanceSeamV3Error(f"{arm_name} run is malformed")
        admissions = arm.get("productive_admissions")
        if not isinstance(admissions, (list, tuple)) or len(admissions) != len(direct):
            raise UgiProductionZeroGuidanceSeamV3Error(
                f"{arm_name} productive reference coverage changed"
            )
        rebound_pool_states: set[str] = set()
        rebound_pool_provenances: set[str] = set()
        for productive_index, (admission, expected) in enumerate(
            zip(admissions, direct, strict=True)
        ):
            if not isinstance(admission, dict):
                raise UgiProductionZeroGuidanceSeamV3Error(
                    f"{arm_name} productive admission is malformed"
                )
            terminal_id = admission.get("terminal_id")
            terminal_sha256 = admission.get("terminal_sha256")
            trace_sha256 = admission.get("generation_trace_sha256")
            observed_id = _parse_productive_terminal_id(
                terminal_id,
                label=f"{arm_name} productive {productive_index}",
            )
            reference_id = _parse_productive_terminal_id(
                expected.get("terminal_id"),
                label=f"selected-v2 direct productive {productive_index}",
            )
            invariant_id_fields = {
                "particle_index",
                "checkpoint_index",
                "invocation_seed",
                "pool_lineage",
                "particle_state",
                "particle_provenance",
                "particle_lineage",
                "productive_seed",
            }
            rebound_pool_states.add(observed_id["pool_state"])
            rebound_pool_provenances.add(observed_id["pool_provenance"])
            if (
                terminal_sha256 != expected.get("terminal_bytes_sha256")
                or observed_id["adapter_identity"] != lane.adapter_identity_sha256
                or reference_id["adapter_identity"] != lane.base_v2_adapter_identity_sha256
                or observed_id["particle_index"] != str(productive_index)
                or observed_id["checkpoint_index"] != str(EXPECTED_DESIGN["sample_steps"])
                or any(observed_id[key] != reference_id[key] for key in invariant_id_fields)
                or observed_id["pool_state"] == reference_id["pool_state"]
                or observed_id["pool_provenance"] == reference_id["pool_provenance"]
                or not isinstance(trace_sha256, str)
                or len(trace_sha256) != 64
                or any(character not in "0123456789abcdef" for character in trace_sha256)
                or trace_sha256 == expected.get("generation_trace_sha256")
            ):
                raise UgiProductionZeroGuidanceSeamV3Error(
                    f"{arm_name} productive terminal differs from selected-v3 direct reference"
                )
            expected_disposition = (
                "invalid_terminal"
                if expected.get("terminal_valid") is not True
                else "nonexact_l1"
                if expected.get("exact_l1") is not True
                else None
            )
            if (
                expected_disposition is not None
                and admission.get("disposition") != expected_disposition
            ):
                raise UgiProductionZeroGuidanceSeamV3Error(
                    f"{arm_name} productive validity differs from selected-v3 direct reference"
                )
            if arm_name == "guided":
                guided_bindings.append((terminal_id, terminal_sha256, trace_sha256))
                manifest_rows.append(
                    {
                        "terminal_bytes_sha256": terminal_sha256,
                        "selected_v2_direct_generation_trace_sha256": (
                            expected["generation_trace_sha256"]
                        ),
                        "selected_v3_generation_trace_sha256": trace_sha256,
                        "selected_v3_terminal_id_sha256": hashlib.sha256(
                            terminal_id.encode()
                        ).hexdigest(),
                        "selected_v3_adapter_identity_sha256": lane.adapter_identity_sha256,
                        "terminal_valid": expected["terminal_valid"],
                        "exact_l1": expected["exact_l1"],
                    }
                )
            elif (terminal_id, terminal_sha256, trace_sha256) != guided_bindings[productive_index]:
                raise UgiProductionZeroGuidanceSeamV3Error(
                    "post_hoc productive identity or trace differs from guided at lambda zero"
                )
        if len(rebound_pool_states) != 1 or len(rebound_pool_provenances) != 1:
            raise UgiProductionZeroGuidanceSeamV3Error(
                f"{arm_name} productive pool identity is not internally consistent"
            )
    return _sha256_payload(manifest_rows)


def _relative_artifact_path(repo: Path, output_dir: Path, filename: str) -> str:
    path = (output_dir / filename).resolve()
    try:
        return str(path.relative_to(repo))
    except ValueError as error:
        raise UgiProductionZeroGuidanceSeamV3Error(
            "seam-v3 output directory must be within the repository"
        ) from error


def _artifact_pin(
    repo: Path, output_dir: Path, filename: str, value: dict[str, Any]
) -> dict[str, Any]:
    return {
        "path": _relative_artifact_path(repo, output_dir, filename),
        "file_sha256": _canonical_file_sha256(value),
        "result_sha256": value["result_sha256"],
    }


def run_production_zero_guidance_seam_v3(
    repo: Path,
    config_path: Path,
    cache_root: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Execute the equivalence-bound production composition at lambda zero."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    cache_root = cache_root.resolve()
    output_dir = output_dir.resolve()
    config, paths = _validate_config(repo, config_path)
    if cache_root.exists() and any(cache_root.iterdir()):
        raise UgiProductionZeroGuidanceSeamV3Error("seam-v3 cache root must be absent or empty")
    cache_root.mkdir(parents=True, exist_ok=True)

    schedule_qualification = load_grouped_smc_schedule_qualification(paths["grouped_schedule"])
    assignments = schedule_qualification.by_seed()
    assignment_seed = config["design"]["assignment_seed"]
    if assignment_seed not in assignments:
        raise UgiProductionZeroGuidanceSeamV3Error("frozen calibration assignment is missing")
    assignment = assignments[assignment_seed]
    lane = build_selected_model_restartable_guidance_lane_v3(repo)
    selected_identities = config["selected_v3_identities"]
    if (
        lane.adapter_identity_sha256 != selected_identities["adapter_identity_sha256"]
        or lane.bound_closure_identity_sha256
        != selected_identities["bound_closure_identity_sha256"]
    ):
        raise UgiProductionZeroGuidanceSeamV3Error("selected-v3 runtime identity changed")
    equivalence_receipt = _load(
        lane.equivalence_binding.result_path,
        label="selected-v3 equivalence result",
    )
    if equivalence_receipt.get("execution", {}).get("assignment_sha256") != (
        assignment.assignment_sha256
    ):
        raise UgiProductionZeroGuidanceSeamV3Error(
            "seam assignment differs from selected-v3 equivalence evidence"
        )

    factory = build_production_ugi_terminal_aware_planner_factory(
        repo_root=repo,
        assessment_as_of_utc=config["assessment_as_of_utc"],
        expected_cumulative_source_inputs_sha256=config["cumulative_source_inputs_sha256"],
        selected_generator_checkpoint_sha256=GENERATOR_CHECKPOINT_SHA256,
        graph_support=lane.callback.graph_support,
        l1_reverifier=lane.callback.l1_reverifier,
        candidate_record_resolver=native_completion_record_from_locked_terminal,
    )
    binding = preflight_lazy_matched_planner_cache_binding(
        FilePlannerCache(cache_root / "base"),
        FilePlannerCache(cache_root / "guided"),
        FilePlannerCache(cache_root / "post_hoc"),
        factory.planner_context,
        assessment_at_utc=config["assessment_as_of_utc"],
    )
    cache_contract = GuidanceCacheIsolationContract(
        base_snapshot_sha256=binding.preflight.base.snapshot_sha256,
        planner_context_sha256=binding.preflight.context_sha256,
        cache_preflight_sha256=binding.preflight.preflight_sha256,
        guided_clone_id=f"v3-seam-guided-{assignment.assignment_sha256[:16]}",
        post_hoc_clone_id=f"v3-seam-post-hoc-{assignment.assignment_sha256[:16]}",
        isolated_overlay_roots=True,
        production_adapter_qualified=False,
    )
    evaluator = ProductionGuidanceRouteEvaluatorV3(binding=binding, factory=factory)
    design = ParticleGroupDesign(morphology_program_count=16, particles_per_program=4)
    schedule = GuidanceSchedule(
        sample_steps=8,
        checkpoints=(2, 4, 6),
        checkpoint_betas=(0.25, 0.5, 0.75),
        rollouts_per_particle_checkpoint=1,
        final_selection_count=32,
    )
    budget = GuidanceComputeBudget(
        product_transition_calls=1280,
        terminal_completions=256,
        logical_planner_calls=768,
        logical_verifier_calls=3328,
        final_candidates=32,
    )
    run = run_development_matched_guidance(
        assignment,
        lane=lane,
        evaluator=evaluator,
        canonical_identity=_canonical_identity,
        design=design,
        schedule=schedule,
        budget=budget,
        guidance_strength=0.0,
        device="cpu",
        expected_value_policy_id=UGI_EXACT_CLOSURE_GUIDANCE_POLICY_SHA256,
        cache_contract=cache_contract,
        cache_finalizer=lambda: finalize_lazy_matched_cache_binding(binding, cache_contract),
    )
    checkpoint_records = run.guided.checkpoint_groups
    if any(record.resampled for record in checkpoint_records) or any(
        tuple(record.global_ancestors) != tuple(record.global_particle_indices)
        for record in checkpoint_records
    ):
        raise UgiProductionZeroGuidanceSeamV3Error(
            "lambda-zero guidance unexpectedly changed an ancestry"
        )
    if (
        run.guided.productive_terminal_ids != run.post_hoc.productive_terminal_ids
        or run.guided.productive_canonical_identities
        != run.post_hoc.productive_canonical_identities
        or run.guided.productive_lock_manifest_sha256
        != run.post_hoc.productive_lock_manifest_sha256
        or run.guided.productive_route_completion_utilities
        != run.post_hoc.productive_route_completion_utilities
    ):
        raise UgiProductionZeroGuidanceSeamV3Error(
            "lambda-zero guided and post-hoc productive pools differ"
        )

    run_artifact = build_zero_guidance_run_artifact(asdict(run))
    selected_reference_manifest = _require_selected_v3_productive_identity(
        run_artifact["run"], lane
    )
    _annotate_support_record_coordinates(run_artifact["run"], evaluator.support_records)
    support_artifact = build_support_audit_artifact(evaluator.support_records)
    typed_counts = derive_typed_counts(run_artifact, support_artifact)
    run_pin = _artifact_pin(repo, output_dir, "run.json", run_artifact)
    support_pin = _artifact_pin(
        repo,
        output_dir,
        "support_audits.json",
        support_artifact,
    )

    result_content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "selected_v3_production_lambda_zero_typed_seam_qualified",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "artifacts": {
            "run": run_pin,
            "support_audits": support_pin,
        },
        "selected_generator": {
            "checkpoint_sha256": GENERATOR_CHECKPOINT_SHA256,
            "terminal_decoder_id": TERMINAL_DECODER_ID,
            "maximum_adjacent_branch_runs": list(MAXIMUM_ADJACENT_BRANCH_RUNS),
            "selected_v3_adapter_schema_version": (SELECTED_GUIDANCE_ADAPTER_V3_SCHEMA_VERSION),
            "selected_v3_adapter_identity_sha256": lane.adapter_identity_sha256,
            "bound_closure_identity": lane.bound_closure_identity.to_dict(),
            "bound_closure_identity_sha256": lane.bound_closure_identity_sha256,
            "equivalence_binding": lane.equivalence_binding.identity_dict(),
        },
        "execution": {
            "assignment_seed": assignment.seed,
            "assignment_sha256": assignment.assignment_sha256,
            "guidance_strength": 0.0,
            "particle_count": design.particle_count,
            "checkpoint_count": len(schedule.checkpoints),
            "selected_v3_productive_reference_manifest_sha256": (selected_reference_manifest),
            "typed_outcomes": typed_counts,
        },
        "invariants": {
            "identity_ancestry_at_every_checkpoint": True,
            "guided_post_hoc_terminal_ids_bitwise_equal": True,
            "guided_post_hoc_canonical_identities_equal": True,
            "guided_post_hoc_productive_lock_equal": True,
            "guided_post_hoc_route_utilities_equal": True,
            "guided_and_post_hoc_equal_selected_v3_direct_reference": True,
            "support_potential_and_receipt_ledger_complete": True,
            "typed_outcomes_partition_every_attempt": True,
            "base_cache_unchanged": run.cache_finalization.base_unchanged,
            "overlay_roots_isolated": run.cache_finalization.overlay_roots_isolated,
            "overlay_start_states_identical": (
                run.cache_finalization.overlay_start_states_identical
            ),
        },
        "scope": {
            "qualification_only": True,
            "production_execution": False,
            "nonzero_guidance_executed": False,
            "nonzero_guidance_authorized_by_this_receipt": False,
            "biological_guidance": False,
            "candidate_selection": False,
            "prospective_candidate_lock": False,
            "sealed_holdout_access": False,
            "synthesis_success_probability": None,
        },
    }
    result = {**result_content, "result_sha256": _sha256_payload(result_content)}
    return result, run_artifact, support_artifact


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "EXPECTED_DESIGN",
    "EXPECTED_INPUT_KEYS",
    "EXPECTED_SCOPE",
    "ProductionGuidanceRouteEvaluatorV3",
    "RESULT_SCHEMA_VERSION",
    "UgiProductionZeroGuidanceSeamV3Error",
    "_annotate_support_record_coordinates",
    "run_production_zero_guidance_seam_v3",
]
