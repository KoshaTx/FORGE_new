"""Combined lambda-zero production seam for the selected v2 Ugi lane.

This qualification composes the selected step-2000 bond-stochastic generator,
the grouped 16-program by 4-particle runner, the current cumulative route
source, isolated planner-cache overlays and the frozen binary exact-dossier
utility.  Guidance strength is exactly zero.  The run may evaluate routes at
checkpoints, but every ancestry decision must remain identity and the guided
and post-hoc productive terminal pools must remain byte-identical.

Passing this module is an integration qualification.  It does not execute or
authorize nonzero guidance, biology, prospective selection or holdout access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.data.r1_prime_audit import sha256_file
from forge.design.flow.ugi_restartable_terminal_support_adapter import (
    native_completion_record_from_locked_terminal,
)
from forge.design.flow.ugi_selected_guidance_adapter_v2 import (
    build_selected_model_restartable_guidance_lane_v2,
)
from forge.design.sampling.ugi_selected_restartable_generator_v2 import (
    GENERATOR_CHECKPOINT_SHA256,
    MAXIMUM_ADJACENT_BRANCH_RUNS,
    PRODUCTION_GENERATOR_MANIFEST_SHA256,
    TERMINAL_DECODER_ID,
)
from forge.design.schedule.ugi_matched_budget_orchestration import (
    MatchedArm,
    MatchedAssessmentContext,
)
from forge.design.schedule.ugi_matched_planner_cache_binding import (
    LazyMatchedPlannerCacheBinding,
    preflight_lazy_matched_planner_cache_binding,
)
from forge.design.schedule.ugi_nonzero_guidance_runner import (
    GuidanceAssessmentContext,
    GuidanceCacheIsolationContract,
    GuidanceComputeBudget,
    GuidanceRouteEvaluation,
    GuidanceSchedule,
    ParticleGroupDesign,
    development_guidance_run_to_dict,
    finalize_lazy_matched_cache_binding,
    load_grouped_smc_schedule_qualification,
    run_development_matched_guidance,
)
from forge.design.schedule.ugi_production_terminal_route_evaluator import (
    ProductionTerminalSupportAudit,
    ProductionUgiTerminalAwarePlannerFactory,
    build_production_ugi_terminal_aware_planner_factory,
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

CONFIG_SCHEMA_VERSION = "phase1_ugi_production_zero_guidance_seam_v2_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_production_zero_guidance_seam_v2.v1"


class UgiProductionZeroGuidanceSeamV2Error(RuntimeError):
    """Raised when the current production seam does not preserve lambda-zero identity."""


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiProductionZeroGuidanceSeamV2Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiProductionZeroGuidanceSeamV2Error(f"{label} must be a JSON object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise UgiProductionZeroGuidanceSeamV2Error(f"{label} pin is malformed")
    path = (repo / record["path"]).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiProductionZeroGuidanceSeamV2Error(f"{label} path escapes repository") from error
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiProductionZeroGuidanceSeamV2Error(f"{label} hash changed")
    return path


def _canonical_identity(terminal_bytes: bytes) -> str:
    return ValidatedUgiTerminalPayload.from_bytes(terminal_bytes).product_smiles


@dataclass
class ProductionGuidanceRouteEvaluator:
    """Adapt the production terminal assessment to the runner's binary utility."""

    binding: LazyMatchedPlannerCacheBinding
    factory: ProductionUgiTerminalAwarePlannerFactory
    support_audits: list[ProductionTerminalSupportAudit] = field(default_factory=list)

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
            post_hoc_lock_manifest_sha256=(context.post_hoc_productive_lock_sha256),
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
        self.support_audits.append(planner.support_audit)
        return GuidanceRouteEvaluation(
            route_completion_utility=bridge.route_completion_utility,
            value_policy_id=UGI_EXACT_CLOSURE_GUIDANCE_POLICY_SHA256,
            usage=receipt.realized_route_usage,
            assessment_receipt_sha256=receipt.assessment_sha256,
            route_dossier_sha256=(receipt.assessment_sha256 if bridge.support_bonus else None),
        )


def _validate_config(repo: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = _load(config_path, label="production zero-guidance seam config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiProductionZeroGuidanceSeamV2Error("unsupported seam config schema")
    if config.get("scope") != {
        "guidance_strength": 0.0,
        "nonzero_guidance": False,
        "biological_guidance": False,
        "candidate_selection": False,
        "sealed_holdout_access": False,
    }:
        raise UgiProductionZeroGuidanceSeamV2Error("seam scope changed")
    if config.get("design") != {
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
    }:
        raise UgiProductionZeroGuidanceSeamV2Error("seam design changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise UgiProductionZeroGuidanceSeamV2Error("seam inputs are missing")
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
        raise UgiProductionZeroGuidanceSeamV2Error(
            "current cumulative route source is not authenticated for this seam"
        )
    utility = _load(paths["route_completion_utility"], label="route utility")
    if (
        utility.get("status") != "binary_exact_dossier_route_completion_utility_qualified"
        or utility.get("implementation_policy", {}).get("policy_sha256")
        != UGI_EXACT_CLOSURE_GUIDANCE_POLICY_SHA256
    ):
        raise UgiProductionZeroGuidanceSeamV2Error("binary route utility is not qualified")
    manifest = _load(paths["production_generator_manifest"], label="production generator")
    if (
        sha256_file(paths["production_generator_manifest"]) != PRODUCTION_GENERATOR_MANIFEST_SHA256
        or manifest.get("identity", {}).get("checkpoint_step") != 2000
        or manifest.get("identity", {}).get("terminal_decoder") != "bond_stochastic"
    ):
        raise UgiProductionZeroGuidanceSeamV2Error("selected production generator changed")
    return config, paths


def run_production_zero_guidance_seam_v2(
    repo: Path,
    config_path: Path,
    cache_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute the current production composition once at exactly lambda zero."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    cache_root = cache_root.resolve()
    config, paths = _validate_config(repo, config_path)
    if cache_root.exists() and any(cache_root.iterdir()):
        raise UgiProductionZeroGuidanceSeamV2Error("seam cache root must be absent or empty")
    cache_root.mkdir(parents=True, exist_ok=True)

    schedule_qualification = load_grouped_smc_schedule_qualification(paths["grouped_schedule"])
    assignments = schedule_qualification.by_seed()
    assignment_seed = config["design"]["assignment_seed"]
    if assignment_seed not in assignments:
        raise UgiProductionZeroGuidanceSeamV2Error("frozen calibration assignment is missing")
    assignment = assignments[assignment_seed]
    lane = build_selected_model_restartable_guidance_lane_v2(repo)
    factory = build_production_ugi_terminal_aware_planner_factory(
        repo_root=repo,
        assessment_as_of_utc=config["assessment_as_of_utc"],
        expected_cumulative_source_inputs_sha256=(config["cumulative_source_inputs_sha256"]),
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
        guided_clone_id=f"v2-seam-guided-{assignment.assignment_sha256[:16]}",
        post_hoc_clone_id=f"v2-seam-post-hoc-{assignment.assignment_sha256[:16]}",
        isolated_overlay_roots=True,
        production_adapter_qualified=False,
    )
    evaluator = ProductionGuidanceRouteEvaluator(binding=binding, factory=factory)
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
        cache_finalizer=lambda: finalize_lazy_matched_cache_binding(
            binding,
            cache_contract,
        ),
    )
    if any(record.resampled for record in run.guided.checkpoint_groups):
        raise UgiProductionZeroGuidanceSeamV2Error(
            "lambda-zero guidance unexpectedly resampled an ancestry"
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
        raise UgiProductionZeroGuidanceSeamV2Error(
            "lambda-zero guided and post-hoc productive pools differ"
        )
    run_value = development_guidance_run_to_dict(run)
    support_audit_manifest = _sha256_payload(
        [audit.to_dict() for audit in evaluator.support_audits]
    )
    result_content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "selected_v2_production_lambda_zero_seam_qualified",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "selected_generator": {
            "checkpoint_sha256": GENERATOR_CHECKPOINT_SHA256,
            "terminal_decoder_id": TERMINAL_DECODER_ID,
            "maximum_adjacent_branch_runs": list(MAXIMUM_ADJACENT_BRANCH_RUNS),
            "adapter_identity_sha256": lane.adapter_identity_sha256,
        },
        "execution": {
            "assignment_seed": assignment.seed,
            "assignment_sha256": assignment.assignment_sha256,
            "guidance_strength": 0.0,
            "particle_count": design.particle_count,
            "checkpoint_count": len(schedule.checkpoints),
            "guided_productive_pool_size": run.guided.productive_pool_size,
            "post_hoc_productive_pool_size": run.post_hoc.productive_pool_size,
            "guided_valid_productive_count": sum(
                item.canonical_identity is not None for item in run.guided.productive_admissions
            ),
            "route_support_bonus_count": sum(
                value == 1.0 for value in run.guided.productive_route_completion_utilities
            ),
            "route_censored_count": sum(
                value is None for value in run.guided.productive_route_completion_utilities
            ),
            "support_audit_count": len(evaluator.support_audits),
            "support_audit_manifest_sha256": support_audit_manifest,
        },
        "invariants": {
            "identity_ancestry_at_every_checkpoint": True,
            "guided_post_hoc_terminal_ids_bitwise_equal": True,
            "guided_post_hoc_canonical_identities_equal": True,
            "guided_post_hoc_productive_lock_equal": True,
            "guided_post_hoc_route_utilities_equal": True,
            "base_cache_unchanged": run.cache_finalization.base_unchanged,
            "overlay_roots_isolated": run.cache_finalization.overlay_roots_isolated,
            "overlay_start_states_identical": (
                run.cache_finalization.overlay_start_states_identical
            ),
        },
        "scope": {
            "nonzero_guidance_executed": False,
            "nonzero_guidance_authorized_by_this_receipt": False,
            "biological_guidance": False,
            "candidate_selection": False,
            "sealed_holdout_access": False,
            "synthesis_success_probability": None,
        },
    }
    result = {**result_content, "result_sha256": _sha256_payload(result_content)}
    return result, run_value


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "ProductionGuidanceRouteEvaluator",
    "RESULT_SCHEMA_VERSION",
    "UgiProductionZeroGuidanceSeamV2Error",
    "run_production_zero_guidance_seam_v2",
]
