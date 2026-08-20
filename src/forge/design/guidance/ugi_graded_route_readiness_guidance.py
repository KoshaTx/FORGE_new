"""Matched synthesis guidance using adjudicated route readiness, not raw proposals."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_file
from forge.core.hashing import sha256_json as _sha256_payload
from forge.core.io import stable_json as _stable_json
from forge.design.ugi_graded_route_readiness_evaluator import (
    UGI_GRADED_ROUTE_READINESS_POLICY_SHA256,
    GradedRouteReadinessEvaluator,
    load_graded_component_index,
    load_semantic_proposal_index,
)
from forge.design.ugi_matched_planner_cache_binding import (
    preflight_lazy_matched_planner_cache_binding,
)
from forge.design.ugi_nonzero_guidance_runner import (
    GuidanceCacheIsolationContract,
    GuidanceComputeBudget,
    GuidanceSchedule,
    ParticleGroupDesign,
    finalize_lazy_matched_cache_binding,
    load_grouped_smc_schedule_qualification,
    run_development_matched_guidance,
)
from forge.design.ugi_production_synthesis_guidance_seam_v4 import (
    EXPECTED_DESIGN,
    _annotate_support_record_coordinates,
    _artifact_pin,
    _build_ancestry_and_difference_diagnostics,
    _build_run_artifact,
    _build_support_artifact,
    _canonical_identity,
    _json_value,
    _require_selected_v3_productive_identity,
)
from forge.design.ugi_production_synthesis_guidance_seam_v4 import (
    _validate_config as _validate_v4_config,
)
from forge.design.ugi_production_terminal_route_evaluator import (
    build_production_ugi_terminal_aware_planner_factory,
)
from forge.design.ugi_production_zero_guidance_seam_v3 import (
    ProductionGuidanceRouteEvaluatorV3,
)
from forge.design.ugi_restartable_terminal_support_adapter import (
    native_completion_record_from_locked_terminal,
)
from forge.design.ugi_selected_guidance_adapter_v3 import (
    build_selected_model_restartable_guidance_lane_v3,
)
from forge.design.ugi_selected_restartable_generator_v2 import (
    GENERATOR_CHECKPOINT_SHA256,
)
from forge.design.ugi_zero_guidance_typed_audit import derive_typed_counts
from forge.route.engine.planner_cache import FilePlannerCache

CONFIG_SCHEMA_VERSION = "phase1_ugi_graded_route_readiness_guidance_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_graded_route_readiness_guidance.v1"
DIAGNOSTICS_SCHEMA_VERSION = "phase1_ugi_graded_route_readiness_guidance_diagnostics.v1"
GUIDANCE_STRENGTH = 0.25
REVIEW_TOKEN = "graded-route-readiness-lambda-0.25-reviewed"


class UgiGradedRouteReadinessGuidanceError(RuntimeError):
    """Raised when the matched graded-readiness experiment changes or fails."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise UgiGradedRouteReadinessGuidanceError(f"JSON object required: {path}")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise UgiGradedRouteReadinessGuidanceError(f"{label} pin is malformed")
    path = (repo / str(record["path"])).resolve()
    if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
        raise UgiGradedRouteReadinessGuidanceError(f"{label} input changed")
    return path


def _readiness_counts(typed: dict[str, Any]) -> dict[str, Any]:
    """Relabel runner-compatible binary fields without changing counts."""

    text = _stable_json(typed)
    text = text.replace('"route_complete"', '"route_ready"')
    text = text.replace('"route_incomplete"', '"route_not_ready"')
    return json.loads(text)


def _generation_only_post_hoc_equivalence(
    current: dict[str, Any], hardened_zero: dict[str, Any]
) -> dict[str, Any]:
    """Verify that changing route evidence did not change unguided generation.

    Route utilities and assessment receipts are expected to differ because this
    experiment deliberately replaces exact-only assessment with the frozen
    graded-readiness policy. Terminal identities, bytes, traces, and checkpoint
    completions must nevertheless reproduce the hardened zero-guidance
    generator exactly.
    """

    current_payload = current["run"] if isinstance(current.get("run"), dict) else current
    zero_payload = (
        hardened_zero["run"] if isinstance(hardened_zero.get("run"), dict) else hardened_zero
    )
    current_post_hoc = current_payload.get("post_hoc")
    zero_post_hoc = zero_payload.get("post_hoc")
    if not isinstance(current_post_hoc, dict) or not isinstance(zero_post_hoc, dict):
        raise UgiGradedRouteReadinessGuidanceError("post-hoc generation records are malformed")

    equality: dict[str, bool] = {}
    for field in ("productive_terminal_ids", "productive_canonical_identities"):
        equality[field] = current_post_hoc.get(field) == zero_post_hoc.get(field)

    def admission_column(payload: dict[str, Any], field: str) -> list[Any]:
        admissions = payload.get("productive_admissions")
        if not isinstance(admissions, list) or any(
            not isinstance(record, dict) for record in admissions
        ):
            raise UgiGradedRouteReadinessGuidanceError("productive admission records are malformed")
        return [record.get(field) for record in admissions]

    for field in ("terminal_sha256", "generation_trace_sha256"):
        equality[f"productive_{field}"] = admission_column(
            current_post_hoc, field
        ) == admission_column(zero_post_hoc, field)

    current_groups = current_post_hoc.get("checkpoint_groups")
    zero_groups = zero_post_hoc.get("checkpoint_groups")
    if (
        not isinstance(current_groups, list)
        or not isinstance(zero_groups, list)
        or len(current_groups) != len(zero_groups)
    ):
        raise UgiGradedRouteReadinessGuidanceError("checkpoint groups are malformed")
    for field in (
        "shadow_terminal_ids",
        "shadow_terminal_sha256s",
        "shadow_trace_sha256s",
    ):
        equality[f"checkpoint_{field}"] = [row.get(field) for row in current_groups] == [
            row.get(field) for row in zero_groups
        ]
    if not all(equality.values()):
        failed = sorted(field for field, passed in equality.items() if not passed)
        raise UgiGradedRouteReadinessGuidanceError(
            "graded evaluator changed unguided generation: " + ", ".join(failed)
        )
    return {
        "fields": equality,
        "all_generation_fields_bitwise_equal": True,
        "route_utility_and_assessment_fields_intentionally_excluded": True,
    }


def _build_graded_diagnostics(
    run_payload: dict[str, Any],
    hardened_zero: dict[str, Any],
    assignment: Any,
    *,
    adapter_identity_sha256: str,
) -> dict[str, Any]:
    """Build ancestry diagnostics under the correct graded-evidence invariant."""

    generation_equivalence = _generation_only_post_hoc_equivalence(run_payload, hardened_zero)
    # The inherited validator still supplies the detailed coordinate, ancestry,
    # and guided-vs-post-hoc checks. Its exact-only route-policy comparison is
    # bypassed only after generation identity against the real hardened zero run
    # has independently passed above.
    diagnostics = _build_ancestry_and_difference_diagnostics(
        run_payload,
        {"run": {"post_hoc": run_payload["post_hoc"]}},
        assignment,
        adapter_identity_sha256=adapter_identity_sha256,
    )
    diagnostics.pop("result_sha256", None)
    diagnostics["schema_version"] = DIAGNOSTICS_SCHEMA_VERSION
    diagnostics["status"] = "graded_route_readiness_ancestry_and_difference_diagnostics_complete"
    diagnostics.pop("post_hoc_vs_hardened_zero", None)
    diagnostics["post_hoc_generation_vs_hardened_zero"] = generation_equivalence
    diagnostics["interpretation"].update(
        {
            "route_policy_changed_from_hardened_zero": True,
            "cross_policy_route_utility_equality_required": False,
            "unguided_generation_identity_required": True,
        }
    )
    diagnostics["result_sha256"] = _sha256_payload(diagnostics)
    return diagnostics


def run_graded_route_readiness_guidance(
    repo: Path,
    config_path: Path,
    cache_root: Path,
    output_dir: Path,
    *,
    review_token: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run one frozen matched route-readiness guidance qualification."""

    if review_token != REVIEW_TOKEN:
        raise UgiGradedRouteReadinessGuidanceError("explicit review token is required")
    repo = repo.resolve()
    config_path = config_path.resolve()
    cache_root = cache_root.resolve()
    output_dir = output_dir.resolve()
    config = _load(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiGradedRouteReadinessGuidanceError("unsupported config schema")
    if float(config.get("guidance_strength")) != GUIDANCE_STRENGTH:
        raise UgiGradedRouteReadinessGuidanceError("guidance strength changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "base_v4_config",
        "graded_evidence_ledger",
        "semantic_proposal_ledger",
        "semantic_proposal_result",
    }:
        raise UgiGradedRouteReadinessGuidanceError("input set changed")
    pinned = {label: _pin(repo, record, label=label) for label, record in inputs.items()}
    semantic_result = _load(pinned["semantic_proposal_result"])
    if (
        semantic_result.get("status") != "frozen_proposal_semantic_readjudication_complete"
        or semantic_result.get("scientific_authority", {}).get("may_enter_synthesis_value")
        is not False
    ):
        raise UgiGradedRouteReadinessGuidanceError("proposal authority contract changed")
    base_config, paths, zero = _validate_v4_config(repo, pinned["base_v4_config"])
    if cache_root.exists() and any(cache_root.iterdir()):
        raise UgiGradedRouteReadinessGuidanceError("cache root must be absent or empty")
    cache_root.mkdir(parents=True, exist_ok=True)

    qualification = load_grouped_smc_schedule_qualification(paths["grouped_schedule"])
    assignment = qualification.by_seed()[EXPECTED_DESIGN["assignment_seed"]]
    lane = build_selected_model_restartable_guidance_lane_v3(repo)
    reference = _require_selected_v3_productive_identity(zero.run["run"], lane)
    if reference != zero.result["execution"]["selected_v3_productive_reference_manifest_sha256"]:
        raise UgiGradedRouteReadinessGuidanceError("selected generator identity changed")

    factory = build_production_ugi_terminal_aware_planner_factory(
        repo_root=repo,
        assessment_as_of_utc=base_config["assessment_as_of_utc"],
        expected_cumulative_source_inputs_sha256=base_config["cumulative_source_inputs_sha256"],
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
        assessment_at_utc=base_config["assessment_as_of_utc"],
    )
    zero_cache = zero.run["run"]["cache_isolation"]
    if (
        binding.preflight.base.snapshot_sha256 != zero_cache["base_snapshot_sha256"]
        or binding.preflight.context_sha256 != zero_cache["planner_context_sha256"]
        or binding.preflight.preflight_sha256 != zero_cache["cache_preflight_sha256"]
    ):
        raise UgiGradedRouteReadinessGuidanceError("planner context changed")
    cache_contract = GuidanceCacheIsolationContract(
        base_snapshot_sha256=binding.preflight.base.snapshot_sha256,
        planner_context_sha256=binding.preflight.context_sha256,
        cache_preflight_sha256=binding.preflight.preflight_sha256,
        guided_clone_id=f"graded-readiness-guided-{assignment.assignment_sha256[:16]}",
        post_hoc_clone_id=f"graded-readiness-posthoc-{assignment.assignment_sha256[:16]}",
        isolated_overlay_roots=True,
        production_adapter_qualified=False,
    )
    strict = ProductionGuidanceRouteEvaluatorV3(binding=binding, factory=factory)
    evaluator = GradedRouteReadinessEvaluator(
        strict_evaluator=strict,
        graded_components=load_graded_component_index(pinned["graded_evidence_ledger"]),
        semantic_proposals=load_semantic_proposal_index(pinned["semantic_proposal_ledger"]),
    )
    design = ParticleGroupDesign(
        morphology_program_count=EXPECTED_DESIGN["morphology_program_count"],
        particles_per_program=EXPECTED_DESIGN["particles_per_program"],
    )
    schedule = GuidanceSchedule(
        sample_steps=EXPECTED_DESIGN["sample_steps"],
        checkpoints=tuple(EXPECTED_DESIGN["checkpoints"]),
        checkpoint_betas=tuple(EXPECTED_DESIGN["checkpoint_betas"]),
        rollouts_per_particle_checkpoint=EXPECTED_DESIGN["rollouts_per_particle_checkpoint"],
        final_selection_count=EXPECTED_DESIGN["final_selection_count"],
    )
    budget = GuidanceComputeBudget(
        product_transition_calls=EXPECTED_DESIGN["product_transition_calls"],
        terminal_completions=EXPECTED_DESIGN["terminal_completions"],
        logical_planner_calls=EXPECTED_DESIGN["logical_planner_calls"],
        logical_verifier_calls=EXPECTED_DESIGN["logical_verifier_calls"],
        final_candidates=EXPECTED_DESIGN["final_selection_count"],
    )
    run = run_development_matched_guidance(
        assignment,
        lane=lane,
        evaluator=evaluator,
        canonical_identity=_canonical_identity,
        design=design,
        schedule=schedule,
        budget=budget,
        guidance_strength=GUIDANCE_STRENGTH,
        device="cpu",
        expected_value_policy_id=UGI_GRADED_ROUTE_READINESS_POLICY_SHA256,
        cache_contract=cache_contract,
        cache_finalizer=lambda: finalize_lazy_matched_cache_binding(binding, cache_contract),
    )
    run_artifact = _build_run_artifact(_json_value(asdict(run)))
    _annotate_support_record_coordinates(run_artifact["run"], evaluator.support_records)
    support_artifact = _build_support_artifact(evaluator.support_records)
    typed = derive_typed_counts(run_artifact, support_artifact)
    diagnostics = _build_graded_diagnostics(
        run_artifact["run"],
        zero.run,
        assignment,
        adapter_identity_sha256=lane.adapter_identity_sha256,
    )
    artifacts = {
        "run": _artifact_pin(repo, output_dir, "run.json", run_artifact),
        "support_audits": _artifact_pin(repo, output_dir, "support_audits.json", support_artifact),
        "diagnostics": _artifact_pin(repo, output_dir, "diagnostics.json", diagnostics),
    }
    result_content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "graded_route_readiness_guidance_qualification_complete",
        "config": {"path": str(config_path.relative_to(repo)), "sha256": sha256_file(config_path)},
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(pinned.items())
        },
        "artifacts": artifacts,
        "execution": {
            "assignment_seed": assignment.seed,
            "assignment_sha256": assignment.assignment_sha256,
            "guidance_strength": GUIDANCE_STRENGTH,
            "value_policy_sha256": UGI_GRADED_ROUTE_READINESS_POLICY_SHA256,
            "typed_route_readiness_outcomes": _readiness_counts(typed),
            "diagnostics_result_sha256": diagnostics["result_sha256"],
        },
        "invariants": {
            "same_frozen_generator": True,
            "same_terminal_completions_and_route_compute_budget": True,
            "same_strict_route_planner_called_in_both_arms": True,
            "proposal_model_score_used": False,
            "proposal_only_hypotheses_authorized": False,
            "family_projection_promoted_to_exact": False,
            "synthesis_success_probability_defined": False,
            "base_cache_unchanged": run.cache_finalization.base_unchanged,
            "overlay_roots_isolated": run.cache_finalization.overlay_roots_isolated,
        },
        "decision": {
            "synthesis_tilting_promoted": False,
            "next_gate": "compare guided and post-hoc unique route-ready yield at matched compute",
        },
        "scope": {
            "qualification_only": True,
            "production_execution": False,
            "candidate_selection": False,
            "biological_guidance": False,
            "prospective_candidate_lock": False,
        },
    }
    result = {**result_content, "result_sha256": _sha256_payload(result_content)}
    return result, run_artifact, support_artifact, diagnostics


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "GUIDANCE_STRENGTH",
    "RESULT_SCHEMA_VERSION",
    "REVIEW_TOKEN",
    "UgiGradedRouteReadinessGuidanceError",
    "run_graded_route_readiness_guidance",
]
