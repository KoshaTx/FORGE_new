"""Selected-v3 synthesis-only guidance qualification at lambda 0.25.

This additive seam executes one preregistered calibration assignment through
the equivalence-bound selected-v3 generator, the same production route
evaluator used by the hardened lambda-zero seam, and isolated guided/post-hoc
planner-cache overlays.  It preserves the exact 16-by-4 particle design,
checkpoint schedule and matched compute ceilings.

The seam is qualification-only.  It performs no biological guidance,
prospective candidate lock, sealed-holdout access or production execution.  A
separate explicit review token is required by the public entrypoint so code,
configuration and tests can be reviewed before this nonzero run is started.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.core.io import stable_json as _stable_json
from forge.data.r1_prime_audit import sha256_file
from forge.product.ugi_matched_planner_cache_binding import (
    preflight_lazy_matched_planner_cache_binding,
)
from forge.product.ugi_nonzero_guidance_runner import (
    FrozenSeedProgramAssignment,
    GuidanceCacheIsolationContract,
    GuidanceComputeBudget,
    GuidanceSchedule,
    ParticleGroupDesign,
    finalize_lazy_matched_cache_binding,
    load_grouped_smc_schedule_qualification,
    run_development_matched_guidance,
)
from forge.product.ugi_production_terminal_route_evaluator import (
    build_production_ugi_terminal_aware_planner_factory,
)
from forge.product.ugi_production_zero_guidance_seam_v3 import (
    ProductionGuidanceRouteEvaluatorV3,
    UgiProductionZeroGuidanceSeamV3Error,
    _annotate_support_record_coordinates,
    _canonical_identity,
    _parse_productive_terminal_id,
    _require_selected_v3_productive_identity,
)
from forge.product.ugi_production_zero_guidance_seam_v3 import (
    _validate_config as _validate_zero_seam_config,
)
from forge.product.ugi_restartable_terminal_support_adapter import (
    native_completion_record_from_locked_terminal,
)
from forge.product.ugi_selected_guidance_adapter_v3 import (
    SELECTED_GUIDANCE_ADAPTER_V3_SCHEMA_VERSION,
    build_selected_model_restartable_guidance_lane_v3,
)
from forge.product.ugi_selected_restartable_generator_v2 import (
    GENERATOR_CHECKPOINT_SHA256,
    MAXIMUM_ADJACENT_BRANCH_RUNS,
    TERMINAL_DECODER_ID,
)
from forge.product.ugi_synthesis_guidance import keyed_random_seed
from forge.product.ugi_zero_guidance_typed_audit import (
    build_support_audit_artifact,
    derive_typed_counts,
)
from forge.route.planner_cache import FilePlannerCache
from forge.value.ugi_exact_closure_guidance import (
    UGI_EXACT_CLOSURE_GUIDANCE_POLICY_SHA256,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi_production_synthesis_guidance_seam_v4_config.v1"
RUN_SCHEMA_VERSION = "forge.ugi_selected_v3_synthesis_guidance_qualification_run.v1"
SUPPORT_SCHEMA_VERSION = "forge.ugi_selected_v3_synthesis_guidance_support_audits.v1"
DIAGNOSTICS_SCHEMA_VERSION = "forge.ugi_selected_v3_synthesis_guidance_diagnostics.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_production_synthesis_guidance_seam_v4.v1"
GUIDANCE_STRENGTH = 0.25
REVIEW_TOKEN = "selected-v3-lambda-0.25-code-config-tests-reviewed"
RERUN_TOKEN = "selected-v3-lambda-0.25-single-operational-rerun-reviewed"
EXPECTED_STRENGTH_SET = [0.0, 0.25, 0.5, 1.0, 2.0]
RETRY_AMENDMENT_FILE_SHA256 = "980b5ceb927dfc1c041dcce1dfdbf03b8c20148cb264432bb775cb8bfc3ff90b"
RETRY_AMENDMENT_LOGICAL_SHA256 = "24cb305e6e7b4eedd14ebfc615a02159646df920ae1293a53209c2c3e90e3b08"
FAILED_ATTEMPT_FILE_SHA256 = "95a5e7f1e5f68f90e110174b641ac1386c4b7919ed940cfcb557b47a3ed89106"
FAILED_ATTEMPT_LOGICAL_SHA256 = "52ef0ea2d817b6849673af0ae0ac39a1fc77c473c63f8d7e880ed2d3d6398a6e"
FAILED_CACHE_MANIFEST_SHA256 = "7b753a29764640fb99f59b1f053be39af15be713305259f961cd4018bf0792e9"
EXPECTED_SELECTED_V3_ADAPTER_IDENTITY_SHA256 = (
    "f2c26cf2a27119a2f2a54d0df8430af7e6a62cacd26cc2fc607586513e5065e6"
)
EXPECTED_BOUND_CLOSURE_IDENTITY_SHA256 = (
    "5195878ece3e13cee68373243037a98ac852fb7b96b9061b39879d9cc7a1acef"
)
ZERO_RESULT_FILE_SHA256 = "ff4fdf36dd94b4e33b6c5e84527c5049578af7ea39ca25440dc33a7d8e32da3a"
ZERO_RESULT_LOGICAL_SHA256 = "0b73258b25f790aeb25fc7a45915dacaa239e682c0582d2b333eb18f6e326611"
ZERO_RUN_FILE_SHA256 = "1627b2cdb47c6d54cc916fbb2327a86ba5420e0a5ca698607f7545335555a087"
ZERO_RUN_LOGICAL_SHA256 = "8f4b907971e197f08b8b0d738451ee7fd3d05be0452b4462d1d9521f958c37ee"
ZERO_SUPPORT_FILE_SHA256 = "66bb85edaa29cb447cc43d359db07a19f092cc8e36110d689e743e191f02f316"
ZERO_SUPPORT_LOGICAL_SHA256 = "1aba7b54bcd353e58dae8de9009ca11a09e0f6191468867398a2cb00f73a4409"
EXPECTED_SCOPE = {
    "guidance_strength": GUIDANCE_STRENGTH,
    "qualification_only": True,
    "production_execution": False,
    "nonzero_synthesis_guidance": True,
    "biological_guidance": False,
    "candidate_selection": False,
    "prospective_candidate_lock": False,
    "sealed_holdout_access": False,
    "no_seed_search": True,
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
EXPECTED_INPUT_KEYS = {
    "binary_utility_source",
    "current_source_requalification",
    "grouped_schedule",
    "matched_cache_binding",
    "matched_preregistration_v1",
    "matched_preregistration_v2",
    "nonzero_runner",
    "production_generator_manifest",
    "production_terminal_route_evaluator",
    "qualification_runner",
    "qualification_source",
    "qualification_tests",
    "retry_amendment",
    "route_completion_utility",
    "selected_guidance_adapter_v2",
    "selected_guidance_adapter_v3",
    "selected_restartable_generator_v2",
    "selected_v3_equivalence_result",
    "selected_v3_equivalence_rows",
    "typed_outcome_audit",
    "zero_seam_config",
    "zero_seam_result",
    "zero_seam_run",
    "zero_seam_runner",
    "zero_seam_source",
    "zero_seam_support",
    "zero_seam_tests",
}
EXPECTED_SELF_PATHS = {
    "qualification_source": "src/forge/product/ugi_production_synthesis_guidance_seam_v4.py",
    "qualification_runner": "scripts/phase1_qualify_ugi_production_synthesis_guidance_seam_v4.py",
    "qualification_tests": "tests/test_ugi_production_synthesis_guidance_seam_v4.py",
}


class UgiProductionSynthesisGuidanceSeamV4Error(RuntimeError):
    """Raised when the reviewed synthesis-only qualification fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return (_stable_json(value) + "\n").encode()


def _canonical_file_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_value(value: Any) -> Any:
    """Normalize dataclass containers to the exact persisted JSON representation."""

    return json.loads(_stable_json(value))


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiProductionSynthesisGuidanceSeamV4Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiProductionSynthesisGuidanceSeamV4Error(f"{label} must be a JSON object")
    return value


def _directory_file_manifest_sha256(repo: Path, root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "failed-attempt cache is missing or is not an ordinary directory"
        )
    records: list[str] = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path.is_symlink():
            raise UgiProductionSynthesisGuidanceSeamV4Error(
                "failed-attempt cache contains a symlink"
            )
        records.append(f"{sha256_file(path)}  {path.relative_to(repo)}\n")
    return hashlib.sha256("".join(records).encode()).hexdigest()


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise UgiProductionSynthesisGuidanceSeamV4Error(f"{label} pin is malformed")
    if not isinstance(record["path"], str) or not isinstance(record["sha256"], str):
        raise UgiProductionSynthesisGuidanceSeamV4Error(f"{label} pin values are malformed")
    path = (repo / record["path"]).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            f"{label} path escapes repository"
        ) from error
    if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
        raise UgiProductionSynthesisGuidanceSeamV4Error(f"{label} hash changed")
    return path


def _require_canonical_self_hash(
    path: Path,
    value: dict[str, Any],
    *,
    expected_file_sha256: str,
    expected_logical_sha256: str,
    label: str,
    logical_hash_field: str = "result_sha256",
) -> None:
    if sha256_file(path) != expected_file_sha256 or path.read_bytes() != _canonical_bytes(value):
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            f"{label} file identity or canonical serialization changed"
        )
    claimed = value.get(logical_hash_field)
    content = {key: item for key, item in value.items() if key != logical_hash_field}
    if claimed != expected_logical_sha256 or _sha256_payload(content) != claimed:
        raise UgiProductionSynthesisGuidanceSeamV4Error(f"{label} logical hash changed")


@dataclass(frozen=True)
class HardenedZeroGuidanceEvidence:
    """Authenticated lambda-zero result, run and support payloads."""

    result: dict[str, Any]
    run: dict[str, Any]
    support: dict[str, Any]


def _validate_hardened_zero_evidence(
    repo: Path,
    paths: dict[str, Path],
) -> HardenedZeroGuidanceEvidence:
    try:
        zero_config, _ = _validate_zero_seam_config(repo, paths["zero_seam_config"])
    except UgiProductionZeroGuidanceSeamV3Error as error:
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "hardened lambda-zero seam config no longer validates"
        ) from error
    result = _load(paths["zero_seam_result"], label="hardened lambda-zero result")
    run = _load(paths["zero_seam_run"], label="hardened lambda-zero run")
    support = _load(paths["zero_seam_support"], label="hardened lambda-zero support")
    _require_canonical_self_hash(
        paths["zero_seam_result"],
        result,
        expected_file_sha256=ZERO_RESULT_FILE_SHA256,
        expected_logical_sha256=ZERO_RESULT_LOGICAL_SHA256,
        label="hardened lambda-zero result",
    )
    _require_canonical_self_hash(
        paths["zero_seam_run"],
        run,
        expected_file_sha256=ZERO_RUN_FILE_SHA256,
        expected_logical_sha256=ZERO_RUN_LOGICAL_SHA256,
        label="hardened lambda-zero run",
    )
    _require_canonical_self_hash(
        paths["zero_seam_support"],
        support,
        expected_file_sha256=ZERO_SUPPORT_FILE_SHA256,
        expected_logical_sha256=ZERO_SUPPORT_LOGICAL_SHA256,
        label="hardened lambda-zero support",
    )
    if (
        result.get("schema_version") != "phase1_ugi_production_zero_guidance_seam_v3.v1"
        or result.get("status") != "selected_v3_production_lambda_zero_typed_seam_qualified"
        or result.get("config", {}).get("sha256") != sha256_file(paths["zero_seam_config"])
        or result.get("execution", {}).get("assignment_seed") != EXPECTED_DESIGN["assignment_seed"]
        or result.get("execution", {}).get("guidance_strength") != 0.0
        or result.get("scope", {}).get("nonzero_guidance_executed") is not False
        or result.get("scope", {}).get("production_execution") is not False
        or not all(result.get("invariants", {}).values())
    ):
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "hardened lambda-zero result is not a passing selected-v3 qualification"
        )
    artifacts = result.get("artifacts")
    if (
        not isinstance(artifacts, dict)
        or artifacts.get("run")
        != {
            "file_sha256": ZERO_RUN_FILE_SHA256,
            "path": str(paths["zero_seam_run"].relative_to(repo)),
            "result_sha256": ZERO_RUN_LOGICAL_SHA256,
        }
        or artifacts.get("support_audits")
        != {
            "file_sha256": ZERO_SUPPORT_FILE_SHA256,
            "path": str(paths["zero_seam_support"].relative_to(repo)),
            "result_sha256": ZERO_SUPPORT_LOGICAL_SHA256,
        }
    ):
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "hardened lambda-zero artifact chain changed"
        )
    zero_run = run.get("run")
    if (
        run.get("schema_version") != "forge.ugi_zero_guidance_qualification_run.v1"
        or run.get("status") != "zero_guidance_qualification_complete_not_production_authorized"
        or not isinstance(zero_run, dict)
        or zero_run.get("guidance_strength") != 0.0
        or zero_run.get("base_seed") != EXPECTED_DESIGN["assignment_seed"]
        or zero_run.get("particle_group_design")
        != {
            "morphology_program_count": EXPECTED_DESIGN["morphology_program_count"],
            "particles_per_program": EXPECTED_DESIGN["particles_per_program"],
        }
        or zero_run.get("budget")
        != {
            "final_candidates": EXPECTED_DESIGN["final_selection_count"],
            "logical_planner_calls": EXPECTED_DESIGN["logical_planner_calls"],
            "logical_verifier_calls": EXPECTED_DESIGN["logical_verifier_calls"],
            "product_transition_calls": EXPECTED_DESIGN["product_transition_calls"],
            "terminal_completions": EXPECTED_DESIGN["terminal_completions"],
        }
    ):
        raise UgiProductionSynthesisGuidanceSeamV4Error("hardened lambda-zero run design changed")
    records = support.get("records")
    if (
        support.get("schema_version") != "phase1_ugi_production_zero_guidance_support_audits.v1"
        or not isinstance(records, list)
        or support.get("record_count") != len(records)
        or support.get("records_manifest_sha256") != _sha256_payload(records)
        or support.get("scalar_value") is not None
        or support.get("success_probability") is not None
    ):
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "hardened lambda-zero support ledger changed"
        )
    typed = derive_typed_counts(run, support)
    if typed != result.get("execution", {}).get("typed_outcomes"):
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "hardened lambda-zero typed outcomes no longer reproduce"
        )
    if zero_config.get("assessment_as_of_utc") is None:
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "hardened lambda-zero assessment time is missing"
        )
    return HardenedZeroGuidanceEvidence(result=result, run=run, support=support)


def _validate_preregistration(
    v1_path: Path,
    v2_path: Path,
    config: dict[str, Any],
) -> None:
    v1 = _load(v1_path, label="matched synthesis-guidance preregistration v1")
    v2 = _load(v2_path, label="matched synthesis-guidance preregistration v2")
    design = v1.get("design")
    if (
        v1.get("schema_version") != "phase1_ugi_matched_synthesis_guidance_preregistration.v1"
        or v1.get("scope") != "nonexecuting_preregistered_matched_synthesis_guidance_experiment"
        or not isinstance(design, dict)
        or design.get("guidance_strengths") != EXPECTED_STRENGTH_SET
        or design.get("calibration_seeds") != [20260821, 20260822, 20260823]
        or design.get("evaluation_seeds") != [20260824, 20260825, 20260826, 20260827, 20260828]
        or design.get("no_retries") is not True
        or design.get("no_seed_search") is not True
    ):
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "matched synthesis-guidance v1 strength/seed contract changed"
        )
    if (
        v2.get("schema_version") != "phase1_ugi_matched_synthesis_guidance_preregistration.v2"
        or v2.get("status") != "frozen_amendment_nonexecuting"
        or v2.get("v1_preregistration")
        != {
            "path": str(v1_path.relative_to(v1_path.parents[2])),
            "sha256": sha256_file(v1_path),
        }
        or v2.get("particle_group_amendment", {}).get("morphology_program_count") != 16
        or v2.get("particle_group_amendment", {}).get("particles_per_program") != 4
        or v2.get("particle_group_amendment", {}).get("cross_program_ancestry") is not False
        or v2.get("schedule_amendment", {}).get("synthesis_checkpoints") != [2, 4, 6]
        or v2.get("schedule_amendment", {}).get("checkpoint_betas") != [0.25, 0.5, 0.75]
        or v2.get("matched_compute_unchanged_from_v1")
        != {
            "product_transition_calls": 1280,
            "terminal_completions": 256,
            "logical_planner_calls": 768,
            "logical_verifier_calls": 3328,
            "final_candidates": 32,
        }
    ):
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "matched synthesis-guidance v2 amendment changed"
        )
    declared = config.get("preregistration")
    if declared != {
        "guidance_strengths": EXPECTED_STRENGTH_SET,
        "qualification_strength": GUIDANCE_STRENGTH,
        "assignment_role": "first_preregistered_calibration_seed",
        "single_assignment_only": True,
        "no_retries": True,
        "no_seed_search": True,
        "not_a_calibration_winner_selection": True,
    }:
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "qualification strength declaration changed"
        )


def _validate_retry_amendment(repo: Path, path: Path) -> None:
    amendment = _load(path, label="single operational-retry amendment")
    _require_canonical_self_hash(
        path,
        amendment,
        expected_file_sha256=RETRY_AMENDMENT_FILE_SHA256,
        expected_logical_sha256=RETRY_AMENDMENT_LOGICAL_SHA256,
        label="single operational-retry amendment",
        logical_hash_field="amendment_sha256",
    )
    prior = amendment.get("prior_failure")
    if prior != {
        "path": "results/phase1/ugi_production_synthesis_guidance_seam_v4_failure_v1/result.json",
        "file_sha256": FAILED_ATTEMPT_FILE_SHA256,
        "result_sha256": FAILED_ATTEMPT_LOGICAL_SHA256,
        "status": "failed_closed_after_execution_before_artifact_publication",
        "failure_stage": "post-run ancestry_and_difference_diagnostics",
        "failure_category": "postprocessing_representation_contract",
    }:
        raise UgiProductionSynthesisGuidanceSeamV4Error("retry failure receipt changed")
    failure_path = (repo / prior["path"]).resolve()
    if (
        not failure_path.is_file()
        or failure_path.is_symlink()
        or sha256_file(failure_path) != FAILED_ATTEMPT_FILE_SHA256
    ):
        raise UgiProductionSynthesisGuidanceSeamV4Error("retry failure receipt hash changed")
    failure = _load(failure_path, label="failed first-attempt receipt")
    if (
        failure.get("result_sha256") != FAILED_ATTEMPT_LOGICAL_SHA256
        or failure.get("status") != prior["status"]
        or failure.get("claims", {}).get("lambda_0_25_qualified") is not False
        or failure.get("claims", {}).get("synthesis_guidance_improved_generation") is not None
    ):
        raise UgiProductionSynthesisGuidanceSeamV4Error("retry failure receipt content changed")
    if amendment.get("scientific_blinding_attestation") != {
        "terminal_outcomes_inspected": False,
        "route_closure_results_inspected": False,
        "ancestry_results_inspected": False,
        "candidate_structures_inspected": False,
        "failed_cache_used_for_parameter_selection": False,
        "failed_cache_used_for_model_or_policy_tuning": False,
        "only_failure_type_and_cache_integrity_metadata_inspected": True,
    }:
        raise UgiProductionSynthesisGuidanceSeamV4Error("retry blinding attestation changed")
    authorization = amendment.get("retry_authorization")
    if (
        not isinstance(authorization, dict)
        or authorization.get("operational_retry_count") != 1
        or authorization.get("additional_retry_count_without_new_decision") != 0
        or authorization.get("guidance_strength") != GUIDANCE_STRENGTH
        or authorization.get("assignment_seed") != EXPECTED_DESIGN["assignment_seed"]
        or not all(
            authorization.get(field) is True
            for field in (
                "strength_change_forbidden",
                "seed_change_forbidden",
                "design_change_forbidden",
                "route_evidence_change_forbidden",
                "generator_change_forbidden",
                "no_seed_search",
                "no_parameter_tuning",
            )
        )
    ):
        raise UgiProductionSynthesisGuidanceSeamV4Error("retry authorization changed")
    preservation = amendment.get("preservation_contract")
    if preservation != {
        "failed_cache_path": "results/phase1/.ugi_production_synthesis_guidance_seam_v4.cache",
        "failed_cache_file_manifest_sha256": FAILED_CACHE_MANIFEST_SHA256,
        "failed_cache_must_remain_unchanged": True,
        "failed_result_must_remain_unchanged": True,
        "retry_output_path": "results/phase1/ugi_production_synthesis_guidance_seam_v4_retry1",
        "retry_cache_path": "results/phase1/.ugi_production_synthesis_guidance_seam_v4_retry1.cache",
        "failed_cache_reuse_forbidden": True,
        "retry_output_overwrite_forbidden": True,
    }:
        raise UgiProductionSynthesisGuidanceSeamV4Error("retry preservation contract changed")
    failed_cache = (repo / preservation["failed_cache_path"]).resolve()
    if _directory_file_manifest_sha256(repo, failed_cache) != FAILED_CACHE_MANIFEST_SHA256:
        raise UgiProductionSynthesisGuidanceSeamV4Error("failed-attempt cache manifest changed")
    if amendment.get("scope") != {
        "qualification_only": True,
        "production_execution": False,
        "nonzero_synthesis_guidance": True,
        "biological_guidance": False,
        "candidate_selection": False,
        "prospective_candidate_lock": False,
        "sealed_holdout_access": False,
        "synthesis_success_probability": None,
    }:
        raise UgiProductionSynthesisGuidanceSeamV4Error("retry scope changed")


def _validate_config(
    repo: Path,
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Path], HardenedZeroGuidanceEvidence]:
    config = _load(config_path, label="selected-v3 synthesis-guidance seam-v4 config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiProductionSynthesisGuidanceSeamV4Error("unsupported seam-v4 config schema")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiProductionSynthesisGuidanceSeamV4Error("seam-v4 scope changed")
    if config.get("design") != EXPECTED_DESIGN:
        raise UgiProductionSynthesisGuidanceSeamV4Error("seam-v4 design changed")
    if config.get("review_gate") != {
        "review_required_before_execution": True,
        "review_token": REVIEW_TOKEN,
        "execution_authorized_by_config_alone": False,
    }:
        raise UgiProductionSynthesisGuidanceSeamV4Error("seam-v4 review gate changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != EXPECTED_INPUT_KEYS:
        raise UgiProductionSynthesisGuidanceSeamV4Error("seam-v4 input set changed")
    paths = {label: _pin(repo, record, label=label) for label, record in inputs.items()}
    for label, expected_path in EXPECTED_SELF_PATHS.items():
        if str(paths[label].relative_to(repo)) != expected_path:
            raise UgiProductionSynthesisGuidanceSeamV4Error(f"{label} path changed")
    _validate_preregistration(
        paths["matched_preregistration_v1"],
        paths["matched_preregistration_v2"],
        config,
    )
    _validate_retry_amendment(repo, paths["retry_amendment"])
    zero = _validate_hardened_zero_evidence(repo, paths)
    if config.get("assessment_as_of_utc") != zero.run.get("run", {}).get("cache_isolation", {}).get(
        "assessment_as_of_utc"
    ) and config.get("assessment_as_of_utc") != _load(
        paths["zero_seam_config"], label="zero config"
    ).get("assessment_as_of_utc"):
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "seam-v4 route-evidence timestamp differs from lambda zero"
        )
    selected = config.get("selected_v3_identities")
    if selected != {
        "adapter_schema_version": SELECTED_GUIDANCE_ADAPTER_V3_SCHEMA_VERSION,
        "adapter_identity_sha256": EXPECTED_SELECTED_V3_ADAPTER_IDENTITY_SHA256,
        "bound_closure_identity_sha256": EXPECTED_BOUND_CLOSURE_IDENTITY_SHA256,
    }:
        raise UgiProductionSynthesisGuidanceSeamV4Error("selected-v3 identities changed")
    zero_declared = config.get("hardened_zero_guidance")
    if zero_declared != {
        "result_file_sha256": ZERO_RESULT_FILE_SHA256,
        "result_logical_sha256": ZERO_RESULT_LOGICAL_SHA256,
        "run_file_sha256": ZERO_RUN_FILE_SHA256,
        "run_logical_sha256": ZERO_RUN_LOGICAL_SHA256,
        "support_file_sha256": ZERO_SUPPORT_FILE_SHA256,
        "support_logical_sha256": ZERO_SUPPORT_LOGICAL_SHA256,
    }:
        raise UgiProductionSynthesisGuidanceSeamV4Error("hardened lambda-zero declaration changed")
    return config, paths, zero


def _parse_terminal_id(value: Any, *, label: str) -> dict[str, str]:
    try:
        return _parse_productive_terminal_id(value, label=label)
    except UgiProductionZeroGuidanceSeamV3Error as error:
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            f"{label} violates the selected-v3 terminal-ID contract"
        ) from error


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise UgiProductionSynthesisGuidanceSeamV4Error(f"{label} is not a SHA-256 digest")
    return value


def _selected_overlap(left: list[Any], right: list[Any]) -> dict[str, Any]:
    left_set = {value for value in left if isinstance(value, str)}
    right_set = {value for value in right if isinstance(value, str)}
    union = left_set | right_set
    intersection = left_set & right_set
    return {
        "left_count": len(left_set),
        "right_count": len(right_set),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "jaccard": 1.0 if not union else len(intersection) / len(union),
    }


def _build_ancestry_and_difference_diagnostics(
    run_payload: dict[str, Any],
    zero_payload: dict[str, Any],
    assignment: FrozenSeedProgramAssignment,
    *,
    adapter_identity_sha256: str,
) -> dict[str, Any]:
    if run_payload.get("base_seed") != assignment.seed:
        raise UgiProductionSynthesisGuidanceSeamV4Error("run seed differs from assignment")
    zero_run = zero_payload.get("run")
    if not isinstance(zero_run, dict):
        raise UgiProductionSynthesisGuidanceSeamV4Error("zero baseline run is malformed")
    terminal_contract_rows: list[dict[str, Any]] = []
    ancestry_by_checkpoint: dict[str, dict[str, int]] = {
        str(step): {
            "groups": 0,
            "resampled_groups": 0,
            "nonuniform_probability_groups": 0,
            "mixed_observed_utility_groups": 0,
            "nonidentity_destination_count": 0,
            "duplicated_ancestor_draw_count": 0,
        }
        for step in EXPECTED_DESIGN["checkpoints"]
    }
    arm_checkpoint_pool_fields: dict[tuple[str, int], set[tuple[str, str, str]]] = {}
    for arm_name in ("guided", "post_hoc"):
        arm = run_payload.get(arm_name)
        if not isinstance(arm, dict):
            raise UgiProductionSynthesisGuidanceSeamV4Error(f"{arm_name} run is malformed")
        groups = arm.get("checkpoint_groups")
        if not isinstance(groups, list) or len(groups) != 48:
            raise UgiProductionSynthesisGuidanceSeamV4Error(
                f"{arm_name} checkpoint-group coverage changed"
            )
        expected_coordinates = [
            (checkpoint, program)
            for checkpoint in EXPECTED_DESIGN["checkpoints"]
            for program in range(EXPECTED_DESIGN["morphology_program_count"])
        ]
        observed_coordinates = [
            (
                (group.get("checkpoint"), group.get("program_index"))
                if isinstance(group, dict)
                else None
            )
            for group in groups
        ]
        if observed_coordinates != expected_coordinates:
            raise UgiProductionSynthesisGuidanceSeamV4Error(
                f"{arm_name} checkpoint-group ordering changed"
            )
        for group in groups:
            checkpoint = group["checkpoint"]
            program_index = group["program_index"]
            start = program_index * EXPECTED_DESIGN["particles_per_program"]
            expected_indices = list(range(start, start + EXPECTED_DESIGN["particles_per_program"]))
            indices = group.get("global_particle_indices")
            ancestors = group.get("global_ancestors")
            local_ancestors = group.get("local_ancestors")
            if (
                indices != expected_indices
                or not isinstance(ancestors, list)
                or not isinstance(local_ancestors, list)
                or len(ancestors) != len(expected_indices)
                or any(value not in expected_indices for value in ancestors)
                or ancestors != [expected_indices[value] for value in local_ancestors]
            ):
                raise UgiProductionSynthesisGuidanceSeamV4Error(
                    f"{arm_name} ancestry escaped its frozen morphology-program group"
                )
            probabilities = group.get("ancestry_probabilities")
            utilities = group.get("route_completion_utilities")
            targets = group.get("tempered_targets")
            carried = group.get("carried_tempered_after")
            if (
                not isinstance(probabilities, list)
                or len(probabilities) != len(expected_indices)
                or abs(sum(probabilities) - 1.0) > 1e-12
                or not isinstance(targets, list)
                or not isinstance(carried, list)
                or carried != [targets[value] for value in local_ancestors]
            ):
                raise UgiProductionSynthesisGuidanceSeamV4Error(
                    f"{arm_name} ancestry probability or carried-potential contract changed"
                )
            if arm_name == "post_hoc":
                if (
                    group.get("resampled") is not False
                    or ancestors != expected_indices
                    or group.get("keyed_seed") is not None
                ):
                    raise UgiProductionSynthesisGuidanceSeamV4Error(
                        "post-hoc matched shadow compute unexpectedly changed ancestry"
                    )
            else:
                row = ancestry_by_checkpoint[str(checkpoint)]
                row["groups"] += 1
                row["resampled_groups"] += int(group.get("resampled") is True)
                row["nonuniform_probability_groups"] += int(
                    len({round(float(value), 15) for value in probabilities}) > 1
                )
                observed_utilities = {value for value in utilities if value is not None}
                row["mixed_observed_utility_groups"] += int(len(observed_utilities) > 1)
                row["nonidentity_destination_count"] += sum(
                    source != destination
                    for source, destination in zip(ancestors, expected_indices, strict=True)
                )
                row["duplicated_ancestor_draw_count"] += len(ancestors) - len(set(ancestors))
                if group.get("resampled") is False and (
                    ancestors != expected_indices or group.get("keyed_seed") is not None
                ):
                    raise UgiProductionSynthesisGuidanceSeamV4Error(
                        "guided no-resampling group changed ancestry"
                    )

            terminal_ids = group.get("shadow_terminal_ids")
            terminal_sha256s = group.get("shadow_terminal_sha256s")
            trace_sha256s = group.get("shadow_trace_sha256s")
            dispositions = group.get("dispositions")
            if not all(
                isinstance(value, list) and len(value) == len(expected_indices)
                for value in (terminal_ids, terminal_sha256s, trace_sha256s, dispositions)
            ):
                raise UgiProductionSynthesisGuidanceSeamV4Error(
                    f"{arm_name} checkpoint terminal columns are malformed"
                )
            for local_index, global_index in enumerate(expected_indices):
                terminal_id = terminal_ids[local_index]
                if terminal_id is None:
                    if dispositions[local_index] != "completion_error":
                        raise UgiProductionSynthesisGuidanceSeamV4Error(
                            "missing shadow terminal lacks completion-error disposition"
                        )
                    continue
                parsed = _parse_terminal_id(
                    terminal_id,
                    label=f"{arm_name} checkpoint {checkpoint} particle {global_index}",
                )
                expected_seed = keyed_random_seed(
                    assignment.seed,
                    arm="matched_checkpoint_completion",
                    program_index=program_index,
                    particle_index=local_index,
                    checkpoint_index=checkpoint,
                    rollout_index=0,
                )
                if (
                    parsed["particle_index"] != str(global_index)
                    or parsed["checkpoint_index"] != str(checkpoint)
                    or parsed["invocation_seed"] != str(expected_seed)
                    or parsed["productive_seed"] != str(expected_seed)
                    or parsed["adapter_identity"] != adapter_identity_sha256
                ):
                    raise UgiProductionSynthesisGuidanceSeamV4Error(
                        "checkpoint terminal ID differs from frozen coordinates"
                    )
                _require_sha256(terminal_sha256s[local_index], label="shadow terminal")
                _require_sha256(trace_sha256s[local_index], label="shadow trace")
                arm_checkpoint_pool_fields.setdefault((arm_name, checkpoint), set()).add(
                    (
                        parsed["pool_state"],
                        parsed["pool_provenance"],
                        parsed["pool_lineage"],
                    )
                )
                terminal_contract_rows.append(
                    {
                        "arm": arm_name,
                        "phase": "checkpoint_shadow",
                        "checkpoint": checkpoint,
                        "program_index": program_index,
                        "particle_index": global_index,
                        "terminal_id_sha256": hashlib.sha256(terminal_id.encode()).hexdigest(),
                        "terminal_sha256": terminal_sha256s[local_index],
                        "generation_trace_sha256": trace_sha256s[local_index],
                        "particle_provenance_sha256": parsed["particle_provenance"],
                        "particle_lineage_sha256": parsed["particle_lineage"],
                    }
                )
    if any(len(values) != 1 for values in arm_checkpoint_pool_fields.values()):
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "checkpoint terminal IDs do not share one authenticated pool state per arm/checkpoint"
        )

    productive_comparisons: dict[str, Any] = {}
    productive_fields: dict[str, list[dict[str, str]]] = {}
    for arm_name in ("guided", "post_hoc"):
        arm = run_payload[arm_name]
        admissions = arm.get("productive_admissions")
        terminal_ids = arm.get("productive_terminal_ids")
        canonical = arm.get("productive_canonical_identities")
        utilities = arm.get("productive_route_completion_utilities")
        if not all(
            isinstance(value, list) and len(value) == 64
            for value in (
                admissions,
                terminal_ids,
                canonical,
                utilities,
            )
        ):
            raise UgiProductionSynthesisGuidanceSeamV4Error(
                f"{arm_name} productive coverage changed"
            )
        fields = []
        for particle_index, admission in enumerate(admissions):
            if (
                not isinstance(admission, dict)
                or admission.get("terminal_id") != terminal_ids[particle_index]
            ):
                raise UgiProductionSynthesisGuidanceSeamV4Error(
                    f"{arm_name} productive admission is misaligned"
                )
            parsed = _parse_terminal_id(
                admission["terminal_id"],
                label=f"{arm_name} productive particle {particle_index}",
            )
            particle_seed = assignment.stochastic_particle_seeds[particle_index]
            if (
                parsed["particle_index"] != str(particle_index)
                or parsed["checkpoint_index"] != str(EXPECTED_DESIGN["sample_steps"])
                or parsed["invocation_seed"] != str(particle_seed + 1)
                or parsed["productive_seed"] != str(particle_seed)
                or parsed["adapter_identity"] != adapter_identity_sha256
            ):
                raise UgiProductionSynthesisGuidanceSeamV4Error(
                    "productive terminal ID differs from frozen particle provenance"
                )
            _require_sha256(admission.get("terminal_sha256"), label="productive terminal")
            _require_sha256(admission.get("generation_trace_sha256"), label="productive trace")
            fields.append(parsed)
            terminal_contract_rows.append(
                {
                    "arm": arm_name,
                    "phase": "productive_final",
                    "checkpoint": EXPECTED_DESIGN["sample_steps"],
                    "program_index": particle_index // EXPECTED_DESIGN["particles_per_program"],
                    "particle_index": particle_index,
                    "terminal_id_sha256": hashlib.sha256(
                        admission["terminal_id"].encode()
                    ).hexdigest(),
                    "terminal_sha256": admission["terminal_sha256"],
                    "generation_trace_sha256": admission["generation_trace_sha256"],
                    "particle_provenance_sha256": parsed["particle_provenance"],
                    "particle_lineage_sha256": parsed["particle_lineage"],
                }
            )
        if (
            len({value["pool_state"] for value in fields}) != 1
            or len({value["pool_provenance"] for value in fields}) != 1
            or len({value["pool_lineage"] for value in fields}) != 1
        ):
            raise UgiProductionSynthesisGuidanceSeamV4Error(
                f"{arm_name} productive pool provenance is internally inconsistent"
            )
        productive_fields[arm_name] = fields

    zero_post_hoc = zero_run.get("post_hoc")
    if not isinstance(zero_post_hoc, dict):
        raise UgiProductionSynthesisGuidanceSeamV4Error("zero post-hoc baseline is malformed")
    post_hoc = run_payload["post_hoc"]
    baseline_columns = (
        "productive_terminal_ids",
        "productive_canonical_identities",
        "productive_route_completion_utilities",
        "productive_admissions",
    )
    if any(post_hoc.get(column) != zero_post_hoc.get(column) for column in baseline_columns):
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "nonzero-run post-hoc pool does not reproduce the hardened lambda-zero baseline"
        )
    guided = run_payload["guided"]
    for particle_index in range(64):
        if (
            productive_fields["guided"][particle_index]["particle_provenance"]
            != productive_fields["post_hoc"][particle_index]["particle_provenance"]
        ):
            raise UgiProductionSynthesisGuidanceSeamV4Error(
                "guided ancestry changed destination-owned particle provenance"
            )
    columns = {
        "terminal_id": "productive_terminal_ids",
        "canonical_identity": "productive_canonical_identities",
        "route_utility": "productive_route_completion_utilities",
    }
    for label, column in columns.items():
        guided_values = guided[column]
        post_hoc_values = post_hoc[column]
        productive_comparisons[label] = {
            "equal_by_index": sum(
                left == right for left, right in zip(guided_values, post_hoc_values, strict=True)
            ),
            "different_by_index": sum(
                left != right for left, right in zip(guided_values, post_hoc_values, strict=True)
            ),
        }
    guided_admissions = guided["productive_admissions"]
    post_hoc_admissions = post_hoc["productive_admissions"]
    for label, field in (
        ("terminal_bytes", "terminal_sha256"),
        ("generation_trace", "generation_trace_sha256"),
        ("particle_lineage", None),
    ):
        if field is None:
            left = [value["particle_lineage"] for value in productive_fields["guided"]]
            right = [value["particle_lineage"] for value in productive_fields["post_hoc"]]
        else:
            left = [value[field] for value in guided_admissions]
            right = [value[field] for value in post_hoc_admissions]
        productive_comparisons[label] = {
            "equal_by_index": sum(a == b for a, b in zip(left, right, strict=True)),
            "different_by_index": sum(a != b for a, b in zip(left, right, strict=True)),
        }
    guided_selected = [value.get("canonical_identity") for value in guided.get("selected", [])]
    post_hoc_selected = [value.get("canonical_identity") for value in post_hoc.get("selected", [])]
    diagnostics_content = {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "status": "selected_v3_synthesis_guidance_ancestry_and_difference_diagnostics_complete",
        "guidance_strength": GUIDANCE_STRENGTH,
        "assignment_seed": assignment.seed,
        "terminal_id_contract": {
            "record_count": len(terminal_contract_rows),
            "manifest_sha256": _sha256_payload(terminal_contract_rows),
            "selected_v3_adapter_identity_sha256": adapter_identity_sha256,
            "all_coordinates_and_provenance_valid": True,
        },
        "guided_ancestry_by_checkpoint": ancestry_by_checkpoint,
        "guided_vs_post_hoc_productive": {
            **productive_comparisons,
            "productive_lock_equal": (
                guided.get("productive_lock_manifest_sha256")
                == post_hoc.get("productive_lock_manifest_sha256")
            ),
            "selected_constitutional_overlap": _selected_overlap(
                guided_selected,
                post_hoc_selected,
            ),
        },
        "post_hoc_vs_hardened_zero": {
            "productive_columns_bitwise_equal": True,
            "checkpoint_terminal_ids_equal": (
                post_hoc.get("checkpoint_groups") == zero_post_hoc.get("checkpoint_groups")
            ),
        },
        "interpretation": {
            "difference_required_for_qualification": False,
            "zero_difference_is_reported_not_repaired": True,
            "seed_search_or_retry_permitted": False,
            "biological_signal_used": False,
        },
    }
    return {
        **diagnostics_content,
        "result_sha256": _sha256_payload(diagnostics_content),
    }


def _build_run_artifact(run_payload: dict[str, Any]) -> dict[str, Any]:
    if (
        run_payload.get("guidance_strength") != GUIDANCE_STRENGTH
        or run_payload.get("production_execution") is not False
        or run_payload.get("biological_guidance") is not False
        or run_payload.get("private_holdout_accessed") is not False
    ):
        raise UgiProductionSynthesisGuidanceSeamV4Error("synthesis-guidance run scope changed")
    content = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "selected_v3_synthesis_only_guidance_qualification_complete",
        "run": run_payload,
        "scope": {
            "qualification_only": True,
            "production_execution": False,
            "nonzero_synthesis_guidance_executed": True,
            "biological_guidance": False,
            "candidate_selection": False,
            "development_endpoint_subset_materialized": True,
            "prospective_candidate_lock": False,
            "sealed_holdout_access": False,
            "synthesis_success_probability": None,
        },
    }
    return {**content, "result_sha256": _sha256_payload(content)}


def _build_support_artifact(records: list[dict[str, Any]]) -> dict[str, Any]:
    validated = build_support_audit_artifact(records)
    content = {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "status": "selected_v3_synthesis_guidance_support_ledger_complete",
        "record_count": validated["record_count"],
        "records": validated["records"],
        "records_manifest_sha256": validated["records_manifest_sha256"],
        "scalar_value": None,
        "success_probability": None,
    }
    return {**content, "result_sha256": _sha256_payload(content)}


def _relative_artifact_path(repo: Path, output_dir: Path, filename: str) -> str:
    path = (output_dir / filename).resolve()
    try:
        return str(path.relative_to(repo))
    except ValueError as error:
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "seam-v4 output directory must be within the repository"
        ) from error


def _artifact_pin(
    repo: Path,
    output_dir: Path,
    filename: str,
    value: dict[str, Any],
) -> dict[str, Any]:
    return {
        "path": _relative_artifact_path(repo, output_dir, filename),
        "file_sha256": _canonical_file_sha256(value),
        "result_sha256": value["result_sha256"],
    }


def run_production_synthesis_guidance_seam_v4(
    repo: Path,
    config_path: Path,
    cache_root: Path,
    output_dir: Path,
    *,
    review_token: str,
    rerun_token: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run the reviewed selected-v3 lambda-0.25 qualification seam."""

    if review_token != REVIEW_TOKEN:
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "explicit root review token is required before nonzero qualification execution"
        )
    if rerun_token != RERUN_TOKEN:
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "explicit single operational-rerun token is required"
        )
    repo = repo.resolve()
    config_path = config_path.resolve()
    cache_root = cache_root.resolve()
    output_dir = output_dir.resolve()
    config, paths, zero = _validate_config(repo, config_path)
    if cache_root.exists() and any(cache_root.iterdir()):
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "seam-v4 cache root must be absent or empty"
        )
    cache_root.mkdir(parents=True, exist_ok=True)

    schedule_qualification = load_grouped_smc_schedule_qualification(paths["grouped_schedule"])
    assignments = schedule_qualification.by_seed()
    assignment_seed = config["design"]["assignment_seed"]
    if assignment_seed not in assignments:
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "first preregistered calibration assignment is missing"
        )
    assignment = assignments[assignment_seed]
    lane = build_selected_model_restartable_guidance_lane_v3(repo)
    if (
        lane.adapter_identity_sha256 != EXPECTED_SELECTED_V3_ADAPTER_IDENTITY_SHA256
        or lane.bound_closure_identity_sha256 != EXPECTED_BOUND_CLOSURE_IDENTITY_SHA256
    ):
        raise UgiProductionSynthesisGuidanceSeamV4Error("selected-v3 runtime identity changed")
    zero_reference_manifest = _require_selected_v3_productive_identity(
        zero.run["run"],
        lane,
    )
    if (
        zero_reference_manifest
        != zero.result["execution"]["selected_v3_productive_reference_manifest_sha256"]
    ):
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "hardened zero productive-reference manifest changed"
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
    zero_cache = zero.run["run"]["cache_isolation"]
    if (
        binding.preflight.base.snapshot_sha256 != zero_cache["base_snapshot_sha256"]
        or binding.preflight.context_sha256 != zero_cache["planner_context_sha256"]
        or binding.preflight.preflight_sha256 != zero_cache["cache_preflight_sha256"]
    ):
        raise UgiProductionSynthesisGuidanceSeamV4Error(
            "route evaluator evidence/cache preflight differs from hardened lambda zero"
        )
    cache_contract = GuidanceCacheIsolationContract(
        base_snapshot_sha256=binding.preflight.base.snapshot_sha256,
        planner_context_sha256=binding.preflight.context_sha256,
        cache_preflight_sha256=binding.preflight.preflight_sha256,
        guided_clone_id=f"v4-l025-guided-{assignment.assignment_sha256[:16]}",
        post_hoc_clone_id=f"v4-l025-post-hoc-{assignment.assignment_sha256[:16]}",
        isolated_overlay_roots=True,
        production_adapter_qualified=False,
    )
    evaluator = ProductionGuidanceRouteEvaluatorV3(binding=binding, factory=factory)
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
        expected_value_policy_id=UGI_EXACT_CLOSURE_GUIDANCE_POLICY_SHA256,
        cache_contract=cache_contract,
        cache_finalizer=lambda: finalize_lazy_matched_cache_binding(binding, cache_contract),
    )
    run_artifact = _build_run_artifact(_json_value(asdict(run)))
    _annotate_support_record_coordinates(run_artifact["run"], evaluator.support_records)
    support_artifact = _build_support_artifact(evaluator.support_records)
    typed_counts = derive_typed_counts(run_artifact, support_artifact)
    diagnostics = _build_ancestry_and_difference_diagnostics(
        run_artifact["run"],
        zero.run,
        assignment,
        adapter_identity_sha256=lane.adapter_identity_sha256,
    )
    artifacts = {
        "run": _artifact_pin(repo, output_dir, "run.json", run_artifact),
        "support_audits": _artifact_pin(
            repo,
            output_dir,
            "support_audits.json",
            support_artifact,
        ),
        "diagnostics": _artifact_pin(
            repo,
            output_dir,
            "diagnostics.json",
            diagnostics,
        ),
    }
    result_content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "selected_v3_lambda_0_25_synthesis_only_qualification_complete",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "artifacts": artifacts,
        "selected_generator": {
            "checkpoint_sha256": GENERATOR_CHECKPOINT_SHA256,
            "terminal_decoder_id": TERMINAL_DECODER_ID,
            "maximum_adjacent_branch_runs": list(MAXIMUM_ADJACENT_BRANCH_RUNS),
            "selected_v3_adapter_schema_version": SELECTED_GUIDANCE_ADAPTER_V3_SCHEMA_VERSION,
            "selected_v3_adapter_identity_sha256": lane.adapter_identity_sha256,
            "bound_closure_identity": lane.bound_closure_identity.to_dict(),
            "bound_closure_identity_sha256": lane.bound_closure_identity_sha256,
            "equivalence_binding": lane.equivalence_binding.identity_dict(),
        },
        "execution": {
            "assignment_seed": assignment.seed,
            "assignment_sha256": assignment.assignment_sha256,
            "guidance_strength": GUIDANCE_STRENGTH,
            "preregistered_strength_set": EXPECTED_STRENGTH_SET,
            "particle_count": design.particle_count,
            "checkpoint_count": len(schedule.checkpoints),
            "typed_outcomes": typed_counts,
            "diagnostics_result_sha256": diagnostics["result_sha256"],
        },
        "invariants": {
            "hardened_lambda_zero_evidence_validated": True,
            "post_hoc_productive_pool_reproduces_lambda_zero": True,
            "selected_v3_terminal_id_and_provenance_contract_complete": True,
            "within_program_ancestry_only": True,
            "same_exact_matched_compute_budget": True,
            "same_route_evaluator_and_current_evidence": True,
            "support_potential_and_receipt_ledger_complete": True,
            "typed_outcomes_partition_every_attempt": True,
            "base_cache_unchanged": run.cache_finalization.base_unchanged,
            "overlay_roots_isolated": run.cache_finalization.overlay_roots_isolated,
            "overlay_start_states_identical": (
                run.cache_finalization.overlay_start_states_identical
            ),
            "no_seed_search_or_retry": True,
        },
        "scope": {
            "qualification_only": True,
            "production_execution": False,
            "nonzero_synthesis_guidance_executed": True,
            "biological_guidance": False,
            "candidate_selection": False,
            "prospective_candidate_lock": False,
            "sealed_holdout_access": False,
            "synthesis_success_probability": None,
            "nonzero_guidance_authorized_beyond_this_single_reviewed_run": False,
        },
    }
    result = {**result_content, "result_sha256": _sha256_payload(result_content)}
    return result, run_artifact, support_artifact, diagnostics


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "DIAGNOSTICS_SCHEMA_VERSION",
    "EXPECTED_DESIGN",
    "EXPECTED_INPUT_KEYS",
    "EXPECTED_SCOPE",
    "GUIDANCE_STRENGTH",
    "RESULT_SCHEMA_VERSION",
    "REVIEW_TOKEN",
    "RERUN_TOKEN",
    "RUN_SCHEMA_VERSION",
    "SUPPORT_SCHEMA_VERSION",
    "UgiProductionSynthesisGuidanceSeamV4Error",
    "run_production_synthesis_guidance_seam_v4",
]
