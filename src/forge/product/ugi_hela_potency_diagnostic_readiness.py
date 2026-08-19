"""Non-executing readiness audit for the bounded HeLa potency diagnostic.

This module authenticates the frozen oracle policy, all eight disjoint SMC
schedules, the selected-v3 generator binding, and the historical lambda-zero
seed.  It also reclassifies every historical checkpoint completion under the
new potency policy.  It never advances a generator trajectory and cannot
authorize nonzero guidance, candidate selection, synthesis guidance, or a
prospective lock.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_file
from forge.core.hashing import sha256_json as _sha256_payload
from forge.potency.ugi_hela_potency_diagnostic import (
    EXPECTED_SELECTED_CANDIDATE,
    POLICY_ID,
    FrozenHeLaOracleWorker,
    HeLaPotencyDiagnosticPolicy,
)
from forge.product.ugi_hela_potency_guidance_seam_v1 import (
    CALIBRATION_SEEDS,
    EVALUATION_SEEDS,
    EXPECTED_SEEDS,
)
from forge.product.ugi_nonzero_guidance_runner import (
    load_grouped_smc_schedule_qualification,
)
from forge.product.ugi_selected_guidance_adapter_v3 import (
    build_selected_model_restartable_guidance_lane_v3,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi_hela_potency_guidance_readiness_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_hela_potency_guidance_readiness.v1"
EXPECTED_INPUTS = {
    "bio_policy",
    "grouped_schedule",
    "historical_result",
    "historical_run",
    "historical_support_audits",
    "potency_policy_source",
    "potency_seam_source",
    "selected_v3_adapter_source",
}
EXPECTED_SCOPE = {
    "diagnostic_only": True,
    "endpoint": "expt_Hela",
    "synthesis_guidance": False,
    "proposal_guidance": False,
    "candidate_selection": False,
    "prospective_candidate_lock": False,
    "generator_trajectories_advanced": False,
    "nonzero_execution_authorized": False,
}
EXPECTED_DESIGN = {
    "oracle_candidate_id": EXPECTED_SELECTED_CANDIDATE,
    "seeds": list(EXPECTED_SEEDS),
    "calibration_seeds": list(CALIBRATION_SEEDS),
    "evaluation_seeds": list(EVALUATION_SEEDS),
    "morphology_program_count": 16,
    "particles_per_program": 4,
    "particles_per_seed": 64,
    "sample_steps": 8,
    "checkpoints": [2, 4, 6],
    "checkpoint_betas": [0.25, 0.5, 0.75],
    "guidance_strength": 0.25,
    "terminal_particles_per_arm_all_seeds": 512,
    "checkpoint_completions_per_arm_all_seeds": 1536,
    "terminal_assessment_attempts_per_arm_all_seeds": 2048,
    "product_transition_calls_per_arm_all_seeds": 10240,
}
EXPECTED_HISTORICAL = {
    "seed": 20260821,
    "scheduled_checkpoint_attempts": 192,
    "invalid_terminal_attempts": 10,
    "nonexact_l1_attempts": 0,
    "classified_exact_l1_attempts": 182,
    "overall_bins": {
        "boundary": 6,
        "extrapolative": 173,
        "interpolative": 3,
    },
    "active_eligible_attempts": 0,
    "interpolative_unseen_role_patterns": {"amine+isocyanide": 3},
    "productive_lock_manifest_sha256": (
        "8e412fa5d67db84bcde57b29830a77f6eadf339d0c88258d8f289153b19ddaa1"
    ),
}


class UgiHeLaPotencyReadinessError(RuntimeError):
    """Raised when the potency diagnostic is not reproducibly ready."""


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiHeLaPotencyReadinessError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiHeLaPotencyReadinessError(f"{label} must contain a JSON object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiHeLaPotencyReadinessError(f"{label} pin is malformed")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiHeLaPotencyReadinessError(f"{label} path escapes repository") from error
    if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
        raise UgiHeLaPotencyReadinessError(f"{label} hash changed")
    return path


def _candidate_from_support_record(record: Mapping[str, Any]) -> dict[str, str]:
    receipt = record.get("assessment_receipt")
    payload = receipt.get("payload") if isinstance(receipt, Mapping) else None
    components = payload.get("components") if isinstance(payload, Mapping) else None
    forward = payload.get("l1_forward_verification") if isinstance(payload, Mapping) else None
    if (
        not isinstance(components, list)
        or not isinstance(forward, Mapping)
        or forward.get("exact_product_reconstructed") is not True
    ):
        raise UgiHeLaPotencyReadinessError("historical checkpoint record lacks exact L1 evidence")
    by_role: dict[str, str] = {}
    for component in components:
        if not isinstance(component, Mapping):
            raise UgiHeLaPotencyReadinessError("historical component record is malformed")
        role = component.get("role")
        smiles = component.get("canonical_smiles")
        if not isinstance(role, str) or not isinstance(smiles, str):
            raise UgiHeLaPotencyReadinessError("historical component identity is malformed")
        if role in by_role:
            raise UgiHeLaPotencyReadinessError("historical component role is duplicated")
        by_role[role] = smiles
    required = {
        "amine_head": "amine_smiles",
        "oxoester_aldehyde_body_tail": "aldehyde_smiles",
        "isocyanide_tail": "isocyanide_smiles",
    }
    if set(by_role) != set(required):
        raise UgiHeLaPotencyReadinessError("historical component roles changed")
    product = payload.get("product_smiles")
    label = record.get("terminal_sha256")
    if not isinstance(product, str) or not isinstance(label, str):
        raise UgiHeLaPotencyReadinessError("historical product identity is malformed")
    return {
        "label": label,
        "product_smiles": product,
        **{field: by_role[role] for role, field in required.items()},
    }


def audit_historical_checkpoint_coverage(
    run: Mapping[str, Any],
    support: Mapping[str, Any],
    policy: HeLaPotencyDiagnosticPolicy,
) -> dict[str, Any]:
    """Reclassify the complete historical guided checkpoint ledger."""

    run_payload = run.get("run")
    guided = run_payload.get("guided") if isinstance(run_payload, Mapping) else None
    groups = guided.get("checkpoint_groups") if isinstance(guided, Mapping) else None
    if not isinstance(groups, list) or len(groups) != 48:
        raise UgiHeLaPotencyReadinessError("historical checkpoint-group design changed")
    scheduled = sum(int(group.get("terminal_completions", -1)) for group in groups)
    invalid = sum(int(group.get("invalid_terminal_count", -1)) for group in groups)
    nonexact = sum(int(group.get("nonexact_l1_count", -1)) for group in groups)
    records = support.get("records")
    if not isinstance(records, list):
        raise UgiHeLaPotencyReadinessError("historical support ledger is malformed")
    selected = [
        record
        for record in records
        if isinstance(record, Mapping)
        and record.get("arm") == "guided"
        and record.get("assessment_phase") == "checkpoint_shadow"
    ]
    overall_bins: Counter[str] = Counter()
    active = 0
    interpolative_patterns: Counter[str] = Counter()
    detailed: Counter[str] = Counter()
    measured = 0
    for record in selected:
        classification = policy.classify_candidate_mapping(_candidate_from_support_record(record))
        overall = str(classification["overall_bin"])
        unseen = tuple(str(value) for value in classification["unseen_roles"])
        pattern_name = "+".join(unseen) if unseen else "none"
        overall_bins[overall] += 1
        detailed[f"{overall}::{pattern_name}"] += 1
        if classification["exact_measured_combination"]:
            measured += 1
        all_views_interpolative = all(
            value == "interpolative" for _, value in classification["view_bins"]
        )
        if overall == "interpolative":
            interpolative_patterns[pattern_name] += 1
        if (
            not classification["exact_measured_combination"]
            and overall == "interpolative"
            and all_views_interpolative
            and classification["pattern_id"] is not None
        ):
            active += 1
    output = {
        "seed": int(run_payload.get("base_seed", -1)),
        "scheduled_checkpoint_attempts": scheduled,
        "invalid_terminal_attempts": invalid,
        "nonexact_l1_attempts": nonexact,
        "classified_exact_l1_attempts": len(selected),
        "overall_bins": dict(sorted(overall_bins.items())),
        "active_eligible_attempts": active,
        "exact_measured_combinations": measured,
        "interpolative_unseen_role_patterns": dict(sorted(interpolative_patterns.items())),
        "detailed_distribution_by_unseen_pattern": dict(sorted(detailed.items())),
        "productive_lock_manifest_sha256": guided.get("productive_lock_manifest_sha256"),
    }
    expected_subset = {
        key: output[key]
        for key in (
            "seed",
            "scheduled_checkpoint_attempts",
            "invalid_terminal_attempts",
            "nonexact_l1_attempts",
            "classified_exact_l1_attempts",
            "overall_bins",
            "active_eligible_attempts",
            "interpolative_unseen_role_patterns",
            "productive_lock_manifest_sha256",
        )
    }
    if expected_subset != EXPECTED_HISTORICAL:
        raise UgiHeLaPotencyReadinessError("historical selected-v3 applicability census changed")
    if scheduled != invalid + nonexact + len(selected):
        raise UgiHeLaPotencyReadinessError(
            "historical checkpoint outcomes do not partition all attempts"
        )
    return output


def _worker_smoke(policy: HeLaPotencyDiagnosticPolicy) -> dict[str, Any]:
    path = policy.paths["curated_agile"]
    with gzip.open(path, "rt", newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if not isinstance(row, dict):
        raise UgiHeLaPotencyReadinessError("curated AGILE table is empty")
    candidate = {
        "label": str(row["label"]),
        "product_smiles": str(row["model_smiles"]),
        "amine_smiles": str(row["A_smiles"]),
        "aldehyde_smiles": str(row["B_smiles"]),
        "isocyanide_smiles": str(row["C_smiles"]),
    }
    with FrozenHeLaOracleWorker(policy) as worker:
        response = worker.predict([candidate])
    prediction = response.get("prediction")
    classifications = response.get("classifications")
    means = prediction.get("ensemble_mean") if isinstance(prediction, Mapping) else None
    deviations = (
        prediction.get("ensemble_standard_deviation") if isinstance(prediction, Mapping) else None
    )
    if (
        not isinstance(classifications, list)
        or len(classifications) != 1
        or classifications[0].get("exact_forward_verified") is not True
        or not isinstance(means, list)
        or len(means) != 1
        or not isinstance(deviations, list)
        or len(deviations) != 1
        or not math.isfinite(float(means[0]))
        or not math.isfinite(float(deviations[0]))
        or float(deviations[0]) < 0
    ):
        raise UgiHeLaPotencyReadinessError("cross-runtime oracle smoke test failed")
    return {
        "records": 1,
        "endpoint": prediction.get("endpoint"),
        "exact_forward_verified": True,
        "worker_receipt_sha256": response.get("receipt_sha256"),
    }


def build_hela_potency_diagnostic_readiness(
    repo: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Build a complete preflight receipt without executing a trajectory."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load(config_path, label="HeLa potency readiness config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiHeLaPotencyReadinessError("unsupported potency readiness config")
    if config.get("scope") != EXPECTED_SCOPE or config.get("design") != EXPECTED_DESIGN:
        raise UgiHeLaPotencyReadinessError("potency readiness contract changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != EXPECTED_INPUTS:
        raise UgiHeLaPotencyReadinessError("potency readiness input set changed")
    paths = {name: _pin(repo, value, label=name) for name, value in inputs.items()}

    policy = HeLaPotencyDiagnosticPolicy(repo, paths["bio_policy"])
    qualification = load_grouped_smc_schedule_qualification(paths["grouped_schedule"])
    assignments = qualification.by_seed()
    if tuple(sorted(assignments)) != EXPECTED_SEEDS:
        raise UgiHeLaPotencyReadinessError("eight-seed frozen schedule changed")
    if len({assignment.assignment_sha256 for assignment in assignments.values()}) != 8:
        raise UgiHeLaPotencyReadinessError("seed assignments are not disjoint")

    historical_result = _load(paths["historical_result"], label="historical result")
    historical_run = _load(paths["historical_run"], label="historical run")
    historical_support = _load(
        paths["historical_support_audits"], label="historical support audits"
    )
    artifacts = historical_result.get("artifacts")
    if (
        historical_result.get("status") != "selected_v3_production_lambda_zero_typed_seam_qualified"
        or not isinstance(artifacts, Mapping)
        or artifacts.get("run", {}).get("file_sha256") != sha256_file(paths["historical_run"])
        or artifacts.get("support_audits", {}).get("file_sha256")
        != sha256_file(paths["historical_support_audits"])
    ):
        raise UgiHeLaPotencyReadinessError("historical lambda-zero evidence changed")
    if (
        historical_run.get("run", {}).get("program_assignment_sha256")
        != assignments[EXPECTED_SEEDS[0]].assignment_sha256
    ):
        raise UgiHeLaPotencyReadinessError("historical seed assignment changed")
    coverage = audit_historical_checkpoint_coverage(historical_run, historical_support, policy)

    lane = build_selected_model_restartable_guidance_lane_v3(repo)
    selected = historical_result.get("selected_generator")
    if (
        not isinstance(selected, Mapping)
        or lane.adapter_identity_sha256 != selected.get("selected_v3_adapter_identity_sha256")
        or lane.bound_closure_identity_sha256 != selected.get("bound_closure_identity_sha256")
    ):
        raise UgiHeLaPotencyReadinessError("selected-v3 runtime identity changed")
    worker = _worker_smoke(policy)

    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "readiness_complete_nonzero_execution_blocked",
        "scope": EXPECTED_SCOPE,
        "design": EXPECTED_DESIGN,
        "inputs": {
            name: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
        "policy": {
            "policy_id": POLICY_ID,
            "policy_config_sha256": policy.policy_sha256,
            "selected_oracle": EXPECTED_SELECTED_CANDIDATE,
            "conditional_calibration_patterns": sorted(policy.scales),
        },
        "schedule": {
            "qualification_file_sha256": qualification.file_sha256,
            "qualification_result_sha256": qualification.result_sha256,
            "assignment_sha256_by_seed": {
                str(seed): assignments[seed].assignment_sha256 for seed in EXPECTED_SEEDS
            },
            "all_eight_assignments_frozen_and_disjoint": True,
        },
        "selected_v3_runtime": {
            "adapter_identity_sha256": lane.adapter_identity_sha256,
            "bound_closure_identity_sha256": lane.bound_closure_identity_sha256,
            "construction_passed": True,
        },
        "oracle_worker_smoke": worker,
        "historical_seed_coverage": coverage,
        "gates": {
            "policy_authenticated": True,
            "all_eight_schedules_frozen": True,
            "selected_v3_runtime_constructed": True,
            "cross_runtime_oracle_smoke_passed": True,
            "historical_seed_20260821_lambda_zero_evidence_authenticated": True,
            "historical_seed_checkpoint_eligibility_audited": True,
            "global_all_eight_lambda_zero_gate_executed": False,
            "nonzero_diagnostic_execution_authorized": False,
        },
        "interpretation": {
            "historical_active_eligible_attempts": 0,
            "historical_seed_would_have_identity_ancestry": True,
            "historical_seed_is_not_evidence_for_nonzero_potency_guidance": True,
            "next_required_gate": (
                "execute all eight lambda-zero identity schedules before any nonzero arm"
            ),
        },
    }
    return {**content, "result_sha256": _sha256_payload(content)}


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "UgiHeLaPotencyReadinessError",
    "audit_historical_checkpoint_coverage",
    "build_hela_potency_diagnostic_readiness",
]
