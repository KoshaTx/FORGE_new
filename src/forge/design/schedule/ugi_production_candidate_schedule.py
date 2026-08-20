"""Freeze the matched two-arm morphology schedule for candidate production.

The production comparison retains only the broad morphology prior and the
independently promoted applicability-enriched proposal.  Potency and synthesis
scores are deliberately absent.  Common uniforms provide paired Monte Carlo
coordinates without forcing the two categorical distributions to select the
same morphology program.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from forge.data.r1_prime_audit import sha256_file
from forge.design.schedule.ugi_matched_morphology_allocation_schedule import (
    _logical_sha256,
    _program,
    _read_jsonl_gzip,
    inverse_cdf_indices,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi_production_candidate_schedule_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_production_candidate_schedule.v1"
SCHEDULE_SCHEMA_VERSION = "forge.ugi_production_candidate_schedule.v1"
ARM_IDS = ("broad_prior", "support_enriched")
EXPECTED_SCOPE = {
    "read_only_distribution_sampling": True,
    "complete_qualified_morphology_support": True,
    "common_uniforms_across_arms": True,
    "terminal_outcomes_consumed": False,
    "generator_calls": 0,
    "oracle_calls": 0,
    "route_calls": 0,
    "synthesis_calls": 0,
    "candidate_selection": False,
    "sealed_holdout_access": False,
}
EXPECTED_INPUTS = {
    "promoted_proposal_ledger",
    "promoted_proposal_result",
    "runner",
    "source",
    "tests",
}


class UgiProductionCandidateScheduleError(RuntimeError):
    """Raised when the production schedule cannot be frozen exactly."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiProductionCandidateScheduleError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiProductionCandidateScheduleError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiProductionCandidateScheduleError(f"malformed pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiProductionCandidateScheduleError(f"pin escapes repository: {label}") from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiProductionCandidateScheduleError(f"pin changed: {label}")
    return path


def _require_design(value: Any) -> tuple[int, int]:
    expected = {"draws_per_arm", "common_uniform_seed", "sampling_with_replacement"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise UgiProductionCandidateScheduleError("production schedule design changed")
    draws = int(value["draws_per_arm"])
    seed = int(value["common_uniform_seed"])
    if draws != 16384 or seed < 0 or value["sampling_with_replacement"] is not True:
        raise UgiProductionCandidateScheduleError("production schedule is not the frozen design")
    return draws, seed


def build_production_candidate_schedule(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the outcome-free broad-versus-support production schedule."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="production schedule config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiProductionCandidateScheduleError("unsupported production schedule schema")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiProductionCandidateScheduleError("production schedule scope changed")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != EXPECTED_INPUTS:
        raise UgiProductionCandidateScheduleError("production schedule input pins changed")
    paths = {label: _pin(repo, record, label=label) for label, record in raw_inputs.items()}

    promoted = _load_json(paths["promoted_proposal_result"], label="promoted proposal result")
    decision = promoted.get("decision", {})
    proposal = promoted.get("proposal", {})
    if (
        decision.get("applicability_proposal_promoted") is not True
        or proposal.get("mixture_rho") != 0.5
        or proposal.get("score_power") != 1.5
    ):
        raise UgiProductionCandidateScheduleError("promoted applicability proposal changed")

    rows = _read_jsonl_gzip(paths["promoted_proposal_ledger"])
    expected_programs = int(config.get("population", {}).get("qualified_programs", -1))
    if expected_programs != 57190 or len(rows) != expected_programs:
        raise UgiProductionCandidateScheduleError("qualified morphology support changed")
    population = []
    for row in rows:
        population.append(
            {
                "program_sha256": str(row["program_sha256"]),
                "program": _program(row["program"]),
                "broad_prior_probability": float(row["broad_prior_probability"]),
                "support_proposal_probability": float(row["proposal_probability"]),
                "support_score": float(row["support_score"]),
            }
        )
    population.sort(key=lambda row: row["program_sha256"])
    if len({row["program_sha256"] for row in population}) != expected_programs:
        raise UgiProductionCandidateScheduleError("morphology program hashes are not unique")

    probability_fields = {
        "broad_prior": "broad_prior_probability",
        "support_enriched": "support_proposal_probability",
    }
    probabilities = {
        arm: np.asarray([float(row[field]) for row in population], dtype=np.float64)
        for arm, field in probability_fields.items()
    }
    for arm, values in probabilities.items():
        if np.any(values <= 0.0) or not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-10):
            raise UgiProductionCandidateScheduleError(
                f"{arm} does not preserve complete normalized morphology support"
            )

    draws, seed = _require_design(config["design"])
    uniforms = np.random.default_rng(seed).random(draws)
    selected = {arm: inverse_cdf_indices(probabilities[arm], uniforms) for arm in ARM_IDS}
    records = []
    for draw_index, uniform in enumerate(uniforms):
        arm_records: dict[str, Any] = {}
        for arm in ARM_IDS:
            support_index = int(selected[arm][draw_index])
            row = population[support_index]
            selected_probability = float(probabilities[arm][support_index])
            arm_records[arm] = {
                "support_index": support_index,
                "program_sha256": row["program_sha256"],
                "program": row["program"],
                "selected_probability": selected_probability,
                "broad_prior_probability": row["broad_prior_probability"],
                "support_proposal_probability": row["support_proposal_probability"],
                "importance_ratio_broad_over_arm": float(
                    row["broad_prior_probability"] / selected_probability
                ),
                "support_score": row["support_score"],
            }
        records.append(
            {"draw_index": draw_index, "common_uniform": float(uniform), "arms": arm_records}
        )

    schedule_content = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "status": "frozen_before_production_candidate_generation",
        "design": {
            "arms": list(ARM_IDS),
            "draws_per_arm": draws,
            "total_terminal_attempts": draws * len(ARM_IDS),
            "common_uniform_seed": seed,
            "sampling_with_replacement": True,
            "support_order": "ascending_program_sha256",
            "common_random_number_role": "morphology_inverse_cdf_only",
        },
        "population": {
            "qualified_programs": expected_programs,
            "all_arms_positive_on_all_programs": True,
            "program_manifest_logical_sha256": _logical_sha256(
                [row["program_sha256"] for row in population]
            ),
        },
        "records": records,
    }
    schedule = {**schedule_content, "schedule_sha256": _logical_sha256(schedule_content)}
    pinned_inputs = {
        label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
        for label, path in sorted(paths.items())
    }
    pinned_inputs["config"] = {
        "path": str(config_path.relative_to(repo)),
        "sha256": sha256_file(config_path),
    }
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "matched_two_arm_production_candidate_schedule_frozen",
        "scope": dict(EXPECTED_SCOPE),
        "inputs": pinned_inputs,
        "design": schedule_content["design"],
        "population": schedule_content["population"],
        "draw_summary": {
            "unique_programs_by_arm": {
                arm: len({int(value) for value in selected[arm]}) for arm in ARM_IDS
            },
            "common_uniforms_identical_across_arms": True,
            "terminal_generator_randomness_not_consumed": True,
        },
        "artifacts": {
            "schedule.json": {
                "schema_version": SCHEDULE_SCHEMA_VERSION,
                "schedule_sha256": schedule["schedule_sha256"],
                "draw_records": draws,
                "arm_records": draws * len(ARM_IDS),
            }
        },
        "decision": {
            "schedule_frozen_before_generation": True,
            "applicability_proposal_promoted": True,
            "potency_tilting_promoted": False,
            "synthesis_tilting_promoted": False,
            "next_gate": "matched_terminal_generation_then_frozen_terminal_funnel",
        },
        "nonclaims": [
            "The schedule contains no terminal outcome or molecular score.",
            "Both arms retain positive probability on every qualified morphology program.",
            "The support-enriched arm is not guaranteed to produce an applicable molecule.",
            "This schedule does not select or lock prospective candidates.",
        ],
    }
    result = {**content, "result_sha256": _logical_sha256(content)}
    return result, schedule


__all__ = [
    "ARM_IDS",
    "UgiProductionCandidateScheduleError",
    "build_production_candidate_schedule",
]
