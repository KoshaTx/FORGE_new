"""Freeze the matched three-arm morphology-allocation generation schedule.

The schedule is the experimental design, not a terminal outcome.  A single
predeclared vector of uniforms is inverse-CDF transformed through the broad,
promoted-support and nested-potency morphology distributions.  This gives the
three arms common random numbers without pretending that they select the same
morphology program.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.io import stable_json as _stable_json
from forge.corpus.r1_prime_audit import sha256_file

CONFIG_SCHEMA_VERSION = "phase1_ugi_matched_morphology_allocation_schedule_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_matched_morphology_allocation_schedule.v1"
SCHEDULE_SCHEMA_VERSION = "forge.ugi_matched_morphology_allocation_schedule.v1"
ARM_IDS = ("broad_prior", "support_enriched", "nested_potency")
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
    "potency_proposal_ledger",
    "potency_proposal_result",
    "promoted_proposal_ledger",
    "promoted_proposal_result",
    "runner",
    "source",
    "tests",
}


class UgiMatchedMorphologyAllocationScheduleError(RuntimeError):
    """Raised when the matched allocation schedule cannot be frozen exactly."""


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiMatchedMorphologyAllocationScheduleError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiMatchedMorphologyAllocationScheduleError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiMatchedMorphologyAllocationScheduleError(f"malformed pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiMatchedMorphologyAllocationScheduleError(
            f"pin escapes repository: {label}"
        ) from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiMatchedMorphologyAllocationScheduleError(f"pin changed: {label}")
    return path


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    try:
        with gzip.open(path, "rt") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiMatchedMorphologyAllocationScheduleError(
            "invalid promoted-proposal ledger"
        ) from error
    if any(not isinstance(row, dict) for row in rows):
        raise UgiMatchedMorphologyAllocationScheduleError(
            "promoted-proposal ledger rows must be objects"
        )
    return rows


def _read_csv_gzip(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise UgiMatchedMorphologyAllocationScheduleError(
            "invalid potency-proposal ledger"
        ) from error


def inverse_cdf_indices(probabilities: Sequence[float], uniforms: Sequence[float]) -> np.ndarray:
    """Map common uniforms to categorical support indices deterministically."""

    probability = np.asarray(probabilities, dtype=np.float64)
    uniform = np.asarray(uniforms, dtype=np.float64)
    if probability.ndim != 1 or not len(probability):
        raise UgiMatchedMorphologyAllocationScheduleError(
            "proposal probabilities must be one nonempty vector"
        )
    if uniform.ndim != 1 or not len(uniform):
        raise UgiMatchedMorphologyAllocationScheduleError(
            "common uniforms must be one nonempty vector"
        )
    if not np.all(np.isfinite(probability)) or np.any(probability <= 0.0):
        raise UgiMatchedMorphologyAllocationScheduleError(
            "every qualified program must have positive finite probability"
        )
    if not np.isclose(probability.sum(), 1.0, rtol=0.0, atol=1e-10):
        raise UgiMatchedMorphologyAllocationScheduleError("proposal probabilities must sum to one")
    if not np.all(np.isfinite(uniform)) or np.any((uniform < 0.0) | (uniform >= 1.0)):
        raise UgiMatchedMorphologyAllocationScheduleError("common uniforms must lie in [0, 1)")
    cdf = np.cumsum(probability, dtype=np.float64)
    cdf[-1] = 1.0
    selected = np.searchsorted(cdf, uniform, side="right")
    if np.any(selected >= len(probability)):
        raise UgiMatchedMorphologyAllocationScheduleError("inverse-CDF selection overflowed")
    return selected.astype(np.int64, copy=False)


def _program(value: Any) -> dict[str, list[int]]:
    fields = {"node_counts", "junction_budgets", "cycle_ranks", "attachment_counts"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise UgiMatchedMorphologyAllocationScheduleError("morphology program schema changed")
    output = {field: [int(item) for item in value[field]] for field in sorted(fields)}
    if any(len(items) != 3 for items in output.values()):
        raise UgiMatchedMorphologyAllocationScheduleError("morphology role width changed")
    return output


def _require_design(value: Any) -> tuple[int, int]:
    if not isinstance(value, Mapping) or set(value) != {
        "draws_per_arm",
        "common_uniform_seed",
        "sampling_with_replacement",
    }:
        raise UgiMatchedMorphologyAllocationScheduleError("schedule design changed")
    draws = int(value["draws_per_arm"])
    seed = int(value["common_uniform_seed"])
    if draws != 3072 or seed < 0 or value["sampling_with_replacement"] is not True:
        raise UgiMatchedMorphologyAllocationScheduleError("schedule design is not preregistered")
    return draws, seed


def build_matched_morphology_allocation_schedule(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze all three morphology-allocation arms before terminal generation."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="matched allocation config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiMatchedMorphologyAllocationScheduleError("unsupported schedule config schema")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiMatchedMorphologyAllocationScheduleError("schedule scope changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != EXPECTED_INPUTS:
        raise UgiMatchedMorphologyAllocationScheduleError("schedule input pins changed")
    paths = {label: _pin(repo, record, label=label) for label, record in inputs.items()}

    promoted_result = _load_json(paths["promoted_proposal_result"], label="promoted proposal")
    potency_result = _load_json(paths["potency_proposal_result"], label="potency proposal")
    if promoted_result.get("decision", {}).get("applicability_proposal_promoted") is not True:
        raise UgiMatchedMorphologyAllocationScheduleError("applicability proposal is not promoted")
    if (
        potency_result.get("decision", {}).get(
            "nested_potency_proposal_frozen_for_matched_diagnostic"
        )
        is not True
        or potency_result.get("decision", {}).get("potency_tilting_promoted") is not False
    ):
        raise UgiMatchedMorphologyAllocationScheduleError(
            "potency proposal is not frozen exclusively for the matched diagnostic"
        )

    promoted_rows = _read_jsonl_gzip(paths["promoted_proposal_ledger"])
    potency_rows = _read_csv_gzip(paths["potency_proposal_ledger"])
    expected_programs = int(config["population"]["qualified_programs"])
    if expected_programs != 57190 or len(promoted_rows) != expected_programs:
        raise UgiMatchedMorphologyAllocationScheduleError(
            "complete qualified morphology support changed"
        )
    potency_by_sha = {row["program_sha256"]: row for row in potency_rows}
    if len(potency_rows) != expected_programs or len(potency_by_sha) != expected_programs:
        raise UgiMatchedMorphologyAllocationScheduleError(
            "potency proposal does not cover every unique program"
        )

    population = []
    for row in promoted_rows:
        program_sha256 = str(row["program_sha256"])
        potency = potency_by_sha.get(program_sha256)
        if potency is None:
            raise UgiMatchedMorphologyAllocationScheduleError(
                "proposal ledgers have different program support"
            )
        broad = float(row["broad_prior_probability"])
        support = float(row["proposal_probability"])
        if not np.isclose(
            support,
            float(potency["support_proposal_probability"]),
            rtol=0.0,
            atol=1e-15,
        ) or not np.isclose(
            broad,
            float(potency["broad_prior_probability"]),
            rtol=0.0,
            atol=1e-15,
        ):
            raise UgiMatchedMorphologyAllocationScheduleError(
                "broad or support probabilities disagree between frozen ledgers"
            )
        population.append(
            {
                "program_sha256": program_sha256,
                "program": _program(row["program"]),
                "broad_prior_probability": broad,
                "support_proposal_probability": support,
                "potency_proposal_probability": float(potency["potency_proposal_probability"]),
                "support_score": float(row["support_score"]),
                "authorized_morphology": str(potency["authorized_morphology"]).lower() == "true",
                "potency_utility": float(potency["potency_utility"]),
            }
        )
    population.sort(key=lambda row: row["program_sha256"])
    if len({row["program_sha256"] for row in population}) != expected_programs:
        raise UgiMatchedMorphologyAllocationScheduleError("program hashes are not unique")

    probability_fields = {
        "broad_prior": "broad_prior_probability",
        "support_enriched": "support_proposal_probability",
        "nested_potency": "potency_proposal_probability",
    }
    probabilities = {
        arm: np.asarray([float(row[field]) for row in population], dtype=np.float64)
        for arm, field in probability_fields.items()
    }
    for arm, values in probabilities.items():
        if np.any(values <= 0.0) or not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-10):
            raise UgiMatchedMorphologyAllocationScheduleError(
                f"{arm} does not preserve the complete normalized support"
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
            arm_records[arm] = {
                "support_index": support_index,
                "program_sha256": row["program_sha256"],
                "program": row["program"],
                "selected_probability": float(probabilities[arm][support_index]),
                "broad_prior_probability": row["broad_prior_probability"],
                "support_proposal_probability": row["support_proposal_probability"],
                "potency_proposal_probability": row["potency_proposal_probability"],
                "importance_ratio_broad_over_arm": float(
                    row["broad_prior_probability"] / probabilities[arm][support_index]
                ),
                "support_score": row["support_score"],
                "authorized_morphology": row["authorized_morphology"],
                "potency_utility": row["potency_utility"],
            }
        records.append(
            {
                "draw_index": draw_index,
                "common_uniform": float(uniform),
                "arms": arm_records,
            }
        )

    schedule_content = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "status": "frozen_before_matched_terminal_generation",
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
    unique_counts = {arm: len({int(value) for value in selected[arm]}) for arm in ARM_IDS}
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "matched_three_arm_morphology_allocation_schedule_frozen",
        "scope": dict(EXPECTED_SCOPE),
        "inputs": pinned_inputs,
        "design": schedule_content["design"],
        "population": schedule_content["population"],
        "draw_summary": {
            "unique_programs_by_arm": unique_counts,
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
            "potency_tilting_promoted": False,
            "next_gate": "matched_terminal_generation_then_frozen_terminal_scoring",
        },
        "nonclaims": [
            "The schedule contains no terminal outcome.",
            "The nested-potency arm changes morphology allocation only.",
            "No potency or applicability model will score an incomplete molecular graph.",
            "This schedule does not promote potency tilting or select prospective candidates.",
        ],
    }
    result = {**content, "result_sha256": _logical_sha256(content)}
    return result, schedule


__all__ = [
    "ARM_IDS",
    "UgiMatchedMorphologyAllocationScheduleError",
    "build_matched_morphology_allocation_schedule",
    "inverse_cdf_indices",
]
