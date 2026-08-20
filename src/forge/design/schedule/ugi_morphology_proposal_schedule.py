"""Freeze a support-preserving morphology proposal on unused Ugi programs."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from forge.data.r1_prime_audit import sha256_file
from forge.design.flow.ugi_dynamic_frozen_prior_terminal_census import (
    load_census_contract,
    load_selected_program_manifest,
)
from forge.design.flow.ugi_restartable_terminal_support_adapter import (
    canonical_morphology_program_bytes,
)
from forge.potency.audit.ugi_dynamic_controller_analysis import BinomialRidge, morphology_features

CONFIG_SCHEMA_VERSION = "phase1_ugi_morphology_proposal_schedule_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_morphology_proposal_schedule.v1"
SCHEDULE_SCHEMA_VERSION = "forge.ugi_morphology_proposal_schedule.v1"
EXPECTED_SCOPE = {
    "read_only": True,
    "unused_program_population_only": True,
    "terminal_outcomes_for_unused_programs_consumed": False,
    "potency_predictions_consumed": False,
    "oracle_calls": 0,
    "route_calls": 0,
    "synthesis_calls": 0,
    "generator_trajectories_advanced": False,
    "candidate_selection": False,
    "sealed_holdout_access": False,
}
EXPECTED_INPUTS = {
    "analysis_result",
    "analysis_terminal_support",
    "census_config",
    "frozen_programs",
    "runner",
    "source",
    "tests",
}


class UgiMorphologyProposalScheduleError(RuntimeError):
    """Raised when an unused-program proposal contract changes."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiMorphologyProposalScheduleError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiMorphologyProposalScheduleError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiMorphologyProposalScheduleError(f"malformed pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiMorphologyProposalScheduleError(f"pin escapes repository: {label}") from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiMorphologyProposalScheduleError(f"pin changed: {label}")
    return path


def _read_support(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise UgiMorphologyProposalScheduleError("invalid terminal-support ledger") from error
    if len(rows) != 24576:
        raise UgiMorphologyProposalScheduleError("terminal-support ledger is incomplete")
    return rows


def _program_dict(value: Any) -> dict[str, tuple[int, int, int]]:
    expected = {"node_counts", "junction_budgets", "cycle_ranks", "attachment_counts"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise UgiMorphologyProposalScheduleError("program fields changed")
    output = {key: tuple(int(item) for item in value[key]) for key in sorted(expected)}
    if any(len(vector) != 3 for vector in output.values()):
        raise UgiMorphologyProposalScheduleError("program role width changed")
    return output


def build_morphology_proposal_schedule(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit M0 on calibration programs and score only unused programs."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="proposal schedule config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiMorphologyProposalScheduleError("unsupported proposal schedule schema")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiMorphologyProposalScheduleError("proposal schedule scope changed")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != EXPECTED_INPUTS:
        raise UgiMorphologyProposalScheduleError("proposal schedule input pins changed")
    paths = {label: _pin(repo, record, label=label) for label, record in raw_inputs.items()}

    analysis = _load_json(paths["analysis_result"], label="controller analysis")
    if analysis.get("result_sha256") != config["policy"]["analysis_result_sha256"]:
        raise UgiMorphologyProposalScheduleError("controller analysis identity changed")
    if analysis.get("decision", {}).get("smc_execution_authorized") is not False:
        raise UgiMorphologyProposalScheduleError("analysis unexpectedly authorized SMC")
    census_contract = load_census_contract(repo, paths["census_config"])
    selected = load_selected_program_manifest(census_contract)
    selected_indices = {record.population_index for record in selected.programs}
    support_rows = _read_support(paths["analysis_terminal_support"])
    successes_by_rank: dict[int, int] = defaultdict(int)
    trials_by_rank: dict[int, int] = defaultdict(int)
    for row in support_rows:
        rank = int(row["selection_rank"])
        trials_by_rank[rank] += 1
        successes_by_rank[rank] += int(row["support"].lower() == "true")
    if set(trials_by_rank.values()) != {24} or set(trials_by_rank) != set(range(1024)):
        raise UgiMorphologyProposalScheduleError("calibration program lattice changed")

    population_value = _load_json(paths["frozen_programs"], label="frozen programs")
    population = population_value.get("samples")
    if not isinstance(population, list) or len(population) != 4096:
        raise UgiMorphologyProposalScheduleError("frozen program population changed")
    selected_by_rank = sorted(selected.programs, key=lambda record: record.selection_rank)
    train_features = []
    train_successes = []
    train_trials = []
    feature_names: tuple[str, ...] | None = None
    for record in selected_by_rank:
        program = _program_dict(population[record.population_index]["program"])
        names, features = morphology_features(program)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise UgiMorphologyProposalScheduleError("morphology feature schema changed")
        train_features.append(features)
        train_successes.append(successes_by_rank[record.selection_rank])
        train_trials.append(trials_by_rank[record.selection_rank])
    policy = config["policy"]
    model = BinomialRidge(l2=float(policy["logistic_l2"])).fit(
        np.asarray(train_features),
        np.asarray(train_successes, dtype=np.float64),
        np.asarray(train_trials, dtype=np.float64),
    )

    unused_records = []
    unused_features = []
    for population_index, row in enumerate(population):
        if population_index in selected_indices:
            continue
        program = _program_dict(row["program"])
        names, features = morphology_features(program)
        if names != feature_names:
            raise UgiMorphologyProposalScheduleError("unused feature schema changed")
        program_bytes = canonical_morphology_program_bytes(dict(row["program"]))
        unused_records.append(
            {
                "population_index": population_index,
                "product_id": str(row["product_id"]),
                "program": {key: list(values) for key, values in program.items()},
                "program_sha256": hashlib.sha256(program_bytes).hexdigest(),
                "metadata": {
                    key: row[key]
                    for key in (
                        "source_stratum",
                        "branch_class",
                        "component_novelty_class",
                        "held_role_class",
                    )
                    if key in row
                },
            }
        )
        unused_features.append(features)
    if len(unused_records) != 3072:
        raise UgiMorphologyProposalScheduleError("unused program count changed")
    scores = model.predict(np.asarray(unused_features))
    rho = float(policy["mixture_rho"])
    epsilon = float(policy["score_floor"])
    kappa = float(policy["score_power"])
    if not 0.0 < rho < 1.0 or epsilon <= 0.0 or kappa <= 0.0:
        raise UgiMorphologyProposalScheduleError("proposal mixture policy is invalid")
    score_mass = np.maximum(scores, epsilon) ** kappa
    score_distribution = score_mass / score_mass.sum()
    prior_probability = 1.0 / len(scores)
    probabilities = (1.0 - rho) * prior_probability + rho * score_distribution
    if not np.isclose(probabilities.sum(), 1.0) or np.any(probabilities <= 0.0):
        raise UgiMorphologyProposalScheduleError("proposal lost population support")
    effective_count = float(1.0 / np.sum(probabilities**2))
    minimum_effective_fraction = float(policy["minimum_effective_program_fraction"])
    if effective_count / len(scores) < minimum_effective_fraction:
        raise UgiMorphologyProposalScheduleError("proposal is too concentrated")
    order = np.argsort(-scores, kind="mergesort")
    rank = np.empty(len(scores), dtype=np.int64)
    rank[order] = np.arange(len(scores))
    for index, record in enumerate(unused_records):
        record.update(
            {
                "proposal_score": float(scores[index]),
                "proposal_probability": float(probabilities[index]),
                "prior_probability": prior_probability,
                "importance_ratio_prior_over_proposal": float(
                    prior_probability / probabilities[index]
                ),
                "score_rank": int(rank[index]),
                "score_quartile": min(4, int(rank[index] * 4 / len(scores)) + 1),
            }
        )
    unused_records.sort(key=lambda row: int(row["population_index"]))
    schedule_content = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "population": "unused_3072_program_complement_of_dynamic_census",
        "feature_names": list(feature_names or ()),
        "records": unused_records,
    }
    schedule = {**schedule_content, "schedule_sha256": _logical_sha256(schedule_content)}
    inputs = {
        label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
        for label, path in sorted(paths.items())
    }
    inputs["config"] = {
        "path": str(config_path.relative_to(repo)),
        "sha256": sha256_file(config_path),
    }
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "frozen_unused_program_morphology_proposal_schedule",
        "scope": dict(EXPECTED_SCOPE),
        "inputs": inputs,
        "training": {
            "program_rows": len(selected_by_rank),
            "unique_program_sha256": len({record.program_sha256 for record in selected_by_rank}),
            "terminal_attempts": int(sum(train_trials)),
            "supported_terminals": int(sum(train_successes)),
            "features": list(feature_names or ()),
            "model": "binomial_logistic_ridge",
            "l2": float(policy["logistic_l2"]),
        },
        "unused_population": {
            "program_rows": len(unused_records),
            "unique_program_sha256": len(
                {str(record["program_sha256"]) for record in unused_records}
            ),
            "proposal_score_minimum": float(np.min(scores)),
            "proposal_score_median": float(np.median(scores)),
            "proposal_score_maximum": float(np.max(scores)),
            "proposal_probability_minimum": float(np.min(probabilities)),
            "proposal_probability_maximum": float(np.max(probabilities)),
            "proposal_effective_program_count": effective_count,
            "proposal_effective_program_fraction": effective_count / len(scores),
            "maximum_importance_ratio": float(np.max(prior_probability / probabilities)),
            "minimum_importance_ratio": float(np.min(prior_probability / probabilities)),
        },
        "proposal": {
            "form": "support_preserving_prior_score_mixture",
            "mixture_rho": rho,
            "score_floor": epsilon,
            "score_power": kappa,
            "prior_probability_positive_for_every_unused_program": True,
            "schedule_sha256": schedule["schedule_sha256"],
        },
        "next_gate": (
            "generate fresh native terminals for every unused program and compare target-free "
            "support across predeclared proposal-score quartiles before potency"
        ),
        "nonclaims": [
            "The proposal score is not potency.",
            "The proposal schedule does not authorize SMC.",
            "No unused-program terminal outcome was read while fitting or scoring.",
            "The proposal is not a production controller until unused-program confirmation.",
        ],
    }
    result = {**content, "result_sha256": _logical_sha256(content)}
    return result, schedule


__all__ = [
    "UgiMorphologyProposalScheduleError",
    "build_morphology_proposal_schedule",
]
