"""Freeze one morphology-strength challenger for a fresh terminal draw."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from experiments._runtime.historical import HistoricalPinArchiveError, resolve_pinned_input
from experiments.phase1.product_l1.sampling.ugi_complete_morphology_proposal import (
    support_preserving_probabilities,
)
from forge.corpus.r1_prime_audit import sha256_file

CONFIG_SCHEMA_VERSION = "phase1_ugi_morphology_proposal_challenger_schedule_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_morphology_proposal_challenger_schedule.v1"
SCHEDULE_SCHEMA_VERSION = "forge.ugi_morphology_proposal_schedule.v1"
EXPECTED_SCOPE = {
    "read_only": True,
    "same_programs_new_terminal_randomness": True,
    "selected_challenger_only": True,
    "terminal_outcomes_consumed": False,
    "potency_predictions_consumed": False,
    "oracle_calls": 0,
    "route_calls": 0,
    "synthesis_calls": 0,
    "generator_trajectories_advanced": False,
    "candidate_selection": False,
    "sealed_holdout_access": False,
}
EXPECTED_INPUTS = {"baseline_schedule", "runner", "source", "sweep_result", "tests"}


class UgiMorphologyProposalChallengerScheduleError(RuntimeError):
    """Raised when the challenger schedule contract changes."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiMorphologyProposalChallengerScheduleError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiMorphologyProposalChallengerScheduleError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiMorphologyProposalChallengerScheduleError(f"malformed pin: {label}")
    try:
        return resolve_pinned_input(repo, str(record["path"]), str(record["sha256"]))
    except HistoricalPinArchiveError as error:
        raise UgiMorphologyProposalChallengerScheduleError(f"pin changed: {label}") from error


def build_morphology_proposal_challenger_schedule(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the one frozen challenger to the original confirmation programs."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="challenger schedule config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiMorphologyProposalChallengerScheduleError("unsupported challenger schedule schema")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiMorphologyProposalChallengerScheduleError("challenger scope changed")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != EXPECTED_INPUTS:
        raise UgiMorphologyProposalChallengerScheduleError("challenger pins changed")
    paths = {label: _pin(repo, record, label=label) for label, record in raw_inputs.items()}
    sweep = _load_json(paths["sweep_result"], label="strength sweep")
    decision = sweep.get("decision", {})
    if (
        decision.get("challenger_selected_for_fresh_confirmation") is not True
        or decision.get("production_replacement_authorized") is not False
    ):
        raise UgiMorphologyProposalChallengerScheduleError(
            "strength sweep did not authorize a fresh challenger draw"
        )
    selected = sweep["development_selected_challenger"]
    expected = config["challenger"]
    if (
        selected.get("candidate_id") != expected["candidate_id"]
        or float(selected["mixture_rho"]) != float(expected["mixture_rho"])
        or float(selected["score_power"]) != float(expected["score_power"])
        or not all(selected.get("floor_checks", {}).values())
    ):
        raise UgiMorphologyProposalChallengerScheduleError("selected challenger identity changed")
    baseline = _load_json(paths["baseline_schedule"], label="baseline schedule")
    raw_records = baseline.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != 3072:
        raise UgiMorphologyProposalChallengerScheduleError("baseline schedule changed")
    records = copy.deepcopy(raw_records)
    broad = np.asarray([float(row["prior_probability"]) for row in records])
    scores = np.asarray([float(row["proposal_score"]) for row in records])
    proposal, importance = support_preserving_probabilities(
        broad,
        scores,
        mixture_rho=float(expected["mixture_rho"]),
        score_floor=float(expected["score_floor"]),
        score_power=float(expected["score_power"]),
    )
    for row, probability, ratio in zip(records, proposal, importance, strict=True):
        row["proposal_probability"] = float(probability)
        row["importance_ratio_prior_over_proposal"] = float(ratio)
    schedule_content = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "population": "same_3072_programs_with_fresh_challenger_terminal_randomness",
        "feature_names": list(baseline["feature_names"]),
        "records": records,
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
        "status": "frozen_strength_challenger_schedule_for_fresh_terminal_draw",
        "scope": dict(EXPECTED_SCOPE),
        "inputs": inputs,
        "population": {
            "program_rows": len(records),
            "unique_program_sha256": len({str(row["program_sha256"]) for row in records}),
            "previously_evaluated_programs": True,
            "fresh_terminal_outcomes": True,
        },
        "proposal": {
            "form": "support_preserving_prior_score_mixture",
            "candidate_id": str(expected["candidate_id"]),
            "mixture_rho": float(expected["mixture_rho"]),
            "score_floor": float(expected["score_floor"]),
            "score_power": float(expected["score_power"]),
            "schedule_sha256": schedule["schedule_sha256"],
            "effective_program_count": float(1.0 / np.sum(proposal**2)),
            "maximum_importance_ratio": float(importance.max()),
            "minimum_importance_ratio": float(importance.min()),
            "all_programs_positive": bool(np.all(proposal > 0.0)),
        },
        "decision": {
            "challenger_frozen": True,
            "production_replacement_authorized": False,
            "next_gate": "independent_fresh_terminal_outcome_confirmation",
        },
        "nonclaims": [
            "The program identities were used in the development sweep.",
            "The new terminal seeds provide independent outcome randomness, not a new program set.",
            "No potency, route, synthesis or applicability outcome was read by this builder.",
        ],
    }
    result = {**content, "result_sha256": _logical_sha256(content)}
    return result, schedule


__all__ = [
    "UgiMorphologyProposalChallengerScheduleError",
    "build_morphology_proposal_challenger_schedule",
]
