"""Adjudicate a morphology-strength challenger on fresh terminal outcomes."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.corpus.r1_prime_audit import sha256_file

CONFIG_SCHEMA_VERSION = "phase1_ugi_morphology_proposal_challenger_adjudication_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_morphology_proposal_challenger_adjudication.v1"
EXPECTED_SCOPE = {
    "read_only": True,
    "fresh_terminal_outcomes_only": True,
    "same_program_population_as_development": True,
    "point_rates_inspected_before_adjudication_config": True,
    "applicability_boundary_frozen": True,
    "potency_predictions_consumed": False,
    "oracle_calls": 0,
    "route_calls": 0,
    "synthesis_calls": 0,
    "proposal_calls": 0,
    "generator_trajectories_advanced": False,
    "candidate_selection": False,
    "sealed_holdout_access": False,
}
EXPECTED_INPUTS = {
    "baseline_schedule",
    "challenger_confirmation_analysis",
    "challenger_schedule",
    "fresh_support_ledger",
    "runner",
    "source",
    "strength_sweep",
    "tests",
}


class UgiMorphologyProposalChallengerAdjudicationError(RuntimeError):
    """Raised when challenger adjudication inputs or policy change."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiMorphologyProposalChallengerAdjudicationError(
            f"invalid {label}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise UgiMorphologyProposalChallengerAdjudicationError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiMorphologyProposalChallengerAdjudicationError(f"malformed pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiMorphologyProposalChallengerAdjudicationError(
            f"pin escapes repository: {label}"
        ) from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiMorphologyProposalChallengerAdjudicationError(f"pin changed: {label}")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise UgiMorphologyProposalChallengerAdjudicationError("invalid support ledger") from error
    if len(rows) != 3072:
        raise UgiMorphologyProposalChallengerAdjudicationError("fresh support ledger changed")
    return rows


def _effective_labeled_count(
    probabilities: np.ndarray, labels: Sequence[str], mask: np.ndarray
) -> float:
    mass: dict[str, float] = defaultdict(float)
    for probability, label, keep in zip(probabilities, labels, mask, strict=True):
        if keep and label:
            mass[label] += float(probability)
    values = np.asarray(tuple(mass.values()), dtype=np.float64)
    if len(values) == 0 or float(values.sum()) <= 0.0:
        return 0.0
    values /= values.sum()
    return float(1.0 / np.sum(values**2))


def paired_cluster_bootstrap(
    *,
    groups: Sequence[str],
    support: np.ndarray,
    current: np.ndarray,
    challenger: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    """Bootstrap the paired challenger-current terminal-yield difference."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[group].append(index)
    clusters = tuple(tuple(indices) for _, indices in sorted(grouped.items()))
    current_den = np.asarray([current[list(indices)].sum() for indices in clusters])
    current_num = np.asarray(
        [(current[list(indices)] * support[list(indices)]).sum() for indices in clusters]
    )
    challenger_den = np.asarray([challenger[list(indices)].sum() for indices in clusters])
    challenger_num = np.asarray(
        [(challenger[list(indices)] * support[list(indices)]).sum() for indices in clusters]
    )
    rng = np.random.default_rng(seed)
    current_rates = np.empty(replicates, dtype=np.float64)
    challenger_rates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        for _ in range(1000):
            counts = np.bincount(
                rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters)
            ).astype(np.float64)
            current_rate = np.dot(counts, current_num) / np.dot(counts, current_den)
            if current_rate > 0.0:
                break
        else:
            raise UgiMorphologyProposalChallengerAdjudicationError(
                "bootstrap could not draw a positive current-support replicate"
            )
        current_rates[replicate] = current_rate
        challenger_rates[replicate] = np.dot(counts, challenger_num) / np.dot(
            counts, challenger_den
        )
    differences = challenger_rates - current_rates
    relative = challenger_rates / current_rates - 1.0
    return {
        "current_support_rate_ci95": [
            float(np.quantile(current_rates, 0.025)),
            float(np.quantile(current_rates, 0.975)),
        ],
        "challenger_support_rate_ci95": [
            float(np.quantile(challenger_rates, 0.025)),
            float(np.quantile(challenger_rates, 0.975)),
        ],
        "absolute_difference_ci95": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "relative_improvement_ci95": [
            float(np.quantile(relative, 0.025)),
            float(np.quantile(relative, 0.975)),
        ],
    }


def build_morphology_proposal_challenger_adjudication(
    repo: Path, config_path: Path
) -> dict[str, Any]:
    """Compare the selected challenger with the current confirmed proposal."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="challenger adjudication config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiMorphologyProposalChallengerAdjudicationError("unsupported adjudication schema")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiMorphologyProposalChallengerAdjudicationError("adjudication scope changed")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != EXPECTED_INPUTS:
        raise UgiMorphologyProposalChallengerAdjudicationError("adjudication pins changed")
    paths = {label: _pin(repo, record, label=label) for label, record in raw_inputs.items()}
    baseline = _load_json(paths["baseline_schedule"], label="baseline schedule")
    challenger = _load_json(paths["challenger_schedule"], label="challenger schedule")
    baseline_rows = baseline.get("records")
    challenger_rows = challenger.get("records")
    if (
        not isinstance(baseline_rows, list)
        or not isinstance(challenger_rows, list)
        or len(baseline_rows) != 3072
        or len(challenger_rows) != 3072
    ):
        raise UgiMorphologyProposalChallengerAdjudicationError("proposal schedules changed")
    for left, right in zip(baseline_rows, challenger_rows, strict=True):
        if (
            left["population_index"] != right["population_index"]
            or left["program_sha256"] != right["program_sha256"]
            or left["prior_probability"] != right["prior_probability"]
            or left["proposal_score"] != right["proposal_score"]
        ):
            raise UgiMorphologyProposalChallengerAdjudicationError(
                "current and challenger programs differ"
            )
    fresh = _read_csv(paths["fresh_support_ledger"])
    for schedule_row, support_row in zip(challenger_rows, fresh, strict=True):
        if int(schedule_row["population_index"]) != int(support_row["population_index"]):
            raise UgiMorphologyProposalChallengerAdjudicationError(
                "fresh outcome alignment changed"
            )
    confirmation = _load_json(
        paths["challenger_confirmation_analysis"], label="challenger confirmation"
    )
    if confirmation.get("decision", {}).get("morphology_proposal_confirmed") is not True:
        raise UgiMorphologyProposalChallengerAdjudicationError(
            "challenger did not pass fresh confirmation"
        )
    sweep = _load_json(paths["strength_sweep"], label="strength sweep")
    selected = sweep["development_selected_challenger"]
    if selected["candidate_id"] != config["policy"]["challenger_candidate_id"]:
        raise UgiMorphologyProposalChallengerAdjudicationError("challenger identity changed")

    broad = np.asarray([float(row["prior_probability"]) for row in baseline_rows])
    current = np.asarray([float(row["proposal_probability"]) for row in baseline_rows])
    proposed = np.asarray([float(row["proposal_probability"]) for row in challenger_rows])
    support = np.asarray([str(row["support"]).lower() == "true" for row in fresh], dtype=np.float64)
    valid = np.asarray(
        [str(row["valid_exact_l1"]).lower() == "true" for row in fresh],
        dtype=np.float64,
    )
    for probabilities in (broad, current, proposed):
        if np.any(probabilities <= 0.0) or not np.isclose(probabilities.sum(), 1.0):
            raise UgiMorphologyProposalChallengerAdjudicationError(
                "proposal probability mass changed"
            )
    rates = {
        "broad": float(np.dot(broad, support)),
        "current": float(np.dot(current, support)),
        "challenger": float(np.dot(proposed, support)),
    }
    validity = {
        "broad": float(np.dot(broad, valid)),
        "current": float(np.dot(current, valid)),
        "challenger": float(np.dot(proposed, valid)),
    }
    labels = [str(row["smiles"]) for row in fresh]
    supported_diversity = {
        "current": _effective_labeled_count(current, labels, support.astype(bool)),
        "challenger": _effective_labeled_count(proposed, labels, support.astype(bool)),
    }
    bootstrap_policy = config["policy"]["clustered_bootstrap"]
    bootstrap = paired_cluster_bootstrap(
        groups=[str(row["program_sha256"]) for row in fresh],
        support=support,
        current=current,
        challenger=proposed,
        replicates=int(bootstrap_policy["replicates"]),
        seed=int(bootstrap_policy["seed"]),
    )
    checks = {
        "point_support_improvement": rates["challenger"] > rates["current"],
        "clustered_absolute_improvement": bootstrap["absolute_difference_ci95"][0]
        > float(config["policy"]["minimum_absolute_improvement_ci_lower"]),
        "validity_retained": validity["challenger"]
        >= validity["current"] - float(config["policy"]["maximum_validity_loss"]),
        "supported_diversity_retained": supported_diversity["challenger"]
        >= float(config["policy"]["minimum_supported_diversity_fraction_of_current"])
        * supported_diversity["current"],
        "development_floors_passed": all(selected["floor_checks"].values()),
    }
    promoted = all(checks.values())
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
        "status": "fresh_terminal_challenger_adjudication_complete",
        "scope": dict(EXPECTED_SCOPE),
        "inputs": inputs,
        "candidate_id": str(selected["candidate_id"]),
        "support_rates": rates,
        "challenger_over_current": {
            "absolute_improvement": rates["challenger"] - rates["current"],
            "relative_improvement": rates["challenger"] / rates["current"] - 1.0,
            "clustered_bootstrap": bootstrap,
        },
        "valid_exact_l1_rates": validity,
        "supported_smiles_effective_counts": supported_diversity,
        "decision": {
            "checks": checks,
            "challenger_promoted": promoted,
            "applicability_boundary_changed": False,
            "potency_guidance_authorized": False,
            "mh_authorized": False,
            "next_gate": (
                "role_restricted_morphology_potency_signal"
                if promoted
                else "retain_current_proposal_and_use_terminal_filtering"
            ),
        },
        "nonclaims": [
            "The fresh outcomes use previously evaluated morphology programs.",
            "Promotion establishes a more efficient proposal, not terminal support certainty.",
            "This result does not evaluate potency, routes, synthesis, SMC or MH.",
        ],
    }
    return {**content, "result_sha256": _logical_sha256(content)}


__all__ = [
    "UgiMorphologyProposalChallengerAdjudicationError",
    "build_morphology_proposal_challenger_adjudication",
    "paired_cluster_bootstrap",
]
