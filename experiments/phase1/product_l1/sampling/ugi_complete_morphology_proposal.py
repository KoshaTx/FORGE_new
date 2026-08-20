"""Freeze a full-support morphology proposal over qualified Ugi role states."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from experiments.phase1.product_l1.sampling.terminal_census import (
    load_census_contract,
    load_selected_program_manifest,
)
from experiments.phase1.synthesis_guidance.adapters.terminal_support import (
    canonical_morphology_program_bytes,
)
from forge.core.io import stable_json as _stable_json
from forge.corpus.r1_prime_audit import sha256_bytes, sha256_file
from forge.model.ugi_morphology_program import UgiMorphologyProgram
from forge.model.ugi_program_prior import UgiProgramPrior, load_program_prior
from forge.potency.controller import BinomialRidge, morphology_features

CONFIG_SCHEMA_VERSION = "phase1_ugi_complete_morphology_proposal_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_complete_morphology_proposal.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_complete_morphology_proposal_ledger.v1"
EXPECTED_SCOPE = {
    "read_only": True,
    "complete_qualified_role_state_cartesian_support": True,
    "observed_terminal_outcomes_for_model_development_only": True,
    "confirmation_result_required": True,
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
    "analysis_terminal_support",
    "census_config",
    "confirmation_analysis",
    "controller_analysis",
    "frozen_programs",
    "program_prior",
    "proposal_result",
    "proposal_schedule",
    "runner",
    "source",
    "tests",
}


class UgiCompleteMorphologyProposalError(RuntimeError):
    """Raised when the complete morphology-support contract changes."""


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiCompleteMorphologyProposalError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiCompleteMorphologyProposalError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiCompleteMorphologyProposalError(f"malformed pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiCompleteMorphologyProposalError(f"pin escapes repository: {label}") from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiCompleteMorphologyProposalError(f"pin changed: {label}")
    return path


def _program_mapping(program: UgiMorphologyProgram) -> dict[str, list[int]]:
    return {
        "node_counts": list(program.node_counts),
        "junction_budgets": list(program.junction_budgets),
        "cycle_ranks": list(program.cycle_ranks),
        "attachment_counts": list(program.attachment_counts),
    }


def enumerate_complete_program_support(
    prior: UgiProgramPrior, bounds: Mapping[str, int]
) -> tuple[dict[str, Any], ...]:
    """Enumerate every qualified role-state tuple with its broad-prior mass."""

    if len(prior.support_by_role) != 3 or len(prior.probabilities_by_role) != 3:
        raise UgiCompleteMorphologyProposalError("program prior role count changed")
    maximum_total = int(bounds["maximum_total_atoms"])
    records = []
    for states_with_mass in product(
        *(
            tuple(zip(states, probabilities, strict=True))
            for states, probabilities in zip(
                prior.support_by_role, prior.probabilities_by_role, strict=True
            )
        )
    ):
        states = tuple(state for state, _ in states_with_mass)
        program = UgiMorphologyProgram(
            node_counts=tuple(state[0] for state in states),
            junction_budgets=tuple(state[1] for state in states),
            cycle_ranks=tuple(state[2] for state in states),
            attachment_counts=tuple(state[3] for state in states),
        )
        if program.node_count > maximum_total:
            continue
        payload = canonical_morphology_program_bytes(program)
        records.append(
            {
                "program_sha256": hashlib.sha256(payload).hexdigest(),
                "program": _program_mapping(program),
                "broad_prior_probability": float(
                    np.prod([probability for _, probability in states_with_mass])
                ),
            }
        )
    records.sort(key=lambda row: str(row["program_sha256"]))
    if len({str(row["program_sha256"]) for row in records}) != len(records):
        raise UgiCompleteMorphologyProposalError("complete program support is not unique")
    total_mass = sum(float(row["broad_prior_probability"]) for row in records)
    if total_mass <= 0.0:
        raise UgiCompleteMorphologyProposalError("complete program support has no probability mass")
    for record in records:
        record["broad_prior_probability"] = float(record["broad_prior_probability"]) / total_mass
    return tuple(records)


def support_preserving_probabilities(
    broad: np.ndarray,
    scores: np.ndarray,
    *,
    mixture_rho: float,
    score_floor: float,
    score_power: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return q and p/q for a broad-prior-relative score mixture."""

    if (
        broad.ndim != 1
        or scores.shape != broad.shape
        or np.any(~np.isfinite(broad))
        or np.any(~np.isfinite(scores))
        or np.any(broad <= 0.0)
        or np.any(scores < 0.0)
        or not np.isclose(broad.sum(), 1.0)
        or not 0.0 < mixture_rho < 1.0
        or score_floor <= 0.0
        or score_power <= 0.0
    ):
        raise UgiCompleteMorphologyProposalError("invalid support-preserving proposal request")
    score_mass = broad * np.maximum(scores, score_floor) ** score_power
    tilted = score_mass / score_mass.sum()
    proposal = (1.0 - mixture_rho) * broad + mixture_rho * tilted
    if np.any(proposal <= 0.0) or not np.isclose(proposal.sum(), 1.0):
        raise UgiCompleteMorphologyProposalError("morphology proposal lost support")
    return proposal, broad / proposal


def _read_support(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise UgiCompleteMorphologyProposalError("invalid controller support ledger") from error
    if len(rows) != 24576:
        raise UgiCompleteMorphologyProposalError("controller support ledger is incomplete")
    return rows


def _fit_confirmed_score_model(
    repo: Path,
    *,
    census_config: Path,
    frozen_programs: Path,
    support_path: Path,
    proposal_result: Mapping[str, Any],
    proposal_schedule: Mapping[str, Any],
) -> tuple[BinomialRidge, tuple[str, ...]]:
    census = load_census_contract(repo, census_config)
    selected = sorted(
        load_selected_program_manifest(census).programs,
        key=lambda record: record.selection_rank,
    )
    population = _load_json(frozen_programs, label="frozen programs").get("samples")
    if not isinstance(population, list) or len(population) != 4096:
        raise UgiCompleteMorphologyProposalError("frozen program population changed")
    support = _read_support(support_path)
    successes: dict[int, int] = defaultdict(int)
    trials: dict[int, int] = defaultdict(int)
    for row in support:
        rank = int(row["selection_rank"])
        trials[rank] += 1
        successes[rank] += int(row["support"].lower() == "true")
    if set(trials) != set(range(1024)) or set(trials.values()) != {24}:
        raise UgiCompleteMorphologyProposalError("controller outcome lattice changed")
    features = []
    names: tuple[str, ...] | None = None
    for record in selected:
        raw_program = population[record.population_index]["program"]
        current_names, current = morphology_features(raw_program)
        if names is None:
            names = current_names
        elif names != current_names:
            raise UgiCompleteMorphologyProposalError("morphology feature schema changed")
        features.append(current)
    l2 = float(proposal_result["training"]["l2"])
    model = BinomialRidge(l2=l2).fit(
        np.asarray(features),
        np.asarray([successes[index] for index in range(1024)], dtype=np.float64),
        np.asarray([trials[index] for index in range(1024)], dtype=np.float64),
    )
    scheduled = proposal_schedule.get("records")
    if not isinstance(scheduled, list) or len(scheduled) != 3072:
        raise UgiCompleteMorphologyProposalError("confirmation schedule changed")
    check_features = []
    expected_scores = []
    for row in scheduled:
        current_names, current = morphology_features(row["program"])
        if current_names != names:
            raise UgiCompleteMorphologyProposalError("schedule feature schema changed")
        check_features.append(current)
        expected_scores.append(float(row["proposal_score"]))
    reproduced = model.predict(np.asarray(check_features))
    if not np.allclose(reproduced, np.asarray(expected_scores), rtol=0.0, atol=1e-14):
        raise UgiCompleteMorphologyProposalError("confirmed morphology score did not reproduce")
    return model, names or ()


def _ledger_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as handle:
        for row in rows:
            handle.write((_stable_json(row) + "\n").encode())
    return output.getvalue()


def build_complete_morphology_proposal(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Build the complete qualified support and confirmed probability proposal."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="complete morphology proposal config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiCompleteMorphologyProposalError("unsupported complete-proposal schema")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiCompleteMorphologyProposalError("complete-proposal scope changed")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != EXPECTED_INPUTS:
        raise UgiCompleteMorphologyProposalError("complete-proposal input pins changed")
    paths = {label: _pin(repo, record, label=label) for label, record in raw_inputs.items()}
    confirmation = _load_json(paths["confirmation_analysis"], label="confirmation analysis")
    if confirmation.get("decision", {}).get("morphology_proposal_confirmed") is not True:
        raise UgiCompleteMorphologyProposalError("morphology proposal is not confirmed")
    controller = _load_json(paths["controller_analysis"], label="controller analysis")
    if controller.get("decision", {}).get("smc_execution_authorized") is not False:
        raise UgiCompleteMorphologyProposalError("controller analysis unexpectedly authorizes SMC")
    proposal_result = _load_json(paths["proposal_result"], label="proposal result")
    proposal_schedule = _load_json(paths["proposal_schedule"], label="proposal schedule")
    if proposal_result.get("proposal", {}).get("schedule_sha256") != proposal_schedule.get(
        "schedule_sha256"
    ):
        raise UgiCompleteMorphologyProposalError("confirmed proposal schedule identity changed")
    model, feature_names = _fit_confirmed_score_model(
        repo,
        census_config=paths["census_config"],
        frozen_programs=paths["frozen_programs"],
        support_path=paths["analysis_terminal_support"],
        proposal_result=proposal_result,
        proposal_schedule=proposal_schedule,
    )
    prior, bounds = load_program_prior(paths["program_prior"])
    expected = config["support"]
    role_counts = [len(states) for states in prior.support_by_role]
    if role_counts != list(expected["role_state_counts"]) or int(np.prod(role_counts)) != int(
        expected["cartesian_programs"]
    ):
        raise UgiCompleteMorphologyProposalError("qualified role-state support changed")
    records = list(enumerate_complete_program_support(prior, bounds))
    if len(records) != int(expected["cartesian_programs"]):
        raise UgiCompleteMorphologyProposalError(
            "total-size filtering removed a qualified role-state combination"
        )
    feature_matrix = []
    for row in records:
        names, values = morphology_features(row["program"])
        if names != feature_names:
            raise UgiCompleteMorphologyProposalError("complete-support feature schema changed")
        feature_matrix.append(values)
    scores = model.predict(np.asarray(feature_matrix))
    broad = np.asarray([float(row["broad_prior_probability"]) for row in records])
    policy = proposal_result["proposal"]
    probabilities, importance = support_preserving_probabilities(
        broad,
        scores,
        mixture_rho=float(policy["mixture_rho"]),
        score_floor=float(policy["score_floor"]),
        score_power=float(policy["score_power"]),
    )
    for row, score, probability, ratio in zip(
        records, scores, probabilities, importance, strict=True
    ):
        row.update(
            {
                "support_score": float(score),
                "proposal_probability": float(probability),
                "importance_ratio_broad_over_proposal": float(ratio),
            }
        )
    ledger = _ledger_bytes(records)
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
        "status": "complete_qualified_morphology_support_and_confirmed_proposal_frozen",
        "scope": dict(EXPECTED_SCOPE),
        "inputs": inputs,
        "support": {
            "definition": "cartesian_product_of_qualified_role_level_program_states",
            "programs": len(records),
            "role_state_counts": role_counts,
            "maximum_combined_exterior_atoms": max(
                sum(row["program"]["node_counts"]) for row in records
            ),
            "maximum_total_atoms_bound": int(bounds["maximum_total_atoms"]),
            "every_program_has_positive_broad_prior_probability": bool(np.all(broad > 0.0)),
            "every_program_has_positive_proposal_probability": bool(np.all(probabilities > 0.0)),
            "frozen_4096_program_population_role": "monte_carlo_evaluation_draw_only",
        },
        "proposal": {
            "form": "broad_prior_relative_support_preserving_score_mixture",
            "mixture_rho": float(policy["mixture_rho"]),
            "score_floor": float(policy["score_floor"]),
            "score_power": float(policy["score_power"]),
            "broad_effective_program_count": float(1.0 / np.sum(broad**2)),
            "proposal_effective_program_count": float(1.0 / np.sum(probabilities**2)),
            "maximum_importance_ratio": float(np.max(importance)),
            "minimum_importance_ratio": float(np.min(importance)),
            "score_minimum": float(np.min(scores)),
            "score_median": float(np.median(scores)),
            "score_maximum": float(np.max(scores)),
            "confirmed_schedule_score_reproduced_exactly": True,
        },
        "artifacts": {
            "proposal_ledger.jsonl.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "records": len(records),
                "sha256": sha256_bytes(ledger),
                "logical_sha256": _logical_sha256(records),
            }
        },
        "decision": {
            "complete_support_frozen": True,
            "morphology_score_changes_probability_not_support": True,
            "partial_state_smc_authorized": False,
            "potency_guidance_authorized": False,
            "next_gate": "oracle reliability under the morphology-enriched supported distribution",
        },
        "nonclaims": [
            "Qualified role-state support is not every integer tuple inside the numerical bounds.",
            "The complete proposal does not evaluate potency, routes or synthesis.",
            "The 4,096-program draw and its 1,024/3,072 partitions do not define production support.",
        ],
    }
    result = {**content, "result_sha256": _logical_sha256(content)}
    return result, ledger


__all__ = [
    "UgiCompleteMorphologyProposalError",
    "build_complete_morphology_proposal",
    "enumerate_complete_program_support",
    "support_preserving_probabilities",
]
