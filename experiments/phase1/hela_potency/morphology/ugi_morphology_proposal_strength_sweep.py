"""Sweep a frozen Ugi morphology proposal without changing applicability."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.phase1.product_l1.sampling.ugi_complete_morphology_proposal import (
    support_preserving_probabilities,
)
from forge.corpus.r1_prime_audit import sha256_bytes, sha256_file
from forge.potency.applicability import ugi_distributional_applicability as applicability_v1
from forge.potency.applicability import ugi_distributional_applicability_v2 as applicability_v2
from forge.potency.controller import ROLE_COMPONENT_KEYS, ROLES, VIEWS

CONFIG_SCHEMA_VERSION = "phase1_ugi_morphology_proposal_strength_sweep_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_morphology_proposal_strength_sweep.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_morphology_proposal_strength_sweep_ledger.v1"
EXPECTED_SCOPE = {
    "read_only": True,
    "reuse_confirmation_for_challenger_selection_only": True,
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
    "applicability_result",
    "confirmation_analysis",
    "confirmation_schedule",
    "confirmation_support_ledger",
    "confirmation_terminal_ledger",
    "curated_agile",
    "full_proposal_ledger",
    "full_proposal_result",
    "runner",
    "source",
    "tests",
}


class UgiMorphologyProposalStrengthSweepError(RuntimeError):
    """Raised when the frozen proposal-strength experiment changes."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiMorphologyProposalStrengthSweepError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiMorphologyProposalStrengthSweepError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiMorphologyProposalStrengthSweepError(f"malformed pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiMorphologyProposalStrengthSweepError(f"pin escapes repository: {label}") from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiMorphologyProposalStrengthSweepError(f"pin changed: {label}")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise UgiMorphologyProposalStrengthSweepError(f"invalid CSV: {path}") from error
    if not rows:
        raise UgiMorphologyProposalStrengthSweepError(f"empty CSV: {path}")
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with gzip.open(path, "rt") as handle:
            rows = [json.loads(line) for line in handle]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiMorphologyProposalStrengthSweepError(f"invalid JSONL: {path}") from error
    if not rows:
        raise UgiMorphologyProposalStrengthSweepError(f"empty JSONL: {path}")
    return rows


def _effective_count(probabilities: np.ndarray) -> float:
    if probabilities.ndim != 1 or np.any(probabilities < 0.0):
        raise UgiMorphologyProposalStrengthSweepError("invalid probability vector")
    total = float(probabilities.sum())
    if total <= 0.0:
        return 0.0
    normalized = probabilities / total
    return float(1.0 / np.sum(normalized**2))


def _shannon_effective_count(probabilities: np.ndarray) -> float:
    total = float(probabilities.sum())
    if total <= 0.0:
        return 0.0
    normalized = probabilities[probabilities > 0.0] / total
    return float(np.exp(-np.sum(normalized * np.log(normalized))))


def _aggregate_effective_count(
    probabilities: np.ndarray, labels: Sequence[str], mask: np.ndarray | None = None
) -> float:
    if len(probabilities) != len(labels):
        raise UgiMorphologyProposalStrengthSweepError("aggregation labels changed")
    selected = np.ones(len(probabilities), dtype=bool) if mask is None else mask
    mass: dict[str, float] = defaultdict(float)
    for probability, label, keep in zip(probabilities, labels, selected, strict=True):
        if keep and label:
            mass[label] += float(probability)
    return _effective_count(np.asarray(tuple(mass.values()), dtype=np.float64))


def _program_role_keys(row: Mapping[str, Any]) -> tuple[str, str, str]:
    program = row["program"]
    output = []
    for role in range(3):
        output.append(
            _stable_json(
                [
                    program["node_counts"][role],
                    program["junction_budgets"][role],
                    program["cycle_ranks"][role],
                    program["attachment_counts"][role],
                ]
            )
        )
    return tuple(output)  # type: ignore[return-value]


def _marginal_metrics(
    rows: Sequence[Mapping[str, Any]], probabilities: np.ndarray
) -> dict[str, dict[str, float]]:
    role_names = ("amine", "aldehyde", "isocyanide")
    keys = [_program_role_keys(row) for row in rows]
    output: dict[str, dict[str, float]] = {}
    for role_index, role_name in enumerate(role_names):
        mass: dict[str, float] = defaultdict(float)
        for probability, current in zip(probabilities, keys, strict=True):
            mass[current[role_index]] += float(probability)
        values = np.asarray(tuple(mass.values()), dtype=np.float64)
        output[role_name] = {
            "states": len(values),
            "effective_state_count": _effective_count(values),
            "shannon_effective_state_count": _shannon_effective_count(values),
            "maximum_state_probability": float(values.max()),
        }
    return output


def _proposal_metrics(
    rows: Sequence[Mapping[str, Any]], broad: np.ndarray, proposal: np.ndarray
) -> dict[str, Any]:
    importance = broad / proposal
    return {
        "effective_program_count": _effective_count(proposal),
        "shannon_effective_program_count": _shannon_effective_count(proposal),
        "maximum_program_probability": float(proposal.max()),
        "importance_ess_fraction": float(1.0 / np.sum((broad**2) / proposal)),
        "maximum_importance_ratio": float(importance.max()),
        "minimum_importance_ratio": float(importance.min()),
        "role_marginals": _marginal_metrics(rows, proposal),
    }


def _terminal_components(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    output = {role: [] for role in ROLES}
    for row in rows:
        terminal = row.get("native_terminal")
        components = (
            terminal.get("component_smiles_by_role") if isinstance(terminal, Mapping) else {}
        )
        for role in ROLES:
            key = ROLE_COMPONENT_KEYS[role]
            output[role].append(
                str(components.get(key, "")) if isinstance(components, Mapping) else ""
            )
    return output


def _confirmation_metrics(
    probabilities: np.ndarray,
    support_rows: Sequence[Mapping[str, Any]],
    terminal_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    support = np.asarray(
        [str(row["support"]).lower() == "true" for row in support_rows], dtype=bool
    )
    valid = np.asarray(
        [str(row["valid_exact_l1"]).lower() == "true" for row in support_rows], dtype=bool
    )
    smiles = [str(row["smiles"]) for row in support_rows]
    components = _terminal_components(terminal_rows)
    support_rate = float(np.dot(probabilities, support))
    valid_rate = float(np.dot(probabilities, valid))
    novelty_mass: dict[str, float] = defaultdict(float)
    for probability, row, keep in zip(probabilities, support_rows, valid, strict=True):
        if keep:
            roles = tuple(json.loads(str(row["exact_new_roles_json"])))
            novelty_mass["+".join(roles) if roles else "none"] += float(probability)
    novelty_total = sum(novelty_mass.values())
    return {
        "expected_support_rate": support_rate,
        "expected_valid_exact_l1_rate": valid_rate,
        "supported_smiles_effective_count": _aggregate_effective_count(
            probabilities, smiles, support
        ),
        "valid_smiles_effective_count": _aggregate_effective_count(probabilities, smiles, valid),
        "valid_component_effective_counts": {
            role: _aggregate_effective_count(probabilities, labels, valid)
            for role, labels in components.items()
        },
        "supported_component_effective_counts": {
            role: _aggregate_effective_count(probabilities, labels, support)
            for role, labels in components.items()
        },
        "valid_novelty_pattern_probability": {
            key: value / novelty_total for key, value in sorted(novelty_mass.items())
        },
    }


def _view_failures(
    terminal_rows: Sequence[Mapping[str, Any]],
    *,
    references: Mapping[str, applicability_v2.CountChemicalReference],
    thresholds: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, float]]]:
    categories: list[str] = []
    ratios_by_row: list[dict[str, float]] = []
    for row in terminal_rows:
        terminal = row.get("native_terminal")
        valid = bool(isinstance(terminal, Mapping) and terminal.get("valid")) and bool(
            terminal.get("l1_forward_verification", {}).get("exact_product_reconstructed")
        )
        if not valid:
            categories.append("invalid_or_non_exact_l1")
            ratios_by_row.append({})
            continue
        components = terminal["component_smiles_by_role"]
        query = {
            "product_smiles": str(terminal["smiles"]),
            "amine_smiles": str(components[ROLE_COMPONENT_KEYS["amine"]]),
            "aldehyde_smiles": str(components[ROLE_COMPONENT_KEYS["aldehyde"]]),
            "isocyanide_smiles": str(components[ROLE_COMPONENT_KEYS["isocyanide"]]),
        }
        distances = applicability_v1._distance_fields(references, query, generated=True)
        ratios: dict[str, float] = {}
        failed = []
        for view in VIEWS:
            fingerprint = distances[view].fingerprint / float(
                thresholds[view]["fingerprint"]["interpolative_max"]
            )
            descriptor = distances[view].descriptor / float(
                thresholds[view]["descriptor"]["interpolative_max"]
            )
            ratios[f"{view}_fingerprint"] = float(fingerprint)
            ratios[f"{view}_descriptor"] = float(descriptor)
            if max(fingerprint, descriptor) > 1.0:
                failed.append(view)
        if not failed:
            category = "supported_all_views"
        elif len(failed) == 1:
            category = f"{failed[0]}_only"
        else:
            category = "multiple_views"
        categories.append(category)
        ratios_by_row.append(ratios)
    return categories, ratios_by_row


def _weighted_failure_summary(
    probabilities: np.ndarray,
    categories: Sequence[str],
    ratios_by_row: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    category_mass: dict[str, float] = defaultdict(float)
    metric_failure_mass: dict[str, float] = defaultdict(float)
    for probability, category, ratios in zip(probabilities, categories, ratios_by_row, strict=True):
        category_mass[category] += float(probability)
        for metric, ratio in ratios.items():
            if ratio > 1.0:
                metric_failure_mass[metric] += float(probability)
    return {
        "exclusive_category_probability": dict(sorted(category_mass.items())),
        "nonexclusive_metric_failure_probability": dict(sorted(metric_failure_mass.items())),
        "uncertainty_failure": "not_part_of_the_frozen_target_free_support_definition",
        "exact_novelty_failure": "exact_novelty_is_reported_but_is_not_a_support_gate",
        "route_failure": "not_evaluated_in_this_read_only_experiment",
    }


def _cluster_bootstrap(
    *,
    groups: Sequence[str],
    support: np.ndarray,
    broad: np.ndarray,
    candidates: np.ndarray,
    replicates: int,
    seed: int,
) -> list[dict[str, list[float]]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[group].append(index)
    clusters = tuple(tuple(indices) for _, indices in sorted(grouped.items()))
    broad_den = np.asarray([broad[list(indices)].sum() for indices in clusters])
    broad_num = np.asarray(
        [(broad[list(indices)] * support[list(indices)]).sum() for indices in clusters]
    )
    candidate_den = np.asarray(
        [[candidate[list(indices)].sum() for candidate in candidates] for indices in clusters]
    )
    candidate_num = np.asarray(
        [
            [(candidate[list(indices)] * support[list(indices)]).sum() for candidate in candidates]
            for indices in clusters
        ]
    )
    rng = np.random.default_rng(seed)
    support_rates = np.empty((replicates, len(candidates)), dtype=np.float64)
    differences = np.empty_like(support_rates)
    relatives = np.empty_like(support_rates)
    for replicate in range(replicates):
        for _ in range(1000):
            counts = np.bincount(
                rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters)
            ).astype(np.float64)
            baseline = float(np.dot(counts, broad_num) / np.dot(counts, broad_den))
            if baseline > 0.0:
                break
        else:
            raise UgiMorphologyProposalStrengthSweepError(
                "bootstrap could not draw a positive-support replicate"
            )
        rates = (counts @ candidate_num) / (counts @ candidate_den)
        support_rates[replicate] = rates
        differences[replicate] = rates - baseline
        relatives[replicate] = rates / baseline - 1.0
    output = []
    for index in range(len(candidates)):
        output.append(
            {
                "support_rate_ci95": [
                    float(np.quantile(support_rates[:, index], 0.025)),
                    float(np.quantile(support_rates[:, index], 0.975)),
                ],
                "absolute_improvement_ci95": [
                    float(np.quantile(differences[:, index], 0.025)),
                    float(np.quantile(differences[:, index], 0.975)),
                ],
                "relative_improvement_ci95": [
                    float(np.quantile(relatives[:, index], 0.025)),
                    float(np.quantile(relatives[:, index], 0.975)),
                ],
            }
        )
    return output


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = (
        "candidate_id",
        "mixture_rho",
        "score_power",
        "expected_support_rate",
        "support_rate_ci95_json",
        "relative_support_improvement",
        "effective_program_count",
        "shannon_effective_program_count",
        "importance_ess_fraction",
        "maximum_program_probability",
        "expected_valid_exact_l1_rate",
        "supported_smiles_effective_count",
        "all_floors_pass",
    )
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as handle:
        handle.write(text.getvalue().encode())
    return output.getvalue()


def build_morphology_proposal_strength_sweep(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Select one stronger proposal challenger for independent confirmation."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="proposal-strength config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiMorphologyProposalStrengthSweepError("unsupported sweep schema")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiMorphologyProposalStrengthSweepError("proposal-strength scope changed")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != EXPECTED_INPUTS:
        raise UgiMorphologyProposalStrengthSweepError("proposal-strength pins changed")
    paths = {label: _pin(repo, record, label=label) for label, record in raw_inputs.items()}

    confirmation_analysis = _load_json(paths["confirmation_analysis"], label="confirmation")
    if confirmation_analysis.get("decision", {}).get("morphology_proposal_confirmed") is not True:
        raise UgiMorphologyProposalStrengthSweepError("baseline proposal was not confirmed")
    full_result = _load_json(paths["full_proposal_result"], label="full proposal result")
    full_rows = _read_jsonl(paths["full_proposal_ledger"])
    if len(full_rows) != 57190 or full_result.get("support", {}).get("programs") != 57190:
        raise UgiMorphologyProposalStrengthSweepError("full morphology support changed")
    schedule = _load_json(paths["confirmation_schedule"], label="confirmation schedule")
    scheduled = schedule.get("records")
    if not isinstance(scheduled, list) or len(scheduled) != 3072:
        raise UgiMorphologyProposalStrengthSweepError("confirmation schedule changed")
    support_rows = _read_csv(paths["confirmation_support_ledger"])
    terminal_rows = _read_jsonl(paths["confirmation_terminal_ledger"])
    if len(support_rows) != len(scheduled) or len(terminal_rows) != len(scheduled):
        raise UgiMorphologyProposalStrengthSweepError("confirmation artifacts differ")
    for expected, support_row, terminal in zip(scheduled, support_rows, terminal_rows, strict=True):
        identity = int(expected["population_index"])
        if identity != int(support_row["population_index"]) or identity != int(
            terminal["population_index"]
        ):
            raise UgiMorphologyProposalStrengthSweepError("confirmation row order changed")

    full_broad = np.asarray(
        [float(row["broad_prior_probability"]) for row in full_rows], dtype=np.float64
    )
    full_scores = np.asarray([float(row["support_score"]) for row in full_rows])
    confirmation_broad = np.asarray(
        [float(row["prior_probability"]) for row in scheduled], dtype=np.float64
    )
    confirmation_scores = np.asarray(
        [float(row["proposal_score"]) for row in scheduled], dtype=np.float64
    )
    if not np.isclose(full_broad.sum(), 1.0) or not np.isclose(confirmation_broad.sum(), 1.0):
        raise UgiMorphologyProposalStrengthSweepError("broad proposal mass changed")

    reference = config["selection"]["reference"]
    grid = [
        (float(rho), float(power))
        for rho in config["grid"]["mixture_rho"]
        for power in config["grid"]["score_power"]
    ]
    if (float(reference["mixture_rho"]), float(reference["score_power"])) not in grid:
        raise UgiMorphologyProposalStrengthSweepError("reference proposal is absent from grid")
    score_floor = float(config["grid"]["score_floor"])

    full_proposals = []
    confirmation_proposals = []
    candidates = []
    for rho, power in grid:
        full_q, _ = support_preserving_probabilities(
            full_broad,
            full_scores,
            mixture_rho=rho,
            score_floor=score_floor,
            score_power=power,
        )
        confirmation_q, _ = support_preserving_probabilities(
            confirmation_broad,
            confirmation_scores,
            mixture_rho=rho,
            score_floor=score_floor,
            score_power=power,
        )
        full_proposals.append(full_q)
        confirmation_proposals.append(confirmation_q)
        candidates.append(
            {
                "candidate_id": f"rho-{rho:.2f}_power-{power:.2f}",
                "mixture_rho": rho,
                "score_power": power,
                "full_support": _proposal_metrics(full_rows, full_broad, full_q),
                "confirmation": _confirmation_metrics(confirmation_q, support_rows, terminal_rows),
            }
        )

    reference_index = grid.index((float(reference["mixture_rho"]), float(reference["score_power"])))
    broad_confirmation = _confirmation_metrics(confirmation_broad, support_rows, terminal_rows)
    reference_confirmation = candidates[reference_index]["confirmation"]
    broad_full = _proposal_metrics(full_rows, full_broad, full_broad)
    floors = config["selection"]["floors"]
    for candidate in candidates:
        full = candidate["full_support"]
        confirmation = candidate["confirmation"]
        checks = {
            "effective_program_count": full["effective_program_count"]
            >= float(floors["minimum_effective_program_count_fraction_of_broad"])
            * broad_full["effective_program_count"],
            "shannon_effective_program_count": full["shannon_effective_program_count"]
            >= float(floors["minimum_shannon_effective_count_fraction_of_broad"])
            * broad_full["shannon_effective_program_count"],
            "importance_ess_fraction": full["importance_ess_fraction"]
            >= float(floors["minimum_importance_ess_fraction"]),
            "maximum_program_probability": full["maximum_program_probability"]
            <= float(floors["maximum_single_program_probability"]),
            "valid_exact_l1": confirmation["expected_valid_exact_l1_rate"]
            >= float(floors["minimum_valid_exact_l1_fraction_of_broad"])
            * broad_confirmation["expected_valid_exact_l1_rate"],
            "supported_smiles_diversity": confirmation["supported_smiles_effective_count"]
            >= float(floors["minimum_supported_smiles_effective_count_fraction_of_reference"])
            * reference_confirmation["supported_smiles_effective_count"],
        }
        for role in ROLES:
            checks[f"role_marginal_{role}"] = (
                full["role_marginals"][role]["effective_state_count"]
                >= float(floors["minimum_role_effective_count_fraction_of_broad"])
                * broad_full["role_marginals"][role]["effective_state_count"]
            )
            checks[f"supported_component_{role}"] = (
                confirmation["supported_component_effective_counts"][role]
                >= float(
                    floors["minimum_supported_component_effective_count_fraction_of_reference"]
                )
                * reference_confirmation["supported_component_effective_counts"][role]
            )
        candidate["floor_checks"] = checks
        candidate["all_floors_pass"] = all(checks.values())
        candidate["confirmation"]["relative_support_improvement_over_broad"] = (
            confirmation["expected_support_rate"] / broad_confirmation["expected_support_rate"]
            - 1.0
        )
        candidate["confirmation"]["relative_support_improvement_over_reference"] = (
            confirmation["expected_support_rate"] / reference_confirmation["expected_support_rate"]
            - 1.0
        )

    support = np.asarray(
        [str(row["support"]).lower() == "true" for row in support_rows], dtype=np.float64
    )
    bootstrap_policy = config["analysis"]["clustered_bootstrap"]
    bootstrap = _cluster_bootstrap(
        groups=[str(row["program_sha256"]) for row in support_rows],
        support=support,
        broad=confirmation_broad,
        candidates=np.asarray(confirmation_proposals),
        replicates=int(bootstrap_policy["replicates"]),
        seed=int(bootstrap_policy["seed"]),
    )
    for candidate, interval in zip(candidates, bootstrap, strict=True):
        candidate["confirmation"]["clustered_bootstrap"] = interval

    eligible = [candidate for candidate in candidates if candidate["all_floors_pass"]]
    if not eligible:
        raise UgiMorphologyProposalStrengthSweepError("no proposal candidate passes frozen floors")
    selected = max(
        eligible,
        key=lambda row: (
            row["confirmation"]["expected_support_rate"],
            row["full_support"]["effective_program_count"],
            -row["mixture_rho"],
            -row["score_power"],
        ),
    )

    applicability = _load_json(paths["applicability_result"], label="applicability result")
    thresholds = applicability.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(VIEWS):
        raise UgiMorphologyProposalStrengthSweepError("applicability thresholds changed")
    curated = _read_csv(paths["curated_agile"])
    references = applicability_v2._references(
        [{**row, "product_smiles": row["model_smiles"]} for row in curated]
    )
    categories, ratios = _view_failures(terminal_rows, references=references, thresholds=thresholds)
    stored_support = np.asarray(
        [str(row["support"]).lower() == "true" for row in support_rows], dtype=bool
    )
    reconstructed_support = np.asarray(
        [category == "supported_all_views" for category in categories], dtype=bool
    )
    if not np.array_equal(stored_support, reconstructed_support):
        raise UgiMorphologyProposalStrengthSweepError(
            "abstention decomposition does not reproduce frozen support labels"
        )
    selected_index = next(
        index for index, candidate in enumerate(candidates) if candidate is selected
    )
    abstention = {
        "broad_prior": _weighted_failure_summary(confirmation_broad, categories, ratios),
        "reference_proposal": _weighted_failure_summary(
            confirmation_proposals[reference_index], categories, ratios
        ),
        "selected_challenger": _weighted_failure_summary(
            confirmation_proposals[selected_index], categories, ratios
        ),
    }

    ledger_rows = []
    for candidate in candidates:
        confirmation = candidate["confirmation"]
        full = candidate["full_support"]
        ledger_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "mixture_rho": candidate["mixture_rho"],
                "score_power": candidate["score_power"],
                "expected_support_rate": confirmation["expected_support_rate"],
                "support_rate_ci95_json": _stable_json(
                    confirmation["clustered_bootstrap"]["support_rate_ci95"]
                ),
                "relative_support_improvement": confirmation[
                    "relative_support_improvement_over_broad"
                ],
                "effective_program_count": full["effective_program_count"],
                "shannon_effective_program_count": full["shannon_effective_program_count"],
                "importance_ess_fraction": full["importance_ess_fraction"],
                "maximum_program_probability": full["maximum_program_probability"],
                "expected_valid_exact_l1_rate": confirmation["expected_valid_exact_l1_rate"],
                "supported_smiles_effective_count": confirmation[
                    "supported_smiles_effective_count"
                ],
                "all_floors_pass": candidate["all_floors_pass"],
            }
        )
    ledger = _csv_bytes(ledger_rows)
    input_records = {
        label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
        for label, path in sorted(paths.items())
    }
    input_records["config"] = {
        "path": str(config_path.relative_to(repo)),
        "sha256": sha256_file(config_path),
    }
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "frozen_proposal_strength_development_sweep_complete",
        "scope": dict(EXPECTED_SCOPE),
        "inputs": input_records,
        "grid": {
            "mixture_rho": list(config["grid"]["mixture_rho"]),
            "score_power": list(config["grid"]["score_power"]),
            "score_floor": score_floor,
            "candidate_count": len(candidates),
        },
        "frozen_boundary": {
            "definition": "valid_exact_l1_and_fixed_multiview_radius_at_most_one",
            "thresholds_unchanged": True,
            "thresholds": thresholds,
        },
        "broad_reference": {
            "full_support": broad_full,
            "confirmation": broad_confirmation,
        },
        "current_reference": candidates[reference_index],
        "candidates": candidates,
        "development_selected_challenger": {
            "candidate_id": selected["candidate_id"],
            "mixture_rho": selected["mixture_rho"],
            "score_power": selected["score_power"],
            "expected_support_rate_on_reused_confirmation": selected["confirmation"][
                "expected_support_rate"
            ],
            "clustered_bootstrap_on_reused_confirmation": selected["confirmation"][
                "clustered_bootstrap"
            ],
            "full_support": selected["full_support"],
            "floor_checks": selected["floor_checks"],
        },
        "abstention_decomposition": abstention,
        "selection": {
            "floors": floors,
            "objective": config["selection"]["objective"],
            "confirmation_set_status": "reused_development_evidence_not_untouched_after_selection",
        },
        "decision": {
            "baseline_proposal_remains_production_reference": True,
            "challenger_selected_for_fresh_confirmation": True,
            "production_replacement_authorized": False,
            "applicability_boundary_changed": False,
            "mh_authorized": False,
            "partial_state_smc_authorized": False,
            "next_gate": "fresh_untouched_program_terminal_confirmation_of_selected_challenger",
        },
        "artifacts": {
            "sweep_ledger.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "records": len(ledger_rows),
                "sha256": sha256_bytes(ledger),
            }
        },
        "nonclaims": [
            "Counterfactual reweighting does not create new terminal chemistry.",
            "The reused confirmation set cannot independently confirm the selected hyperparameters.",
            "Morphology role-state diversity is not identical to atom-level component diversity.",
            "This experiment does not evaluate potency, routes, synthesis, MH or SMC.",
        ],
    }
    result = {**content, "result_sha256": _logical_sha256(content)}
    return result, ledger


__all__ = [
    "UgiMorphologyProposalStrengthSweepError",
    "_cluster_bootstrap",
    "_effective_count",
    "build_morphology_proposal_strength_sweep",
]
