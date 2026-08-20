"""Freeze a conservative potency proposal nested inside the promoted support proposal."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import sha256_file as _sha256_file
from forge.potency.morphology.ugi_morphology_potency_signal import _ridge_roles

CONFIG_SCHEMA_VERSION = "phase1_ugi_morphology_potency_proposal_sweep_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_morphology_potency_proposal_sweep.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_morphology_potency_proposal_sweep_ledger.v1"
ROLE_FIELDS = ("amine_role_state", "aldehyde_role_state", "isocyanide_role_state")


class UgiMorphologyPotencyProposalSweepError(RuntimeError):
    """Raised when the frozen potency-proposal sweep contract changes."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise UgiMorphologyPotencyProposalSweepError(f"JSON object required: {path}")
    return value


def _read_csv_gzip(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise UgiMorphologyPotencyProposalSweepError("invalid proposal ledger")
    return rows


def _stable_state(program: Mapping[str, Any], role: int) -> str:
    return json.dumps(
        [
            int(program["node_counts"][role]),
            int(program["junction_budgets"][role]),
            int(program["cycle_ranks"][role]),
            int(program["attachment_counts"][role]),
        ],
        separators=(",", ":"),
    )


def _effective_count(probabilities: np.ndarray) -> float:
    probabilities = probabilities / probabilities.sum()
    return float(1.0 / np.square(probabilities).sum())


def _shannon_effective_count(probabilities: np.ndarray) -> float:
    probabilities = probabilities / probabilities.sum()
    positive = probabilities[probabilities > 0.0]
    return float(math.exp(-np.dot(positive, np.log(positive))))


def _importance_ess_fraction(broad: np.ndarray, proposal: np.ndarray) -> float:
    ratio = broad / proposal
    weighted = proposal * ratio
    return float(weighted.sum() ** 2 / np.dot(proposal, ratio**2))


def _marginal_effective_counts(
    rows: Sequence[Mapping[str, Any]], probabilities: np.ndarray
) -> dict[str, float]:
    output: dict[str, float] = {}
    for role, name in enumerate(("amine", "aldehyde", "isocyanide")):
        mass: dict[str, float] = defaultdict(float)
        for row, probability in zip(rows, probabilities, strict=True):
            mass[_stable_state(row["program"], role)] += float(probability)
        output[name] = _effective_count(np.asarray(tuple(mass.values()), dtype=np.float64))
    return output


def proposal_metrics(
    rows: Sequence[Mapping[str, Any]],
    broad: np.ndarray,
    proposal: np.ndarray,
    expected_yield: np.ndarray,
    authorized: np.ndarray,
) -> dict[str, Any]:
    """Summarize support, diversity and expected conservative yield."""

    return {
        "expected_supported_potency_value": float(np.dot(proposal, expected_yield)),
        "authorized_morphology_mass": float(proposal[authorized].sum()),
        "inverse_simpson_effective_program_count": _effective_count(proposal),
        "shannon_effective_program_count": _shannon_effective_count(proposal),
        "importance_ess_fraction_broad_over_proposal": _importance_ess_fraction(broad, proposal),
        "maximum_program_probability": float(proposal.max()),
        "role_marginal_effective_state_counts": _marginal_effective_counts(rows, proposal),
    }


def _finite_conformal_radius(residuals: np.ndarray, coverage: float) -> float:
    if not 0.0 < coverage < 1.0 or len(residuals) < 2:
        raise UgiMorphologyPotencyProposalSweepError("invalid conformal inputs")
    probability = min(1.0, math.ceil((len(residuals) + 1) * coverage) / len(residuals))
    return float(np.quantile(np.abs(residuals), probability, method="higher"))


def _calibrated_utility(values: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    ordered = np.sort(calibration)
    ecdf = np.searchsorted(ordered, values, side="right") / len(ordered)
    return np.maximum(0.0, 2.0 * ecdf - 1.0)


def _candidate_distribution(
    support: np.ndarray,
    value: np.ndarray,
    *,
    gamma: float,
    beta: float,
) -> np.ndarray:
    tilted = support * np.exp(beta * value)
    tilted /= tilted.sum()
    output = (1.0 - gamma) * support + gamma * tilted
    return output / output.sum()


def _csv_gzip_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
        handle.write(buffer.getvalue().encode())
    return raw.getvalue()


def build_morphology_potency_proposal_sweep(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Fit the frozen low-capacity value and sweep only proposal strength."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiMorphologyPotencyProposalSweepError("unsupported config schema")
    paths: dict[str, Path] = {}
    for label, record in config["inputs"].items():
        path = repo / str(record["path"])
        if _sha256_file(path) != record["sha256"]:
            raise UgiMorphologyPotencyProposalSweepError(f"input hash changed: {label}")
        paths[label] = path
    signal = _read_json(paths["potency_signal_result"])
    promoted = _read_json(paths["promoted_proposal_result"])
    if (
        signal.get("decision", {}).get("morphology_potency_signal_passed") is not True
        or promoted.get("decision", {}).get("applicability_proposal_promoted") is not True
    ):
        raise UgiMorphologyPotencyProposalSweepError("prerequisite gate has not passed")
    selected_model = str(signal["selection"]["selected_low_capacity_model"])
    if selected_model != "role_factorized_additive":
        raise UgiMorphologyPotencyProposalSweepError("selected model changed")

    oof = [
        row
        for row in _read_csv_gzip(paths["potency_oof_ledger"])
        if row["scheme"] == "exact_program_hash" and row["model"] == selected_model
    ]
    if len(oof) != int(signal["cohort"]["records"]):
        raise UgiMorphologyPotencyProposalSweepError("OOF cohort changed")
    categorical = np.asarray(
        [
            [
                row["amine_role_state"],
                row["aldehyde_role_state"],
                row["isocyanide_role_state"],
                row["aldehyde_role_state"] + "|" + row["isocyanide_role_state"],
            ]
            for row in oof
        ],
        dtype=object,
    )
    observed = np.asarray([float(row["observed_hela_mtp"]) for row in oof], dtype=np.float64)
    oof_prediction = np.asarray([float(row["oof_prediction"]) for row in oof], dtype=np.float64)
    alpha = float(config["model"]["ridge_alpha"])
    model = _ridge_roles(alpha)
    model.fit(categorical, observed)
    q90 = _finite_conformal_radius(
        observed - oof_prediction, float(config["model"]["conformal_coverage"])
    )
    calibration_lcb = oof_prediction - q90
    seen = [set(categorical[:, index]) for index in range(3)]

    rows = _read_jsonl_gzip(paths["promoted_proposal_ledger"])
    if len(rows) != 57190:
        raise UgiMorphologyPotencyProposalSweepError("complete morphology support changed")
    broad = np.asarray([float(row["broad_prior_probability"]) for row in rows])
    support = np.asarray([float(row["proposal_probability"]) for row in rows])
    support_scores = np.asarray([float(row["support_score"]) for row in rows])
    all_categorical = np.asarray(
        [
            [
                _stable_state(row["program"], 0),
                _stable_state(row["program"], 1),
                _stable_state(row["program"], 2),
                _stable_state(row["program"], 1) + "|" + _stable_state(row["program"], 2),
            ]
            for row in rows
        ],
        dtype=object,
    )
    authorized = np.asarray(
        [all(state[index] in seen[index] for index in range(3)) for state in all_categorical],
        dtype=bool,
    )
    prediction = np.asarray(model.predict(all_categorical), dtype=np.float64)
    lcb90 = prediction - q90
    utility = _calibrated_utility(lcb90, calibration_lcb)
    utility[~authorized] = 0.0
    expected_yield = support_scores * utility
    baseline = proposal_metrics(rows, broad, support, expected_yield, authorized)

    candidates = []
    floors = config["safeguards"]
    for gamma in config["sweep"]["gamma"]:
        for beta in config["sweep"]["beta"]:
            proposal = _candidate_distribution(
                support, expected_yield, gamma=float(gamma), beta=float(beta)
            )
            metrics = proposal_metrics(rows, broad, proposal, expected_yield, authorized)
            checks = {
                "full_support": bool(np.all(proposal > 0.0)),
                "effective_program_count": metrics["inverse_simpson_effective_program_count"]
                >= baseline["inverse_simpson_effective_program_count"]
                * float(floors["minimum_effective_program_fraction"]),
                "shannon_support": metrics["shannon_effective_program_count"]
                >= baseline["shannon_effective_program_count"]
                * float(floors["minimum_shannon_effective_fraction"]),
                "importance_ess": metrics["importance_ess_fraction_broad_over_proposal"]
                >= float(floors["minimum_importance_ess_fraction"]),
                "maximum_probability": metrics["maximum_program_probability"]
                <= baseline["maximum_program_probability"]
                * float(floors["maximum_program_probability_multiplier"]),
                "role_marginals": all(
                    metrics["role_marginal_effective_state_counts"][role]
                    >= baseline["role_marginal_effective_state_counts"][role]
                    * float(floors["minimum_role_effective_state_fraction"])
                    for role in ("amine", "aldehyde", "isocyanide")
                ),
                "expected_value_improvement": metrics["expected_supported_potency_value"]
                >= baseline["expected_supported_potency_value"]
                * (1.0 + float(floors["minimum_expected_value_relative_improvement"])),
            }
            candidates.append(
                {
                    "gamma": float(gamma),
                    "beta": float(beta),
                    "metrics": metrics,
                    "checks": checks,
                    "all_safeguards_pass": all(checks.values()),
                    "proposal": proposal,
                }
            )
    passing = [row for row in candidates if row["all_safeguards_pass"]]
    selected = max(
        passing,
        key=lambda row: (
            row["metrics"]["expected_supported_potency_value"],
            row["metrics"]["inverse_simpson_effective_program_count"],
            -row["gamma"],
            -row["beta"],
        ),
        default=None,
    )
    selected_proposal = support if selected is None else selected["proposal"]
    ledger_rows = []
    for index, row in enumerate(rows):
        ledger_rows.append(
            {
                "program_sha256": row["program_sha256"],
                "broad_prior_probability": broad[index],
                "support_proposal_probability": support[index],
                "potency_proposal_probability": selected_proposal[index],
                "authorized_morphology": bool(authorized[index]),
                "predicted_hela_mtp": prediction[index] if authorized[index] else "",
                "morphology_lcb90": lcb90[index] if authorized[index] else "",
                "potency_utility": utility[index],
                "support_score": support_scores[index],
                "expected_supported_potency_value": expected_yield[index],
            }
        )
    fields = (
        "program_sha256",
        "broad_prior_probability",
        "support_proposal_probability",
        "potency_proposal_probability",
        "authorized_morphology",
        "predicted_hela_mtp",
        "morphology_lcb90",
        "potency_utility",
        "support_score",
        "expected_supported_potency_value",
    )
    ledger = _csv_gzip_bytes(ledger_rows, fields)
    serial_candidates = [
        {key: value for key, value in row.items() if key != "proposal"} for row in candidates
    ]
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "nested_morphology_potency_proposal_sweep_complete",
        "scope": dict(config["scope"]),
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": _sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "value_model": {
            "model": selected_model,
            "ridge_alpha": alpha,
            "fit_records": len(oof),
            "conformal_coverage": float(config["model"]["conformal_coverage"]),
            "oof_absolute_residual_q90": q90,
            "utility": "positive half of the calibration LCB90 ECDF",
            "expected_yield": "frozen applicability support score times conservative potency utility",
            "unauthorized_morphology_utility": 0.0,
        },
        "authorized_morphology_support": {
            "programs": int(authorized.sum()),
            "fraction_of_programs": float(authorized.mean()),
            "broad_prior_mass": float(broad[authorized].sum()),
            "support_proposal_mass": float(support[authorized].sum()),
            "exact_new_component_identity_guidance_authorized": False,
        },
        "baseline_support_proposal": baseline,
        "sweep": serial_candidates,
        "selection": (
            None
            if selected is None
            else {
                "gamma": selected["gamma"],
                "beta": selected["beta"],
                "metrics": selected["metrics"],
                "checks": selected["checks"],
                "expected_value_relative_improvement": selected["metrics"][
                    "expected_supported_potency_value"
                ]
                / baseline["expected_supported_potency_value"]
                - 1.0,
            }
        ),
        "artifacts": {
            "proposal_ledger.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "rows": len(ledger_rows),
                "sha256": hashlib.sha256(ledger).hexdigest(),
            }
        },
        "decision": {
            "nested_potency_proposal_frozen_for_matched_diagnostic": selected is not None,
            "potency_tilting_promoted": False,
            "prospective_candidate_selection_authorized": False,
            "next_gate": (
                "matched_broad_vs_support_vs_nested_potency_terminal_comparison"
                if selected is not None
                else "retain_support_proposal_plus_terminal_conservative_ranking"
            ),
        },
        "nonclaims": [
            "This distribution calculation does not show that guided generation beats post-hoc ranking.",
            "Potency authority remains terminal, role-restricted and abstaining.",
            "All 57,190 qualified programs retain nonzero probability.",
        ],
    }
    content["result_sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return content, ledger


__all__ = [
    "UgiMorphologyPotencyProposalSweepError",
    "build_morphology_potency_proposal_sweep",
    "proposal_metrics",
]
