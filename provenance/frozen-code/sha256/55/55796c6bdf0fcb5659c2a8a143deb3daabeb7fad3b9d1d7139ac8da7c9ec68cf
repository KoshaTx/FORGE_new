"""Freeze the independently adjudicated Ugi morphology proposal."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.data.r1_prime_audit import sha256_bytes, sha256_file
from forge.product.ugi_complete_morphology_proposal import support_preserving_probabilities

CONFIG_SCHEMA_VERSION = "phase1_ugi_promoted_morphology_proposal_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_promoted_morphology_proposal.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_promoted_morphology_proposal_ledger.v1"
EXPECTED_SCOPE = {
    "read_only": True,
    "complete_qualified_morphology_support": True,
    "confirmed_challenger_only": True,
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
    "adjudication_result",
    "baseline_full_ledger",
    "baseline_full_result",
    "runner",
    "source",
    "strength_sweep",
    "tests",
}


class UgiPromotedMorphologyProposalError(RuntimeError):
    """Raised when the promoted morphology proposal contract changes."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiPromotedMorphologyProposalError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiPromotedMorphologyProposalError(f"{label} must contain one object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with gzip.open(path, "rt") as handle:
            rows = [json.loads(line) for line in handle]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiPromotedMorphologyProposalError("invalid baseline proposal ledger") from error
    if len(rows) != 57190:
        raise UgiPromotedMorphologyProposalError("baseline proposal support changed")
    return rows


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiPromotedMorphologyProposalError(f"malformed pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiPromotedMorphologyProposalError(f"pin escapes repository: {label}") from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiPromotedMorphologyProposalError(f"pin changed: {label}")
    return path


def _ledger_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as handle:
        for row in rows:
            handle.write((_stable_json(row) + "\n").encode())
    return output.getvalue()


def _shannon_effective_count(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    return float(np.exp(-np.sum(positive * np.log(positive))))


def build_promoted_morphology_proposal(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Apply the confirmed challenger to all 57,190 qualified programs."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="promoted proposal config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiPromotedMorphologyProposalError("unsupported promoted proposal schema")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiPromotedMorphologyProposalError("promoted proposal scope changed")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != EXPECTED_INPUTS:
        raise UgiPromotedMorphologyProposalError("promoted proposal pins changed")
    paths = {label: _pin(repo, record, label=label) for label, record in raw_inputs.items()}
    adjudication = _load_json(paths["adjudication_result"], label="adjudication result")
    if adjudication.get("decision", {}).get("challenger_promoted") is not True:
        raise UgiPromotedMorphologyProposalError("challenger was not promoted")
    sweep = _load_json(paths["strength_sweep"], label="strength sweep")
    selected = sweep["development_selected_challenger"]
    expected = config["proposal"]
    if (
        adjudication.get("candidate_id") != expected["candidate_id"]
        or selected.get("candidate_id") != expected["candidate_id"]
        or float(selected["mixture_rho"]) != float(expected["mixture_rho"])
        or float(selected["score_power"]) != float(expected["score_power"])
    ):
        raise UgiPromotedMorphologyProposalError("promoted challenger identity changed")
    baseline_result = _load_json(paths["baseline_full_result"], label="baseline result")
    rows = _read_jsonl(paths["baseline_full_ledger"])
    if baseline_result.get("support", {}).get("programs") != len(rows):
        raise UgiPromotedMorphologyProposalError("baseline result and ledger differ")
    broad = np.asarray([float(row["broad_prior_probability"]) for row in rows])
    scores = np.asarray([float(row["support_score"]) for row in rows])
    probabilities, importance = support_preserving_probabilities(
        broad,
        scores,
        mixture_rho=float(expected["mixture_rho"]),
        score_floor=float(expected["score_floor"]),
        score_power=float(expected["score_power"]),
    )
    for row, probability, ratio in zip(rows, probabilities, importance, strict=True):
        row["proposal_probability"] = float(probability)
        row["importance_ratio_broad_over_proposal"] = float(ratio)
    ledger = _ledger_bytes(rows)
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
        "status": "confirmed_applicability_morphology_proposal_promoted",
        "scope": dict(EXPECTED_SCOPE),
        "inputs": inputs,
        "support": {
            "programs": len(rows),
            "all_broad_probabilities_positive": bool(np.all(broad > 0.0)),
            "all_proposal_probabilities_positive": bool(np.all(probabilities > 0.0)),
            "support_identical_to_baseline": True,
        },
        "proposal": {
            "candidate_id": str(expected["candidate_id"]),
            "form": "broad_prior_relative_support_preserving_score_mixture",
            "mixture_rho": float(expected["mixture_rho"]),
            "score_floor": float(expected["score_floor"]),
            "score_power": float(expected["score_power"]),
            "effective_program_count": float(1.0 / np.sum(probabilities**2)),
            "shannon_effective_program_count": _shannon_effective_count(probabilities),
            "importance_ess_fraction": float(1.0 / np.sum((broad**2) / probabilities)),
            "maximum_program_probability": float(probabilities.max()),
            "maximum_importance_ratio": float(importance.max()),
            "minimum_importance_ratio": float(importance.min()),
        },
        "fresh_confirmation": {
            "broad_support_rate": adjudication["support_rates"]["broad"],
            "previous_proposal_support_rate": adjudication["support_rates"]["current"],
            "promoted_proposal_support_rate": adjudication["support_rates"]["challenger"],
            "absolute_improvement_over_previous": adjudication["challenger_over_current"][
                "absolute_improvement"
            ],
            "relative_improvement_over_previous": adjudication["challenger_over_current"][
                "relative_improvement"
            ],
            "paired_clustered_bootstrap": adjudication["challenger_over_current"][
                "clustered_bootstrap"
            ],
        },
        "decision": {
            "applicability_proposal_promoted": True,
            "applicability_boundary_changed": False,
            "potency_guidance_authorized": False,
            "mh_authorized": False,
            "partial_state_smc_authorized": False,
            "next_gate": "role_restricted_morphology_potency_signal",
        },
        "artifacts": {
            "proposal_ledger.jsonl.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "records": len(rows),
                "sha256": sha256_bytes(ledger),
                "logical_sha256": _logical_sha256(rows),
            }
        },
        "nonclaims": [
            "The proposal enriches but does not guarantee terminal applicability.",
            "The applicability boundary and molecular generator are unchanged.",
            "Promotion does not authorize potency, route, synthesis, SMC or MH guidance.",
        ],
    }
    result = {**content, "result_sha256": _logical_sha256(content)}
    return result, ledger


__all__ = [
    "UgiPromotedMorphologyProposalError",
    "build_promoted_morphology_proposal",
]
