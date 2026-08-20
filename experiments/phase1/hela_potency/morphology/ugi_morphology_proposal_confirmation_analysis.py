"""Read-only support analysis for the unused-program morphology confirmation."""

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
from sklearn.metrics import average_precision_score, roc_auc_score

from forge.corpus.r1_prime_audit import sha256_bytes, sha256_file
from forge.potency.applicability import ugi_distributional_applicability_v2 as applicability_v2
from forge.potency.controller import _terminal_support_rows

CONFIG_SCHEMA_VERSION = "phase1_ugi_morphology_proposal_confirmation_analysis_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_morphology_proposal_confirmation_analysis.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_morphology_proposal_confirmation_support.v1"
EXPECTED_SCOPE = {
    "read_only": True,
    "fresh_unused_program_terminals_only": True,
    "target_free_terminal_support_only": True,
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
    "confirmation_result",
    "confirmation_terminal_ledger",
    "curated_agile",
    "proposal_result",
    "proposal_schedule",
    "runner",
    "source",
    "tests",
}


class UgiMorphologyProposalConfirmationAnalysisError(RuntimeError):
    """Raised when the confirmation-analysis contract changes."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiMorphologyProposalConfirmationAnalysisError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiMorphologyProposalConfirmationAnalysisError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiMorphologyProposalConfirmationAnalysisError(f"malformed pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiMorphologyProposalConfirmationAnalysisError(
            f"pin escapes repository: {label}"
        ) from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiMorphologyProposalConfirmationAnalysisError(f"pin changed: {label}")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise UgiMorphologyProposalConfirmationAnalysisError("curated reference is empty")
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with gzip.open(path, "rt") as handle:
            rows = [json.loads(line) for line in handle]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiMorphologyProposalConfirmationAnalysisError(
            "invalid confirmation terminal ledger"
        ) from error
    if len(rows) != 3072:
        raise UgiMorphologyProposalConfirmationAnalysisError(
            "confirmation terminal ledger is incomplete"
        )
    return rows


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as handle:
        handle.write(text.getvalue().encode())
    return output.getvalue()


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attempts = len(rows)
    if attempts == 0:
        raise UgiMorphologyProposalConfirmationAnalysisError(
            "cannot summarize an empty confirmation stratum"
        )
    supported = sum(bool(row["support"]) for row in rows)
    unique_supported = len({str(row["smiles"]) for row in rows if row["support"] and row["smiles"]})
    return {
        "attempts": attempts,
        "valid_exact_l1": sum(bool(row["valid_exact_l1"]) for row in rows),
        "supported": supported,
        "support_rate": supported / attempts,
        "unique_supported_smiles": unique_supported,
        "unique_supported_smiles_per_attempt": unique_supported / attempts,
    }


def _proposal_weighted_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise UgiMorphologyProposalConfirmationAnalysisError(
            "cannot weight an empty confirmation population"
        )
    support = np.asarray([bool(row["support"]) for row in rows], dtype=np.float64)
    valid = np.asarray([bool(row["valid_exact_l1"]) for row in rows], dtype=np.float64)
    proposal = np.asarray([float(row["proposal_probability"]) for row in rows], dtype=np.float64)
    prior = np.asarray([float(row["prior_probability"]) for row in rows], dtype=np.float64)
    importance = np.asarray(
        [float(row["importance_ratio_prior_over_proposal"]) for row in rows],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(proposal))
        or not np.all(np.isfinite(prior))
        or not np.all(np.isfinite(importance))
        or np.any(proposal <= 0.0)
        or np.any(prior <= 0.0)
        or not np.isclose(proposal.sum(), 1.0)
        or not np.isclose(prior.sum(), 1.0)
        or not np.allclose(importance, prior / proposal)
    ):
        raise UgiMorphologyProposalConfirmationAnalysisError(
            "proposal probabilities or importance ratios changed"
        )
    prior_support = float(np.dot(prior, support))
    proposal_support = float(np.dot(proposal, support))
    if prior_support <= 0.0:
        raise UgiMorphologyProposalConfirmationAnalysisError(
            "fresh confirmation contains no supported terminal"
        )
    return {
        "prior_expected_support_rate": prior_support,
        "proposal_expected_support_rate": proposal_support,
        "proposal_support_absolute_improvement": proposal_support - prior_support,
        "proposal_support_relative_improvement": proposal_support / prior_support - 1.0,
        "prior_expected_valid_exact_l1_rate": float(np.dot(prior, valid)),
        "proposal_expected_valid_exact_l1_rate": float(np.dot(proposal, valid)),
    }


def _cluster_bootstrap_differences(
    rows: Sequence[Mapping[str, Any]], *, replicates: int, seed: int
) -> dict[str, Any]:
    if replicates < 1:
        raise UgiMorphologyProposalConfirmationAnalysisError(
            "bootstrap requires at least one replicate"
        )
    support = np.asarray([bool(row["support"]) for row in rows], dtype=np.float64)
    top = np.asarray([int(row["score_quartile"]) == 1 for row in rows], dtype=bool)
    proposal = np.asarray([float(row["proposal_probability"]) for row in rows], dtype=np.float64)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["program_sha256"])].append(index)
    clusters = tuple(tuple(indices) for _, indices in sorted(groups.items()))
    if len(clusters) < 2 or not np.any(top) or np.all(top):
        raise UgiMorphologyProposalConfirmationAnalysisError(
            "confirmation bootstrap strata are degenerate"
        )
    rng = np.random.default_rng(seed)
    top_differences = np.empty(replicates, dtype=np.float64)
    top_relative = np.empty(replicates, dtype=np.float64)
    proposal_differences = np.empty(replicates, dtype=np.float64)
    proposal_relative = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        for _ in range(1000):
            sampled_clusters = rng.integers(0, len(clusters), len(clusters))
            indices = np.fromiter(
                (
                    index
                    for cluster_index in sampled_clusters
                    for index in clusters[int(cluster_index)]
                ),
                dtype=np.int64,
            )
            sampled_support = support[indices]
            sampled_top = top[indices]
            if np.any(sampled_top) and float(np.mean(sampled_support)) > 0.0:
                break
        else:
            raise UgiMorphologyProposalConfirmationAnalysisError(
                "bootstrap could not draw a nondegenerate clustered replicate"
            )
        overall_rate = float(np.mean(sampled_support))
        top_rate = float(np.mean(sampled_support[sampled_top]))
        sampled_proposal = proposal[indices]
        proposal_rate = float(np.dot(sampled_proposal, sampled_support) / sampled_proposal.sum())
        top_differences[replicate] = top_rate - overall_rate
        top_relative[replicate] = top_rate / overall_rate - 1.0
        proposal_differences[replicate] = proposal_rate - overall_rate
        proposal_relative[replicate] = proposal_rate / overall_rate - 1.0
    return {
        "replicates": replicates,
        "unit": "program_sha256",
        "clusters": len(clusters),
        "top_quartile": {
            "absolute_difference_ci95": [
                float(np.quantile(top_differences, 0.025)),
                float(np.quantile(top_differences, 0.975)),
            ],
            "relative_improvement_ci95": [
                float(np.quantile(top_relative, 0.025)),
                float(np.quantile(top_relative, 0.975)),
            ],
        },
        "support_preserving_proposal": {
            "absolute_difference_ci95": [
                float(np.quantile(proposal_differences, 0.025)),
                float(np.quantile(proposal_differences, 0.975)),
            ],
            "relative_improvement_ci95": [
                float(np.quantile(proposal_relative, 0.025)),
                float(np.quantile(proposal_relative, 0.975)),
            ],
        },
    }


def build_morphology_proposal_confirmation_analysis(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Score fresh terminals and adjudicate the morphology proposal."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="confirmation-analysis config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiMorphologyProposalConfirmationAnalysisError(
            "unsupported confirmation-analysis schema"
        )
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiMorphologyProposalConfirmationAnalysisError("confirmation-analysis scope changed")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != EXPECTED_INPUTS:
        raise UgiMorphologyProposalConfirmationAnalysisError(
            "confirmation-analysis input pins changed"
        )
    paths = {label: _pin(repo, record, label=label) for label, record in raw_inputs.items()}
    applicability = _load_json(paths["applicability_result"], label="applicability result")
    confirmation = _load_json(paths["confirmation_result"], label="confirmation result")
    proposal = _load_json(paths["proposal_result"], label="proposal result")
    schedule = _load_json(paths["proposal_schedule"], label="proposal schedule")
    if confirmation.get("schedule_sha256") != schedule.get("schedule_sha256") or proposal.get(
        "proposal", {}
    ).get("schedule_sha256") != schedule.get("schedule_sha256"):
        raise UgiMorphologyProposalConfirmationAnalysisError(
            "confirmation and proposal schedules differ"
        )
    terminal_artifact = confirmation.get("artifacts", {}).get("terminal_ledger", {})
    if (
        terminal_artifact.get("sha256") != sha256_file(paths["confirmation_terminal_ledger"])
        or terminal_artifact.get("rows") != 3072
        or confirmation.get("counts", {}).get("programs") != 3072
        or confirmation.get("counts", {}).get("terminal_attempts") != 3072
    ):
        raise UgiMorphologyProposalConfirmationAnalysisError(
            "confirmation result and terminal ledger differ"
        )
    thresholds = applicability.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != {
        "product",
        "amine",
        "aldehyde",
        "isocyanide",
    }:
        raise UgiMorphologyProposalConfirmationAnalysisError("applicability thresholds changed")
    curated = _read_csv(paths["curated_agile"])
    references = applicability_v2._references(
        [{**row, "product_smiles": row["model_smiles"]} for row in curated]
    )
    raw = _read_jsonl(paths["confirmation_terminal_ledger"])
    adapted = [
        {
            **row,
            "selection_rank": int(row["score_rank"]),
            "state_replicate": 0,
            "checkpoint_index": 6,
            "rollout_index": 0,
        }
        for row in raw
    ]
    scored = _terminal_support_rows(adapted, references=references, thresholds=thresholds)
    raw_by_population = {int(row["population_index"]): row for row in raw}
    for row in scored:
        source = raw_by_population[int(row["population_index"])]
        row["proposal_score"] = float(source["proposal_score"])
        row["proposal_probability"] = float(source["proposal_probability"])
        row["prior_probability"] = float(source["prior_probability"])
        row["importance_ratio_prior_over_proposal"] = float(
            source["importance_ratio_prior_over_proposal"]
        )
        row["score_rank"] = int(source["score_rank"])
        row["score_quartile"] = int(source["score_quartile"])
    scored.sort(key=lambda row: int(row["population_index"]))

    overall = _summarize(scored)
    if overall["supported"] == 0 or overall["unique_supported_smiles"] == 0:
        raise UgiMorphologyProposalConfirmationAnalysisError(
            "fresh confirmation contains no supported terminal diversity"
        )
    quartiles = {
        str(quartile): _summarize([row for row in scored if int(row["score_quartile"]) == quartile])
        for quartile in (1, 2, 3, 4)
    }
    top = quartiles["1"]
    top_support_improvement = top["support_rate"] / overall["support_rate"] - 1.0
    top_diverse_improvement = (
        top["unique_supported_smiles_per_attempt"] / overall["unique_supported_smiles_per_attempt"]
        - 1.0
    )
    weighted = _proposal_weighted_summary(scored)
    labels = np.asarray([bool(row["support"]) for row in scored], dtype=np.int8)
    scores = np.asarray([float(row["proposal_score"]) for row in scored], dtype=np.float64)
    if len(np.unique(labels)) != 2:
        raise UgiMorphologyProposalConfirmationAnalysisError(
            "fresh confirmation support labels are degenerate"
        )
    bootstrap_policy = config["analysis"]["clustered_bootstrap"]
    if bootstrap_policy.get("unit") != "program_sha256":
        raise UgiMorphologyProposalConfirmationAnalysisError(
            "confirmation bootstrap must be clustered by program_sha256"
        )
    bootstrap = _cluster_bootstrap_differences(
        scored,
        replicates=int(bootstrap_policy["replicates"]),
        seed=int(bootstrap_policy["seed"]),
    )
    gates = config["analysis"]["gates"]
    checks = {
        "valid_exact_l1_rate": overall["valid_exact_l1"] / overall["attempts"]
        >= float(gates["minimum_valid_exact_l1_rate"]),
        "auroc": float(roc_auc_score(labels, scores)) >= float(gates["minimum_auroc"]),
        "top_quartile_support_improvement": top_support_improvement
        >= float(gates["minimum_top_quartile_relative_support_improvement"]),
        "top_quartile_diverse_yield_improvement": top_diverse_improvement
        >= float(gates["minimum_top_quartile_unique_supported_yield_improvement"]),
        "proposal_weighted_support_improvement": weighted["proposal_support_relative_improvement"]
        >= float(gates["minimum_proposal_weighted_support_relative_improvement"]),
        "top_quartile_clustered_bootstrap": bootstrap["top_quartile"]["absolute_difference_ci95"][0]
        > 0.0,
        "proposal_clustered_bootstrap": bootstrap["support_preserving_proposal"][
            "absolute_difference_ci95"
        ][0]
        > 0.0,
    }
    confirmed = all(checks.values())
    fields = (
        "terminal_index",
        "shard_index",
        "population_index",
        "particle_index",
        "program_sha256",
        "proposal_score",
        "proposal_probability",
        "prior_probability",
        "importance_ratio_prior_over_proposal",
        "score_rank",
        "score_quartile",
        "valid_exact_l1",
        "radius",
        "support",
        "exact_measured_product",
        "exact_new_roles_json",
        "smiles",
    )
    ledger = _csv_bytes(scored, fields)
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
        "status": "complete_unused_program_morphology_proposal_confirmation_analysis",
        "scope": dict(EXPECTED_SCOPE),
        "inputs": inputs,
        "support_definition": "valid_exact_l1_and_fixed_multiview_radius_at_most_one",
        "overall": overall,
        "score_quartiles": quartiles,
        "support_preserving_proposal": weighted,
        "ranking": {
            "auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "support_prevalence": float(np.mean(labels)),
            "top_quartile_support_relative_improvement": top_support_improvement,
            "top_quartile_diverse_yield_relative_improvement": top_diverse_improvement,
            "bootstrap": bootstrap,
            "quartile_support_is_monotone": all(
                quartiles[str(left)]["support_rate"] >= quartiles[str(left + 1)]["support_rate"]
                for left in (1, 2, 3)
            ),
        },
        "decision": {
            "checks": checks,
            "morphology_proposal_confirmed": confirmed,
            "partial_state_smc_authorized": False,
            "potency_guidance_authorized": False,
            "next_gate": (
                "matched applicability-bounded potency tilt versus terminal post-hoc ranking"
                if confirmed
                else "retain plain terminal screening and do not promote morphology proposal"
            ),
        },
        "artifacts": {
            "support_ledger.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "records": len(scored),
                "sha256": sha256_bytes(ledger),
            }
        },
        "nonclaims": [
            "This analysis does not evaluate potency.",
            "This analysis does not evaluate routes or synthesis.",
            "Confirmation of morphology allocation does not authorize partial-state SMC.",
        ],
    }
    result = {**content, "result_sha256": _logical_sha256(content)}
    return result, ledger


__all__ = [
    "UgiMorphologyProposalConfirmationAnalysisError",
    "build_morphology_proposal_confirmation_analysis",
]
