"""Test whether coarse Ugi morphology predicts observed HeLa potency."""

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
from scipy.stats import rankdata
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from forge.core.hashing import sha256_file as _sha256_file
from forge.core.io import stable_json as _stable_json
from forge.product.ugi_restartable_terminal_support_adapter import (
    canonical_morphology_program_bytes,
)
from forge.product.ugi_training_cache import load_ugi_training_cache

CONFIG_SCHEMA_VERSION = "phase1_ugi_morphology_potency_signal_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_morphology_potency_signal.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_morphology_potency_signal_oof.v1"
PAIR_SCHEME = "held_aldehyde_isocyanide_pair_5fold"
FEATURE_FIELDS = (
    "amine_nodes",
    "aldehyde_nodes",
    "isocyanide_nodes",
    "amine_junctions",
    "aldehyde_junctions",
    "isocyanide_junctions",
    "amine_cycles",
    "aldehyde_cycles",
    "isocyanide_cycles",
    "amine_attachments",
    "aldehyde_attachments",
    "isocyanide_attachments",
)


class UgiMorphologyPotencySignalError(RuntimeError):
    """Raised when the frozen morphology-potency gate cannot be evaluated."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise UgiMorphologyPotencySignalError(f"JSON object required: {path}")
    return value


def _read_csv_gzip(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise UgiMorphologyPotencySignalError("proposal ledger must contain JSON objects")
    return rows


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


def _program_by_product(training_cache: Path) -> dict[str, str]:
    corpus, records_by_fold = load_ugi_training_cache(training_cache)
    output: dict[str, str] = {}
    for fold in sorted(corpus.assignments_by_fold):
        assignments = corpus.assignments_by_fold[fold]
        records = records_by_fold[fold]
        for assignment, record in zip(assignments, records, strict=True):
            product = str(assignment["canonical_product_smiles"])
            digest = hashlib.sha256(canonical_morphology_program_bytes(record.program)).hexdigest()
            previous = output.get(product)
            if previous is not None and previous != digest:
                raise UgiMorphologyPotencySignalError("conflicting morphology programs")
            output[product] = digest
    return output


def _role_state(program: Mapping[str, Any], role: int) -> str:
    return _stable_json(
        [
            int(program["node_counts"][role]),
            int(program["junction_budgets"][role]),
            int(program["cycle_ranks"][role]),
            int(program["attachment_counts"][role]),
        ]
    )


def _features(program: Mapping[str, Any]) -> list[float]:
    return [
        *[float(value) for value in program["node_counts"]],
        *[float(value) for value in program["junction_budgets"]],
        *[float(value) for value in program["cycle_ranks"]],
        *[float(value) for value in program["attachment_counts"]],
    ]


def _correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 2 or np.all(observed == observed[0]) or np.all(predicted == predicted[0]):
        return 0.0
    left = rankdata(observed, method="average")
    right = rankdata(predicted, method="average")
    return float(np.corrcoef(left, right)[0, 1])


def prediction_metrics(
    observed: np.ndarray, predicted: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, float]:
    """Return frozen prediction and top-quartile allocation metrics."""

    if len(observed) != len(predicted) or len(observed) < 2:
        raise UgiMorphologyPotencySignalError("invalid metric arrays")
    if weights is None:
        weights = np.ones(len(observed), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    residual = observed - predicted
    observed_mean = float(np.dot(weights, observed))
    denominator = float(np.dot(weights, (observed - observed_mean) ** 2))
    threshold = float(np.quantile(predicted, 0.75))
    top = predicted >= threshold
    top_weights = weights[top] / weights[top].sum()
    top_mean = float(np.dot(top_weights, observed[top]))
    return {
        "records": float(len(observed)),
        "mae": float(np.dot(weights, np.abs(residual))),
        "rmse": float(math.sqrt(np.dot(weights, residual**2))),
        "r2": 1.0 - float(np.dot(weights, residual**2)) / denominator if denominator > 0 else 0.0,
        "midrank_spearman": _correlation(observed, predicted),
        "observed_mean": observed_mean,
        "predicted_top_quartile_threshold": threshold,
        "observed_mean_in_predicted_top_quartile": top_mean,
        "top_quartile_observed_gain": top_mean - observed_mean,
    }


def _ridge_numeric(alpha: float) -> Any:
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
        transformer=StandardScaler(),
    )


def _ridge_roles(alpha: float) -> Any:
    return TransformedTargetRegressor(
        regressor=make_pipeline(
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            Ridge(alpha=alpha),
        ),
        transformer=StandardScaler(),
    )


def _shallow_nonlinear(seed: int) -> Any:
    return TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            HistGradientBoostingRegressor(
                max_depth=2,
                max_iter=60,
                learning_rate=0.05,
                min_samples_leaf=20,
                l2_regularization=10.0,
                random_state=seed,
            ),
        ),
        transformer=StandardScaler(),
    )


def _inner_alpha(
    kind: str,
    numeric: np.ndarray,
    categorical: np.ndarray,
    observed: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float],
) -> float:
    unique = np.unique(groups)
    if len(unique) < 3:
        return float(alphas[len(alphas) // 2])
    splitter = GroupKFold(n_splits=min(4, len(unique)))
    losses: dict[float, list[float]] = defaultdict(list)
    source = numeric if kind == "ridge_numeric" else categorical
    for train, validation in splitter.split(source, observed, groups):
        for alpha in alphas:
            model = (
                _ridge_numeric(float(alpha))
                if kind == "ridge_numeric"
                else _ridge_roles(float(alpha))
            )
            model.fit(source[train], observed[train])
            prediction = np.asarray(model.predict(source[validation]), dtype=np.float64)
            losses[float(alpha)].append(float(np.mean(np.abs(observed[validation] - prediction))))
    return min((float(np.mean(values)), alpha) for alpha, values in losses.items())[1]


def _cross_validated_predictions(
    *,
    kind: str,
    numeric: np.ndarray,
    categorical: np.ndarray,
    observed: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float],
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    unique = np.unique(groups)
    if len(unique) < 2:
        raise UgiMorphologyPotencySignalError(f"too few groups for {kind}")
    splitter = GroupKFold(n_splits=min(5, len(unique)))
    predictions = np.zeros(len(observed), dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    source = categorical if kind == "role_factorized_additive" else numeric
    for fold, (train, test) in enumerate(splitter.split(source, observed, groups)):
        if kind == "constant":
            predictions[test] = float(np.mean(observed[train]))
            alpha = None
        elif kind in {"ridge_numeric", "role_factorized_additive"}:
            inner_kind = "ridge_numeric" if kind == "ridge_numeric" else "role_factorized_additive"
            alpha = _inner_alpha(
                inner_kind,
                numeric[train],
                categorical[train],
                observed[train],
                groups[train],
                alphas,
            )
            model = _ridge_numeric(alpha) if kind == "ridge_numeric" else _ridge_roles(alpha)
            model.fit(source[train], observed[train])
            predictions[test] = np.asarray(model.predict(source[test]), dtype=np.float64)
        elif kind == "shallow_nonlinear":
            alpha = None
            model = _shallow_nonlinear(seed + fold)
            model.fit(source[train], observed[train])
            predictions[test] = np.asarray(model.predict(source[test]), dtype=np.float64)
        else:
            raise UgiMorphologyPotencySignalError(f"unknown model: {kind}")
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "test_rows": len(test),
                "test_groups": len(np.unique(groups[test])),
                "selected_alpha": alpha,
            }
        )
    return predictions, fold_rows


def _cluster_bootstrap(
    observed: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    spearman: list[float] = []
    top_gain: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        metrics = prediction_metrics(observed[selected], predicted[selected])
        spearman.append(metrics["midrank_spearman"])
        top_gain.append(metrics["top_quartile_observed_gain"])
    return {
        "midrank_spearman_ci95": [float(value) for value in np.quantile(spearman, [0.025, 0.975])],
        "top_quartile_observed_gain_ci95": [
            float(value) for value in np.quantile(top_gain, [0.025, 0.975])
        ],
    }


def _head_stability(
    labels: Sequence[str], observed: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    by_head: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_head[label.split("B", 1)[0]].append(index)
    rows = []
    for head, indices in sorted(by_head.items()):
        if len(indices) < 10:
            continue
        selected = np.asarray(indices, dtype=np.int64)
        metrics = prediction_metrics(observed[selected], predicted[selected])
        rows.append({"head": head, **metrics})
    correlations = [float(row["midrank_spearman"]) for row in rows]
    gains = [float(row["top_quartile_observed_gain"]) for row in rows]
    return {
        "heads_evaluated": len(rows),
        "positive_spearman_head_fraction": float(np.mean(np.asarray(correlations) > 0.0)),
        "median_head_spearman": float(np.median(correlations)),
        "positive_top_quartile_gain_head_fraction": float(np.mean(np.asarray(gains) > 0.0)),
        "per_head": rows,
    }


def _cohort(
    *,
    curated: Sequence[Mapping[str, str]],
    applicability: Sequence[Mapping[str, str]],
    program_by_product: Mapping[str, str],
    proposal_by_program: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source_by_label = {str(row["label"]): row for row in curated}
    rows = [
        row
        for row in applicability
        if row.get("scheme") == PAIR_SCHEME and row.get("distribution_bin") == "interpolative"
    ]
    output = []
    for row in rows:
        label = str(row["label"])
        source = source_by_label[label]
        digest = program_by_product.get(str(source["model_smiles"]))
        proposal = proposal_by_program.get(str(digest)) if digest is not None else None
        if proposal is None:
            continue
        program = proposal["program"]
        role_states = [_role_state(program, role) for role in range(3)]
        output.append(
            {
                "label": label,
                "head_id": label.split("B", 1)[0],
                "program_sha256": digest,
                "amine_role_state": role_states[0],
                "aldehyde_role_state": role_states[1],
                "isocyanide_role_state": role_states[2],
                "tail_role_state_pair": role_states[1] + "|" + role_states[2],
                "role_factorized_features": [
                    role_states[0],
                    role_states[1],
                    role_states[2],
                    role_states[1] + "|" + role_states[2],
                ],
                "numeric_features": _features(program),
                "observed_hela_mtp": float(source["expt_Hela"]),
                "promoted_over_broad_weight": float(proposal["proposal_probability"])
                / float(proposal["broad_prior_probability"]),
            }
        )
    output.sort(key=lambda row: str(row["label"]))
    return output


def build_morphology_potency_signal(repo: Path, config_path: Path) -> tuple[dict[str, Any], bytes]:
    """Execute the preregistered, read-only observed-potency signal gate."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiMorphologyPotencySignalError("unsupported config schema")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping):
        raise UgiMorphologyPotencySignalError("inputs are missing")
    paths: dict[str, Path] = {}
    for label, record in raw_inputs.items():
        if not isinstance(record, Mapping):
            raise UgiMorphologyPotencySignalError(f"invalid input pin: {label}")
        path = repo / str(record["path"])
        if _sha256_file(path) != record["sha256"]:
            raise UgiMorphologyPotencySignalError(f"input hash changed: {label}")
        paths[str(label)] = path
    novelty = _read_json(paths["novelty_lane_audit"])
    promoted = _read_json(paths["promoted_proposal_result"])
    if (
        novelty.get("decision", {}).get("morphology_potency_signal_gate_may_proceed") is not True
        or promoted.get("decision", {}).get("applicability_proposal_promoted") is not True
    ):
        raise UgiMorphologyPotencySignalError("prerequisite decision is not frozen")

    program_by_product = _program_by_product(paths["training_cache"])
    proposal_rows = _read_jsonl_gzip(paths["promoted_proposal_ledger"])
    proposal_by_program = {str(row["program_sha256"]): row for row in proposal_rows}
    if len(proposal_by_program) != 57190:
        raise UgiMorphologyPotencySignalError("promoted support changed")
    cohort = _cohort(
        curated=_read_csv_gzip(paths["curated_agile"]),
        applicability=_read_csv_gzip(paths["heldout_applicability"]),
        program_by_product=program_by_product,
        proposal_by_program=proposal_by_program,
    )
    minimum = int(config["analysis"]["minimum_cohort_records"])
    if len(cohort) < minimum:
        raise UgiMorphologyPotencySignalError("authorized cohort is too small")

    numeric = np.asarray([row["numeric_features"] for row in cohort], dtype=np.float64)
    categorical = np.asarray([row["role_factorized_features"] for row in cohort], dtype=object)
    observed = np.asarray([row["observed_hela_mtp"] for row in cohort], dtype=np.float64)
    weights = np.asarray([row["promoted_over_broad_weight"] for row in cohort], dtype=np.float64)
    labels = [str(row["label"]) for row in cohort]
    group_arrays = {
        "exact_program_hash": np.asarray([row["program_sha256"] for row in cohort]),
        "aldehyde_role_state": np.asarray([row["aldehyde_role_state"] for row in cohort]),
        "isocyanide_role_state": np.asarray([row["isocyanide_role_state"] for row in cohort]),
        "tail_role_state_pair": np.asarray([row["tail_role_state_pair"] for row in cohort]),
    }
    models = tuple(config["analysis"]["models"])
    alphas = tuple(float(value) for value in config["analysis"]["ridge_alphas"])
    seed = int(config["analysis"]["seed"])
    summaries: dict[str, Any] = {}
    prediction_store: dict[tuple[str, str], np.ndarray] = {}
    fold_store: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for scheme, groups in group_arrays.items():
        scheme_rows: dict[str, Any] = {}
        for model in models:
            prediction, folds = _cross_validated_predictions(
                kind=model,
                numeric=numeric,
                categorical=categorical,
                observed=observed,
                groups=groups,
                alphas=alphas,
                seed=seed,
            )
            prediction_store[(scheme, model)] = prediction
            fold_store[(scheme, model)] = folds
            scheme_rows[model] = {
                "unweighted": prediction_metrics(observed, prediction),
                "promoted_proposal_reweighted": prediction_metrics(observed, prediction, weights),
                "folds": folds,
            }
        summaries[scheme] = {
            "groups": len(np.unique(groups)),
            "models": scheme_rows,
        }

    primary_scheme = str(config["analysis"]["primary_scheme"])
    selectable = tuple(config["analysis"]["selectable_low_capacity_models"])
    selected_model = min(
        selectable,
        key=lambda model: (
            float(summaries[primary_scheme]["models"][model]["unweighted"]["mae"]),
            selectable.index(model),
        ),
    )
    primary_prediction = prediction_store[(primary_scheme, selected_model)]
    bootstrap_policy = config["analysis"]["clustered_bootstrap"]
    bootstrap = _cluster_bootstrap(
        observed,
        primary_prediction,
        group_arrays[primary_scheme],
        replicates=int(bootstrap_policy["replicates"]),
        seed=int(bootstrap_policy["seed"]),
    )
    head_stability = _head_stability(labels, observed, primary_prediction)
    primary = summaries[primary_scheme]["models"][selected_model]["unweighted"]
    baseline = summaries[primary_scheme]["models"]["constant"]["unweighted"]
    role_blocked = {
        scheme: summaries[scheme]["models"][selected_model]["unweighted"]
        for scheme in ("aldehyde_role_state", "isocyanide_role_state", "tail_role_state_pair")
    }
    gates = config["analysis"]["gates"]
    checks = {
        "primary_rank_ci": bootstrap["midrank_spearman_ci95"][0]
        > float(gates["minimum_primary_spearman_ci_lower"]),
        "primary_mae_improvement": primary["mae"]
        <= baseline["mae"] * (1.0 - float(gates["minimum_primary_mae_relative_improvement"])),
        "top_quartile_gain_ci": bootstrap["top_quartile_observed_gain_ci95"][0]
        > float(gates["minimum_top_quartile_gain_ci_lower"]),
        "role_state_blocking": all(
            float(metrics["midrank_spearman"]) >= float(gates["minimum_role_blocked_spearman"])
            for metrics in role_blocked.values()
        ),
        "head_stability": head_stability["positive_spearman_head_fraction"]
        >= float(gates["minimum_positive_spearman_head_fraction"])
        and head_stability["median_head_spearman"] >= float(gates["minimum_median_head_spearman"]),
        "low_capacity_signal": selected_model in selectable,
    }
    passed = all(checks.values())

    ledger_rows = []
    for index, row in enumerate(cohort):
        for scheme in group_arrays:
            for model in models:
                ledger_rows.append(
                    {
                        "label": row["label"],
                        "program_sha256": row["program_sha256"],
                        "scheme": scheme,
                        "model": model,
                        "observed_hela_mtp": row["observed_hela_mtp"],
                        "oof_prediction": float(prediction_store[(scheme, model)][index]),
                        "promoted_over_broad_weight": row["promoted_over_broad_weight"],
                        "amine_role_state": row["amine_role_state"],
                        "aldehyde_role_state": row["aldehyde_role_state"],
                        "isocyanide_role_state": row["isocyanide_role_state"],
                    }
                )
    ledger_fields = (
        "label",
        "program_sha256",
        "scheme",
        "model",
        "observed_hela_mtp",
        "oof_prediction",
        "promoted_over_broad_weight",
        "amine_role_state",
        "aldehyde_role_state",
        "isocyanide_role_state",
    )
    ledger = _csv_gzip_bytes(ledger_rows, ledger_fields)
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "morphology_to_observed_hela_signal_gate_complete",
        "scope": dict(config["scope"]),
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": _sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "cohort": {
            "records": len(cohort),
            "unique_exact_programs": len(np.unique(group_arrays["exact_program_hash"])),
            "unique_amine_states": len({row["amine_role_state"] for row in cohort}),
            "unique_aldehyde_states": len({row["aldehyde_role_state"] for row in cohort}),
            "unique_isocyanide_states": len({row["isocyanide_role_state"] for row in cohort}),
            "unique_tail_state_pairs": len({row["tail_role_state_pair"] for row in cohort}),
            "target": "observed expt_Hela MTP",
            "oracle_predictions_used_as_target": False,
            "identity_scope": novelty["authorized_potency_signal_cohort"],
        },
        "models": {
            "constant": "training-fold mean",
            "ridge_numeric": "standardized ridge over the 12 morphology coordinates",
            "role_factorized_additive": "ridge over one-hot role states plus the aldehyde-isocyanide state pair",
            "shallow_nonlinear": "depth-2 regularized histogram gradient boosting; diagnostic only",
        },
        "cross_validation": summaries,
        "selection": {
            "primary_scheme": primary_scheme,
            "selected_low_capacity_model": selected_model,
            "shallow_nonlinear_selection_eligible": False,
            "primary_metrics": primary,
            "constant_metrics": baseline,
            "clustered_bootstrap": bootstrap,
            "role_state_blocked_metrics": role_blocked,
            "head_stability": head_stability,
        },
        "gates": {"criteria": dict(gates), "checks": checks, "all_pass": passed},
        "artifacts": {
            "oof_predictions.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "rows": len(ledger_rows),
                "sha256": _sha256_bytes(ledger),
            }
        },
        "decision": {
            "morphology_potency_signal_passed": passed,
            "potency_tilting_authorized": False,
            "mh_authorized": False,
            "partial_state_smc_authorized": False,
            "next_gate": (
                "freeze_nested_potency_proposal_with_anti_collapse_safeguards"
                if passed
                else "retain_promoted_applicability_proposal_plus_terminal_conservative_ranking"
            ),
        },
        "nonclaims": [
            "Cross-validation does not establish prospective potency of generated molecules.",
            "This gate does not support potency claims for exact-new aldehyde or isocyanide identities.",
            "Production-proposal reweighting is a covariate-shift diagnostic, not new biological evidence.",
            "No potency proposal or molecular trajectory was changed by this read-only gate.",
        ],
    }
    content["result_sha256"] = hashlib.sha256(_stable_json(content).encode()).hexdigest()
    return content, ledger


__all__ = [
    "UgiMorphologyPotencySignalError",
    "build_morphology_potency_signal",
    "prediction_metrics",
]
