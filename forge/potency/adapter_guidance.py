"""Deterministic statistics and objectives for the bounded Ugi potency adapter."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]


class PotencyAdapterGuidanceError(ValueError):
    """The potency-adapter analysis violates its leak-aware diagnostic contract."""


@dataclass(frozen=True)
class PotencyObservation:
    """One LNPDB observation joined to one separately constructed Ugi graph record."""

    label: str
    cache_index: int
    model_smiles: str
    potency: float
    cluster_id: str


def normalized_agile_label(value: str) -> str:
    """Normalize LNPDB's A1_B1_C1 spelling to the frozen split label A1B1C1."""

    label = value.replace("_", "").strip()
    if not label or any(
        character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" for character in label
    ):
        raise PotencyAdapterGuidanceError(f"invalid Ugi observation label: {value!r}")
    return label


def training_ecdf_quantiles(values: Sequence[float]) -> np.ndarray:
    """Return deterministic mid-ECDF targets, including tied-value midranks."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise PotencyAdapterGuidanceError("ECDF values must be one finite vector")
    order = np.argsort(array, kind="stable")
    quantiles = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        quantile = (0.5 * (start + stop - 1) + 0.5) / len(array)
        quantiles[order[start:stop]] = quantile
        start = stop
    return quantiles


def map_to_training_ecdf(train_values: Sequence[float], values: Sequence[float]) -> np.ndarray:
    """Map held-out values through a right-continuous training-fold ECDF."""

    train = np.sort(np.asarray(train_values, dtype=np.float64), kind="stable")
    query = np.asarray(values, dtype=np.float64)
    if (
        train.ndim != 1
        or query.ndim != 1
        or not len(train)
        or not np.isfinite(train).all()
        or not np.isfinite(query).all()
    ):
        raise PotencyAdapterGuidanceError("ECDF mapping requires finite one-dimensional values")
    return np.searchsorted(train, query, side="right").astype(np.float64) / len(train)


def quartile_balanced_sample(
    quantiles: Sequence[float],
    *,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample equal target mass from the four training-fold potency quartiles."""

    values = np.asarray(quantiles, dtype=np.float64)
    if values.ndim != 1 or not len(values) or size < 4 or size % 4:
        raise PotencyAdapterGuidanceError("quartile-balanced batch geometry is invalid")
    bins = np.minimum((values * 4.0).astype(np.int64), 3)
    per_bin = size // 4
    selected: list[int] = []
    for quartile in range(4):
        support = np.flatnonzero(bins == quartile)
        if not len(support):
            raise PotencyAdapterGuidanceError(f"training fold has no quartile {quartile}")
        selected.extend(rng.choice(support, size=per_bin, replace=True).tolist())
    output = np.asarray(selected, dtype=np.int64)
    rng.shuffle(output)
    return output


def deterministic_permutation(size: int, *, seed: int, namespace: str) -> np.ndarray:
    if size < 2:
        raise PotencyAdapterGuidanceError("shuffle control requires at least two observations")
    derived = int.from_bytes(hashlib.sha256(f"{namespace}:{seed}".encode()).digest()[:8], "big")
    return np.random.default_rng(derived).permutation(size)


def _midranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    truth = np.asarray(labels, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if truth.shape != values.shape or truth.ndim != 1 or not np.isfinite(values).all():
        raise PotencyAdapterGuidanceError("AUROC inputs are malformed")
    positives = truth == 1
    negatives = truth == 0
    if not bool(positives.any()) or not bool(negatives.any()) or np.any(~(positives | negatives)):
        raise PotencyAdapterGuidanceError("AUROC requires both binary classes")
    ranks = _midranks(values) + 1.0
    count_positive = int(positives.sum())
    count_negative = int(negatives.sum())
    return float(
        (ranks[positives].sum() - count_positive * (count_positive + 1) / 2)
        / (count_positive * count_negative)
    )


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    first = _midranks(np.asarray(left, dtype=np.float64))
    second = _midranks(np.asarray(right, dtype=np.float64))
    if first.shape != second.shape or first.ndim != 1 or len(first) < 3:
        raise PotencyAdapterGuidanceError("Spearman inputs are malformed")
    first -= first.mean()
    second -= second.mean()
    denominator = float(np.sqrt(np.dot(first, first) * np.dot(second, second)))
    if denominator == 0.0:
        raise PotencyAdapterGuidanceError("Spearman correlation is undefined")
    return float(np.dot(first, second) / denominator)


def morphology_residuals(
    train_features: np.ndarray,
    train_values: Sequence[float],
    test_features: np.ndarray,
    test_values: Sequence[float],
) -> np.ndarray:
    """Residualize held-out potency against a train-only linear morphology model."""

    train_x = np.asarray(train_features, dtype=np.float64)
    test_x = np.asarray(test_features, dtype=np.float64)
    train_y = np.asarray(train_values, dtype=np.float64)
    test_y = np.asarray(test_values, dtype=np.float64)
    if (
        train_x.ndim != 2
        or test_x.ndim != 2
        or train_x.shape[1] != test_x.shape[1]
        or train_x.shape[0] != len(train_y)
        or test_x.shape[0] != len(test_y)
        or not all(np.isfinite(value).all() for value in (train_x, test_x, train_y, test_y))
    ):
        raise PotencyAdapterGuidanceError("morphology residual inputs are malformed")
    train_design = np.column_stack((np.ones(len(train_x)), train_x))
    test_design = np.column_stack((np.ones(len(test_x)), test_x))
    coefficients = np.linalg.lstsq(train_design, train_y, rcond=None)[0]
    return test_y - test_design @ coefficients


def _cluster_bootstrap_members(groups: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Index each cluster once instead of rescanning every row in every bootstrap replicate."""

    unique = np.asarray(sorted(set(str(value) for value in groups)), dtype=object)
    members = {str(cluster): np.flatnonzero(groups == cluster) for cluster in unique}
    return unique, members


def _resampled_cluster_indices(
    rng: np.random.Generator,
    unique: np.ndarray,
    members: Mapping[str, np.ndarray],
) -> np.ndarray:
    chosen = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([members[str(cluster)] for cluster in chosen])


def clustered_bootstrap_interval(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    statistic: str,
    labels: Sequence[int] | None = None,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """Cluster-resample AUROC or Spearman without splitting component groups."""

    scores = np.asarray(values, dtype=np.float64)
    groups = np.asarray(clusters, dtype=object)
    if scores.ndim != 1 or groups.shape != scores.shape or replicates < 100:
        raise PotencyAdapterGuidanceError("clustered bootstrap contract is invalid")
    unique, members = _cluster_bootstrap_members(groups)
    if len(unique) < 2:
        raise PotencyAdapterGuidanceError("clustered bootstrap needs at least two clusters")
    truth = None if labels is None else np.asarray(labels)
    rng = np.random.default_rng(seed)
    sampled: list[float] = []
    for _ in range(replicates):
        indices = _resampled_cluster_indices(rng, unique, members)
        try:
            value = (
                roc_auc(truth[indices], scores[indices])
                if statistic == "auroc" and truth is not None
                else (
                    spearman(truth[indices], scores[indices])
                    if statistic == "spearman" and truth is not None
                    else None
                )
            )
        except PotencyAdapterGuidanceError:
            continue
        if value is None:
            raise PotencyAdapterGuidanceError(f"unsupported bootstrap statistic: {statistic}")
        sampled.append(value)
    if len(sampled) < max(50, replicates // 2):
        raise PotencyAdapterGuidanceError("too many clustered bootstrap replicates were undefined")
    return {
        "clusters": int(len(unique)),
        "valid_replicates": len(sampled),
        "lower_95": float(np.quantile(sampled, 0.025)),
        "median": float(np.quantile(sampled, 0.5)),
        "upper_95": float(np.quantile(sampled, 0.975)),
    }


def clustered_auroc_difference_interval(
    labels: Sequence[int],
    real_scores: Sequence[float],
    shuffled_scores: Sequence[float],
    clusters: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """Pair both adapter scores while resampling complete component clusters."""

    truth = np.asarray(labels, dtype=np.int64)
    real = np.asarray(real_scores, dtype=np.float64)
    shuffled = np.asarray(shuffled_scores, dtype=np.float64)
    groups = np.asarray(clusters, dtype=object)
    if not (truth.shape == real.shape == shuffled.shape == groups.shape) or replicates < 100:
        raise PotencyAdapterGuidanceError("paired AUROC bootstrap inputs are malformed")
    unique, members = _cluster_bootstrap_members(groups)
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(replicates):
        indices = _resampled_cluster_indices(rng, unique, members)
        try:
            differences.append(
                roc_auc(truth[indices], real[indices]) - roc_auc(truth[indices], shuffled[indices])
            )
        except PotencyAdapterGuidanceError:
            continue
    if len(differences) < max(50, replicates // 2):
        raise PotencyAdapterGuidanceError("paired AUROC bootstrap was usually undefined")
    return {
        "clusters": int(len(unique)),
        "valid_replicates": len(differences),
        "lower_95": float(np.quantile(differences, 0.025)),
        "median": float(np.quantile(differences, 0.5)),
        "upper_95": float(np.quantile(differences, 0.975)),
    }


def per_record_masked_nll(predictions: Mapping[str, Any], clean: Mapping[str, Any]) -> Any:
    """Return one equally-family-weighted denoising NLL per molecular record."""

    if torch is None or functional is None:
        raise PotencyAdapterGuidanceError("per-record denoising requires torch")
    fields = {
        "nodes": ("nodes", "atom_variable_mask"),
        "parents": ("parents", "parent_variable_mask"),
        "parent_bonds": ("parent_bonds", "parent_bond_variable_mask"),
        "closure_left": ("closure_left", "closure_endpoint_variable_mask"),
        "closure_right": ("closure_right", "closure_endpoint_variable_mask"),
        "closure_bonds": ("closure_bonds", "closure_bond_variable_mask"),
    }
    losses = predictions["nodes"].new_zeros(predictions["nodes"].shape[0])
    present = torch.zeros_like(losses)
    for output, (target, mask_name) in fields.items():
        logits = predictions[output]
        mask = clean[mask_name]
        point = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            clean[target].reshape(-1),
            reduction="none",
        ).reshape(mask.shape)
        counts = mask.sum(dim=1)
        active = counts > 0
        losses += (point * mask).sum(dim=1) / counts.clamp(min=1)
        present += active.to(losses.dtype)
    return losses / present.clamp(min=1)


def ordinal_nll_matrix(predictions: Sequence[Mapping[str, Any]], clean: Mapping[str, Any]) -> Any:
    """Stack per-record reconstruction NLL under ordered percentile requests."""

    if torch is None or len(predictions) < 2:
        raise PotencyAdapterGuidanceError("ordinal NLL requires at least two predictions")
    values = [per_record_masked_nll(prediction, clean) for prediction in predictions]
    if len({tuple(value.shape) for value in values}) != 1:
        raise PotencyAdapterGuidanceError("ordinal prediction batches changed shape")
    return torch.stack(values, dim=1)


def ordinal_contrastive_loss(
    nll: Any,
    target_quantiles: Any,
    *,
    anchors: Sequence[float],
    margin: float,
) -> tuple[Any, Any]:
    """Prefer requests closer to the observed percentile over counterfactual requests."""

    if torch is None:
        raise PotencyAdapterGuidanceError("ordinal contrast requires torch")
    anchor_values = torch.as_tensor(anchors, dtype=nll.dtype, device=nll.device)
    targets = torch.as_tensor(target_quantiles, dtype=nll.dtype, device=nll.device)
    if (
        nll.ndim != 2
        or targets.ndim != 1
        or nll.shape[0] != targets.shape[0]
        or nll.shape[1] != len(anchor_values)
        or len(anchor_values) < 2
        or not bool(torch.isfinite(nll).all())
        or not bool(torch.isfinite(targets).all())
        or not bool(torch.all(anchor_values[1:] > anchor_values[:-1]))
        or margin <= 0.0
    ):
        raise PotencyAdapterGuidanceError("ordinal contrast inputs are malformed")
    distances = torch.abs(targets[:, None] - anchor_values[None, :])
    nearest = torch.argmin(distances, dim=1)
    primary = nll.gather(1, nearest[:, None]).mean()
    comparisons: list[Any] = []
    for closer in range(len(anchor_values)):
        for farther in range(len(anchor_values)):
            ordered = distances[:, closer] + 1e-6 < distances[:, farther]
            if bool(ordered.any()):
                comparisons.append(
                    functional.relu(float(margin) + nll[ordered, closer] - nll[ordered, farther])
                )
    if not comparisons:
        raise PotencyAdapterGuidanceError("ordinal targets induce no ordered anchor pairs")
    contrast = torch.cat(comparisons).mean()
    return primary, contrast


def ordinal_direction_score(nll: Any) -> Any:
    """Positive means the molecule is reconstructed better under the highest request."""

    if torch is None or nll.ndim != 2 or nll.shape[1] < 2:
        raise PotencyAdapterGuidanceError("ordinal direction requires an N-by-K NLL matrix")
    return nll[:, 0] - nll[:, -1]


def ordinal_monotonicity_rows(
    nll: Sequence[Sequence[float]],
    target_quantiles: Sequence[float],
    *,
    anchors: Sequence[float],
) -> dict[str, np.ndarray]:
    """Return direct per-record anchor-choice and strict outer-quartile monotonicity checks."""

    values = np.asarray(nll, dtype=np.float64)
    targets = np.asarray(target_quantiles, dtype=np.float64)
    anchor_values = np.asarray(anchors, dtype=np.float64)
    if (
        values.ndim != 2
        or targets.ndim != 1
        or values.shape[0] != len(targets)
        or values.shape[1] != len(anchor_values)
        or len(anchor_values) != 3
        or not np.isfinite(values).all()
        or not np.isfinite(targets).all()
        or not np.all(anchor_values[1:] > anchor_values[:-1])
    ):
        raise PotencyAdapterGuidanceError("ordinal monotonicity inputs are malformed")
    nearest = np.argmin(np.abs(targets[:, None] - anchor_values[None, :]), axis=1)
    anchor_correct = np.argmin(values, axis=1) == nearest
    low = targets <= 0.25
    high = targets >= 0.75
    outer = low | high
    monotonic = np.zeros(len(targets), dtype=bool)
    monotonic[low] = (values[low, 0] < values[low, 1]) & (values[low, 1] < values[low, 2])
    monotonic[high] = (values[high, 2] < values[high, 1]) & (values[high, 1] < values[high, 0])
    return {
        "anchor_correct": anchor_correct,
        "outer_quartile": outer,
        "outer_monotonic": monotonic,
        "direction_score": values[:, 0] - values[:, -1],
    }


def masked_retention_kl(
    student: Mapping[str, Any],
    teacher: Mapping[str, Any],
    clean: Mapping[str, Any],
) -> Any:
    """Distill the null generator into the median condition on exactly variable coordinates."""

    if torch is None or functional is None:
        raise PotencyAdapterGuidanceError("retention loss requires torch")
    fields = {
        "nodes": "atom_variable_mask",
        "parents": "parent_variable_mask",
        "parent_bonds": "parent_bond_variable_mask",
        "closure_left": "closure_endpoint_variable_mask",
        "closure_right": "closure_endpoint_variable_mask",
        "closure_bonds": "closure_bond_variable_mask",
    }
    total = student["nodes"].new_zeros(())
    families = 0
    for field, mask_name in fields.items():
        mask = clean[mask_name]
        if not bool(mask.any()):
            continue
        teacher_probability = torch.softmax(teacher[field].detach(), dim=-1)
        divergence = functional.kl_div(
            torch.log_softmax(student[field], dim=-1),
            teacher_probability,
            reduction="none",
        ).sum(dim=-1)
        total = total + divergence[mask].mean()
        families += 1
    if not families:
        raise PotencyAdapterGuidanceError("retention batch has no variable coordinates")
    return total / families


def signal_gate(
    *,
    real_auroc: float,
    real_auroc_lower: float,
    residual_spearman: float,
    residual_spearman_lower: float,
    shuffled_auroc: float,
    shuffled_difference_lower: float,
    thresholds: Mapping[str, float],
) -> dict[str, bool]:
    checks = {
        "minimum_auroc": real_auroc >= float(thresholds["minimum_auroc"]),
        "auroc_lcb_above_chance": real_auroc_lower > 0.5,
        "minimum_residual_spearman": residual_spearman
        >= float(thresholds["minimum_residual_spearman"]),
        "residual_spearman_lcb_positive": residual_spearman_lower > 0.0,
        "minimum_shuffle_auroc_gap": real_auroc - shuffled_auroc
        >= float(thresholds["minimum_shuffle_auroc_gap"]),
        "shuffle_gap_lcb_positive": shuffled_difference_lower > 0.0,
    }
    return {**checks, "passes": all(checks.values())}


def clustered_mean_difference_interval(
    real_values: Sequence[float],
    shuffled_values: Sequence[float],
    clusters: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """Pair two per-record metrics while resampling complete component clusters."""

    real = np.asarray(real_values, dtype=np.float64)
    shuffled = np.asarray(shuffled_values, dtype=np.float64)
    groups = np.asarray(clusters, dtype=object)
    if (
        not (real.shape == shuffled.shape == groups.shape)
        or real.ndim != 1
        or not len(real)
        or not np.isfinite(real).all()
        or not np.isfinite(shuffled).all()
        or replicates < 100
    ):
        raise PotencyAdapterGuidanceError("paired mean bootstrap inputs are malformed")
    unique, members = _cluster_bootstrap_members(groups)
    if len(unique) < 2:
        raise PotencyAdapterGuidanceError("paired mean bootstrap needs two component clusters")
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(replicates):
        indices = _resampled_cluster_indices(rng, unique, members)
        differences.append(float(np.mean(real[indices] - shuffled[indices])))
    return {
        "clusters": int(len(unique)),
        "valid_replicates": len(differences),
        "lower_95": float(np.quantile(differences, 0.025)),
        "median": float(np.quantile(differences, 0.5)),
        "upper_95": float(np.quantile(differences, 0.975)),
    }


def monotonicity_gate(
    *,
    real_outer_fraction: float,
    shuffled_outer_fraction: float,
    outer_difference_lower: float,
    real_anchor_accuracy: float,
    shuffled_anchor_accuracy: float,
    anchor_difference_lower: float,
    thresholds: Mapping[str, float],
) -> dict[str, bool]:
    """Require direct q10/q50/q90 ordering beyond the paired shuffled-label control."""

    checks = {
        "minimum_outer_monotonic_fraction": real_outer_fraction
        >= float(thresholds["minimum_outer_monotonic_fraction"]),
        "minimum_outer_monotonic_gap": real_outer_fraction - shuffled_outer_fraction
        >= float(thresholds["minimum_outer_monotonic_gap"]),
        "outer_monotonic_gap_lcb_positive": outer_difference_lower > 0.0,
        "minimum_anchor_accuracy": real_anchor_accuracy
        >= float(thresholds["minimum_anchor_accuracy"]),
        "minimum_anchor_accuracy_gap": real_anchor_accuracy - shuffled_anchor_accuracy
        >= float(thresholds["minimum_anchor_accuracy_gap"]),
        "anchor_accuracy_gap_lcb_positive": anchor_difference_lower > 0.0,
    }
    return {**checks, "passes": all(checks.values())}


def promotion_gate(
    metrics: Mapping[str, float], thresholds: Mapping[str, float]
) -> dict[str, bool]:
    """Apply the prespecified seed-0 generation promotion rule without relaxing failures."""

    null_yield = float(metrics["null_unique_conservative_high_exact_l1_per_1000"])
    guided_yield = float(metrics["guided_unique_conservative_high_exact_l1_per_1000"])
    relative = (
        math.inf
        if null_yield == 0.0 and guided_yield > 0.0
        else (0.0 if null_yield == 0.0 else (guided_yield - null_yield) / null_yield)
    )
    checks = {
        "minimum_relative_gain": relative >= float(thresholds["minimum_relative_gain"]),
        "minimum_absolute_gain_per_1000": guided_yield - null_yield
        >= float(thresholds["minimum_absolute_gain_per_1000"]),
        "paired_program_lcb_positive": float(metrics["paired_program_bootstrap_lower"]) > 0.0,
        "validity_noninferior": float(metrics["guided_valid_fraction"])
        - float(metrics["null_valid_fraction"])
        >= -float(thresholds["maximum_validity_drop"]),
        "exact_l1_noninferior": float(metrics["guided_exact_l1_fraction"])
        - float(metrics["null_exact_l1_fraction"])
        >= -float(thresholds["maximum_exact_l1_drop"]),
        "minimum_distinct_fraction": float(metrics["guided_distinct_fraction"])
        >= float(thresholds["minimum_distinct_fraction"]),
        "minimum_effective_count_fraction": float(metrics["guided_effective_count_fraction"])
        >= float(thresholds["minimum_effective_count_fraction"]),
        "fixed_state_failures_zero": int(metrics["fixed_state_failures"]) == 0,
        "support_overflows_zero": int(metrics["support_overflows"]) == 0,
    }
    return {**checks, "passes": all(checks.values())}


__all__ = [
    "PotencyAdapterGuidanceError",
    "PotencyObservation",
    "clustered_mean_difference_interval",
    "clustered_bootstrap_interval",
    "clustered_auroc_difference_interval",
    "deterministic_permutation",
    "map_to_training_ecdf",
    "masked_retention_kl",
    "monotonicity_gate",
    "morphology_residuals",
    "normalized_agile_label",
    "ordinal_contrastive_loss",
    "ordinal_direction_score",
    "ordinal_monotonicity_rows",
    "ordinal_nll_matrix",
    "per_record_masked_nll",
    "promotion_gate",
    "quartile_balanced_sample",
    "roc_auc",
    "signal_gate",
    "spearman",
    "training_ecdf_quantiles",
]
