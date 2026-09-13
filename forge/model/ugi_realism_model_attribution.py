"""Small, train-only reconstruction and probability-law attribution primitives.

These diagnostics do not generate molecules. In particular, a categorical training support is
not the complete set of feasible terminal-decoder arrangements.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from forge.core.io import stable_json
from forge.model.ugi_mog_semantic_guidance import UgiMogSemanticGuidancePolicy


class UgiRealismModelAttributionError(ValueError):
    """The bounded diagnostic or its attribution denominators are invalid."""


def select_measured_train_rows(
    rows: Iterable[Mapping[str, str]],
    *,
    roles: Sequence[str],
    records: int,
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Greedily cover unique measured components, breaking ties with seeded order.

    Only train/measured metadata is inspected before reading any structural fields. Product
    enumeration frequency does not determine selection probability. This is an intentionally
    diverse diagnostic panel, not an unbiased population estimator.
    """

    if records < 1 or not roles or len(set(roles)) != len(roles):
        raise UgiRealismModelAttributionError("invalid diagnostic selection geometry")
    candidates: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for row in rows:
        if row["primary_product_fold"] != "train":
            continue
        if row["is_source_adjudicated_measured_product"] != "true":
            continue
        record_id = row["product_id"]
        if record_id in seen_ids:
            raise UgiRealismModelAttributionError(f"duplicate measured train ID: {record_id}")
        if any(not row.get(f"{role}_smiles") for role in roles):
            raise UgiRealismModelAttributionError(f"missing train component: {record_id}")
        if any(row[f"{role}_family_fold"] != "train" for role in roles):
            raise UgiRealismModelAttributionError(f"component fold disagrees: {record_id}")
        seen_ids.add(record_id)
        candidates.append(dict(row))
    candidates.sort(key=lambda row: row["product_id"])
    if len(candidates) < records:
        raise UgiRealismModelAttributionError("measured train support is smaller than records")
    order = np.random.default_rng(seed).permutation(len(candidates)).tolist()
    covered: dict[str, set[str]] = {role: set() for role in roles}
    selected: list[dict[str, str]] = []
    for _ in range(records):
        chosen = max(
            order,
            key=lambda index: sum(
                candidates[index][f"{role}_smiles"] not in covered[role] for role in roles
            ),
        )
        order.remove(chosen)
        row = candidates[chosen]
        selected.append(row)
        for role in roles:
            covered[role].add(row[f"{role}_smiles"])
    return selected, {
        "algorithm": "greedy_unique_component_coverage_seeded_ties_v1",
        "eligible_measured_train_products": len(candidates),
        "available_components_by_role": {
            role: len({row[f"{role}_smiles"] for row in candidates}) for role in roles
        },
        "selected_components_by_role": {role: len(covered[role]) for role in roles},
        "selected_product_ids": [row["product_id"] for row in selected],
        "selected_product_ids_sha256": hashlib.sha256(
            stable_json([row["product_id"] for row in selected]).encode()
        ).hexdigest(),
        "population_estimator": False,
    }


def classification_summary(
    probabilities: np.ndarray, targets: np.ndarray, *, calibration_bins: int = 10
) -> dict[str, Any]:
    """Report proper scores and descriptive calibration for one explicit denominator."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets)
    if (
        probabilities.ndim != 2
        or targets.shape != probabilities.shape[:1]
        or not np.issubdtype(targets.dtype, np.integer)
        or calibration_bins < 1
        or probabilities.shape[1] < 1
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(axis=1), 1.0)
        or np.any(targets < 0)
        or np.any(targets >= probabilities.shape[1])
    ):
        raise UgiRealismModelAttributionError("classification probabilities/targets are invalid")
    total = len(targets)
    if total == 0:
        return {"total": 0, "accuracy": None, "nll": None, "ece": None}
    target_probability = probabilities[np.arange(total), targets]
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    bins = np.minimum((confidence * calibration_bins).astype(int), calibration_bins - 1)
    calibration = []
    ece = 0.0
    for index in range(calibration_bins):
        mask = bins == index
        count = int(mask.sum())
        if not count:
            continue
        accuracy = float(correct[mask].mean())
        mean_confidence = float(confidence[mask].mean())
        ece += count / total * abs(accuracy - mean_confidence)
        calibration.append(
            {
                "bin": index,
                "total": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
        )
    log_probability = np.log(np.maximum(probabilities, np.finfo(np.float64).tiny))
    return {
        "total": total,
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.maximum(target_probability, np.finfo(np.float64).tiny)).mean()),
        "mean_target_probability": float(target_probability.mean()),
        "mean_confidence": float(confidence.mean()),
        "mean_entropy_nats": float(-(probabilities * log_probability).sum(axis=1).mean()),
        "multiclass_brier": float(
            (np.square(probabilities).sum(axis=1) - 2 * target_probability + 1).mean()
        ),
        "ece": ece,
        "calibration_bins": calibration,
    }


def masked_probabilities(logits: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Normalize only identical declared categorical supports; never resurrect masked classes."""

    logits = np.asarray(logits, dtype=np.float64)
    allowed = np.asarray(allowed)
    if (
        logits.ndim != 2
        or logits.shape != allowed.shape
        or allowed.dtype != np.bool_
        or np.any(allowed.sum(axis=1) == 0)
        or not np.isfinite(logits[allowed]).all()
    ):
        raise UgiRealismModelAttributionError("logits or candidate masks are invalid")
    masked = np.where(allowed, logits, -np.inf)
    if not len(masked):
        return np.zeros_like(masked)
    mass = np.exp(masked - masked.max(axis=1, keepdims=True))
    return mass / mass.sum(axis=1, keepdims=True)


def rank_law_probabilities(
    logits: np.ndarray,
    allowed: np.ndarray,
    *,
    model_rank_weight: float,
    rank_temperature: float,
    uniform_probability_mass: float,
) -> np.ndarray:
    """Use the existing decoder rank law with every non-neural score held equal.

    This isolates rank normalization plus uniform mixing on the SAME supplied candidate set. It
    is an off-policy mathematical diagnostic, not the actual terminal decoder distribution.
    """

    probabilities = masked_probabilities(logits, allowed)
    policy = UgiMogSemanticGuidancePolicy(
        amine_heavy_atom_graph_diameter_tolerance=0,
        amine_carbon_skeleton_diameter_tolerance=0,
        aldehyde_ester_side_carbons_tolerance=0,
        model_rank_weight=model_rank_weight,
        semantic_rank_weight=0.0,
        rank_temperature=rank_temperature,
        uniform_probability_mass=uniform_probability_mass,
    )
    for index in range(len(probabilities)):
        mask = allowed[index]
        probabilities[index, mask] = policy.probabilities(
            logits[index, mask], np.zeros(int(mask.sum()))
        )
    return probabilities


def coordinate_summaries(
    *,
    logits: np.ndarray,
    targets: np.ndarray,
    noisy: np.ndarray,
    active_mask: np.ndarray,
    role_states: np.ndarray,
    roles: Mapping[str, int],
    allowed: np.ndarray,
    rank_law: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Partition variable targets by role and actual corruption, preserving every denominator."""

    if (
        targets.shape != active_mask.shape
        or noisy.shape != targets.shape
        or role_states.shape != targets.shape
        or logits.shape[:-1] != targets.shape
        or allowed.shape != logits.shape
        or active_mask.dtype != np.bool_
    ):
        raise UgiRealismModelAttributionError("coordinate tensors are not aligned")
    active_roles = set(np.unique(role_states[active_mask]).tolist())
    if not active_roles.issubset(set(roles.values())):
        raise UgiRealismModelAttributionError("active coordinates have an unreported role")
    result = {}
    for role, role_state in sorted(roles.items()):
        role_mask = active_mask & (role_states == role_state)
        result[role] = {}
        for subset, subset_mask in (
            ("all_variable", role_mask),
            ("actually_corrupted", role_mask & (noisy != targets)),
            ("unchanged", role_mask & (noisy == targets)),
        ):
            selected_logits = logits[subset_mask]
            selected_targets = targets[subset_mask]
            selected_allowed = allowed[subset_mask]
            if (
                len(selected_targets)
                and not selected_allowed[np.arange(len(selected_targets)), selected_targets].all()
            ):
                raise UgiRealismModelAttributionError("target excluded by categorical support")
            probabilities = masked_probabilities(selected_logits, selected_allowed)
            entry = {"softmax": classification_summary(probabilities, selected_targets)}
            if rank_law is not None:
                ranked = rank_law_probabilities(selected_logits, selected_allowed, **rank_law)
                entry["rank_uniform_equal_other_scores"] = classification_summary(
                    ranked, selected_targets
                )
                entry["candidate_counts"] = sorted(set(selected_allowed.sum(axis=1).tolist()))
                entry["argmax_changes"] = int(
                    (ranked.argmax(axis=1) != probabilities.argmax(axis=1)).sum()
                )
            result[role][subset] = entry
    return result
