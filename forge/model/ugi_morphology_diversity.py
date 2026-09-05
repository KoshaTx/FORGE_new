"""Entropy-calibrated sampling over measured, identity-free Ugi morphology modes.

The joint semantic prior already preserves correlations among coarse role sizes, head shape,
ester position and tail chemistry.  Independent draws from that law can nevertheless concentrate
on a small number of high-mass modes.  This module exposes a bounded temperature transform of the
same train-fold support.  It changes probabilities, never support: no product, component, graph,
SMILES or fragment identity enters a sampled program.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


class UgiMorphologyDiversityError(ValueError):
    """A morphology-diversity policy is invalid or cannot satisfy its frozen gate."""


def _probability_vector(values: Sequence[float] | np.ndarray) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if (
        probabilities.ndim != 1
        or probabilities.size < 2
        or not np.isfinite(probabilities).all()
        or np.any(probabilities <= 0)
        or not np.isclose(probabilities.sum(), 1.0)
    ):
        raise UgiMorphologyDiversityError("morphology probabilities are invalid")
    return probabilities


def tempered_probabilities(
    probabilities: Sequence[float] | np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    """Flatten one categorical law without adding or removing support."""

    source = _probability_vector(probabilities)
    if not math.isfinite(temperature) or temperature < 1.0:
        raise UgiMorphologyDiversityError("morphology temperature must be finite and at least one")
    logits = np.log(source) / float(temperature)
    logits -= logits.max()
    output = np.exp(logits)
    output /= output.sum()
    return output


def expected_unique_modes(probabilities: Sequence[float] | np.ndarray, *, count: int) -> float:
    """Return the expected number of distinct modes under ``count`` IID requests."""

    source = _probability_vector(probabilities)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise UgiMorphologyDiversityError("morphology draw count must be positive")
    return float(np.sum(-np.expm1(count * np.log1p(-source))))


def categorical_kl(
    treatment: Sequence[float] | np.ndarray,
    reference: Sequence[float] | np.ndarray,
) -> float:
    """Return ``KL(treatment || reference)`` on one common full support."""

    left = _probability_vector(treatment)
    right = _probability_vector(reference)
    if left.shape != right.shape:
        raise UgiMorphologyDiversityError("morphology distributions have different support")
    return float(np.dot(left, np.log(left) - np.log(right)))


@dataclass(frozen=True)
class UgiMorphologyTemperatureSelection:
    """Frozen analytic choice of an entropy-increasing train-support temperature."""

    temperature: float
    probabilities: np.ndarray
    source_expected_unique: float
    treatment_expected_unique: float
    expected_unique_ratio: float
    kl_from_source: float
    target_ratio: float
    maximum_kl: float
    candidates: tuple[dict[str, float | bool], ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "source_expected_unique": self.source_expected_unique,
            "treatment_expected_unique": self.treatment_expected_unique,
            "expected_unique_ratio": self.expected_unique_ratio,
            "kl_from_source": self.kl_from_source,
            "target_ratio": self.target_ratio,
            "maximum_kl": self.maximum_kl,
            "target_attained": self.expected_unique_ratio >= self.target_ratio,
            "candidates": [dict(row) for row in self.candidates],
        }


def select_morphology_temperature(
    probabilities: Sequence[float] | np.ndarray,
    *,
    count: int,
    candidate_temperatures: Sequence[float],
    minimum_expected_unique_ratio: float,
    maximum_kl: float,
) -> UgiMorphologyTemperatureSelection:
    """Choose the smallest prespecified temperature satisfying diversity and drift gates."""

    source = _probability_vector(probabilities)
    candidates = tuple(float(value) for value in candidate_temperatures)
    if (
        not candidates
        or tuple(sorted(set(candidates))) != candidates
        or candidates[0] != 1.0
        or not math.isfinite(minimum_expected_unique_ratio)
        or minimum_expected_unique_ratio <= 1.0
        or not math.isfinite(maximum_kl)
        or maximum_kl <= 0
    ):
        raise UgiMorphologyDiversityError("morphology temperature-selection policy is invalid")
    source_unique = expected_unique_modes(source, count=count)
    audit_rows: list[dict[str, float | bool]] = []
    eligible: list[tuple[float, np.ndarray, float, float, float]] = []
    for temperature in candidates:
        treatment = tempered_probabilities(source, temperature=temperature)
        unique = expected_unique_modes(treatment, count=count)
        ratio = unique / source_unique
        divergence = categorical_kl(treatment, source)
        within_kl = divergence <= maximum_kl
        reaches_target = ratio >= minimum_expected_unique_ratio
        audit_rows.append(
            {
                "temperature": temperature,
                "expected_unique": unique,
                "expected_unique_ratio": ratio,
                "kl_from_source": divergence,
                "within_kl": within_kl,
                "reaches_target": reaches_target,
            }
        )
        if within_kl and reaches_target:
            eligible.append((temperature, treatment, unique, ratio, divergence))
    if not eligible:
        raise UgiMorphologyDiversityError(
            "no prespecified morphology temperature satisfies the diversity and KL gates"
        )
    temperature, treatment, unique, ratio, divergence = eligible[0]
    return UgiMorphologyTemperatureSelection(
        temperature=temperature,
        probabilities=treatment,
        source_expected_unique=source_unique,
        treatment_expected_unique=unique,
        expected_unique_ratio=ratio,
        kl_from_source=divergence,
        target_ratio=float(minimum_expected_unique_ratio),
        maximum_kl=float(maximum_kl),
        candidates=tuple(audit_rows),
    )


def sample_tempered_indices(
    probabilities: Sequence[float] | np.ndarray,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    """Draw IID support indices exactly once per requested attempt."""

    source = _probability_vector(probabilities)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise UgiMorphologyDiversityError("morphology seed must be non-negative")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise UgiMorphologyDiversityError("morphology draw count must be positive")
    return np.random.default_rng(seed).choice(
        source.size,
        size=count,
        replace=True,
        p=source,
    )


def summarize_sampled_modes(
    indices: Sequence[int] | np.ndarray, *, support_size: int
) -> dict[str, Any]:
    """Summarize realized categorical concentration without reading molecular identities."""

    values = [int(value) for value in indices]
    if support_size < 2 or not values or min(values) < 0 or max(values) >= support_size:
        raise UgiMorphologyDiversityError("sampled morphology indices are invalid")
    counts = Counter(values)
    probabilities = np.asarray(tuple(counts.values()), dtype=np.float64) / len(values)
    entropy = -float(np.dot(probabilities, np.log(probabilities)))
    return {
        "attempts": len(values),
        "support_size": support_size,
        "unique_modes": len(counts),
        "effective_mode_count": float(math.exp(entropy)),
        "maximum_mode_multiplicity": max(counts.values()),
        "maximum_mode_fraction": max(counts.values()) / len(values),
    }


__all__ = [
    "UgiMorphologyDiversityError",
    "UgiMorphologyTemperatureSelection",
    "categorical_kl",
    "expected_unique_modes",
    "sample_tempered_indices",
    "select_morphology_temperature",
    "summarize_sampled_modes",
    "tempered_probabilities",
]
