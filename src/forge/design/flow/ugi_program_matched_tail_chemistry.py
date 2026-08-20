"""Program-matched tail-chemistry evaluation for Ugi generator samples.

The global training marginal is not a fair reference when a sampling audit uses
a fixed morphology schedule. This module balances the frozen training measure
to the role-level morphology schedule represented among valid, exact-L1
generated products. Exact joint-cell matching is retained as a sensitivity view.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from forge.design.ugi_tail_chemotype_audit import component_chemotype_metrics
from forge.potency.audit.ugi_semantic_annotations import ROLE_NAMES

TAIL_ROLES = ("oxoester_aldehyde_body_tail", "isocyanide_tail")
MATCHING_COORDINATES = (
    "role_node_count",
    "role_junction_budget",
    "role_cycle_rank",
    "role_attachment_count",
)
FEATURE_FIELDS = {
    "has_adjacent_carbon_branches": "adjacent_carbon_branch_edges",
    "has_carbon_branch": "carbon_branch_atoms",
    "has_carbon_carbon_double_bond": "carbon_carbon_double_bonds",
    "has_carbon_carbon_triple_bond": "carbon_carbon_triple_bonds",
    "has_ester_like_carbonyl": "ester_like_carbonyl_count",
    "has_ether_oxygen": "ether_oxygen_count",
    "has_ring": "ring_count",
}
LOCAL_CHEMISTRY_FEATURES = (
    "has_carbon_carbon_double_bond",
    "has_carbon_carbon_triple_bond",
    "has_ester_like_carbonyl",
    "has_ether_oxygen",
)


class UgiProgramMatchedTailChemistryError(ValueError):
    """Raised when a program-matched chemistry comparison is malformed."""


def _program_coordinate(program: Any, field: str, role_index: int) -> int:
    values = program[field] if isinstance(program, Mapping) else getattr(program, field)
    return int(values[role_index])


def role_program_key(program: Any, role_index: int) -> tuple[int, int, int, int]:
    """Return the exact role-level morphology coordinates used for matching."""

    return (
        _program_coordinate(program, "node_counts", role_index),
        _program_coordinate(program, "junction_budgets", role_index),
        _program_coordinate(program, "cycle_ranks", role_index),
        _program_coordinate(program, "attachment_counts", role_index),
    )


def eligible_generated_rows(sample: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return valid, reconstructed, exact-L1 rows without hiding failures."""

    rows = sample.get("samples")
    if not isinstance(rows, list) or not rows:
        raise UgiProgramMatchedTailChemistryError("sample must contain generated rows")
    return [
        row
        for row in rows
        if row.get("valid") is True
        and row.get("component_reconstruction_valid") is True
        and (row.get("l1_forward_verification") or {}).get("exact_product_reconstructed") is True
    ]


def _weighted_feature_fractions(
    records: Sequence[tuple[Mapping[str, int], float]],
) -> dict[str, float]:
    total = sum(weight for _, weight in records)
    if total <= 0:
        raise UgiProgramMatchedTailChemistryError("feature summary has no positive mass")
    return {
        name: sum(weight * (int(metrics[field]) > 0) for metrics, weight in records) / total
        for name, field in FEATURE_FIELDS.items()
    }


def _weighted_component_diversity(
    records: Sequence[tuple[str, float]],
) -> dict[str, float | int]:
    mass: Counter[str] = Counter()
    for smiles, weight in records:
        mass[smiles] += float(weight)
    total = float(sum(mass.values()))
    if total <= 0:
        raise UgiProgramMatchedTailChemistryError("component summary has no positive mass")
    probabilities = sorted((value / total for value in mass.values()), reverse=True)
    return {
        "unique_exact_components": len(mass),
        "effective_exact_component_count": math.exp(
            -sum(value * math.log(value) for value in probabilities if value > 0)
        ),
        "top_1_component_fraction": probabilities[0],
        "top_5_component_fraction": sum(probabilities[:5]),
    }


def _serialized_key(key: tuple[int, int, int, int]) -> dict[str, int]:
    return {
        "role_node_count": key[0],
        "role_junction_budget": key[1],
        "role_cycle_rank": key[2],
        "role_attachment_count": key[3],
    }


def _mean_absolute_error(
    generated: Mapping[str, float],
    reference: Mapping[str, float],
    features: Sequence[str],
) -> float:
    return float(
        np.mean([abs(float(generated[name]) - float(reference[name])) for name in features])
    )


def _rake_training_weights_to_generated_marginals(
    *,
    training_keys: Sequence[tuple[int, int, int, int]],
    generated_keys: Sequence[tuple[int, int, int, int]],
    base_weights: Sequence[float],
    iterations: int = 200,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find the closest multiplicative weights matching schedule marginals."""

    if not training_keys or not generated_keys or len(training_keys) != len(base_weights):
        raise UgiProgramMatchedTailChemistryError("marginal raking inputs are malformed")
    training = np.asarray(training_keys, dtype=np.int64)
    generated = np.asarray(generated_keys, dtype=np.int64)
    weights = np.asarray(base_weights, dtype=np.float64).copy()
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise UgiProgramMatchedTailChemistryError("marginal raking weights are malformed")
    weights /= weights.sum()

    targets: list[dict[int, float]] = []
    unsupported: list[dict[str, int | str]] = []
    for coordinate, name in enumerate(MATCHING_COORDINATES):
        values, counts = np.unique(generated[:, coordinate], return_counts=True)
        target = {int(value): float(count) / len(generated) for value, count in zip(values, counts)}
        observed_values = set(int(value) for value in training[:, coordinate])
        for value, mass in target.items():
            if value not in observed_values:
                unsupported.append(
                    {
                        "coordinate": name,
                        "value": value,
                        "generated_rows": int(round(mass * len(generated))),
                    }
                )
        targets.append(target)
    if unsupported:
        raise UgiProgramMatchedTailChemistryError(
            f"generated schedule contains unsupported coordinate values: {unsupported}"
        )

    converged_at = iterations
    maximum_error = float("inf")
    for iteration in range(1, iterations + 1):
        for coordinate, target in enumerate(targets):
            active_values = np.asarray(tuple(target), dtype=np.int64)
            weights[~np.isin(training[:, coordinate], active_values)] = 0.0
            total = weights.sum()
            if total <= 0:
                raise UgiProgramMatchedTailChemistryError(
                    "marginal raking removed all training support"
                )
            weights /= total
            for value, target_mass in target.items():
                mask = training[:, coordinate] == value
                current_mass = float(weights[mask].sum())
                if current_mass <= 0:
                    raise UgiProgramMatchedTailChemistryError(
                        "marginal targets are infeasible on the training joint support"
                    )
                weights[mask] *= target_mass / current_mass
            weights /= weights.sum()

        maximum_error = max(
            abs(float(weights[training[:, coordinate] == value].sum()) - target_mass)
            for coordinate, target in enumerate(targets)
            for value, target_mass in target.items()
        )
        if maximum_error <= tolerance:
            converged_at = iteration
            break

    positive = weights > 0
    effective_count = float(np.exp(-np.sum(weights[positive] * np.log(weights[positive]))))
    return weights, {
        "method": "iterative_proportional_fitting_from_frozen_training_weights",
        "iterations": converged_at,
        "maximum_absolute_marginal_error": maximum_error,
        "effective_weighted_training_product_count": effective_count,
        "target_marginals": {
            name: {str(value): mass for value, mass in sorted(target.items())}
            for name, target in zip(MATCHING_COORDINATES, targets, strict=True)
        },
    }


def compare_program_matched_tail_chemistry(
    *,
    sample: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    joint_records: Sequence[Any],
    training_weights: Sequence[float],
) -> dict[str, Any]:
    """Compare generated tail chemistry with the exact scheduled training measure.

    Training rows retain the frozen source-balanced weights within each exact
    role-program cell.  Cell masses are then replaced by the generated program
    frequencies, which isolates chemistry realization from the sampling schedule.
    """

    if not (len(assignments) == len(joint_records) == len(training_weights)):
        raise UgiProgramMatchedTailChemistryError(
            "training assignments, graph records, and weights must align"
        )
    eligible = eligible_generated_rows(sample)
    if not eligible:
        raise UgiProgramMatchedTailChemistryError("sample has no valid exact-L1 products")

    unique_smiles = {str(row[f"{role}_smiles"]) for row in assignments for role in TAIL_ROLES}
    unique_smiles.update(
        str(row["component_smiles_by_role"][role]) for row in eligible for role in TAIL_ROLES
    )
    metrics_cache = {smiles: component_chemotype_metrics(smiles) for smiles in unique_smiles}

    by_role: dict[str, Any] = {}
    all_feature_errors: list[float] = []
    local_chemistry_errors: list[float] = []
    exact_cell_all_feature_errors: list[float] = []
    exact_cell_local_chemistry_errors: list[float] = []
    for role in TAIL_ROLES:
        role_index = ROLE_NAMES.index(role)
        training_groups: dict[tuple[int, int, int, int], list[tuple[Mapping[str, int], float]]] = {}
        for assignment, record, weight in zip(
            assignments, joint_records, training_weights, strict=True
        ):
            numeric_weight = float(weight)
            if numeric_weight < 0:
                raise UgiProgramMatchedTailChemistryError("training weights must be nonnegative")
            key = role_program_key(record.program, role_index)
            smiles = str(assignment[f"{role}_smiles"])
            training_groups.setdefault(key, []).append((metrics_cache[smiles], numeric_weight))

        generated_groups: dict[
            tuple[int, int, int, int], list[tuple[Mapping[str, int], float]]
        ] = {}
        for row in eligible:
            key = role_program_key(row["program"], role_index)
            smiles = str(row["component_smiles_by_role"][role])
            generated_groups.setdefault(key, []).append((metrics_cache[smiles], 1.0))

        training_keys = [role_program_key(record.program, role_index) for record in joint_records]
        training_coordinate_support = tuple(
            {key[coordinate] for key in training_keys} for coordinate in range(4)
        )
        marginal_supported_groups = {
            key: records
            for key, records in generated_groups.items()
            if all(
                key[coordinate] in training_coordinate_support[coordinate]
                for coordinate in range(4)
            )
        }
        marginal_unsupported_groups = {
            key: records
            for key, records in generated_groups.items()
            if key not in marginal_supported_groups
        }
        marginal_supported_rows = sum(
            len(records) for records in marginal_supported_groups.values()
        )
        if marginal_supported_rows <= 0:
            raise UgiProgramMatchedTailChemistryError(
                f"no generated {role} rows have selection-visible coordinate support"
            )
        generated_keys = [
            key for key, records in marginal_supported_groups.items() for _ in records
        ]
        raked_weights, raking_diagnostics = _rake_training_weights_to_generated_marginals(
            training_keys=training_keys,
            generated_keys=generated_keys,
            base_weights=training_weights,
        )
        marginal_supported_key_set = set(marginal_supported_groups)
        generated_component_diversity = _weighted_component_diversity(
            [
                (str(row["component_smiles_by_role"][role]), 1.0)
                for row in eligible
                if role_program_key(row["program"], role_index) in marginal_supported_key_set
            ]
        )
        reference_component_diversity = _weighted_component_diversity(
            [
                (str(assignment[f"{role}_smiles"]), float(weight))
                for assignment, weight in zip(assignments, raked_weights, strict=True)
                if weight > 0
            ]
        )
        all_generated_records = [
            record for records in marginal_supported_groups.values() for record in records
        ]
        generated_features_all = _weighted_feature_fractions(all_generated_records)
        marginal_reference = _weighted_feature_fractions(
            [
                (metrics_cache[str(assignment[f"{role}_smiles"])], float(weight))
                for assignment, weight in zip(assignments, raked_weights, strict=True)
                if weight > 0
            ]
        )
        marginal_absolute_errors = {
            name: abs(generated_features_all[name] - marginal_reference[name])
            for name in FEATURE_FIELDS
        }
        all_feature_errors.extend(marginal_absolute_errors.values())
        local_chemistry_errors.extend(
            marginal_absolute_errors[name] for name in LOCAL_CHEMISTRY_FEATURES
        )

        generated_key_counts = Counter(
            {key: len(records) for key, records in generated_groups.items()}
        )
        matched_keys = set(generated_groups).intersection(training_groups)
        matched_generated_rows = sum(generated_key_counts[key] for key in matched_keys)
        if matched_generated_rows <= 0:
            raise UgiProgramMatchedTailChemistryError(
                f"no generated {role} programs occur in the training reference"
            )

        generated_records = [record for key in matched_keys for record in generated_groups[key]]
        matched_reference: list[tuple[Mapping[str, int], float]] = []
        for key in matched_keys:
            group = training_groups[key]
            group_mass = sum(weight for _, weight in group)
            if group_mass <= 0:
                raise UgiProgramMatchedTailChemistryError(
                    f"training program cell for {role} has no positive mass"
                )
            target_mass = generated_key_counts[key] / matched_generated_rows
            matched_reference.extend(
                (metrics, target_mass * weight / group_mass) for metrics, weight in group
            )

        generated_features = _weighted_feature_fractions(generated_records)
        reference_features = _weighted_feature_fractions(matched_reference)
        absolute_errors = {
            name: abs(generated_features[name] - reference_features[name])
            for name in FEATURE_FIELDS
        }
        exact_cell_all_feature_errors.extend(absolute_errors.values())
        exact_cell_local_chemistry_errors.extend(
            absolute_errors[name] for name in LOCAL_CHEMISTRY_FEATURES
        )
        missing = sorted(set(generated_groups).difference(training_groups))
        by_role[role] = {
            "matching_coordinates": list(MATCHING_COORDINATES),
            "generated_exact_l1_rows": len(eligible),
            "exactly_matched_generated_rows": matched_generated_rows,
            "exact_match_fraction": matched_generated_rows / len(eligible),
            "matched_program_cells": len(matched_keys),
            "unmatched_generated_program_cells": [
                {"program": _serialized_key(key), "generated_rows": generated_key_counts[key]}
                for key in missing
            ],
            "marginally_balanced_coordinate_supported_cohort": {
                "generated_rows": marginal_supported_rows,
                "coordinate_support_fraction": marginal_supported_rows / len(eligible),
                "unsupported_generated_program_cells": [
                    {"program": _serialized_key(key), "generated_rows": len(records)}
                    for key, records in sorted(marginal_unsupported_groups.items())
                ],
                "generated_feature_occurrence_fractions": generated_features_all,
                "training_feature_occurrence_fractions": marginal_reference,
                "component_diversity": {
                    "generated": generated_component_diversity,
                    "training_reference": reference_component_diversity,
                },
                "absolute_error": marginal_absolute_errors,
                "mean_absolute_error_all_features": _mean_absolute_error(
                    generated_features_all, marginal_reference, tuple(FEATURE_FIELDS)
                ),
                "mean_absolute_error_local_chemistry": _mean_absolute_error(
                    generated_features_all, marginal_reference, LOCAL_CHEMISTRY_FEATURES
                ),
                "raking_diagnostics": raking_diagnostics,
            },
            "exact_joint_cell_sensitivity": {
                "generated_feature_occurrence_fractions": generated_features,
                "training_feature_occurrence_fractions": reference_features,
                "absolute_error": absolute_errors,
                "mean_absolute_error_all_features": _mean_absolute_error(
                    generated_features, reference_features, tuple(FEATURE_FIELDS)
                ),
                "mean_absolute_error_local_chemistry": _mean_absolute_error(
                    generated_features, reference_features, LOCAL_CHEMISTRY_FEATURES
                ),
            },
        }

    return {
        "matching_coordinates": list(MATCHING_COORDINATES),
        "feature_definitions": dict(FEATURE_FIELDS),
        "local_chemistry_features": list(LOCAL_CHEMISTRY_FEATURES),
        "by_role": by_role,
        "mean_absolute_error_across_tail_roles_all_features": float(np.mean(all_feature_errors)),
        "mean_absolute_error_across_tail_roles_local_chemistry": float(
            np.mean(local_chemistry_errors)
        ),
        "exact_joint_cell_sensitivity_mean_absolute_error_across_tail_roles_all_features": float(
            np.mean(exact_cell_all_feature_errors)
        ),
        "exact_joint_cell_sensitivity_mean_absolute_error_across_tail_roles_local_chemistry": float(
            np.mean(exact_cell_local_chemistry_errors)
        ),
    }
