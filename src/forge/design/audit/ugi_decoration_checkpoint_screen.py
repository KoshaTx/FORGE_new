"""Selection-ready summaries for the Ugi decoration-coupling checkpoint screen."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from forge.design.ugi_program_matched_tail_chemistry import eligible_generated_rows
from forge.potency.audit.ugi_semantic_annotations import ROLE_NAMES

MINIMUM_VALID_FRACTION = 0.95
MINIMUM_UNIQUE_VALID_FRACTION = 0.98
MINIMUM_EFFECTIVE_COMPONENT_COUNT = 20.0
MINIMUM_UNIQUE_COMPONENTS = {
    "amine_head": 64,
    "oxoester_aldehyde_body_tail": 32,
    "isocyanide_tail": 24,
}
TAIL_ROLES = ("oxoester_aldehyde_body_tail", "isocyanide_tail")


class UgiDecorationCheckpointScreenError(ValueError):
    """Raised when a checkpoint-screen artifact is inconsistent."""


def _effective_count(counts: Mapping[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        raise UgiDecorationCheckpointScreenError("effective count requires observations")
    return math.exp(
        -sum((count / total) * math.log(count / total) for count in counts.values() if count > 0)
    )


def _component_diversity(rows: Sequence[Mapping[str, Any]], role: str) -> dict[str, Any]:
    counts = Counter(str(row["component_smiles_by_role"][role]) for row in rows)
    total = sum(counts.values())
    descending = sorted(counts.values(), reverse=True)
    return {
        "component_occurrences": total,
        "unique_exact_components": len(counts),
        "effective_exact_component_count": _effective_count(counts),
        "top_1_component_fraction": descending[0] / total,
        "top_5_component_fraction": sum(descending[:5]) / total,
    }


def _branch_geometry_deltas(branch_arm: Mapping[str, Any], role: str) -> dict[str, Any]:
    comparison = branch_arm["program_matched_training_reference"][role]
    generated = comparison["generated_summary_on_exactly_matched_programs"]
    reference = comparison["summary_on_exactly_matched_program_mass"]
    scalar_paths = {
        "first_outward_branch_fraction": ("first_outward_branch_fraction",),
        "mean_carbon_branch_atoms": ("mean_carbon_branch_atoms",),
        "adjacent_branch_component_fraction": ("adjacent_branch_component_fraction",),
        "mean_branch_depth_from_handle": ("branch_depth_from_handle", "mean"),
        "mean_longer_distal_arm_carbons": ("longer_distal_arm_carbons", "mean"),
        "mean_shorter_distal_arm_carbons": ("shorter_distal_arm_carbons", "mean"),
        "mean_distal_arm_balance": ("distal_arm_balance", "mean"),
        "twig_like_branch_fraction": (
            "first_branch_shape_fractions",
            "twig_like_balance_at_most_0.25",
        ),
        "balanced_fork_fraction": (
            "first_branch_shape_fractions",
            "balanced_fork_at_least_0.67",
        ),
    }

    def value(source: Mapping[str, Any], path: tuple[str, ...]) -> float | None:
        current: Any = source
        for field in path:
            current = current[field]
        return None if current is None else float(current)

    metrics = {}
    for name, path in scalar_paths.items():
        observed = value(generated, path)
        target = value(reference, path)
        metrics[name] = {
            "generated": observed,
            "reference": target,
            "absolute_error": (
                None if observed is None or target is None else abs(observed - target)
            ),
        }
    return {
        "exact_program_match_fraction": float(comparison["exact_match_fraction"]),
        "metrics": metrics,
    }


def evaluate_checkpoint_arm(
    *,
    sample: Mapping[str, Any],
    chemistry_arm: Mapping[str, Any],
    branch_arm: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one arm with hard integrity/collapse gates and descriptive axes."""

    rows = sample.get("samples")
    statistics = sample.get("statistics")
    if not isinstance(rows, list) or not isinstance(statistics, Mapping):
        raise UgiDecorationCheckpointScreenError("sample lacks rows or statistics")
    eligible = eligible_generated_rows(sample)
    valid = int(statistics["valid_molecules"])
    unique_valid = int(statistics["unique_valid_molecules"])
    if valid <= 0 or len(rows) <= 0:
        raise UgiDecorationCheckpointScreenError("sample has no valid denominator")
    component_diversity = {role: _component_diversity(eligible, role) for role in ROLE_NAMES}
    branch_geometry = {role: _branch_geometry_deltas(branch_arm, role) for role in TAIL_ROLES}
    mean_branch_geometry_jsd = sum(
        float(
            branch_arm["program_matched_training_reference"][role]["branch_geometry_signature"][
                "jensen_shannon_divergence_bits"
            ]
        )
        for role in TAIL_ROLES
    ) / len(TAIL_ROLES)
    maximum_adjacent_branch_fraction = max(
        float(branch_geometry[role]["metrics"]["adjacent_branch_component_fraction"]["generated"])
        for role in TAIL_ROLES
    )
    gate_checks: dict[str, bool] = {
        "minimum_valid_fraction": float(statistics["valid_fraction"]) >= MINIMUM_VALID_FRACTION,
        "exact_l1_fraction_of_valid": len(eligible) / valid == 1.0,
        "minimum_unique_valid_fraction": unique_valid / valid >= MINIMUM_UNIQUE_VALID_FRACTION,
        "no_adjacent_tail_branch_inflation": maximum_adjacent_branch_fraction == 0.0,
    }
    for role, summary in component_diversity.items():
        gate_checks[f"{role}_minimum_unique_components"] = (
            int(summary["unique_exact_components"]) >= MINIMUM_UNIQUE_COMPONENTS[role]
        )
        gate_checks[f"{role}_minimum_effective_component_count"] = (
            float(summary["effective_exact_component_count"]) >= MINIMUM_EFFECTIVE_COMPONENT_COUNT
        )
    return {
        "attempted_molecules": len(rows),
        "valid_molecules": valid,
        "valid_fraction": float(statistics["valid_fraction"]),
        "unique_valid_fraction": unique_valid / valid,
        "exact_l1_fraction_of_valid": len(eligible) / valid,
        "component_diversity": component_diversity,
        "program_balanced_tail_chemistry": chemistry_arm,
        "branch_geometry": branch_geometry,
        "hard_gate_checks": gate_checks,
        "passes_all_hard_gates": all(gate_checks.values()),
        "pareto_axes": {
            "maximize_valid_fraction": float(statistics["valid_fraction"]),
            "minimize_program_balanced_local_chemistry_mae": float(
                chemistry_arm["mean_absolute_error_across_tail_roles_local_chemistry"]
            ),
            "minimize_program_matched_branch_geometry_jsd_bits": mean_branch_geometry_jsd,
        },
    }


def pareto_frontier(arms: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Return hard-gate-passing arms nondominated on validity and chemistry."""

    passing = {name: arm for name, arm in arms.items() if arm.get("passes_all_hard_gates") is True}
    frontier = []
    for name, candidate in passing.items():
        candidate_validity = float(candidate["pareto_axes"]["maximize_valid_fraction"])
        candidate_error = float(
            candidate["pareto_axes"]["minimize_program_balanced_local_chemistry_mae"]
        )
        candidate_branch = float(
            candidate["pareto_axes"]["minimize_program_matched_branch_geometry_jsd_bits"]
        )
        dominated = False
        for other_name, other in passing.items():
            if other_name == name:
                continue
            other_validity = float(other["pareto_axes"]["maximize_valid_fraction"])
            other_error = float(
                other["pareto_axes"]["minimize_program_balanced_local_chemistry_mae"]
            )
            other_branch = float(
                other["pareto_axes"]["minimize_program_matched_branch_geometry_jsd_bits"]
            )
            no_worse = (
                other_validity >= candidate_validity
                and other_error <= candidate_error
                and other_branch <= candidate_branch
            )
            strictly_better = (
                other_validity > candidate_validity
                or other_error < candidate_error
                or other_branch < candidate_branch
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(name)
    return sorted(frontier)


def select_confirmed_training_duration(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    chemistry_noninferiority_margin: float,
    branch_jsd_noninferiority_margin_bits: float,
    validity_tie_margin: float,
) -> dict[str, Any]:
    """Apply the frozen lexicographic confirmation rule to aggregate arms."""

    if any(
        value < 0
        for value in (
            chemistry_noninferiority_margin,
            branch_jsd_noninferiority_margin_bits,
            validity_tie_margin,
        )
    ):
        raise UgiDecorationCheckpointScreenError("selection margins must be nonnegative")
    passing = {
        str(name): arm for name, arm in arms.items() if arm.get("passes_all_hard_gates") is True
    }
    if not passing:
        return {
            "selected_checkpoint_step": None,
            "reason": "all finalists failed at least one hard gate",
            "hard_gate_passing_steps": [],
        }

    chemistry_best = min(
        float(arm["mean_axes"]["program_balanced_local_chemistry_mae"]) for arm in passing.values()
    )
    chemistry_noninferior = {
        name: arm
        for name, arm in passing.items()
        if float(arm["mean_axes"]["program_balanced_local_chemistry_mae"])
        <= chemistry_best + chemistry_noninferiority_margin
    }
    branch_best = min(
        float(arm["mean_axes"]["program_matched_branch_geometry_jsd_bits"])
        for arm in chemistry_noninferior.values()
    )
    branch_noninferior = {
        name: arm
        for name, arm in chemistry_noninferior.items()
        if float(arm["mean_axes"]["program_matched_branch_geometry_jsd_bits"])
        <= branch_best + branch_jsd_noninferiority_margin_bits
    }
    validity_best = max(
        float(arm["mean_axes"]["valid_fraction"]) for arm in branch_noninferior.values()
    )
    validity_tied = {
        name: arm
        for name, arm in branch_noninferior.items()
        if validity_best - float(arm["mean_axes"]["valid_fraction"]) < validity_tie_margin
    }
    selected = min(int(name) for name in validity_tied)
    return {
        "selected_checkpoint_step": selected,
        "reason": "frozen chemistry, branch-geometry, validity, then earlier-step rule",
        "hard_gate_passing_steps": sorted(int(name) for name in passing),
        "chemistry_best_mae": chemistry_best,
        "chemistry_noninferior_steps": sorted(int(name) for name in chemistry_noninferior),
        "branch_best_jsd_bits": branch_best,
        "branch_noninferior_steps": sorted(int(name) for name in branch_noninferior),
        "validity_best_fraction": validity_best,
        "validity_tied_steps": sorted(int(name) for name in validity_tied),
    }
