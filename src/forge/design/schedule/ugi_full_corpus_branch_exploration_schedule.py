"""Freeze full-corpus Ugi morphology support and a branched exploration schedule.

The promoted biological proposal remains defined on its independently confirmed
57,190-program support.  This module does not extrapolate that score.  It builds
an outcome-free exploration schedule from the larger role-level program prior
used by the final all-fold generator, conditioned only on a tail-origin junction.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.io import stable_json as _stable_json
from forge.data.r1_prime_audit import sha256_bytes, sha256_file
from forge.design.ugi_complete_morphology_proposal import enumerate_complete_program_support
from forge.design.ugi_constrained_stochastic_production_candidates import branch_class
from forge.design.ugi_program_prior import load_program_prior

CONFIG_SCHEMA_VERSION = "phase1_ugi_full_corpus_branch_exploration_schedule_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_full_corpus_branch_exploration_schedule.v1"
SUPPORT_LEDGER_SCHEMA_VERSION = "forge.ugi_full_corpus_morphology_support_ledger.v1"
SCHEDULE_SCHEMA_VERSION = "forge.ugi_branch_exploration_schedule.v1"
BRANCH_CLASSES = (
    "aldehyde_origin_branched",
    "isocyanide_origin_branched",
    "both_tail_origins_branched",
)
EXPECTED_SCOPE = {
    "read_only_distribution_sampling": True,
    "full_corpus_role_state_support": True,
    "branch_exploration_only": True,
    "biological_applicability_policy_changed": False,
    "promoted_applicability_proposal_changed": False,
    "terminal_outcomes_consumed": False,
    "generator_calls": 0,
    "oracle_calls": 0,
    "route_calls": 0,
    "synthesis_calls": 0,
    "candidate_selection": False,
    "sealed_holdout_access": False,
}
EXPECTED_INPUTS = {"all_fold_program_prior", "previous_complete_support_result"}
EXPECTED_IMPLEMENTATION = {"runner", "source", "tests"}


class UgiFullCorpusBranchExplorationScheduleError(RuntimeError):
    """Raised when the expanded support or exploration contract changes."""


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiFullCorpusBranchExplorationScheduleError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiFullCorpusBranchExplorationScheduleError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiFullCorpusBranchExplorationScheduleError(f"malformed pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiFullCorpusBranchExplorationScheduleError(
            f"pin escapes repository: {label}"
        ) from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiFullCorpusBranchExplorationScheduleError(f"pin changed: {label}")
    return path


def _gzip_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as handle:
        for row in rows:
            handle.write((_stable_json(row) + "\n").encode())
    return output.getvalue()


def _largest_remainder_quotas(masses: Mapping[str, float], *, total_draws: int) -> dict[str, int]:
    """Allocate integer stratum counts without changing their relative target mass."""

    if set(masses) != set(BRANCH_CLASSES) or total_draws < len(BRANCH_CLASSES):
        raise UgiFullCorpusBranchExplorationScheduleError("invalid branch-stratum quota request")
    total_mass = float(sum(masses.values()))
    if total_mass <= 0.0 or any(
        not np.isfinite(value) or value <= 0.0 for value in masses.values()
    ):
        raise UgiFullCorpusBranchExplorationScheduleError("branch strata lack probability mass")
    exact = {key: total_draws * float(masses[key]) / total_mass for key in BRANCH_CLASSES}
    quotas = {key: int(np.floor(exact[key])) for key in BRANCH_CLASSES}
    remaining = total_draws - sum(quotas.values())
    order = sorted(BRANCH_CLASSES, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    if sum(quotas.values()) != total_draws or any(value < 1 for value in quotas.values()):
        raise UgiFullCorpusBranchExplorationScheduleError("branch quotas did not close")
    return quotas


def _node_count_coverage(records: Sequence[Mapping[str, Any]], *, role_index: int) -> list[int]:
    return sorted(
        {
            int(row["program"]["node_counts"][role_index])
            for row in records
            if int(row["program"]["junction_budgets"][role_index]) > 0
        }
    )


def build_full_corpus_branch_exploration_schedule(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    """Build the expanded support ledger and branch-conditioned finite schedule."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="branch exploration config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiFullCorpusBranchExplorationScheduleError("unsupported config schema")
    if config.get("status") != "frozen_before_branch_exploration_schedule":
        raise UgiFullCorpusBranchExplorationScheduleError("branch schedule is not frozen")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiFullCorpusBranchExplorationScheduleError("branch exploration scope changed")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != EXPECTED_INPUTS:
        raise UgiFullCorpusBranchExplorationScheduleError("branch exploration inputs changed")
    raw_implementation = config.get("implementation")
    if (
        not isinstance(raw_implementation, Mapping)
        or set(raw_implementation) != EXPECTED_IMPLEMENTATION
    ):
        raise UgiFullCorpusBranchExplorationScheduleError("implementation pins changed")
    paths = {label: _pin(repo, record, label=label) for label, record in raw_inputs.items()}
    implementation_paths = {
        label: _pin(repo, record, label=f"implementation.{label}")
        for label, record in raw_implementation.items()
    }

    prior_payload = _load_json(paths["all_fold_program_prior"], label="all-fold program prior")
    if (
        prior_payload.get("population") != "all_fold_production_refit"
        or prior_payload.get("products") != 112386
        or prior_payload.get("support_audit", {}).get("status") != "pass"
    ):
        raise UgiFullCorpusBranchExplorationScheduleError("all-fold program prior changed")
    previous = _load_json(
        paths["previous_complete_support_result"], label="previous complete support"
    )
    if previous.get("support", {}).get("programs") != 57190:
        raise UgiFullCorpusBranchExplorationScheduleError("previous support denominator changed")

    prior, bounds = load_program_prior(paths["all_fold_program_prior"])
    expected = config.get("support", {})
    role_counts = [len(states) for states in prior.support_by_role]
    if role_counts != list(expected.get("role_state_counts", ())):
        raise UgiFullCorpusBranchExplorationScheduleError("full-corpus role support changed")
    records = [dict(row) for row in enumerate_complete_program_support(prior, bounds)]
    if len(records) != int(expected.get("cartesian_programs", -1)):
        raise UgiFullCorpusBranchExplorationScheduleError("full program support changed")

    by_class: dict[str, list[dict[str, Any]]] = {key: [] for key in BRANCH_CLASSES}
    class_mass: Counter[str] = Counter()
    for row in records:
        current_class = branch_class(row["program"])
        eligible = current_class in by_class
        row["schema_version"] = SUPPORT_LEDGER_SCHEMA_VERSION
        row["branch_class"] = current_class
        row["eligible_for_branch_exploration"] = eligible
        if eligible:
            by_class[current_class].append(row)
            class_mass[current_class] += float(row["broad_prior_probability"])
    branch_mass = float(sum(class_mass.values()))
    if not 0.0 < branch_mass < 1.0 or any(not by_class[key] for key in BRANCH_CLASSES):
        raise UgiFullCorpusBranchExplorationScheduleError("branch-conditioned support is empty")

    design = config.get("design", {})
    if set(design) != {"draws", "seed", "sampling_with_replacement", "stratification"}:
        raise UgiFullCorpusBranchExplorationScheduleError("branch design changed")
    draws = int(design["draws"])
    seed = int(design["seed"])
    if (
        draws != 4096
        or seed < 0
        or design["sampling_with_replacement"] is not True
        or design["stratification"] != "largest_remainder_of_broad_prior_branch_class_mass"
    ):
        raise UgiFullCorpusBranchExplorationScheduleError("unsupported branch design")
    quotas = _largest_remainder_quotas(class_mass, total_draws=draws)
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    for current_class in BRANCH_CLASSES:
        population = by_class[current_class]
        probabilities = np.asarray(
            [float(row["broad_prior_probability"]) for row in population], dtype=np.float64
        )
        probabilities /= probabilities.sum()
        indices = rng.choice(
            len(population), size=quotas[current_class], replace=True, p=probabilities
        )
        stratum_draw_fraction = float(quotas[current_class]) / draws
        for stratum_index, population_index in enumerate(indices.tolist()):
            source = population[int(population_index)]
            broad_probability = float(source["broad_prior_probability"])
            conditional_probability = broad_probability / branch_mass
            schedule_probability = (
                stratum_draw_fraction * broad_probability / float(class_mass[current_class])
            )
            selected.append(
                {
                    "branch_class": current_class,
                    "stratum_draw_index": stratum_index,
                    "program_sha256": source["program_sha256"],
                    "program": source["program"],
                    "broad_prior_probability": broad_probability,
                    "branch_conditional_probability": conditional_probability,
                    "stratified_schedule_probability": schedule_probability,
                    "importance_ratio_conditional_over_schedule": (
                        conditional_probability / schedule_probability
                    ),
                }
            )
    permutation = rng.permutation(len(selected))
    schedule_records = []
    for draw_index, source_index in enumerate(permutation.tolist()):
        schedule_records.append({"draw_index": draw_index, **selected[int(source_index)]})

    support_aldehyde_sizes = _node_count_coverage(records, role_index=1)
    support_isocyanide_sizes = _node_count_coverage(records, role_index=2)
    draw_aldehyde_sizes = _node_count_coverage(schedule_records, role_index=1)
    draw_isocyanide_sizes = _node_count_coverage(schedule_records, role_index=2)
    required_reference_state = (16, 1, 0, 1)
    reference_state_draws = sum(
        tuple(
            int(row["program"][field][1])
            for field in (
                "node_counts",
                "junction_budgets",
                "cycle_ranks",
                "attachment_counts",
            )
        )
        == required_reference_state
        for row in schedule_records
    )
    if (
        draw_aldehyde_sizes != support_aldehyde_sizes
        or draw_isocyanide_sizes != support_isocyanide_sizes
        or reference_state_draws < 1
    ):
        raise UgiFullCorpusBranchExplorationScheduleError(
            "finite branch schedule failed the frozen morphology coverage gate"
        )

    schedule_content = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "status": "frozen_before_branch_exploration_generation",
        "design": {
            **dict(design),
            "branch_classes": list(BRANCH_CLASSES),
            "branch_class_quotas": dict(quotas),
            "conditioning": "at_least_one_tail_origin_has_positive_junction_budget",
            "record_order": "seeded_shuffle_after_within_stratum_sampling",
        },
        "population": {
            "full_corpus_programs": len(records),
            "branch_eligible_programs": sum(len(by_class[key]) for key in BRANCH_CLASSES),
            "branch_conditional_broad_prior_mass": branch_mass,
            "program_prior_population": "all_fold_production_refit",
        },
        "records": schedule_records,
    }
    schedule = {**schedule_content, "schedule_sha256": _logical_sha256(schedule_content)}
    support_ledger = _gzip_jsonl(records)
    pins = {
        label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
        for label, path in sorted(paths.items())
    }
    implementation_pins = {
        label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
        for label, path in sorted(implementation_paths.items())
    }
    pins["config"] = {
        "path": str(config_path.relative_to(repo)),
        "sha256": sha256_file(config_path),
    }
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "full_corpus_morphology_support_and_branch_schedule_frozen",
        "scope": dict(EXPECTED_SCOPE),
        "inputs": pins,
        "implementation": implementation_pins,
        "support": {
            "definition": "cartesian_product_of_all_fold_production_refit_role_states",
            "role_state_counts": role_counts,
            "programs": len(records),
            "previous_programs": int(previous["support"]["programs"]),
            "program_support_expansion_factor": len(records) / int(previous["support"]["programs"]),
            "maximum_combined_exterior_atoms": max(
                sum(row["program"]["node_counts"]) for row in records
            ),
            "maximum_total_atoms_bound": int(bounds["maximum_total_atoms"]),
            "branch_class_program_counts": {key: len(by_class[key]) for key in BRANCH_CLASSES},
            "branch_class_broad_prior_mass": {
                key: float(class_mass[key]) for key in BRANCH_CLASSES
            },
            "linear_tail_origins_broad_prior_mass": 1.0 - branch_mass,
            "aldehyde_branch_node_count_support": support_aldehyde_sizes,
            "isocyanide_branch_node_count_support": support_isocyanide_sizes,
            "required_aldehyde_reference_state": list(required_reference_state),
        },
        "schedule": {
            "draws": draws,
            "seed": seed,
            "branch_class_draw_counts": dict(
                sorted(Counter(row["branch_class"] for row in schedule_records).items())
            ),
            "unique_programs": len({row["program_sha256"] for row in schedule_records}),
            "aldehyde_branch_node_counts_drawn": draw_aldehyde_sizes,
            "isocyanide_branch_node_counts_drawn": draw_isocyanide_sizes,
            "required_aldehyde_reference_state_draws": reference_state_draws,
            "minimum_importance_ratio_conditional_over_schedule": min(
                row["importance_ratio_conditional_over_schedule"] for row in schedule_records
            ),
            "maximum_importance_ratio_conditional_over_schedule": max(
                row["importance_ratio_conditional_over_schedule"] for row in schedule_records
            ),
        },
        "artifacts": {
            "support_ledger.jsonl.gz": {
                "schema_version": SUPPORT_LEDGER_SCHEMA_VERSION,
                "rows": len(records),
                "sha256": sha256_bytes(support_ledger),
                "logical_sha256": _logical_sha256(records),
            },
            "schedule.json": {
                "schema_version": SCHEDULE_SCHEMA_VERSION,
                "draw_records": len(schedule_records),
                "schedule_sha256": schedule["schedule_sha256"],
            },
        },
        "decision": {
            "full_corpus_generator_support_restored_for_branch_exploration": True,
            "generator_retraining_required": False,
            "biological_applicability_thresholds_relaxed": False,
            "promoted_applicability_proposal_extrapolated_to_new_states": False,
            "previous_branching_shortlist_superseded": True,
            "next_gate": "fresh_stochastic_generation_then_realized_branch_chemotype_audit",
        },
        "nonclaims": [
            "A positive junction budget does not guarantee a carbon branch after chemistry decoding.",
            "The branch exploration schedule is not a potency-guided or applicability-guided proposal.",
            "The expanded full-corpus support does not retroactively alter the frozen causal potency comparison.",
            "Schedule inclusion is not evidence of route closure or experimental activity.",
        ],
    }
    result = {**content, "result_sha256": _logical_sha256(content)}
    return result, support_ledger, schedule


__all__ = [
    "BRANCH_CLASSES",
    "SCHEDULE_SCHEMA_VERSION",
    "UgiFullCorpusBranchExplorationScheduleError",
    "build_full_corpus_branch_exploration_schedule",
]
