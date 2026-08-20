"""Qualify grouped SMC schedules with nontrivial within-program ancestry.

The joint sparse sampler forbids ancestry across different morphology programs.
Therefore a nominal 64-particle run is not a meaningful SMC experiment if all
64 particles have unique programs.  This module freezes 16 morphology programs
per seed and four stochastic chemistry particles per program.  It is a
nonexecuting schedule contract: it does not generate molecules, evaluate routes,
select candidates, use biology, or access a holdout.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.data.r1_prime_audit import sha256_file
from forge.design.flow.ugi_morphology_program import UgiMorphologyProgram
from forge.design.flow.ugi_restartable_terminal_support_adapter import (
    canonical_morphology_program_bytes,
)
from forge.design.guidance.ugi_synthesis_guidance import keyed_random_seed

CONFIG_SCHEMA_VERSION = "phase1_ugi_grouped_smc_schedule_qualification_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_grouped_smc_schedule_qualification.v1"
_EXPECTED_ALLOCATION = {
    "linear_tail_origins": 8,
    "isocyanide_origin_branched": 6,
    "aldehyde_origin_branched": 1,
    "both_tail_origins_branched": 1,
}
_EXPECTED_POLICY = {
    "programs_per_seed": 16,
    "particles_per_program": 4,
    "particles_per_seed": 64,
    "program_selection": "sha256_keyed_without_replacement_within_frozen_branch_strata",
    "ancestry_group_key": ["seed", "program_index"],
    "ancestry_crosses_programs": False,
    "candidate_selection": False,
    "nonzero_guidance_execution": False,
    "biology_used": False,
    "sealed_holdout_accessed": False,
}


class UgiGroupedSMCScheduleQualificationError(RuntimeError):
    """Raised when the grouped SMC schedule cannot be frozen exactly."""


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise UgiGroupedSMCScheduleQualificationError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiGroupedSMCScheduleQualificationError(f"{label} must be an object")
    return value


def _pin(repo: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise UgiGroupedSMCScheduleQualificationError(f"{label} pin is malformed")
    path = (repo / str(value.get("path"))).resolve()
    try:
        relative = path.relative_to(repo)
    except ValueError as error:
        raise UgiGroupedSMCScheduleQualificationError(f"{label} escapes repository") from error
    lowered = "/".join(relative.parts).lower()
    if "holdout" in lowered or "sealed" in lowered:
        raise UgiGroupedSMCScheduleQualificationError(f"{label} is forbidden")
    if sha256_file(path) != value.get("sha256"):
        raise UgiGroupedSMCScheduleQualificationError(f"{label} hash changed")
    return path


def _program(value: Any) -> UgiMorphologyProgram:
    if not isinstance(value, dict) or set(value) != {
        "node_counts",
        "junction_budgets",
        "cycle_ranks",
        "attachment_counts",
    }:
        raise UgiGroupedSMCScheduleQualificationError("morphology program is malformed")
    return UgiMorphologyProgram(
        node_counts=tuple(value["node_counts"]),
        junction_budgets=tuple(value["junction_budgets"]),
        cycle_ranks=tuple(value["cycle_ranks"]),
        attachment_counts=tuple(value["attachment_counts"]),
    )


def _program_dict(value: UgiMorphologyProgram) -> dict[str, list[int]]:
    return {
        "node_counts": list(value.node_counts),
        "junction_budgets": list(value.junction_budgets),
        "cycle_ranks": list(value.cycle_ranks),
        "attachment_counts": list(value.attachment_counts),
    }


def _rank(seed: int, branch_class: str, sample: Mapping[str, Any]) -> str:
    return _sha256_payload(
        {
            "seed": seed,
            "branch_class": branch_class,
            "product_id": sample.get("product_id"),
            "program": sample.get("program"),
        }
    )


def _select_programs(
    samples: list[Any],
    *,
    seed: int,
    excluded_programs: set[bytes],
) -> tuple[tuple[str, str, UgiMorphologyProgram], ...]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if not isinstance(sample, dict):
            raise UgiGroupedSMCScheduleQualificationError("program sample is malformed")
        branch_class = sample.get("branch_class")
        product_id = sample.get("product_id")
        if branch_class not in _EXPECTED_ALLOCATION or not isinstance(product_id, str):
            raise UgiGroupedSMCScheduleQualificationError("program sample identity is malformed")
        grouped[branch_class].append(sample)
    selected: list[tuple[str, str, UgiMorphologyProgram]] = []
    seen_programs: set[bytes] = set(excluded_programs)
    for branch_class, count in _EXPECTED_ALLOCATION.items():
        for sample in sorted(
            grouped[branch_class], key=lambda item: _rank(seed, branch_class, item)
        ):
            program = _program(sample.get("program"))
            canonical = canonical_morphology_program_bytes(program)
            if canonical in seen_programs:
                continue
            seen_programs.add(canonical)
            selected.append((branch_class, sample["product_id"], program))
            if sum(item[0] == branch_class for item in selected) == count:
                break
        if sum(item[0] == branch_class for item in selected) != count:
            raise UgiGroupedSMCScheduleQualificationError(
                f"branch stratum {branch_class} lacks enough unique programs"
            )
    if len(selected) != 16 or len(seen_programs) != len(excluded_programs) + 16:
        raise UgiGroupedSMCScheduleQualificationError(
            "each seed must retain 16 unique morphology programs"
        )
    return tuple(selected)


def build_grouped_smc_schedule_qualification(
    repo: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Return the authenticated, nonexecuting grouped-particle schedule receipt."""

    repo = repo.resolve()
    config = _load(config_path, label="grouped SMC schedule config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiGroupedSMCScheduleQualificationError("unsupported config schema")
    if config.get("policy") != _EXPECTED_POLICY:
        raise UgiGroupedSMCScheduleQualificationError("grouped SMC policy changed")
    inputs = config.get("inputs")
    expected_inputs = {
        "program_draw",
        "matched_preregistration",
        "restartable_sampling",
        "joint_sparse_sampling",
        "synthesis_guidance",
        "route_completion_utility_qualification",
        "current_source_zero_guidance_requalification",
        "builder_source",
        "builder_runner",
        "builder_tests",
    }
    if not isinstance(inputs, dict) or set(inputs) != expected_inputs:
        raise UgiGroupedSMCScheduleQualificationError("input set changed")
    paths = {label: _pin(repo, record, label=label) for label, record in inputs.items()}
    preregistration = _load(paths["matched_preregistration"], label="matched preregistration")
    design = preregistration.get("design")
    if not isinstance(design, dict) or design.get("particles_per_seed") != 64:
        raise UgiGroupedSMCScheduleQualificationError(
            "matched preregistration does not retain 64 particles per seed"
        )
    seeds = tuple(config.get("seeds", ()))
    expected_seeds = tuple(design.get("calibration_seeds", ())) + tuple(
        design.get("evaluation_seeds", ())
    )
    if seeds != expected_seeds or len(seeds) != len(set(seeds)):
        raise UgiGroupedSMCScheduleQualificationError(
            "schedule seeds must exactly match calibration then evaluation seeds"
        )
    draw = _load(paths["program_draw"], label="program draw")
    samples = draw.get("samples")
    if not isinstance(samples, list) or len(samples) != 1024:
        raise UgiGroupedSMCScheduleQualificationError("program draw is incomplete")

    seed_schedules: list[dict[str, Any]] = []
    all_particle_seeds: set[int] = set()
    all_programs: set[bytes] = set()
    for seed in seeds:
        selected = _select_programs(
            samples,
            seed=seed,
            excluded_programs=all_programs,
        )
        program_records: list[dict[str, Any]] = []
        particles: list[dict[str, Any]] = []
        for program_index, (branch_class, product_id, program) in enumerate(selected):
            canonical = canonical_morphology_program_bytes(program)
            if canonical in all_programs:
                raise UgiGroupedSMCScheduleQualificationError(
                    "morphology programs must be disjoint across frozen seeds"
                )
            all_programs.add(canonical)
            program_sha256 = hashlib.sha256(canonical).hexdigest()
            program_records.append(
                {
                    "program_index": program_index,
                    "source_product_id": product_id,
                    "branch_class": branch_class,
                    "program": _program_dict(program),
                    "morphology_program_sha256": program_sha256,
                }
            )
            for within_program_particle_index in range(4):
                particle_seed = keyed_random_seed(
                    seed,
                    schedule="phase1_ugi_grouped_smc_schedule_v1",
                    program_index=program_index,
                    particle_index=within_program_particle_index,
                )
                if particle_seed in all_particle_seeds:
                    raise UgiGroupedSMCScheduleQualificationError(
                        "particle seed collision detected"
                    )
                all_particle_seeds.add(particle_seed)
                particles.append(
                    {
                        "global_particle_index": program_index * 4 + within_program_particle_index,
                        "program_index": program_index,
                        "within_program_particle_index": within_program_particle_index,
                        "morphology_program_sha256": program_sha256,
                        "stochastic_particle_seed": particle_seed,
                        "ancestry_group": {
                            "seed": seed,
                            "program_index": program_index,
                        },
                    }
                )
        counts = Counter(item["branch_class"] for item in program_records)
        group_sizes = Counter(item["program_index"] for item in particles)
        if (
            len(program_records) != 16
            or len(particles) != 64
            or dict(counts) != _EXPECTED_ALLOCATION
            or set(group_sizes.values()) != {4}
            or len({item["morphology_program_sha256"] for item in particles}) != 16
        ):
            raise UgiGroupedSMCScheduleQualificationError(
                "grouped schedule does not contain 16 unique programs times four particles"
            )
        schedule_content = {
            "seed": seed,
            "programs": program_records,
            "particles": particles,
        }
        seed_schedules.append(
            {**schedule_content, "schedule_sha256": _sha256_payload(schedule_content)}
        )

    receipt = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "grouped_smc_schedule_qualified_nonexecuting",
        "config": {
            "path": str(config_path.resolve().relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "design": {
            "programs_per_seed": 16,
            "particles_per_program": 4,
            "particles_per_seed": 64,
            "branch_stratum_program_allocation": _EXPECTED_ALLOCATION,
            "ancestry_group_key": ["seed", "program_index"],
            "within_program_ancestry_has_four_choices": True,
            "cross_program_ancestry_forbidden": True,
            "unique_stochastic_particle_seeds": len(all_particle_seeds),
            "unique_morphology_programs_across_seeds": len(all_programs),
            "morphology_program_overlap_across_seeds": 0,
        },
        "integration_contract": {
            "sampler_program_batch_order": (
                "program_index_major_then_within_program_particle_index"
            ),
            "sampler_receives_each_program_four_times": True,
            "ancestry_selection_invoked_separately_per_program_group": True,
            "postselection_group_key": ["seed", "program_index"],
            "particle_index_must_not_be_reinterpreted_as_program_index": True,
            "sixty_four_unique_programs_forbidden": True,
        },
        "seed_schedules": seed_schedules,
        "anti_collapse_safeguards_inherited": preregistration.get("safeguards"),
        "interpretation": {
            "this_receipt_executes_guidance": False,
            "this_receipt_authorizes_guidance": False,
            "schedule_makes_nontrivial_within_program_ancestry_possible": True,
            "schedule_does_not_guarantee_nonuniform_weights": True,
            "grouped_lambda_zero_bitwise_run_qualified": False,
            "remaining_blocker": (
                "selected-model 64-particle grouped lambda=0 execution must be bitwise "
                "identical through closures, exact L1, and all three synthesis checkpoints"
            ),
        },
        "scope": {
            "candidate_selection": False,
            "biology_used": False,
            "sealed_holdout_accessed": False,
            "success_probability": None,
        },
    }
    return {**receipt, "result_sha256": _sha256_payload(receipt)}


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "UgiGroupedSMCScheduleQualificationError",
    "build_grouped_smc_schedule_qualification",
]
