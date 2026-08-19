"""Freeze a bounded whole-program Ugi schedule without guidance.

The candidate pool is the deduplicated union of measured exact morphology
anchors and complete exact/local tuples from the already-frozen valid 1,024
program draw.  Programs are accepted, weighted and selected only as complete
tuples.  This module never constructs precursor-role marginals, reads biology,
advances a generator, or computes a synthesis value.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.data.r1_prime_audit import sha256_file
from forge.product.ugi_joint_program_support_audit import (
    JointProgramKernel,
    build_joint_program_kernel,
    joint_program_affinities,
    joint_program_distance,
    joint_program_neighborhood,
)
from forge.product.ugi_morphology_program import (
    UgiMorphologyProgram,
    attached_program_feasible,
)
from forge.product.ugi_nonzero_guidance_runner import (
    GROUPED_SMC_SCHEDULE_SCHEMA_VERSION,
    FrozenSeedProgramAssignment,
    GroupedSMCScheduleQualification,
)
from forge.product.ugi_restartable_terminal_support_adapter import (
    canonical_morphology_program_bytes,
)
from forge.product.ugi_synthesis_guidance import keyed_random_seed

CONFIG_SCHEMA_VERSION = "phase1_ugi_joint_kernel_bounded_schedule_config.v1"
RESULT_SCHEMA_VERSION = GROUPED_SMC_SCHEDULE_SCHEMA_VERSION

ROLE_ORDER = (
    "amine_head",
    "oxoester_aldehyde_body_tail",
    "isocyanide_tail",
)
NEIGHBORHOOD_ORDER = ("exact_anchor", "local_smoothed_neighborhood")
EXPECTED_SEEDS = tuple(range(20260821, 20260829))
EXPECTED_POLICY = {
    "programs_per_seed": 16,
    "particles_per_program": 4,
    "whole_program_sampling": True,
    "role_marginal_recombination": False,
    "program_selection": "deterministic_kernel_weighted_without_replacement",
    "program_overlap_across_seeds": False,
    "seed_search_retry_or_exclusion_allowed": False,
    "all_three_component_graphs_generated": True,
    "component_ids_enter_program_state": False,
    "biological_targets_read": False,
    "oracle_calls": 0,
    "synthesis_calls": 0,
    "proposal_calls": 0,
    "generator_trajectories_advanced": False,
    "nonzero_guidance": False,
    "candidate_selection": False,
    "prospective_candidate_lock": False,
}
EXPECTED_INPUTS = {
    "frozen_valid_program_draw",
    "joint_program_audit",
    "joint_program_ledger",
    "model_config",
    "prior_grouped_schedule",
    "runner",
    "source",
    "tests",
}


class UgiJointKernelBoundedScheduleError(RuntimeError):
    """Raised when the bounded whole-program schedule changes contract."""


@dataclass(frozen=True)
class ProgramCandidate:
    """One exact or calibrated-local complete morphology program."""

    program: UgiMorphologyProgram
    neighborhood: str
    joint_distance: float
    joint_affinity: float
    sources: tuple[str, ...]

    @property
    def payload(self) -> bytes:
        return canonical_morphology_program_bytes(self.program)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def total_nodes(self) -> int:
        return sum(self.program.node_counts)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiJointKernelBoundedScheduleError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiJointKernelBoundedScheduleError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiJointKernelBoundedScheduleError(f"{label} pin is malformed")
    path = (repo / str(record["path"])).resolve()
    try:
        relative = path.relative_to(repo)
    except ValueError as error:
        raise UgiJointKernelBoundedScheduleError(f"{label} path escapes repository") from error
    lowered = "/".join(relative.parts).lower()
    if "holdout" in lowered or "sealed" in lowered:
        raise UgiJointKernelBoundedScheduleError(f"{label} is forbidden")
    if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
        raise UgiJointKernelBoundedScheduleError(f"{label} hash changed")
    return path


def _read_gzip_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise UgiJointKernelBoundedScheduleError(f"invalid {label}: {path}") from error
    if not rows:
        raise UgiJointKernelBoundedScheduleError(f"{label} is empty")
    return rows


def _program_mapping(program: UgiMorphologyProgram) -> dict[str, list[int]]:
    return {
        "node_counts": list(program.node_counts),
        "junction_budgets": list(program.junction_budgets),
        "cycle_ranks": list(program.cycle_ranks),
        "attachment_counts": list(program.attachment_counts),
    }


def _program_from_mapping(value: Any) -> UgiMorphologyProgram:
    required = {
        "node_counts",
        "junction_budgets",
        "cycle_ranks",
        "attachment_counts",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise UgiJointKernelBoundedScheduleError("morphology program is malformed")
    try:
        program = UgiMorphologyProgram(
            node_counts=tuple(int(item) for item in value["node_counts"]),
            junction_budgets=tuple(int(item) for item in value["junction_budgets"]),
            cycle_ranks=tuple(int(item) for item in value["cycle_ranks"]),
            attachment_counts=tuple(int(item) for item in value["attachment_counts"]),
        )
    except (TypeError, ValueError) as error:
        raise UgiJointKernelBoundedScheduleError("morphology program values are invalid") from error
    if any(len(values) != 3 for values in _program_mapping(program).values()):
        raise UgiJointKernelBoundedScheduleError("morphology program role count changed")
    return program


def _program_from_ledger(row: Mapping[str, str]) -> tuple[UgiMorphologyProgram, int]:
    try:
        program = _program_from_mapping(
            {
                "node_counts": json.loads(row["node_counts_json"]),
                "junction_budgets": json.loads(row["junction_budgets_json"]),
                "cycle_ranks": json.loads(row["cycle_ranks_json"]),
                "attachment_counts": json.loads(row["attachment_counts_json"]),
            }
        )
        occurrences = int(row["occurrences"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise UgiJointKernelBoundedScheduleError("joint-program ledger row is malformed") from error
    if occurrences < 1 or hashlib.sha256(
        canonical_morphology_program_bytes(program)
    ).hexdigest() != row.get("program_sha256"):
        raise UgiJointKernelBoundedScheduleError("joint-program ledger identity changed")
    return program, occurrences


def _validate_model_bounds(model_config: Mapping[str, Any]) -> dict[str, int]:
    model = model_config.get("model")
    required = {
        "maximum_children",
        "maximum_component_atoms",
        "maximum_total_atoms",
        "maximum_junction_budget",
        "maximum_cycle_rank",
        "maximum_attachment_count",
    }
    if not isinstance(model, Mapping) or not required <= set(model):
        raise UgiJointKernelBoundedScheduleError("model support bounds are incomplete")
    output = {name: int(model[name]) for name in required}
    if any(value < 1 for name, value in output.items() if name != "maximum_cycle_rank"):
        raise UgiJointKernelBoundedScheduleError("model support bound is invalid")
    if output["maximum_cycle_rank"] < 0:
        raise UgiJointKernelBoundedScheduleError("cycle-rank support bound is invalid")
    return output


def _program_feasible(program: UgiMorphologyProgram, bounds: Mapping[str, int]) -> bool:
    if (
        any(value < 1 or value > bounds["maximum_component_atoms"] for value in program.node_counts)
        or sum(program.node_counts) > bounds["maximum_total_atoms"]
        or any(
            value < 0 or value > bounds["maximum_junction_budget"]
            for value in program.junction_budgets
        )
        or any(value < 0 or value > bounds["maximum_cycle_rank"] for value in program.cycle_ranks)
        or any(
            value < 1 or value > bounds["maximum_attachment_count"]
            for value in program.attachment_counts
        )
    ):
        return False
    for node_count, junctions, cycles, attachments in zip(
        program.node_counts,
        program.junction_budgets,
        program.cycle_ranks,
        program.attachment_counts,
        strict=True,
    ):
        if not attached_program_feasible(
            node_count,
            junctions,
            bounds["maximum_children"],
            attachments,
        ):
            return False
        tree_edges = node_count - attachments
        available_nontree_pairs = node_count * (node_count - 1) // 2 - tree_edges
        if cycles > max(0, available_nontree_pairs):
            return False
    return True


def build_complete_program_pool(
    kernel: JointProgramKernel,
    *,
    anchors: Sequence[UgiMorphologyProgram],
    frozen_draw: Sequence[UgiMorphologyProgram],
    bounds: Mapping[str, int],
) -> tuple[ProgramCandidate, ...]:
    """Build exact anchors plus admitted whole tuples from the frozen draw."""

    by_payload: dict[bytes, UgiMorphologyProgram] = {}
    sources: dict[bytes, set[str]] = {}
    for program in anchors:
        payload = canonical_morphology_program_bytes(program)
        by_payload[payload] = program
        sources.setdefault(payload, set()).add("measured_exact_anchor")
    for program in frozen_draw:
        if not _program_feasible(program, bounds):
            raise UgiJointKernelBoundedScheduleError(
                "frozen valid 1,024-program draw contains an infeasible program"
            )
        neighborhood = joint_program_neighborhood(kernel, program)
        if neighborhood not in NEIGHBORHOOD_ORDER:
            continue
        payload = canonical_morphology_program_bytes(program)
        by_payload[payload] = program
        sources.setdefault(payload, set()).add("frozen_valid_1024_draw")
    payloads = tuple(sorted(by_payload))
    programs = tuple(by_payload[payload] for payload in payloads)
    affinities = joint_program_affinities(kernel, programs)
    output = []
    for payload, program, affinity in zip(payloads, programs, affinities, strict=True):
        if not _program_feasible(program, bounds):
            raise UgiJointKernelBoundedScheduleError("measured anchor violates selected bounds")
        neighborhood = joint_program_neighborhood(kernel, program)
        if neighborhood not in NEIGHBORHOOD_ORDER:
            raise UgiJointKernelBoundedScheduleError(
                "bounded candidate pool contains an outside-neighborhood program"
            )
        output.append(
            ProgramCandidate(
                program=program,
                neighborhood=neighborhood,
                joint_distance=joint_program_distance(kernel, program),
                joint_affinity=float(affinity),
                sources=tuple(sorted(sources[payload])),
            )
        )
    return tuple(output)


def _weighted_priority(seed: int, candidate: ProgramCandidate) -> float:
    digest = hashlib.sha256(
        _stable_json(
            {
                "protocol": "phase1_ugi_joint_kernel_bounded_schedule_v1",
                "seed": seed,
                "program_sha256": candidate.sha256,
            }
        ).encode()
    ).digest()
    integer = int.from_bytes(digest, "big")
    uniform = (integer + 1.0) / (2**256 + 1.0)
    return -math.log(uniform) / candidate.joint_affinity


def _branch_class(program: UgiMorphologyProgram) -> str:
    aldehyde = program.junction_budgets[1] > 0
    isocyanide = program.junction_budgets[2] > 0
    if aldehyde and isocyanide:
        return "both_tail_origins_branched"
    if aldehyde:
        return "aldehyde_origin_branched"
    if isocyanide:
        return "isocyanide_origin_branched"
    return "linear_tail_origins"


def select_bounded_schedule(
    pool: Sequence[ProgramCandidate],
    *,
    seeds: Sequence[int] = EXPECTED_SEEDS,
) -> tuple[dict[str, Any], ...]:
    """Select all eight schedules without quotas, seed search or overlap."""

    selected_payloads: set[bytes] = set()
    particle_seeds: set[int] = set()
    schedules = []
    for seed in seeds:
        ranked = sorted(
            (
                (_weighted_priority(int(seed), candidate), candidate)
                for candidate in pool
                if candidate.payload not in selected_payloads
            ),
            key=lambda item: (item[0], item[1].sha256),
        )
        if len(ranked) < EXPECTED_POLICY["programs_per_seed"]:
            raise UgiJointKernelBoundedScheduleError("candidate pool exhausted across seeds")
        selected = ranked[: EXPECTED_POLICY["programs_per_seed"]]
        programs = []
        particles = []
        for program_index, (priority, candidate) in enumerate(selected):
            if candidate.payload in selected_payloads:
                raise UgiJointKernelBoundedScheduleError("program overlap across seeds")
            selected_payloads.add(candidate.payload)
            programs.append(
                {
                    "program_index": program_index,
                    "source_product_id": f"joint-program-{candidate.sha256[:16]}",
                    "branch_class": _branch_class(candidate.program),
                    "program": _program_mapping(candidate.program),
                    "morphology_program_sha256": candidate.sha256,
                    "bounded_neighborhood": candidate.neighborhood,
                    "total_nodes": candidate.total_nodes,
                    "joint_distance": candidate.joint_distance,
                    "joint_affinity": candidate.joint_affinity,
                    "candidate_sources": list(candidate.sources),
                    "selection_priority": priority,
                }
            )
            for within_index in range(EXPECTED_POLICY["particles_per_program"]):
                particle_seed = keyed_random_seed(
                    int(seed),
                    schedule="phase1_ugi_joint_kernel_bounded_schedule_v1",
                    program_index=program_index,
                    particle_index=within_index,
                )
                if particle_seed in particle_seeds:
                    raise UgiJointKernelBoundedScheduleError("particle seed collision")
                particle_seeds.add(particle_seed)
                particles.append(
                    {
                        "global_particle_index": program_index * 4 + within_index,
                        "program_index": program_index,
                        "within_program_particle_index": within_index,
                        "morphology_program_sha256": candidate.sha256,
                        "stochastic_particle_seed": particle_seed,
                        "ancestry_group": {"seed": int(seed), "program_index": program_index},
                    }
                )
        content = {"seed": int(seed), "programs": programs, "particles": particles}
        schedules.append({**content, "schedule_sha256": _sha256_payload(content)})
    return tuple(schedules)


def _effective_count(values: Sequence[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    return 0.0 if total == 0 else 1.0 / sum((count / total) ** 2 for count in counts.values())


def _validate_loader_compatibility(receipt: Mapping[str, Any]) -> None:
    content = dict(receipt)
    content.pop("result_sha256", None)
    complete = {**content, "result_sha256": _sha256_payload(content)}
    # The public loader authenticates from disk; reconstruct the same typed
    # assignment contract in memory before materialization.
    assignments = []
    for schedule in complete["seed_schedules"]:
        programs = tuple(
            canonical_morphology_program_bytes(record["program"]) for record in schedule["programs"]
        )
        assignments.append(
            FrozenSeedProgramAssignment(
                seed=int(schedule["seed"]),
                schedule_sha256=str(schedule["schedule_sha256"]),
                programs=programs,
                program_sha256s=tuple(
                    str(record["morphology_program_sha256"]) for record in schedule["programs"]
                ),
                stochastic_particle_seeds=tuple(
                    int(record["stochastic_particle_seed"]) for record in schedule["particles"]
                ),
            )
        )
    GroupedSMCScheduleQualification(
        file_sha256="0" * 64,
        result_sha256=str(complete["result_sha256"]),
        assignments=tuple(assignments),
    )


def build_joint_kernel_bounded_schedule(repo: Path, config_path: Path) -> dict[str, Any]:
    """Return one authenticated, nonexecuting eight-seed bounded schedule."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="joint-kernel bounded schedule config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiJointKernelBoundedScheduleError("unsupported bounded schedule config schema")
    if config.get("policy") != EXPECTED_POLICY:
        raise UgiJointKernelBoundedScheduleError("bounded schedule policy changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != EXPECTED_INPUTS:
        raise UgiJointKernelBoundedScheduleError("bounded schedule input set changed")
    paths = {label: _pin(repo, record, label=label) for label, record in inputs.items()}

    audit = _load_json(paths["joint_program_audit"], label="joint-program audit")
    if (
        audit.get("schema_version") != "phase1_ugi_joint_program_support_audit.v2"
        or audit.get("status") != "complete_joint_measured_program_audit_nonzero_blocked"
        or audit.get("adjudication", {}).get("new_joint_schedule_required") is not True
        or audit.get("adjudication", {}).get("nonzero_guidance_authorized") is not False
    ):
        raise UgiJointKernelBoundedScheduleError("joint-program audit is not the frozen precursor")
    ledger_rows = _read_gzip_csv(paths["joint_program_ledger"], label="joint-program ledger")
    anchors_with_counts = [_program_from_ledger(row) for row in ledger_rows]
    anchors = tuple(program for program, _ in anchors_with_counts)
    measured = tuple(program for program, count in anchors_with_counts for _ in range(count))
    if len(anchors) != 336 or len(measured) != 1_100:
        raise UgiJointKernelBoundedScheduleError("measured anchor census changed")
    kernel = build_joint_program_kernel(measured)
    calibration = audit.get("joint_kernel_calibration")
    if not isinstance(calibration, Mapping) or (
        kernel.bandwidth != calibration.get("bandwidth")
        or kernel.local_radius != calibration.get("local_radius")
    ):
        raise UgiJointKernelBoundedScheduleError("joint kernel no longer reproduces its audit")

    draw = _load_json(paths["frozen_valid_program_draw"], label="frozen valid program draw")
    samples = draw.get("samples")
    if (
        draw.get("schema_version") != "phase1_ugi_program_probe.v1"
        or not isinstance(samples, list)
        or len(samples) != 1_024
    ):
        raise UgiJointKernelBoundedScheduleError("frozen valid program draw changed")
    frozen_draw = tuple(_program_from_mapping(sample.get("program")) for sample in samples)
    bounds = _validate_model_bounds(_load_json(paths["model_config"], label="model config"))
    pool = build_complete_program_pool(
        kernel,
        anchors=anchors,
        frozen_draw=frozen_draw,
        bounds=bounds,
    )
    if len(pool) < 128:
        raise UgiJointKernelBoundedScheduleError("bounded program pool is too small")
    schedules = select_bounded_schedule(pool)

    prior_schedule = _load_json(paths["prior_grouped_schedule"], label="prior grouped schedule")
    prior_seeds = tuple(record["seed"] for record in prior_schedule.get("seed_schedules", ()))
    if prior_seeds != EXPECTED_SEEDS:
        raise UgiJointKernelBoundedScheduleError("calibration/evaluation seed structure changed")

    flat_programs = [record for schedule in schedules for record in schedule["programs"]]
    pool_counts = Counter(candidate.neighborhood for candidate in pool)
    selected_counts = Counter(record["bounded_neighborhood"] for record in flat_programs)
    per_seed_counts = {
        str(schedule["seed"]): dict(
            sorted(Counter(row["bounded_neighborhood"] for row in schedule["programs"]).items())
        )
        for schedule in schedules
    }
    branch_counts = Counter(record["branch_class"] for record in flat_programs)
    design = {
        "programs_per_seed": 16,
        "particles_per_program": 4,
        "particles_per_seed": 64,
        "ancestry_group_key": ["seed", "program_index"],
        "within_program_ancestry_has_four_choices": True,
        "cross_program_ancestry_forbidden": True,
        "unique_stochastic_particle_seeds": 512,
        "unique_morphology_programs_across_seeds": 128,
        "morphology_program_overlap_across_seeds": 0,
    }
    receipt: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "grouped_smc_schedule_qualified_nonexecuting",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "design": design,
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
        "bounded_joint_program_contract": {
            "candidate_pool": (
                "deduplicated measured exact anchors plus exact/local complete tuples from "
                "the frozen valid 1,024-program draw"
            ),
            "frozen_draw_programs": 1_024,
            "candidate_pool_programs": len(pool),
            "candidate_pool_by_neighborhood": dict(sorted(pool_counts.items())),
            "selected_by_neighborhood": dict(sorted(selected_counts.items())),
            "selected_by_neighborhood_per_seed": per_seed_counts,
            "selected_empirical_branch_classes": dict(sorted(branch_counts.items())),
            "selection_unit": "complete_joint_morphology_program",
            "selection_weight": "occurrence-weighted_joint_kernel_affinity",
            "role_marginal_recombination": False,
            "schedule_builder_mutation_or_role_composition": False,
            "model_bound_and_attached_tree_feasibility_revalidated": True,
            "all_three_component_graphs_still_generated": True,
            "effective_program_count_inverse_simpson": _effective_count(
                [record["morphology_program_sha256"] for record in flat_programs]
            ),
            "maximum_program_concentration": 1.0 / len(flat_programs),
            "program_overlap_across_seeds": 0,
            "biological_labels_or_oracle_outputs_read": False,
        },
        "seed_schedules": list(schedules),
        "anti_collapse_safeguards_inherited": prior_schedule.get(
            "anti_collapse_safeguards_inherited"
        ),
        "interpretation": {
            "this_receipt_executes_guidance": False,
            "this_receipt_authorizes_guidance": False,
            "schedule_makes_nontrivial_within_program_ancestry_possible": True,
            "schedule_does_not_guarantee_nonuniform_weights": True,
            "grouped_lambda_zero_bitwise_run_qualified": False,
            "historical_old_schedule_manifest_is_not_a_new_schedule_reference": True,
            "remaining_blocker": (
                "all eight new schedules must pass paired selected-v3 lambda-zero identity; "
                "terminal biological applicability is bound separately"
            ),
        },
        "scope": {
            "candidate_selection": False,
            "biology_used": False,
            "oracle_calls": 0,
            "synthesis_calls": 0,
            "proposal_calls": 0,
            "generator_trajectories_advanced": False,
            "nonzero_guidance": False,
            "sealed_holdout_accessed": False,
            "success_probability": None,
        },
    }
    complete = {**receipt, "result_sha256": _sha256_payload(receipt)}
    _validate_loader_compatibility(complete)
    return complete


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "EXPECTED_POLICY",
    "EXPECTED_SEEDS",
    "ProgramCandidate",
    "RESULT_SCHEMA_VERSION",
    "UgiJointKernelBoundedScheduleError",
    "build_complete_program_pool",
    "build_joint_kernel_bounded_schedule",
    "select_bounded_schedule",
]
