"""Derive a diagnostic-only known-head/generated-tail ablation schedule.

The selected measured amine is projected from the frozen joint training cache.
Only its exact exterior topology and chemistry are clamped.  Tail morphology
programs remain sampled from the previously frozen 1,024-program draw, and
all 128 tail programs are unique after the head fields are replaced.

This module is not the primary FORGE path.  The production design generates
the amine, aldehyde and isocyanide graphs; a fixed-head schedule may be used
only as an explicitly labeled diagnostic ablation.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.data.r1_prime_audit import sha256_file
from forge.potency import ugi_distributional_applicability as applicability
from forge.product.ugi_morphology_program import UgiMorphologyProgram
from forge.product.ugi_restartable_terminal_support_adapter import (
    canonical_morphology_program_bytes,
)
from forge.product.ugi_synthesis_guidance import keyed_random_seed
from forge.product.ugi_training_cache import load_ugi_training_cache

CONFIG_SCHEMA_VERSION = "phase1_ugi_known_head_tail_schedule_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_grouped_smc_schedule_qualification.v1"
KNOWN_HEAD_SPEC_SCHEMA_VERSION = "forge.ugi_known_head_clamp_spec.v1"
EXPECTED_SEEDS = tuple(range(20260821, 20260829))
EXPECTED_ALLOCATION = {
    "linear_tail_origins": 8,
    "isocyanide_origin_branched": 6,
    "aldehyde_origin_branched": 1,
    "both_tail_origins_branched": 1,
}
EXPECTED_POLICY = {
    "diagnostic_ablation_only": True,
    "primary_fully_generated_role_path": False,
    "target_amine_smiles": "CN(C)CCN",
    "programs_per_seed": 16,
    "particles_per_program": 4,
    "particles_per_seed": 64,
    "program_selection": "sha256_keyed_without_replacement_after_exact_head_projection",
    "tail_programs_disjoint_across_seeds": True,
    "candidate_selection": False,
    "biological_guidance": False,
    "nonzero_guidance_execution": False,
    "sealed_holdout_accessed": False,
}
EXPECTED_INPUTS = {
    "curated_agile",
    "program_draw",
    "schedule_source",
    "training_cache",
}


class UgiKnownHeadTailScheduleError(RuntimeError):
    """Raised when the bounded head clamp or tail schedule is not exact."""


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiKnownHeadTailScheduleError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiKnownHeadTailScheduleError(f"{label} must contain a JSON object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiKnownHeadTailScheduleError(f"{label} pin is malformed")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiKnownHeadTailScheduleError(f"{label} path escapes repository") from error
    if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
        raise UgiKnownHeadTailScheduleError(f"{label} hash changed")
    return path


@dataclass(frozen=True)
class KnownHeadClampSpec:
    """Exact core-excluded topology and chemistry of one measured amine."""

    canonical_smiles: str
    offspring: tuple[int, ...]
    atom_states: tuple[int, ...]
    parent_bond_states: tuple[int, ...]
    node_count: int
    junction_budget: int
    cycle_rank: int
    attachment_count: int
    measured_row_count: int
    projected_product_count: int
    source_product_ids_sha256: str
    schema_version: str = KNOWN_HEAD_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.node_count < 1
            or len(self.offspring) != self.node_count
            or len(self.atom_states) != self.node_count
            or len(self.parent_bond_states) != self.node_count
            or self.cycle_rank != 0
            or self.attachment_count < 1
            or self.measured_row_count < 1
            or self.projected_product_count < 1
        ):
            raise UgiKnownHeadTailScheduleError("known-head clamp specification is invalid")

    @property
    def program_fields(self) -> tuple[int, int, int, int]:
        return (
            self.node_count,
            self.junction_budget,
            self.cycle_rank,
            self.attachment_count,
        )

    @property
    def identity_sha256(self) -> str:
        return _sha256_payload(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "canonical_smiles": self.canonical_smiles,
            "offspring": list(self.offspring),
            "atom_states": list(self.atom_states),
            "parent_bond_states": list(self.parent_bond_states),
            "node_count": self.node_count,
            "junction_budget": self.junction_budget,
            "cycle_rank": self.cycle_rank,
            "attachment_count": self.attachment_count,
            "measured_row_count": self.measured_row_count,
            "projected_product_count": self.projected_product_count,
            "source_product_ids_sha256": self.source_product_ids_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> KnownHeadClampSpec:
        if not isinstance(value, Mapping):
            raise UgiKnownHeadTailScheduleError("known-head clamp record is malformed")
        expected = {
            "schema_version",
            "canonical_smiles",
            "offspring",
            "atom_states",
            "parent_bond_states",
            "node_count",
            "junction_budget",
            "cycle_rank",
            "attachment_count",
            "measured_row_count",
            "projected_product_count",
            "source_product_ids_sha256",
        }
        if set(value) != expected or value.get("schema_version") != KNOWN_HEAD_SPEC_SCHEMA_VERSION:
            raise UgiKnownHeadTailScheduleError("known-head clamp fields changed")
        return cls(
            canonical_smiles=str(value["canonical_smiles"]),
            offspring=tuple(int(item) for item in value["offspring"]),
            atom_states=tuple(int(item) for item in value["atom_states"]),
            parent_bond_states=tuple(int(item) for item in value["parent_bond_states"]),
            node_count=int(value["node_count"]),
            junction_budget=int(value["junction_budget"]),
            cycle_rank=int(value["cycle_rank"]),
            attachment_count=int(value["attachment_count"]),
            measured_row_count=int(value["measured_row_count"]),
            projected_product_count=int(value["projected_product_count"]),
            source_product_ids_sha256=str(value["source_product_ids_sha256"]),
        )


def _derive_head_spec(training_cache: Path, curated_agile: Path, target: str) -> KnownHeadClampSpec:
    target = applicability._canonical(target)
    measured = 0
    with gzip.open(curated_agile, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            measured += int(applicability._canonical(row["A_smiles"]) == target)
    corpus, joint_by_fold = load_ugi_training_cache(training_cache)
    signatures: set[tuple[Any, ...]] = set()
    product_ids: list[str] = []
    for fold in corpus.assignments_by_fold:
        for assignment, record in zip(
            corpus.assignments_by_fold[fold], joint_by_fold[fold], strict=True
        ):
            if applicability._canonical(assignment["amine_head_smiles"]) != target:
                continue
            count = record.program.node_counts[0]
            if record.program.cycle_ranks[0] != 0 or any(
                int(anchor) < count for anchor in record.decoration_anchors
            ):
                raise UgiKnownHeadTailScheduleError(
                    "selected head has a closure or terminal decoration"
                )
            signatures.add(
                (
                    tuple(int(value) for value in record.offspring[:count]),
                    tuple(int(value) for value in record.atom_states[:count]),
                    tuple(int(value) for value in record.parent_bond_states[:count]),
                    int(record.program.junction_budgets[0]),
                    int(record.program.cycle_ranks[0]),
                    int(record.program.attachment_counts[0]),
                )
            )
            product_ids.append(record.product_id)
    if measured < 1 or len(signatures) != 1 or not product_ids:
        raise UgiKnownHeadTailScheduleError(
            "selected measured head lacks one exact projected representation"
        )
    offspring, atoms, bonds, junction, cycle, attachment = signatures.pop()
    return KnownHeadClampSpec(
        canonical_smiles=target,
        offspring=offspring,
        atom_states=atoms,
        parent_bond_states=bonds,
        node_count=len(offspring),
        junction_budget=junction,
        cycle_rank=cycle,
        attachment_count=attachment,
        measured_row_count=measured,
        projected_product_count=len(product_ids),
        source_product_ids_sha256=_sha256_payload(sorted(product_ids)),
    )


def _program(value: Any) -> UgiMorphologyProgram:
    if not isinstance(value, Mapping):
        raise UgiKnownHeadTailScheduleError("tail morphology program is malformed")
    return UgiMorphologyProgram(
        node_counts=tuple(int(item) for item in value["node_counts"]),
        junction_budgets=tuple(int(item) for item in value["junction_budgets"]),
        cycle_ranks=tuple(int(item) for item in value["cycle_ranks"]),
        attachment_counts=tuple(int(item) for item in value.get("attachment_counts", (1, 1, 1))),
    )


def _clamp_program(program: UgiMorphologyProgram, spec: KnownHeadClampSpec) -> UgiMorphologyProgram:
    return UgiMorphologyProgram(
        node_counts=(spec.node_count, *program.node_counts[1:]),
        junction_budgets=(spec.junction_budget, *program.junction_budgets[1:]),
        cycle_ranks=(spec.cycle_rank, *program.cycle_ranks[1:]),
        attachment_counts=(spec.attachment_count, *program.attachment_counts[1:]),
    )


def _program_dict(program: UgiMorphologyProgram) -> dict[str, list[int]]:
    return {
        "node_counts": list(program.node_counts),
        "junction_budgets": list(program.junction_budgets),
        "cycle_ranks": list(program.cycle_ranks),
        "attachment_counts": list(program.attachment_counts),
    }


def build_known_head_tail_schedule(repo: Path, config_path: Path) -> dict[str, Any]:
    """Freeze 128 unique tail programs around one exact measured head."""

    repo = repo.resolve()
    config = _load(config_path.resolve(), label="known-head tail schedule config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiKnownHeadTailScheduleError("unsupported known-head schedule config")
    if config.get("policy") != EXPECTED_POLICY or tuple(config.get("seeds", ())) != EXPECTED_SEEDS:
        raise UgiKnownHeadTailScheduleError("known-head schedule policy changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != EXPECTED_INPUTS:
        raise UgiKnownHeadTailScheduleError("known-head schedule input set changed")
    paths = {name: _pin(repo, value, label=name) for name, value in inputs.items()}
    spec = _derive_head_spec(
        paths["training_cache"],
        paths["curated_agile"],
        EXPECTED_POLICY["target_amine_smiles"],
    )
    draw = _load(paths["program_draw"], label="frozen morphology draw")
    samples = draw.get("samples")
    if not isinstance(samples, list) or len(samples) != 1024:
        raise UgiKnownHeadTailScheduleError("frozen morphology draw changed")
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("branch_class") not in EXPECTED_ALLOCATION:
            raise UgiKnownHeadTailScheduleError("frozen morphology sample is malformed")
        grouped[str(sample["branch_class"])].append(sample)

    all_programs: set[bytes] = set()
    all_particle_seeds: set[int] = set()
    seed_schedules = []
    for seed in EXPECTED_SEEDS:
        selected: list[tuple[str, str, UgiMorphologyProgram]] = []
        for branch_class, target_count in EXPECTED_ALLOCATION.items():
            ranked = sorted(
                grouped[branch_class],
                key=lambda sample: _sha256_payload(
                    {
                        "seed": seed,
                        "branch_class": branch_class,
                        "product_id": sample.get("product_id"),
                        "program": sample.get("program"),
                        "head_spec_sha256": spec.identity_sha256,
                    }
                ),
            )
            retained = 0
            for sample in ranked:
                program = _clamp_program(_program(sample.get("program")), spec)
                canonical = canonical_morphology_program_bytes(program)
                if canonical in all_programs or any(
                    canonical == canonical_morphology_program_bytes(item[2]) for item in selected
                ):
                    continue
                selected.append((branch_class, str(sample["product_id"]), program))
                retained += 1
                if retained == target_count:
                    break
            if retained != target_count:
                raise UgiKnownHeadTailScheduleError(
                    f"tail stratum {branch_class} lacks disjoint projected programs"
                )
        if len(selected) != 16:
            raise UgiKnownHeadTailScheduleError("known-head seed lacks 16 tail programs")
        program_records = []
        particles = []
        for program_index, (branch_class, product_id, program) in enumerate(selected):
            canonical = canonical_morphology_program_bytes(program)
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
            for within_index in range(4):
                particle_seed = keyed_random_seed(
                    seed,
                    schedule="phase1_ugi_known_head_generated_tail_schedule_v1",
                    program_index=program_index,
                    particle_index=within_index,
                )
                if particle_seed in all_particle_seeds:
                    raise UgiKnownHeadTailScheduleError("particle-seed collision detected")
                all_particle_seeds.add(particle_seed)
                particles.append(
                    {
                        "global_particle_index": program_index * 4 + within_index,
                        "program_index": program_index,
                        "within_program_particle_index": within_index,
                        "morphology_program_sha256": program_sha256,
                        "stochastic_particle_seed": particle_seed,
                        "ancestry_group": {"seed": seed, "program_index": program_index},
                    }
                )
        content = {"seed": seed, "programs": program_records, "particles": particles}
        seed_schedules.append({**content, "schedule_sha256": _sha256_payload(content)})
    if len(all_programs) != 128 or len(all_particle_seeds) != 512:
        raise UgiKnownHeadTailScheduleError("derived schedule is not globally disjoint")
    branch_counts = Counter(
        record["branch_class"] for schedule in seed_schedules for record in schedule["programs"]
    )
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "grouped_smc_schedule_qualified_nonexecuting",
        "config": {
            "path": str(config_path.resolve().relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "known_head_clamp": {
            "spec": spec.to_dict(),
            "spec_sha256": spec.identity_sha256,
            "head_generated": False,
            "aldehyde_tail_generated": True,
            "isocyanide_tail_generated": True,
        },
        "design": {
            "programs_per_seed": 16,
            "particles_per_program": 4,
            "particles_per_seed": 64,
            "branch_stratum_program_allocation": EXPECTED_ALLOCATION,
            "cross_program_ancestry_forbidden": True,
            "within_program_ancestry_has_four_choices": True,
            "ancestry_group_key": ["seed", "program_index"],
            "unique_morphology_programs_across_seeds": 128,
            "morphology_program_overlap_across_seeds": 0,
            "unique_stochastic_particle_seeds": 512,
            "head_fields_identical_across_programs": True,
            "tail_branch_counts_all_seeds": dict(sorted(branch_counts.items())),
        },
        "integration_contract": {
            "sampler_receives_each_program_four_times": True,
            "ancestry_selection_invoked_separately_per_program_group": True,
            "particle_index_must_not_be_reinterpreted_as_program_index": True,
            "sixty_four_unique_programs_forbidden": True,
            "head_channels_clamped_at_every_flow_step": True,
            "head_terminal_logits_clamped": True,
        },
        "inputs": {
            name: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
        "seed_schedules": seed_schedules,
        "scope": {
            "nonexecuting_schedule_only": True,
            "diagnostic_ablation_only": True,
            "primary_fully_generated_role_path": False,
            "head_generated": False,
            "biological_guidance": False,
            "candidate_selection": False,
            "prospective_candidate_lock": False,
        },
    }
    return {**content, "result_sha256": _sha256_payload(content)}


def load_known_head_spec(schedule_path: Path) -> KnownHeadClampSpec:
    """Load a canonical schedule-owned clamp specification."""

    value = _load(schedule_path, label="known-head schedule")
    content = {key: item for key, item in value.items() if key != "result_sha256"}
    clamp = value.get("known_head_clamp")
    if (
        value.get("schema_version") != RESULT_SCHEMA_VERSION
        or value.get("status") != "grouped_smc_schedule_qualified_nonexecuting"
        or value.get("result_sha256") != _sha256_payload(content)
        or not isinstance(clamp, Mapping)
    ):
        raise UgiKnownHeadTailScheduleError("known-head schedule receipt changed")
    spec = KnownHeadClampSpec.from_dict(clamp.get("spec"))
    if clamp.get("spec_sha256") != spec.identity_sha256:
        raise UgiKnownHeadTailScheduleError("known-head clamp identity changed")
    return spec


__all__ = [
    "KnownHeadClampSpec",
    "UgiKnownHeadTailScheduleError",
    "build_known_head_tail_schedule",
    "load_known_head_spec",
]
