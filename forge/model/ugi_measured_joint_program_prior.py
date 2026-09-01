"""Group-balanced joint morphology programs from measured Ugi training products.

The ordinary measured-Ugi layout support gives every eligible product row equal mass.  Product
rows are not exchangeable evidence for morphology because the same component-family triple may be
repeated many more times than another.  This module estimates a joint law that gives each observed
component-family triple equal mass and then gives each eligible product within that group equal
mass.  Family identifiers are estimator metadata only: sampled programs contain four integer
semantics per role and never contain a product, component, graph, SMILES, family or fragment ID.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import sha256_file, sha256_json
from forge.core.io import iter_csv
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.reaction_program_flow import derive_role_morphology_states
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_morphology_program import UgiMorphologyProgram
from forge.potency.annotations import ROLE_NAMES


class UgiMeasuredJointProgramPriorError(ValueError):
    """The measured joint-program estimator violates its frozen training-only contract."""


ProgramKey = tuple[
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
]


def summarize_program_distribution(
    programs: tuple[UgiMorphologyProgram, ...],
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Summarize only the coarse semantics carried by a morphology-program law."""

    if (
        not programs
        or probabilities.shape != (len(programs),)
        or np.any(probabilities < 0)
        or not np.isfinite(probabilities).all()
        or not np.isclose(probabilities.sum(), 1.0)
    ):
        raise UgiMeasuredJointProgramPriorError("program summary distribution is invalid")
    node_counts = np.asarray([program.node_counts for program in programs], dtype=np.float64)
    junctions = np.asarray([program.junction_budgets for program in programs], dtype=np.float64)
    cycles = np.asarray([program.cycle_ranks for program in programs], dtype=np.float64)
    attachments = np.asarray([program.attachment_counts for program in programs], dtype=np.float64)
    aldehyde_index = ROLE_NAMES.index("oxoester_aldehyde_body_tail")
    isocyanide_index = ROLE_NAMES.index("isocyanide_tail")

    def expected(values: np.ndarray) -> float:
        return float(np.dot(probabilities, values))

    def role_rows(values: np.ndarray) -> dict[str, float]:
        return {role: expected(values[:, index]) for index, role in enumerate(ROLE_NAMES)}

    tail_total = node_counts[:, aldehyde_index] + node_counts[:, isocyanide_index]
    tail_asymmetry = np.abs(node_counts[:, aldehyde_index] - node_counts[:, isocyanide_index])
    return {
        "expected_node_count_by_role": role_rows(node_counts),
        "expected_junction_budget_by_role": role_rows(junctions),
        "expected_cycle_rank_by_role": role_rows(cycles),
        "expected_attachment_count_by_role": role_rows(attachments),
        "expected_two_tail_exterior_node_total": expected(tail_total),
        "expected_two_tail_exterior_node_asymmetry": expected(tail_asymmetry),
    }


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true"}


def _program_key(program: UgiMorphologyProgram) -> ProgramKey:
    return (
        tuple(int(value) for value in program.node_counts),
        tuple(int(value) for value in program.junction_budgets),
        tuple(int(value) for value in program.cycle_ranks),
        tuple(int(value) for value in program.attachment_counts),
    )


def build_group_balanced_program_distribution(
    programs_by_group: Mapping[str, Sequence[UgiMorphologyProgram]],
) -> tuple[tuple[UgiMorphologyProgram, ...], np.ndarray]:
    """Give each nonempty family group equal mass, then each row equal within its group."""

    if not programs_by_group or any(
        not isinstance(group, str) or not group or not rows
        for group, rows in programs_by_group.items()
    ):
        raise UgiMeasuredJointProgramPriorError(
            "group-balanced program support contains an empty group"
        )
    mass: dict[ProgramKey, float] = {}
    group_mass = 1.0 / len(programs_by_group)
    for rows in programs_by_group.values():
        row_mass = group_mass / len(rows)
        for program in rows:
            key = _program_key(program)
            mass[key] = mass.get(key, 0.0) + row_mass
    support = tuple(sorted(mass))
    programs = tuple(
        UgiMorphologyProgram(
            node_counts=key[0],
            junction_budgets=key[1],
            cycle_ranks=key[2],
            attachment_counts=key[3],
        )
        for key in support
    )
    probabilities = np.asarray([mass[key] for key in support], dtype=np.float64)
    probabilities /= probabilities.sum()
    if (
        probabilities.shape != (len(programs),)
        or np.any(probabilities <= 0)
        or not np.isfinite(probabilities).all()
        or not np.isclose(probabilities.sum(), 1.0)
    ):
        raise UgiMeasuredJointProgramPriorError("joint-program probabilities are invalid")
    return programs, probabilities


def program_from_layout_record(
    record: Any,
    *,
    vocabulary: Any,
) -> UgiMorphologyProgram:
    morphology = derive_role_morphology_states(record)
    values_by_role: dict[str, tuple[int, int, int, int]] = {}
    for role in ROLE_NAMES:
        try:
            role_state = int(vocabulary.role_to_index[role])
        except KeyError as error:
            raise UgiMeasuredJointProgramPriorError(
                f"production cache lacks Ugi role {role!r}"
            ) from error
        exterior = (record.role_states == role_state) & (record.core_position_states == 1)
        values = np.unique(morphology[exterior], axis=0)
        if values.shape != (1, 4) or np.any(values[0] < 1):
            raise UgiMeasuredJointProgramPriorError(
                f"measured record has inconsistent {role} morphology"
            )
        values_by_role[role] = tuple(int(value) - 1 for value in values[0])
    columns = tuple(
        tuple(values_by_role[role][column] for role in ROLE_NAMES) for column in range(4)
    )
    return UgiMorphologyProgram(
        node_counts=columns[0],
        junction_budgets=columns[1],
        cycle_ranks=columns[2],
        attachment_counts=columns[3],
    )


def program_admitted_by_ester_policy(
    program: UgiMorphologyProgram,
    policy: UgiEsterChemotypePolicy,
) -> bool:
    values_by_role = {
        role: (
            int(program.node_counts[index]),
            int(program.junction_budgets[index]),
            int(program.cycle_ranks[index]),
            int(program.attachment_counts[index]),
        )
        for index, role in enumerate(ROLE_NAMES)
    }
    if any(
        not policy.minimum_topology_exterior_atoms(role)
        <= values_by_role[role][0]
        <= policy.maximum_exterior_atoms(role)
        for role in ROLE_NAMES
    ):
        return False
    return values_by_role[policy.aldehyde_role][1:] == (1, 0, 1)


@dataclass(frozen=True)
class UgiMeasuredJointProgramPrior:
    """Identity-free categorical support over complete Ugi morphology programs."""

    programs: tuple[UgiMorphologyProgram, ...]
    probabilities: np.ndarray
    audit: Mapping[str, Any]

    @classmethod
    def from_training_data(
        cls,
        *,
        assignments_path: Path,
        cache: SynthesisProgramProductionCache,
        ester_policy: UgiEsterChemotypePolicy,
        reaction_id: str = "ugi_3cr_agile",
        expected_assignments_sha256: str | None = None,
    ) -> UgiMeasuredJointProgramPrior:
        """Compile equal family-group mass from source-adjudicated measured train rows."""

        assignments = assignments_path.resolve()
        observed_assignments_sha256 = str(sha256_file(assignments))
        if (
            expected_assignments_sha256 is not None
            and observed_assignments_sha256 != expected_assignments_sha256
        ):
            raise UgiMeasuredJointProgramPriorError(
                "Ugi assignments changed: expected "
                f"{expected_assignments_sha256}, found {observed_assignments_sha256}"
            )
        required = {
            "product_id",
            "primary_product_fold",
            "is_source_adjudicated_measured_product",
            *(f"{role}_family_id" for role in ROLE_NAMES),
        }
        group_by_product: dict[str, str] = {}
        for row in iter_csv(assignments):
            if not required.issubset(row):
                raise UgiMeasuredJointProgramPriorError("Ugi assignment schema changed")
            if row["primary_product_fold"] != "train" or not _truthy(
                row["is_source_adjudicated_measured_product"]
            ):
                continue
            product_id = str(row["product_id"]).strip()
            group = "|".join(str(row[f"{role}_family_id"]).strip() for role in ROLE_NAMES)
            if not product_id or any(not part for part in group.split("|")):
                raise UgiMeasuredJointProgramPriorError(
                    "measured Ugi training row lacks product or family identity"
                )
            previous = group_by_product.setdefault(product_id, group)
            if previous != group:
                raise UgiMeasuredJointProgramPriorError(
                    "one measured product maps to multiple component-family groups"
                )
        if not group_by_product:
            raise UgiMeasuredJointProgramPriorError(
                "Ugi assignments contain no measured training products"
            )
        if reaction_id != ester_policy.reaction_id:
            raise UgiMeasuredJointProgramPriorError("reaction and ester policy disagree")

        observed: set[str] = set()
        eligible_by_group: dict[str, list[UgiMorphologyProgram]] = {}
        eligible_programs_in_occurrence_order: list[UgiMorphologyProgram] = []
        exclusion_counts: Counter[str] = Counter()
        for raw_index in cache.indices(program_id=reaction_id, fold="train"):
            index = int(raw_index)
            product_id = cache.record_id(index)
            group = group_by_product.get(product_id)
            if group is None:
                continue
            observed.add(product_id)
            program = program_from_layout_record(
                cache.record(index), vocabulary=cache.vocabulary
            )
            if not program_admitted_by_ester_policy(program, ester_policy):
                exclusion_counts["outside_existing_ester_decoder_program_support"] += 1
                continue
            eligible_by_group.setdefault(group, []).append(program)
            eligible_programs_in_occurrence_order.append(program)
        missing = set(group_by_product).difference(observed)
        if missing:
            raise UgiMeasuredJointProgramPriorError(
                f"production cache lacks {len(missing)} measured training products"
            )
        if not eligible_by_group:
            raise UgiMeasuredJointProgramPriorError(
                "no measured training program is eligible for the existing exact decoder"
            )

        eligible_rows = 0
        group_sizes: list[int] = []
        for rows in eligible_by_group.values():
            group_sizes.append(len(rows))
            eligible_rows += len(rows)
        programs, probabilities = build_group_balanced_program_distribution(eligible_by_group)

        all_groups = set(group_by_product.values())
        audit = {
            "schema_version": "forge.ugi_measured_joint_program_prior.v1",
            "reaction_id": reaction_id,
            "training_fold": "train",
            "estimator": (
                "equal component-family-triple mass, equal eligible product mass within group"
            ),
            "measured_training_rows": len(group_by_product),
            "measured_training_groups": len(all_groups),
            "eligible_rows": eligible_rows,
            "eligible_groups": len(eligible_by_group),
            "excluded_rows": len(group_by_product) - eligible_rows,
            "excluded_groups": len(all_groups.difference(eligible_by_group)),
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
            "eligible_group_size_histogram": {
                str(size): count for size, count in sorted(Counter(group_sizes).items())
            },
            "program_support_size": len(programs),
            "program_support_sha256": str(
                sha256_json(
                    [
                        {
                            "program": _program_key(program),
                            "probability": float(probability),
                        }
                        for program, probability in zip(
                            programs, probabilities.tolist(), strict=True
                        )
                    ]
                )
            ),
            "assignments_path": str(assignments),
            "assignments_sha256": observed_assignments_sha256,
            "component_or_product_ids_in_sampled_program": False,
            "stored_component_graphs_in_sampled_program": False,
            "fragment_tokens_in_sampled_program": False,
            "program_distribution_projection": {
                "occurrence_weighted_eligible_rows": summarize_program_distribution(
                    tuple(eligible_programs_in_occurrence_order),
                    np.full(
                        len(eligible_programs_in_occurrence_order),
                        1.0 / len(eligible_programs_in_occurrence_order),
                        dtype=np.float64,
                    ),
                ),
                "group_balanced_eligible_rows": summarize_program_distribution(
                    programs,
                    probabilities,
                ),
            },
        }
        return cls(programs=programs, probabilities=probabilities, audit=audit)

    def sample(
        self,
        *,
        count: int,
        seed: int,
    ) -> tuple[UgiMorphologyProgram, ...]:
        """Draw programs once from the aggregate semantic support."""

        if count < 1 or seed < 0:
            raise UgiMeasuredJointProgramPriorError("sample count or seed is invalid")
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.programs), size=count, replace=True, p=self.probabilities)
        return tuple(self.programs[int(index)] for index in indices)


__all__ = [
    "UgiMeasuredJointProgramPrior",
    "UgiMeasuredJointProgramPriorError",
    "build_group_balanced_program_distribution",
    "program_admitted_by_ester_policy",
    "program_from_layout_record",
    "summarize_program_distribution",
]
