"""Identity-free joint semantics for complete measured-Ugi lipid architecture.

The existing Ugi morphology program fixes only role-local counts, junction budgets, cycle ranks,
and attachment counts.  The amine-semantic extension adds head compactness and composition, but it
does not describe where the aldehyde ester divides its two carbon arms or which measured tail
unsaturation class accompanies the head and tail sizes.  This module estimates that missing joint
law from source-adjudicated measured Ugi products in the training fold.

Family identifiers are used only to balance the estimator.  A sampled target contains integers;
it never contains a product or component identifier, molecular graph, SMILES, or fragment token.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forge.chemistry.descriptors import component_chemotype_metrics
from forge.core.hashing import sha256_file, sha256_json
from forge.core.io import iter_csv
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.ugi_amine_semantic_program import (
    UgiAmineSemanticTarget,
    amine_semantic_target,
)
from forge.model.ugi_ester_chemotype import (
    UgiEsterChemotypePolicy,
    aldehyde_ester_directional_carbon_counts,
    ester_side_carbon_counts,
)
from forge.model.ugi_measured_joint_program_prior import (
    program_admitted_by_ester_policy,
    program_from_layout_record,
)
from forge.model.ugi_morphology_program import UgiMorphologyProgram
from forge.potency.annotations import ROLE_NAMES


class UgiAllRoleSemanticProgramError(ValueError):
    """A complete-lipid semantic target violates the training-only contract."""


ProgramKey = tuple[
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
]
TailTargetKey = tuple[int, int, int, int, int, int, int, int, int]
AllRoleTargetKey = tuple[tuple[int, ...], TailTargetKey]
RoleMorphologyKey = tuple[int, int, int, int]
AldehydeSemanticKey = tuple[int, int, int, int, int, int]
IsocyanideSemanticKey = tuple[int, int, int]


@dataclass(frozen=True, order=True)
class UgiTailPairSemanticTarget:
    """Low-dimensional joint chemistry and morphology of both tail-bearing roles."""

    aldehyde_ester_short_side_carbons: int
    aldehyde_ester_long_side_carbons: int
    aldehyde_carbon_carbon_double_bonds: int
    aldehyde_carbon_carbon_triple_bonds: int
    isocyanide_carbon_carbon_double_bonds: int
    isocyanide_carbon_carbon_triple_bonds: int
    isocyanide_carbon_skeleton_diameter: int
    aldehyde_alkoxy_handle_side_carbons: int = 0
    aldehyde_acyl_side_carbons: int = 0

    def __post_init__(self) -> None:
        values = self.key
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise UgiAllRoleSemanticProgramError("tail-pair semantic target must contain integers")
        if (
            self.aldehyde_ester_short_side_carbons < 1
            or self.aldehyde_ester_long_side_carbons < self.aldehyde_ester_short_side_carbons
            or self.aldehyde_carbon_carbon_double_bonds < 0
            or self.aldehyde_carbon_carbon_triple_bonds < 0
            or self.isocyanide_carbon_carbon_double_bonds < 0
            or self.isocyanide_carbon_carbon_triple_bonds < 0
            or self.isocyanide_carbon_skeleton_diameter < 1
            or self.aldehyde_alkoxy_handle_side_carbons < 0
            or self.aldehyde_acyl_side_carbons < 0
        ):
            raise UgiAllRoleSemanticProgramError(
                "tail-pair semantic target is outside graph support"
            )
        directional = (
            self.aldehyde_alkoxy_handle_side_carbons,
            self.aldehyde_acyl_side_carbons,
        )
        if (directional[0] == 0) != (directional[1] == 0):
            raise UgiAllRoleSemanticProgramError(
                "directional aldehyde ester semantics must be jointly specified"
            )
        if directional[0] > 0 and tuple(sorted(directional)) != (
            self.aldehyde_ester_short_side_carbons,
            self.aldehyde_ester_long_side_carbons,
        ):
            raise UgiAllRoleSemanticProgramError(
                "directional aldehyde ester semantics disagree with the undirected counts"
            )

    @property
    def key(self) -> TailTargetKey:
        return (
            self.aldehyde_ester_short_side_carbons,
            self.aldehyde_ester_long_side_carbons,
            self.aldehyde_carbon_carbon_double_bonds,
            self.aldehyde_carbon_carbon_triple_bonds,
            self.isocyanide_carbon_carbon_double_bonds,
            self.isocyanide_carbon_carbon_triple_bonds,
            self.isocyanide_carbon_skeleton_diameter,
            self.aldehyde_alkoxy_handle_side_carbons,
            self.aldehyde_acyl_side_carbons,
        )

    def to_mapping(self) -> dict[str, int]:
        output = {
            "aldehyde_ester_short_side_carbons": self.aldehyde_ester_short_side_carbons,
            "aldehyde_ester_long_side_carbons": self.aldehyde_ester_long_side_carbons,
            "aldehyde_carbon_carbon_double_bonds": (self.aldehyde_carbon_carbon_double_bonds),
            "aldehyde_carbon_carbon_triple_bonds": (self.aldehyde_carbon_carbon_triple_bonds),
            "isocyanide_carbon_carbon_double_bonds": (self.isocyanide_carbon_carbon_double_bonds),
            "isocyanide_carbon_carbon_triple_bonds": (self.isocyanide_carbon_carbon_triple_bonds),
            "isocyanide_carbon_skeleton_diameter": (self.isocyanide_carbon_skeleton_diameter),
        }
        if self.aldehyde_alkoxy_handle_side_carbons > 0:
            output.update(
                {
                    "aldehyde_alkoxy_handle_side_carbons": (
                        self.aldehyde_alkoxy_handle_side_carbons
                    ),
                    "aldehyde_acyl_side_carbons": self.aldehyde_acyl_side_carbons,
                }
            )
        return output

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> UgiTailPairSemanticTarget:
        base = set(cls(1, 1, 0, 0, 0, 0, 1).to_mapping())
        directional = {
            "aldehyde_alkoxy_handle_side_carbons",
            "aldehyde_acyl_side_carbons",
        }
        if set(value) not in {frozenset(base), frozenset(base | directional)}:
            raise UgiAllRoleSemanticProgramError("tail-pair semantic target fields changed")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value.values()):
            raise UgiAllRoleSemanticProgramError("tail-pair semantic target must contain integers")
        return cls(**dict(value))


@dataclass(frozen=True, order=True)
class UgiAllRoleSemanticTarget:
    """One identity-free joint target spanning the head and both tail-bearing roles."""

    amine: UgiAmineSemanticTarget
    tail_pair: UgiTailPairSemanticTarget

    @property
    def key(self) -> AllRoleTargetKey:
        return self.amine.key, self.tail_pair.key

    def to_mapping(self) -> dict[str, dict[str, int]]:
        return {
            "amine": self.amine.to_mapping(),
            "tail_pair": self.tail_pair.to_mapping(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> UgiAllRoleSemanticTarget:
        if set(value) != {"amine", "tail_pair"} or not all(
            isinstance(value[field], Mapping) for field in ("amine", "tail_pair")
        ):
            raise UgiAllRoleSemanticProgramError("all-role semantic target fields changed")
        return cls(
            amine=UgiAmineSemanticTarget.from_mapping(value["amine"]),
            tail_pair=UgiTailPairSemanticTarget.from_mapping(value["tail_pair"]),
        )


def _program_key(program: UgiMorphologyProgram) -> ProgramKey:
    return (
        tuple(int(value) for value in program.node_counts),
        tuple(int(value) for value in program.junction_budgets),
        tuple(int(value) for value in program.cycle_ranks),
        tuple(int(value) for value in program.attachment_counts),
    )


def all_role_semantic_target(
    *,
    amine_smiles: str,
    aldehyde_smiles: str,
    isocyanide_smiles: str,
    include_amine_substitution_semantics: bool = False,
) -> UgiAllRoleSemanticTarget:
    """Project one measured precursor triple to graph-derived integer semantics."""

    sides = ester_side_carbon_counts(aldehyde_smiles)
    if sides is None:
        raise UgiAllRoleSemanticProgramError(
            "measured aldehyde lacks one unambiguous ester-side decomposition"
        )
    aldehyde = component_chemotype_metrics(aldehyde_smiles)
    isocyanide = component_chemotype_metrics(isocyanide_smiles)
    if any(
        int(aldehyde[field])
        for field in ("nitrogen_atoms", "sulfur_atoms", "phosphorus_atoms", "halogen_atoms")
    ) or any(
        int(isocyanide[field])
        for field in ("oxygen_atoms", "sulfur_atoms", "phosphorus_atoms", "halogen_atoms")
    ):
        raise UgiAllRoleSemanticProgramError(
            "tail semantic support currently admits the measured C/O aldehyde and C/N "
            "isocyanide strata only"
        )
    if int(aldehyde["oxygen_atoms"]) != 3 or int(isocyanide["nitrogen_atoms"]) != 1:
        raise UgiAllRoleSemanticProgramError("measured Ugi tail handle composition changed")
    return UgiAllRoleSemanticTarget(
        amine=amine_semantic_target(
            amine_smiles,
            include_substitution_semantics=include_amine_substitution_semantics,
        ),
        tail_pair=UgiTailPairSemanticTarget(
            aldehyde_ester_short_side_carbons=int(sides[0]),
            aldehyde_ester_long_side_carbons=int(sides[1]),
            aldehyde_carbon_carbon_double_bonds=int(aldehyde["carbon_carbon_double_bonds"]),
            aldehyde_carbon_carbon_triple_bonds=int(aldehyde["carbon_carbon_triple_bonds"]),
            isocyanide_carbon_carbon_double_bonds=int(isocyanide["carbon_carbon_double_bonds"]),
            isocyanide_carbon_carbon_triple_bonds=int(isocyanide["carbon_carbon_triple_bonds"]),
            isocyanide_carbon_skeleton_diameter=(int(isocyanide["carbon_subgraph_diameter"]) + 1),
        ),
    )


def directional_all_role_semantic_target(
    *,
    amine_smiles: str,
    aldehyde_smiles: str,
    isocyanide_smiles: str,
    include_amine_substitution_semantics: bool = False,
) -> UgiAllRoleSemanticTarget:
    """Project one measured triple while preserving the aldehyde-to-ester direction."""

    target = all_role_semantic_target(
        amine_smiles=amine_smiles,
        aldehyde_smiles=aldehyde_smiles,
        isocyanide_smiles=isocyanide_smiles,
        include_amine_substitution_semantics=include_amine_substitution_semantics,
    )
    directional = aldehyde_ester_directional_carbon_counts(aldehyde_smiles)
    if directional is None:
        raise UgiAllRoleSemanticProgramError(
            "measured aldehyde lacks one alkoxy-side aldehyde-to-ester direction"
        )
    return UgiAllRoleSemanticTarget(
        amine=target.amine,
        tail_pair=UgiTailPairSemanticTarget(
            **target.tail_pair.to_mapping(),
            aldehyde_alkoxy_handle_side_carbons=int(directional[0]),
            aldehyde_acyl_side_carbons=int(directional[1]),
        ),
    )


def _validate_target_matches_program(
    program: UgiMorphologyProgram,
    target: UgiAllRoleSemanticTarget,
) -> None:
    role_index = {role: ROLE_NAMES.index(role) for role in ROLE_NAMES}
    amine_nodes = int(program.node_counts[role_index["amine_head"]])
    aldehyde_nodes = int(program.node_counts[role_index["oxoester_aldehyde_body_tail"]])
    isocyanide_nodes = int(program.node_counts[role_index["isocyanide_tail"]])
    if target.amine.heavy_atom_graph_diameter > amine_nodes + 1:
        raise UgiAllRoleSemanticProgramError("amine semantics exceed the coarse program")
    # The precursor has one retained core carbon and one aldehyde oxygen absent from the product;
    # therefore its two ester-side carbon counts sum to exterior nodes minus one.
    if (
        target.tail_pair.aldehyde_ester_short_side_carbons
        + target.tail_pair.aldehyde_ester_long_side_carbons
        != aldehyde_nodes - 1
    ):
        raise UgiAllRoleSemanticProgramError(
            "aldehyde ester-side semantics disagree with the coarse program"
        )
    if target.tail_pair.isocyanide_carbon_skeleton_diameter != isocyanide_nodes:
        raise UgiAllRoleSemanticProgramError(
            "isocyanide skeleton semantics disagree with the coarse program"
        )


def _role_morphology_key(program: UgiMorphologyProgram, role: str) -> RoleMorphologyKey:
    index = ROLE_NAMES.index(role)
    return (
        int(program.node_counts[index]),
        int(program.junction_budgets[index]),
        int(program.cycle_ranks[index]),
        int(program.attachment_counts[index]),
    )


def _aldehyde_semantic_key(target: UgiAllRoleSemanticTarget) -> AldehydeSemanticKey:
    tail = target.tail_pair
    return (
        tail.aldehyde_ester_short_side_carbons,
        tail.aldehyde_ester_long_side_carbons,
        tail.aldehyde_carbon_carbon_double_bonds,
        tail.aldehyde_carbon_carbon_triple_bonds,
        tail.aldehyde_alkoxy_handle_side_carbons,
        tail.aldehyde_acyl_side_carbons,
    )


def _isocyanide_semantic_key(target: UgiAllRoleSemanticTarget) -> IsocyanideSemanticKey:
    tail = target.tail_pair
    return (
        tail.isocyanide_carbon_carbon_double_bonds,
        tail.isocyanide_carbon_carbon_triple_bonds,
        tail.isocyanide_carbon_skeleton_diameter,
    )


def build_equal_group_conditional_distribution(
    rows: Sequence[tuple[ProgramKey, str, UgiAllRoleSemanticTarget]],
) -> dict[ProgramKey, tuple[tuple[UgiAllRoleSemanticTarget, ...], np.ndarray]]:
    """Balance family triples within each exact complete morphology program."""

    if not rows:
        raise UgiAllRoleSemanticProgramError("all-role semantic support is empty")
    by_program: dict[ProgramKey, dict[str, list[UgiAllRoleSemanticTarget]]] = {}
    for program, group, target in rows:
        if not group:
            raise UgiAllRoleSemanticProgramError("all-role estimator group is empty")
        by_program.setdefault(program, {}).setdefault(group, []).append(target)
    output: dict[ProgramKey, tuple[tuple[UgiAllRoleSemanticTarget, ...], np.ndarray]] = {}
    for program, by_group in sorted(by_program.items()):
        mass: dict[AllRoleTargetKey, float] = {}
        group_mass = 1.0 / len(by_group)
        for targets in by_group.values():
            row_mass = group_mass / len(targets)
            for target in targets:
                mass[target.key] = mass.get(target.key, 0.0) + row_mass
        keys = tuple(sorted(mass))
        targets = tuple(
            UgiAllRoleSemanticTarget(
                amine=UgiAmineSemanticTarget.from_key(key[0]),
                tail_pair=UgiTailPairSemanticTarget(*key[1]),
            )
            for key in keys
        )
        probabilities = np.asarray([mass[key] for key in keys], dtype=np.float64)
        probabilities /= probabilities.sum()
        if np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1.0):
            raise UgiAllRoleSemanticProgramError(
                "conditional all-role semantic probabilities are invalid"
            )
        output[program] = targets, probabilities
    return output


def build_equal_group_joint_distribution(
    rows: Sequence[tuple[ProgramKey, str, UgiAllRoleSemanticTarget]],
) -> tuple[
    tuple[tuple[UgiMorphologyProgram, UgiAllRoleSemanticTarget], ...],
    np.ndarray,
]:
    """Estimate one coherent joint ``q(program, semantics)`` with equal group mass."""

    if not rows:
        raise UgiAllRoleSemanticProgramError("joint all-role semantic support is empty")
    by_group: dict[str, list[tuple[ProgramKey, UgiAllRoleSemanticTarget]]] = {}
    for program, group, target in rows:
        if not group:
            raise UgiAllRoleSemanticProgramError("all-role estimator group is empty")
        by_group.setdefault(group, []).append((program, target))
    mass: dict[tuple[ProgramKey, AllRoleTargetKey], float] = {}
    group_mass = 1.0 / len(by_group)
    for group_rows in by_group.values():
        row_mass = group_mass / len(group_rows)
        for program, target in group_rows:
            key = (program, target.key)
            mass[key] = mass.get(key, 0.0) + row_mass
    keys = tuple(sorted(mass))
    support = tuple(
        (
            UgiMorphologyProgram(
                node_counts=program[0],
                junction_budgets=program[1],
                cycle_ranks=program[2],
                attachment_counts=program[3],
            ),
            UgiAllRoleSemanticTarget(
                amine=UgiAmineSemanticTarget.from_key(target[0]),
                tail_pair=UgiTailPairSemanticTarget(*target[1]),
            ),
        )
        for program, target in keys
    )
    probabilities = np.asarray([mass[key] for key in keys], dtype=np.float64)
    probabilities /= probabilities.sum()
    if np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1.0):
        raise UgiAllRoleSemanticProgramError("joint all-role probabilities are invalid")
    return support, probabilities


@dataclass(frozen=True)
class UgiMeasuredAllRoleSemanticPrior:
    """Joint target law estimated only from source-adjudicated measured train products."""

    support: Mapping[ProgramKey, tuple[tuple[UgiAllRoleSemanticTarget, ...], np.ndarray]]
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
        include_amine_substitution_semantics: bool = False,
    ) -> UgiMeasuredAllRoleSemanticPrior:
        assignments = assignments_path.resolve()
        assignments_sha256 = str(sha256_file(assignments))
        if (
            expected_assignments_sha256 is not None
            and assignments_sha256 != expected_assignments_sha256
        ):
            raise UgiAllRoleSemanticProgramError(
                "Ugi assignments changed: expected "
                f"{expected_assignments_sha256}, found {assignments_sha256}"
            )
        required = {
            "product_id",
            "primary_product_fold",
            "is_source_adjudicated_measured_product",
            *(f"{role}_smiles" for role in ROLE_NAMES),
            *(f"{role}_family_id" for role in ROLE_NAMES),
        }
        assignments_by_product: dict[str, tuple[str, UgiAllRoleSemanticTarget]] = {}
        for row in iter_csv(assignments):
            if not required.issubset(row):
                raise UgiAllRoleSemanticProgramError("Ugi assignment schema changed")
            if row["primary_product_fold"] != "train" or str(
                row["is_source_adjudicated_measured_product"]
            ).strip().lower() not in {"1", "true"}:
                continue
            product_id = str(row["product_id"]).strip()
            group = "|".join(str(row[f"{role}_family_id"]).strip() for role in ROLE_NAMES)
            smiles = {role: str(row[f"{role}_smiles"]).strip() for role in ROLE_NAMES}
            if not product_id or any(not value for value in (*group.split("|"), *smiles.values())):
                raise UgiAllRoleSemanticProgramError(
                    "measured Ugi train row lacks complete role metadata"
                )
            value = (
                group,
                all_role_semantic_target(
                    amine_smiles=smiles["amine_head"],
                    aldehyde_smiles=smiles["oxoester_aldehyde_body_tail"],
                    isocyanide_smiles=smiles["isocyanide_tail"],
                    include_amine_substitution_semantics=(include_amine_substitution_semantics),
                ),
            )
            previous = assignments_by_product.setdefault(product_id, value)
            if previous != value:
                raise UgiAllRoleSemanticProgramError(
                    "one measured product maps to inconsistent all-role semantics"
                )
        if not assignments_by_product:
            raise UgiAllRoleSemanticProgramError("no measured Ugi train products were found")

        observed: set[str] = set()
        rows: list[tuple[ProgramKey, str, UgiAllRoleSemanticTarget]] = []
        excluded = 0
        for raw_index in cache.indices(program_id=reaction_id, fold="train"):
            index = int(raw_index)
            product_id = cache.record_id(index)
            assignment = assignments_by_product.get(product_id)
            if assignment is None:
                continue
            observed.add(product_id)
            program = program_from_layout_record(cache.record(index), vocabulary=cache.vocabulary)
            if not program_admitted_by_ester_policy(program, ester_policy):
                excluded += 1
                continue
            group, target = assignment
            _validate_target_matches_program(program, target)
            rows.append((_program_key(program), group, target))
        missing = set(assignments_by_product).difference(observed)
        if missing:
            raise UgiAllRoleSemanticProgramError(
                f"production cache lacks {len(missing)} measured train products"
            )
        support = build_equal_group_conditional_distribution(rows)
        payload = [
            {
                "program": program,
                "targets": [
                    {"target": target.to_mapping(), "probability": float(probability)}
                    for target, probability in zip(targets, probabilities.tolist(), strict=True)
                ],
            }
            for program, (targets, probabilities) in sorted(support.items())
        ]
        audit = {
            "schema_version": (
                "forge.ugi_measured_all_role_substitution_semantic_prior.v2"
                if include_amine_substitution_semantics
                else "forge.ugi_measured_all_role_semantic_prior.v1"
            ),
            "reaction_id": reaction_id,
            "training_fold": "train",
            "reference": "source-adjudicated measured Ugi train products only",
            "weighting": (
                "equal component-family-triple mass within exact complete coarse program"
            ),
            "measured_training_rows": len(assignments_by_product),
            "eligible_rows": len(rows),
            "excluded_by_existing_decoder_program_support": excluded,
            "conditional_complete_programs": len(support),
            "unique_joint_semantic_targets": len({target.key for _, _, target in rows}),
            "support_sha256": str(sha256_json(payload)),
            "assignments_path": str(assignments),
            "assignments_sha256": assignments_sha256,
            "sampled_target_fields": UgiAllRoleSemanticTarget(
                UgiAmineSemanticTarget(1, 0, 1, 0),
                UgiTailPairSemanticTarget(1, 1, 0, 0, 0, 0, 1),
            ).to_mapping(),
            "component_or_family_ids_in_sampled_target": False,
            "stored_component_graphs_in_sampled_target": False,
            "smiles_or_fragment_tokens_in_sampled_target": False,
        }
        return cls(support=support, audit=audit)

    def sample_for_programs(
        self,
        programs: Sequence[UgiMorphologyProgram],
        *,
        seed: int,
    ) -> tuple[UgiAllRoleSemanticTarget, ...]:
        """Draw one complete semantic target per coarse program, without retry."""

        if not programs or seed < 0:
            raise UgiAllRoleSemanticProgramError("all-role semantic target request is invalid")
        rng = np.random.default_rng(seed)
        output: list[UgiAllRoleSemanticTarget] = []
        for program in programs:
            key = _program_key(program)
            if key not in self.support:
                raise UgiAllRoleSemanticProgramError(
                    f"complete coarse program {key} lacks measured semantic support"
                )
            targets, probabilities = self.support[key]
            output.append(targets[int(rng.choice(len(targets), p=probabilities))])
        return tuple(output)


@dataclass(frozen=True)
class UgiMeasuredRoleFactorizedSemanticPrior:
    """Identity-free semantics sampled once per role from unique measured components.

    The virtual Ugi library repeats a component across many Cartesian combinations.  This prior
    removes that replication before estimating the role-local semantic law.  Generation receives
    only low-dimensional integer semantics conditioned on the already frozen role morphology; it
    never receives the representative component constitution used during compilation.
    """

    amine_support: Mapping[
        RoleMorphologyKey, tuple[tuple[UgiAmineSemanticTarget, ...], np.ndarray]
    ]
    aldehyde_support: Mapping[
        RoleMorphologyKey, tuple[tuple[AldehydeSemanticKey, ...], np.ndarray]
    ]
    isocyanide_support: Mapping[
        RoleMorphologyKey, tuple[tuple[IsocyanideSemanticKey, ...], np.ndarray]
    ]
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
        include_amine_substitution_semantics: bool = False,
    ) -> UgiMeasuredRoleFactorizedSemanticPrior:
        """Compile equal unique-component mass within each exact role morphology."""

        assignments = assignments_path.resolve()
        assignments_sha256 = str(sha256_file(assignments))
        if (
            expected_assignments_sha256 is not None
            and assignments_sha256 != expected_assignments_sha256
        ):
            raise UgiAllRoleSemanticProgramError(
                "Ugi assignments changed: expected "
                f"{expected_assignments_sha256}, found {assignments_sha256}"
            )
        required = {
            "product_id",
            "primary_product_fold",
            "is_source_adjudicated_measured_product",
            *(f"{role}_smiles" for role in ROLE_NAMES),
        }
        assignments_by_product: dict[
            str, tuple[dict[str, str], UgiAllRoleSemanticTarget]
        ] = {}
        component_products: dict[str, dict[str, set[str]]] = {
            role: {} for role in ROLE_NAMES
        }
        for row in iter_csv(assignments):
            if not required.issubset(row):
                raise UgiAllRoleSemanticProgramError("Ugi assignment schema changed")
            if row["primary_product_fold"] != "train" or str(
                row["is_source_adjudicated_measured_product"]
            ).strip().lower() not in {"1", "true"}:
                continue
            product_id = str(row["product_id"]).strip()
            smiles = {role: str(row[f"{role}_smiles"]).strip() for role in ROLE_NAMES}
            if not product_id or any(not value for value in smiles.values()):
                raise UgiAllRoleSemanticProgramError(
                    "measured Ugi train row lacks complete role metadata"
                )
            value = (
                smiles,
                directional_all_role_semantic_target(
                    amine_smiles=smiles["amine_head"],
                    aldehyde_smiles=smiles["oxoester_aldehyde_body_tail"],
                    isocyanide_smiles=smiles["isocyanide_tail"],
                    include_amine_substitution_semantics=(
                        include_amine_substitution_semantics
                    ),
                ),
            )
            previous = assignments_by_product.setdefault(product_id, value)
            if previous != value:
                raise UgiAllRoleSemanticProgramError(
                    "one measured product maps to inconsistent role semantics"
                )
            for role, component_smiles in smiles.items():
                component_products[role].setdefault(component_smiles, set()).add(product_id)
        if not assignments_by_product or any(not values for values in component_products.values()):
            raise UgiAllRoleSemanticProgramError(
                "measured Ugi train fold does not cover every precursor role"
            )

        eligible: dict[str, tuple[UgiMorphologyProgram, UgiAllRoleSemanticTarget]] = {}
        excluded = 0
        for raw_index in cache.indices(program_id=reaction_id, fold="train"):
            index = int(raw_index)
            product_id = cache.record_id(index)
            assignment = assignments_by_product.get(product_id)
            if assignment is None:
                continue
            program = program_from_layout_record(cache.record(index), vocabulary=cache.vocabulary)
            if not program_admitted_by_ester_policy(program, ester_policy):
                excluded += 1
                continue
            target = assignment[1]
            _validate_target_matches_program(program, target)
            eligible[product_id] = (program, target)

        role_observations: dict[str, dict[RoleMorphologyKey, list[tuple[int, ...]]]] = {
            role: {} for role in ROLE_NAMES
        }
        admitted_component_counts: dict[str, int] = {}
        excluded_component_counts: dict[str, int] = {}
        for role, by_component in component_products.items():
            admitted = 0
            excluded_components = 0
            for product_ids in by_component.values():
                representatives = sorted(product_ids & eligible.keys())
                if not representatives:
                    excluded_components += 1
                    continue
                program, target = eligible[representatives[0]]
                morphology = _role_morphology_key(program, role)
                if role == "amine_head":
                    semantic = target.amine.key
                elif role == "oxoester_aldehyde_body_tail":
                    semantic = _aldehyde_semantic_key(target)
                else:
                    semantic = _isocyanide_semantic_key(target)
                role_observations[role].setdefault(morphology, []).append(semantic)
                admitted += 1
            admitted_component_counts[role] = admitted
            excluded_component_counts[role] = excluded_components

        def compile_support(
            observations: Mapping[RoleMorphologyKey, Sequence[tuple[int, ...]]],
        ) -> dict[RoleMorphologyKey, tuple[tuple[tuple[int, ...], ...], np.ndarray]]:
            output = {}
            for morphology, values in sorted(observations.items()):
                counts = Counter(values)
                keys = tuple(sorted(counts))
                probabilities = np.asarray([counts[key] for key in keys], dtype=np.float64)
                probabilities /= probabilities.sum()
                output[morphology] = keys, probabilities
            if not output:
                raise UgiAllRoleSemanticProgramError("role-factorized semantic support is empty")
            return output

        raw_amine = compile_support(role_observations["amine_head"])
        raw_aldehyde = compile_support(role_observations["oxoester_aldehyde_body_tail"])
        raw_isocyanide = compile_support(role_observations["isocyanide_tail"])
        amine_support = {
            morphology: (
                tuple(UgiAmineSemanticTarget.from_key(value) for value in values),
                probabilities,
            )
            for morphology, (values, probabilities) in raw_amine.items()
        }
        aldehyde_support = {
            morphology: (tuple(values), probabilities)
            for morphology, (values, probabilities) in raw_aldehyde.items()
        }
        isocyanide_support = {
            morphology: (tuple(values), probabilities)
            for morphology, (values, probabilities) in raw_isocyanide.items()
        }
        payload = {
            "amine": [
                [morphology, [target.to_mapping() for target in targets], probabilities.tolist()]
                for morphology, (targets, probabilities) in sorted(amine_support.items())
            ],
            "aldehyde": [
                [morphology, targets, probabilities.tolist()]
                for morphology, (targets, probabilities) in sorted(aldehyde_support.items())
            ],
            "isocyanide": [
                [morphology, targets, probabilities.tolist()]
                for morphology, (targets, probabilities) in sorted(isocyanide_support.items())
            ],
        }
        audit = {
            "schema_version": "forge.ugi_measured_role_factorized_semantic_prior.v1",
            "reaction_id": reaction_id,
            "training_fold": "train",
            "reference": "source-adjudicated measured Ugi train products only",
            "weighting": "equal unique component constitution within exact role morphology",
            "measured_training_rows": len(assignments_by_product),
            "eligible_product_rows": len(eligible),
            "excluded_product_rows_by_decoder_support": excluded,
            "admitted_unique_components_by_role": admitted_component_counts,
            "excluded_unique_components_by_role": excluded_component_counts,
            "support_sha256": str(sha256_json(payload)),
            "assignments_path": str(assignments),
            "assignments_sha256": assignments_sha256,
            "component_or_family_ids_in_sampled_target": False,
            "stored_component_graphs_in_sampled_target": False,
            "smiles_or_fragment_tokens_in_sampled_target": False,
        }
        return cls(
            amine_support=amine_support,
            aldehyde_support=aldehyde_support,
            isocyanide_support=isocyanide_support,
            audit=audit,
        )

    def sample_for_programs(
        self,
        programs: Sequence[UgiMorphologyProgram],
        *,
        seed: int,
    ) -> tuple[UgiAllRoleSemanticTarget, ...]:
        """Sample one independent role-local semantic tuple per frozen coarse program."""

        if not programs or seed < 0:
            raise UgiAllRoleSemanticProgramError("role-factorized sample request is invalid")
        rng = np.random.default_rng(seed)
        output = []
        for program in programs:
            role_values = {}
            for role, support in (
                ("amine_head", self.amine_support),
                ("oxoester_aldehyde_body_tail", self.aldehyde_support),
                ("isocyanide_tail", self.isocyanide_support),
            ):
                morphology = _role_morphology_key(program, role)
                if morphology not in support:
                    raise UgiAllRoleSemanticProgramError(
                        f"role morphology {role}/{morphology} lacks measured semantic support"
                    )
                values, probabilities = support[morphology]
                role_values[role] = values[int(rng.choice(len(values), p=probabilities))]
            aldehyde = role_values["oxoester_aldehyde_body_tail"]
            isocyanide = role_values["isocyanide_tail"]
            target = UgiAllRoleSemanticTarget(
                amine=role_values["amine_head"],
                tail_pair=UgiTailPairSemanticTarget(
                    aldehyde_ester_short_side_carbons=aldehyde[0],
                    aldehyde_ester_long_side_carbons=aldehyde[1],
                    aldehyde_carbon_carbon_double_bonds=aldehyde[2],
                    aldehyde_carbon_carbon_triple_bonds=aldehyde[3],
                    isocyanide_carbon_carbon_double_bonds=isocyanide[0],
                    isocyanide_carbon_carbon_triple_bonds=isocyanide[1],
                    isocyanide_carbon_skeleton_diameter=isocyanide[2],
                    aldehyde_alkoxy_handle_side_carbons=aldehyde[4],
                    aldehyde_acyl_side_carbons=aldehyde[5],
                ),
            )
            _validate_target_matches_program(program, target)
            output.append(target)
        return tuple(output)


@dataclass(frozen=True)
class UgiMeasuredJointAllRoleSemanticPrior:
    """One train-only joint law over coarse programs and complete role semantics."""

    support: tuple[tuple[UgiMorphologyProgram, UgiAllRoleSemanticTarget], ...]
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
        include_amine_substitution_semantics: bool = False,
    ) -> UgiMeasuredJointAllRoleSemanticPrior:
        """Compile equal family-group mass without factorizing program from semantics."""

        assignments = assignments_path.resolve()
        assignments_sha256 = str(sha256_file(assignments))
        if (
            expected_assignments_sha256 is not None
            and assignments_sha256 != expected_assignments_sha256
        ):
            raise UgiAllRoleSemanticProgramError(
                "Ugi assignments changed: expected "
                f"{expected_assignments_sha256}, found {assignments_sha256}"
            )
        required = {
            "product_id",
            "primary_product_fold",
            "is_source_adjudicated_measured_product",
            *(f"{role}_smiles" for role in ROLE_NAMES),
            *(f"{role}_family_id" for role in ROLE_NAMES),
        }
        assignments_by_product: dict[str, tuple[str, UgiAllRoleSemanticTarget]] = {}
        for row in iter_csv(assignments):
            if not required.issubset(row):
                raise UgiAllRoleSemanticProgramError("Ugi assignment schema changed")
            if row["primary_product_fold"] != "train" or str(
                row["is_source_adjudicated_measured_product"]
            ).strip().lower() not in {"1", "true"}:
                continue
            product_id = str(row["product_id"]).strip()
            group = "|".join(str(row[f"{role}_family_id"]).strip() for role in ROLE_NAMES)
            smiles = {role: str(row[f"{role}_smiles"]).strip() for role in ROLE_NAMES}
            if not product_id or any(not value for value in (*group.split("|"), *smiles.values())):
                raise UgiAllRoleSemanticProgramError(
                    "measured Ugi train row lacks complete role metadata"
                )
            value = (
                group,
                directional_all_role_semantic_target(
                    amine_smiles=smiles["amine_head"],
                    aldehyde_smiles=smiles["oxoester_aldehyde_body_tail"],
                    isocyanide_smiles=smiles["isocyanide_tail"],
                    include_amine_substitution_semantics=(include_amine_substitution_semantics),
                ),
            )
            previous = assignments_by_product.setdefault(product_id, value)
            if previous != value:
                raise UgiAllRoleSemanticProgramError(
                    "one measured product maps to inconsistent directional semantics"
                )
        if not assignments_by_product:
            raise UgiAllRoleSemanticProgramError("no measured Ugi train products were found")

        observed: set[str] = set()
        rows: list[tuple[ProgramKey, str, UgiAllRoleSemanticTarget]] = []
        excluded = 0
        eligible_groups: Counter[str] = Counter()
        for raw_index in cache.indices(program_id=reaction_id, fold="train"):
            index = int(raw_index)
            product_id = cache.record_id(index)
            assignment = assignments_by_product.get(product_id)
            if assignment is None:
                continue
            observed.add(product_id)
            program = program_from_layout_record(cache.record(index), vocabulary=cache.vocabulary)
            if not program_admitted_by_ester_policy(program, ester_policy):
                excluded += 1
                continue
            group, target = assignment
            _validate_target_matches_program(program, target)
            rows.append((_program_key(program), group, target))
            eligible_groups[group] += 1
        missing = set(assignments_by_product).difference(observed)
        if missing:
            raise UgiAllRoleSemanticProgramError(
                f"production cache lacks {len(missing)} measured train products"
            )
        support, probabilities = build_equal_group_joint_distribution(rows)
        payload = [
            {
                "program": _program_key(program),
                "target": target.to_mapping(),
                "probability": float(probability),
            }
            for (program, target), probability in zip(support, probabilities.tolist(), strict=True)
        ]
        audit = {
            "schema_version": (
                "forge.ugi_measured_joint_all_role_substitution_semantic_prior.v3"
                if include_amine_substitution_semantics
                else "forge.ugi_measured_joint_all_role_semantic_prior.v2"
            ),
            "reaction_id": reaction_id,
            "training_fold": "train",
            "reference": "source-adjudicated measured Ugi train products only",
            "estimator": (
                "equal component-family-triple mass then equal eligible product mass; "
                "program and role semantics sampled jointly"
            ),
            "measured_training_rows": len(assignments_by_product),
            "eligible_rows": len(rows),
            "excluded_by_existing_decoder_program_support": excluded,
            "eligible_groups": len(eligible_groups),
            "eligible_group_size_histogram": {
                str(size): count
                for size, count in sorted(Counter(eligible_groups.values()).items())
            },
            "joint_support_size": len(support),
            "joint_support_sha256": str(sha256_json(payload)),
            "assignments_path": str(assignments),
            "assignments_sha256": assignments_sha256,
            "paired_baseline_and_treatment_amine_target": True,
            "aldehyde_direction": ("reactive aldehyde fixed to the alkoxy side of the ester"),
            "directionally_supported_measured_training_rows": len(assignments_by_product),
            "component_or_family_ids_in_sampled_target": False,
            "stored_component_graphs_in_sampled_target": False,
            "smiles_or_fragment_tokens_in_sampled_target": False,
        }
        return cls(support=support, probabilities=probabilities, audit=audit)

    def sample(
        self,
        *,
        count: int,
        seed: int,
    ) -> tuple[
        tuple[UgiMorphologyProgram, ...],
        tuple[UgiAllRoleSemanticTarget, ...],
    ]:
        """Draw each complete program and its semantics together, exactly once."""

        if count < 1 or seed < 0 or not self.support:
            raise UgiAllRoleSemanticProgramError("joint all-role sample request is invalid")
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.support), size=count, p=self.probabilities)
        selected = tuple(self.support[int(index)] for index in indices.tolist())
        return (
            tuple(program for program, _ in selected),
            tuple(target for _, target in selected),
        )


__all__ = [
    "UgiAllRoleSemanticProgramError",
    "UgiAllRoleSemanticTarget",
    "UgiMeasuredAllRoleSemanticPrior",
    "UgiMeasuredJointAllRoleSemanticPrior",
    "UgiMeasuredRoleFactorizedSemanticPrior",
    "UgiTailPairSemanticTarget",
    "all_role_semantic_target",
    "build_equal_group_joint_distribution",
    "build_equal_group_conditional_distribution",
    "directional_all_role_semantic_target",
]
