"""Identity-free measured-Ugi amine semantics conditioned on coarse programs.

The ordinary morphology program fixes only node, junction, cycle and attachment counts.  Those
coordinates cannot distinguish a compact measured head from an elongated head with the same
counts.  This module compiles a train-fold-only conditional law over graph-derived amine
semantics: heavy-atom graph diameter, carbon-skeleton diameter, total nitrogen/oxygen counts and,
in the optional substitution-aware representation, hydrogen-bond-donor and heavy-atom branch
counts.

Component and family identifiers are estimator metadata only.  A sampled target contains a few
integers and no product identity, component identity, graph, SMILES or fragment token.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.chemistry.descriptors import component_chemotype_metrics, connected_molecule
from forge.core.hashing import sha256_file, sha256_json
from forge.core.io import iter_csv
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_measured_joint_program_prior import (
    program_admitted_by_ester_policy,
    program_from_layout_record,
)
from forge.model.ugi_morphology_program import UgiMorphologyProgram
from forge.potency.annotations import ROLE_NAMES


class UgiAmineSemanticProgramError(ValueError):
    """Measured amine semantics violate the train-fold-only program contract."""


AmineProgramKey = tuple[int, int, int, int]
TargetKey = tuple[int, ...]


@dataclass(frozen=True, order=True)
class UgiAmineSemanticTarget:
    """Graph-derived head coordinates with no component identity."""

    heavy_atom_graph_diameter: int
    carbon_skeleton_diameter: int
    nitrogen_atoms: int
    oxygen_atoms: int
    hydrogen_bond_donors: int | None = None
    heavy_branch_atoms: int | None = None

    def __post_init__(self) -> None:
        core_values = (
            self.heavy_atom_graph_diameter,
            self.carbon_skeleton_diameter,
            self.nitrogen_atoms,
            self.oxygen_atoms,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in core_values):
            raise UgiAmineSemanticProgramError("amine semantic target must contain integers")
        if (
            self.heavy_atom_graph_diameter < 1
            or self.carbon_skeleton_diameter < 0
            or self.nitrogen_atoms < 1
            or self.oxygen_atoms < 0
        ):
            raise UgiAmineSemanticProgramError("amine semantic target is outside graph support")
        substitution_values = (self.hydrogen_bond_donors, self.heavy_branch_atoms)
        if (substitution_values[0] is None) != (substitution_values[1] is None):
            raise UgiAmineSemanticProgramError(
                "amine substitution semantics must be jointly specified"
            )
        if substitution_values[0] is not None:
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in substitution_values
            ):
                raise UgiAmineSemanticProgramError("amine semantic target must contain integers")
            if (
                self.hydrogen_bond_donors < 0
                or self.heavy_branch_atoms < 0
                or self.hydrogen_bond_donors > self.nitrogen_atoms + self.oxygen_atoms
            ):
                raise UgiAmineSemanticProgramError(
                    "amine substitution semantics are outside graph support"
                )

    @property
    def key(self) -> TargetKey:
        base = (
            self.heavy_atom_graph_diameter,
            self.carbon_skeleton_diameter,
            self.nitrogen_atoms,
            self.oxygen_atoms,
        )
        if self.hydrogen_bond_donors is None:
            return base
        return (*base, self.hydrogen_bond_donors, self.heavy_branch_atoms)

    def to_mapping(self) -> dict[str, int]:
        output = {
            "heavy_atom_graph_diameter": self.heavy_atom_graph_diameter,
            "carbon_skeleton_diameter": self.carbon_skeleton_diameter,
            "nitrogen_atoms": self.nitrogen_atoms,
            "oxygen_atoms": self.oxygen_atoms,
        }
        if self.hydrogen_bond_donors is not None:
            output.update(
                {
                    "hydrogen_bond_donors": self.hydrogen_bond_donors,
                    "heavy_branch_atoms": self.heavy_branch_atoms,
                }
            )
        return output

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> UgiAmineSemanticTarget:
        base = {
            "heavy_atom_graph_diameter",
            "carbon_skeleton_diameter",
            "nitrogen_atoms",
            "oxygen_atoms",
        }
        substitution = {"hydrogen_bond_donors", "heavy_branch_atoms"}
        if set(value) not in {frozenset(base), frozenset(base | substitution)}:
            raise UgiAmineSemanticProgramError("amine semantic target fields changed")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value.values()):
            raise UgiAmineSemanticProgramError("amine semantic target must contain integers")
        return cls(**dict(value))

    @classmethod
    def from_key(cls, value: TargetKey) -> UgiAmineSemanticTarget:
        """Restore either the frozen four-coordinate or substitution-aware target key."""

        if len(value) not in {4, 6}:
            raise UgiAmineSemanticProgramError("amine semantic target key width changed")
        return cls(*value)


def amine_local_substitution_metrics(
    symbols: Sequence[str], degrees: Sequence[int]
) -> tuple[int, int]:
    """Return donor and branch counts on the decoder's neutral saturated head support.

    This is deliberately not a generic HBD calculator.  The terminal Ugi head decoder admits
    non-aromatic, formally neutral C/N/O atoms and training-supported saturated amine-head bonds.
    Within that declared support, neutral N with heavy degree at most two and neutral O with heavy
    degree one are hydrogen-bond donors.
    """

    if len(symbols) != len(degrees) or not symbols:
        raise UgiAmineSemanticProgramError("amine local substitution state is malformed")
    if any(symbol not in {"C", "N", "O"} for symbol in symbols) or any(
        isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or degree < 1
        for degree in degrees
    ):
        raise UgiAmineSemanticProgramError(
            "amine local substitution state is outside decoder support"
        )
    donors = sum(
        (symbol == "N" and int(degree) <= 2) or (symbol == "O" and int(degree) == 1)
        for symbol, degree in zip(symbols, degrees, strict=True)
    )
    branches = sum(int(degree) >= 3 for degree in degrees)
    return int(donors), int(branches)


def amine_program_key(program: UgiMorphologyProgram) -> AmineProgramKey:
    """Return the existing coarse amine program in role order."""

    index = ROLE_NAMES.index("amine_head")
    return (
        int(program.node_counts[index]),
        int(program.junction_budgets[index]),
        int(program.cycle_ranks[index]),
        int(program.attachment_counts[index]),
    )


def amine_semantic_target(
    smiles: str, *, include_substitution_semantics: bool = False
) -> UgiAmineSemanticTarget:
    """Project one complete measured amine precursor to identity-free graph coordinates."""

    molecule = connected_molecule(smiles, label="measured Ugi amine")
    metrics = component_chemotype_metrics(smiles)
    if int(metrics["heavy_atoms"]) != sum(
        int(metrics[field]) for field in ("carbon_atoms", "nitrogen_atoms", "oxygen_atoms")
    ):
        raise UgiAmineSemanticProgramError(
            "measured Ugi amine semantic support currently admits only C/N/O heads"
        )
    distances = Chem.GetDistanceMatrix(molecule)
    heavy_diameter = int(np.max(distances)) + 1
    carbon_atoms = int(metrics["carbon_atoms"])
    return UgiAmineSemanticTarget(
        heavy_atom_graph_diameter=heavy_diameter,
        carbon_skeleton_diameter=(int(metrics["carbon_subgraph_diameter"]) + int(carbon_atoms > 0)),
        nitrogen_atoms=int(metrics["nitrogen_atoms"]),
        oxygen_atoms=int(metrics["oxygen_atoms"]),
        hydrogen_bond_donors=(
            int(rdMolDescriptors.CalcNumHBD(molecule)) if include_substitution_semantics else None
        ),
        heavy_branch_atoms=(
            sum(atom.GetAtomicNum() > 1 and atom.GetDegree() >= 3 for atom in molecule.GetAtoms())
            if include_substitution_semantics
            else None
        ),
    )


def build_equal_family_conditional_distribution(
    rows: Sequence[tuple[AmineProgramKey, str, UgiAmineSemanticTarget]],
) -> dict[AmineProgramKey, tuple[tuple[UgiAmineSemanticTarget, ...], np.ndarray]]:
    """Give each amine family equal mass within one exact coarse amine program."""

    if not rows:
        raise UgiAmineSemanticProgramError("amine semantic support is empty")
    by_program: dict[AmineProgramKey, dict[str, list[UgiAmineSemanticTarget]]] = {}
    for program, family, target in rows:
        if len(program) != 4 or any(value < 0 for value in program) or not family:
            raise UgiAmineSemanticProgramError("amine semantic estimator row is malformed")
        by_program.setdefault(program, {}).setdefault(family, []).append(target)

    output: dict[AmineProgramKey, tuple[tuple[UgiAmineSemanticTarget, ...], np.ndarray]] = {}
    for program, by_family in sorted(by_program.items()):
        mass: dict[TargetKey, float] = {}
        family_mass = 1.0 / len(by_family)
        for targets in by_family.values():
            row_mass = family_mass / len(targets)
            for target in targets:
                mass[target.key] = mass.get(target.key, 0.0) + row_mass
        keys = tuple(sorted(mass))
        targets = tuple(UgiAmineSemanticTarget.from_key(key) for key in keys)
        probabilities = np.asarray([mass[key] for key in keys], dtype=np.float64)
        probabilities /= probabilities.sum()
        if np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1.0):
            raise UgiAmineSemanticProgramError(
                "conditional amine semantic probabilities are invalid"
            )
        output[program] = (targets, probabilities)
    return output


@dataclass(frozen=True)
class UgiMeasuredAmineSemanticPrior:
    """Conditional identity-free amine semantics estimated from measured train products."""

    support: Mapping[AmineProgramKey, tuple[tuple[UgiAmineSemanticTarget, ...], np.ndarray]]
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
    ) -> UgiMeasuredAmineSemanticPrior:
        assignments = assignments_path.resolve()
        assignments_sha256 = str(sha256_file(assignments))
        if (
            expected_assignments_sha256 is not None
            and assignments_sha256 != expected_assignments_sha256
        ):
            raise UgiAmineSemanticProgramError(
                "Ugi assignments changed: expected "
                f"{expected_assignments_sha256}, found {assignments_sha256}"
            )
        required = {
            "product_id",
            "primary_product_fold",
            "is_source_adjudicated_measured_product",
            "amine_head_smiles",
            "amine_head_family_id",
        }
        assignments_by_product: dict[str, tuple[str, UgiAmineSemanticTarget]] = {}
        for row in iter_csv(assignments):
            if not required.issubset(row):
                raise UgiAmineSemanticProgramError("Ugi assignment schema changed")
            if row["primary_product_fold"] != "train" or str(
                row["is_source_adjudicated_measured_product"]
            ).strip().lower() not in {"1", "true"}:
                continue
            product_id = str(row["product_id"]).strip()
            family = str(row["amine_head_family_id"]).strip()
            smiles = str(row["amine_head_smiles"]).strip()
            if not product_id or not family or not smiles:
                raise UgiAmineSemanticProgramError(
                    "measured Ugi train row lacks amine identity metadata"
                )
            value = (family, amine_semantic_target(smiles))
            previous = assignments_by_product.setdefault(product_id, value)
            if previous != value:
                raise UgiAmineSemanticProgramError(
                    "one measured product maps to inconsistent amine semantics"
                )
        if not assignments_by_product:
            raise UgiAmineSemanticProgramError("no measured Ugi train products were found")

        observed: set[str] = set()
        rows: list[tuple[AmineProgramKey, str, UgiAmineSemanticTarget]] = []
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
            family, target = assignment
            expected_heavy_atoms = amine_program_key(program)[0] + 1
            if target.nitrogen_atoms + target.oxygen_atoms > expected_heavy_atoms:
                raise UgiAmineSemanticProgramError(
                    "amine semantic element counts exceed the coarse program size"
                )
            rows.append((amine_program_key(program), family, target))
        missing = set(assignments_by_product).difference(observed)
        if missing:
            raise UgiAmineSemanticProgramError(
                f"production cache lacks {len(missing)} measured train products"
            )
        support = build_equal_family_conditional_distribution(rows)
        unique_targets = {target.key for _, _, target in rows}
        support_payload = [
            {
                "amine_program": list(program),
                "targets": [
                    {"target": target.to_mapping(), "probability": float(probability)}
                    for target, probability in zip(targets, probabilities.tolist(), strict=True)
                ],
            }
            for program, (targets, probabilities) in sorted(support.items())
        ]
        audit = {
            "schema_version": "forge.ugi_measured_amine_semantic_prior.v1",
            "reaction_id": reaction_id,
            "training_fold": "train",
            "reference": "source-adjudicated measured Ugi train products only",
            "weighting": "equal amine-family mass within exact coarse amine program",
            "measured_training_rows": len(assignments_by_product),
            "eligible_rows": len(rows),
            "excluded_by_existing_decoder_program_support": excluded,
            "conditional_coarse_programs": len(support),
            "unique_semantic_targets": len(unique_targets),
            "support_sha256": str(sha256_json(support_payload)),
            "assignments_path": str(assignments),
            "assignments_sha256": assignments_sha256,
            "sampled_target_fields": list(UgiAmineSemanticTarget(1, 0, 1, 0).to_mapping()),
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
    ) -> tuple[UgiAmineSemanticTarget, ...]:
        """Draw one semantic target per existing coarse program, without retry."""

        if not programs or seed < 0:
            raise UgiAmineSemanticProgramError("semantic target request is invalid")
        rng = np.random.default_rng(seed)
        output: list[UgiAmineSemanticTarget] = []
        for program in programs:
            key = amine_program_key(program)
            if key not in self.support:
                raise UgiAmineSemanticProgramError(
                    f"coarse amine program {key} lacks measured semantic support"
                )
            targets, probabilities = self.support[key]
            selected = int(rng.choice(len(targets), p=probabilities))
            output.append(targets[selected])
        return tuple(output)


__all__ = [
    "UgiAmineSemanticProgramError",
    "UgiAmineSemanticTarget",
    "UgiMeasuredAmineSemanticPrior",
    "amine_local_substitution_metrics",
    "amine_program_key",
    "amine_semantic_target",
    "build_equal_family_conditional_distribution",
]
