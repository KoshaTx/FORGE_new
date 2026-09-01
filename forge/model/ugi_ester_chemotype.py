"""Registry-bound request for the ester-bearing AGILE aldehyde stratum.

The AGILE Ugi transform admits aldehydes both with and without an ester.  This
policy therefore does not alter exact-L1 chemistry and does not claim that an
ester is required by the reaction.  It makes one optional coarse chemotype
coordinate explicit: when requested, terminal decoding must realize one
descriptor-consistent ``C(=O)-O-C`` motif inside the registry-declared
aldehyde-derived role or abstain.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem

from forge.core.hashing import sha256_file
from forge.core.io import read_json_object

QUALIFIED_STATUS = "qualified_for_enumeration"


class UgiEsterChemotypeError(ValueError):
    """The qualified registry cannot bind the requested chemotype coordinate."""


def _lower_quantile(values: list[int], quantile: float) -> int:
    if not values or not 0.0 <= quantile <= 1.0:
        raise UgiEsterChemotypeError("chemotype morphology quantile is undefined")
    ordered = sorted(values)
    return int(ordered[math.floor(quantile * (len(ordered) - 1))])


def _ester_side_carbon_counts(smiles: str) -> tuple[int, int] | None:
    """Return carbon counts on both sides of one unambiguous C(=O)-O-C bond."""

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise UgiEsterChemotypeError(f"training component no longer parses: {smiles}")
    candidates: list[tuple[int, int]] = []
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        double_oxygen = []
        single_oxygen = []
        carbon_neighbors = []
        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            if neighbor.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_oxygen.append(neighbor)
            elif neighbor.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
                single_oxygen.append(neighbor)
            elif neighbor.GetSymbol() == "C":
                carbon_neighbors.append(neighbor)
        for oxygen in single_oxygen:
            if (
                len(double_oxygen) == 1
                and carbon_neighbors
                and any(
                    neighbor.GetSymbol() == "C" and neighbor.GetIdx() != atom.GetIdx()
                    for neighbor in oxygen.GetNeighbors()
                )
            ):
                candidates.append((atom.GetIdx(), oxygen.GetIdx()))
    if len(candidates) != 1:
        return None
    center, ester_oxygen = candidates[0]
    blocked = frozenset((center, ester_oxygen))

    def component(start: int) -> set[int]:
        visited = {start}
        frontier = [start]
        while frontier:
            node = frontier.pop()
            for neighbor in molecule.GetAtomWithIdx(node).GetNeighbors():
                target = int(neighbor.GetIdx())
                if frozenset((node, target)) == blocked or target in visited:
                    continue
                visited.add(target)
                frontier.append(target)
        return visited

    sides = (component(center), component(ester_oxygen))
    if sides[0] & sides[1] or len(sides[0] | sides[1]) != molecule.GetNumAtoms():
        return None
    counts = sorted(
        sum(molecule.GetAtomWithIdx(node).GetSymbol() == "C" for node in side) for side in sides
    )
    return int(counts[0]), int(counts[1])


@dataclass(frozen=True)
class UgiEsterChemotypePolicy:
    """One optional Ugi chemotype coordinate bound to a registry role."""

    reaction_id: str
    amine_role: str
    aldehyde_role: str
    isocyanide_role: str
    registry_path: Path
    registry_sha256: str
    training_assignments_path: Path
    training_assignments_sha256: str
    minimum_role_exterior_atoms: tuple[tuple[str, int], ...]
    maximum_role_exterior_atoms: tuple[tuple[str, int], ...]
    minimum_ester_side_carbons: int
    minimum_ester_long_side_carbons: int
    minimum_amine_exterior_nitrogens: int
    maximum_amine_exterior_nitrogens: int
    morphology_quantile: float
    measured_training_product_ids: tuple[str, ...] = ()
    amine_cycle_sizes_by_exterior_count: tuple[tuple[int, tuple[int, ...]], ...] = ()

    @classmethod
    def from_qualified_registry(
        cls,
        registry_path: Path,
        *,
        training_assignments_path: Path,
        reaction_id: str = "ugi_3cr_agile",
        expected_sha256: str | None = None,
        expected_training_assignments_sha256: str | None = None,
        morphology_quantile: float = 0.25,
    ) -> UgiEsterChemotypePolicy:
        resolved = Path(registry_path).resolve()
        observed = str(sha256_file(resolved))
        if expected_sha256 is not None and observed != expected_sha256:
            raise UgiEsterChemotypeError(
                f"qualified reaction registry changed: expected {expected_sha256}, "
                f"found {observed}"
            )
        registry = read_json_object(
            resolved, error=UgiEsterChemotypeError, label="qualified reaction registry"
        )
        definitions = [
            value
            for value in registry.get("reactions", [])
            if str(value.get("reaction_id")) == reaction_id
        ]
        if len(definitions) != 1 or definitions[0].get("status") != QUALIFIED_STATUS:
            raise UgiEsterChemotypeError(
                f"registry does not expose one qualified reaction {reaction_id!r}"
            )
        role_names = [str(value.get("name")) for value in definitions[0].get("reactant_roles", [])]
        by_kind = {
            kind: [role for role in role_names if kind in role.lower()]
            for kind in ("amine", "aldehyde", "isocyanide")
        }
        if any(len(values) != 1 for values in by_kind.values()):
            raise UgiEsterChemotypeError(
                f"reaction {reaction_id!r} does not expose one amine, aldehyde and isocyanide role"
            )
        if not 0.0 < morphology_quantile < 0.5:
            raise UgiEsterChemotypeError(
                "morphology quantile must be strictly between zero and 0.5"
            )
        assignments = Path(training_assignments_path).resolve()
        assignments_sha256 = str(sha256_file(assignments))
        if (
            expected_training_assignments_sha256 is not None
            and assignments_sha256 != expected_training_assignments_sha256
        ):
            raise UgiEsterChemotypeError(
                "Ugi assignments changed: expected "
                f"{expected_training_assignments_sha256}, found {assignments_sha256}"
            )
        role_definitions = {
            str(value["name"]): value for value in definitions[0].get("reactant_roles", [])
        }
        roles = (by_kind["amine"][0], by_kind["aldehyde"][0], by_kind["isocyanide"][0])
        unique_components = {role: set() for role in roles}
        with gzip.open(assignments, mode="rt", newline="") as handle:
            reader = csv.DictReader(handle)
            expected_columns = {
                "product_id",
                "primary_product_fold",
                "is_source_adjudicated_measured_product",
                *(f"{role}_smiles" for role in roles),
            }
            if reader.fieldnames is None or not expected_columns.issubset(reader.fieldnames):
                raise UgiEsterChemotypeError(
                    "Ugi assignments do not expose the required role fields"
                )
            measured_training_product_ids: set[str] = set()
            for row in reader:
                measured = str(row["is_source_adjudicated_measured_product"]).lower()
                if row["primary_product_fold"] != "train" or measured not in {"1", "true"}:
                    continue
                product_id = str(row["product_id"]).strip()
                if not product_id:
                    raise UgiEsterChemotypeError("measured Ugi training row has no product ID")
                measured_training_product_ids.add(product_id)
                for role in roles:
                    smiles = str(row[f"{role}_smiles"]).strip()
                    if smiles:
                        unique_components[role].add(smiles)
        if any(not values for values in unique_components.values()):
            raise UgiEsterChemotypeError("Ugi train fold has an empty role component support")
        if not measured_training_product_ids:
            raise UgiEsterChemotypeError(
                "Ugi train fold has no source-adjudicated measured products"
            )
        exterior_support: list[tuple[str, int]] = []
        exterior_ceiling: list[tuple[str, int]] = []
        handle_queries: dict[str, Any] = {}
        for role in roles:
            handle_smarts = str(role_definitions[role].get("required_handle_smarts", ""))
            handle_query = Chem.MolFromSmarts(handle_smarts)
            if handle_query is None or handle_query.GetNumAtoms() < 1:
                raise UgiEsterChemotypeError(
                    f"qualified role {role!r} has no parseable reactive-handle SMARTS"
                )
            handle_queries[role] = handle_query
            exterior_counts = []
            for smiles in sorted(unique_components[role]):
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    raise UgiEsterChemotypeError(
                        f"training component no longer parses for {role!r}: {smiles}"
                    )
                exterior = int(molecule.GetNumHeavyAtoms() - handle_query.GetNumAtoms())
                if exterior < 1:
                    raise UgiEsterChemotypeError(
                        f"training component has no exterior atoms for {role!r}: {smiles}"
                    )
                exterior_counts.append(exterior)
            exterior_support.append((role, _lower_quantile(exterior_counts, morphology_quantile)))
            exterior_ceiling.append(
                (role, _lower_quantile(exterior_counts, 1.0 - morphology_quantile))
            )
        ester_sides = [
            value
            for smiles in sorted(unique_components[by_kind["aldehyde"][0]])
            if (value := _ester_side_carbon_counts(smiles)) is not None
        ]
        if not ester_sides:
            raise UgiEsterChemotypeError(
                "Ugi train fold contains no unambiguous ester-bearing aldehyde components"
            )
        amine_handle_nitrogens = sum(
            atom.GetSymbol() == "N" for atom in handle_queries[by_kind["amine"][0]].GetAtoms()
        )
        amine_exterior_nitrogens = []
        amine_cycle_sizes: dict[int, set[int]] = {}
        for smiles in sorted(unique_components[by_kind["amine"][0]]):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:  # guarded above; retain the local type invariant
                raise UgiEsterChemotypeError(f"training amine no longer parses: {smiles}")
            count = (
                sum(atom.GetSymbol() == "N" for atom in molecule.GetAtoms())
                - amine_handle_nitrogens
            )
            if count < 0:
                raise UgiEsterChemotypeError("amine handle exceeds its molecular nitrogen count")
            amine_exterior_nitrogens.append(int(count))
            exterior_count = int(
                molecule.GetNumHeavyAtoms() - handle_queries[by_kind["amine"][0]].GetNumAtoms()
            )
            rings = {len(ring) for ring in molecule.GetRingInfo().AtomRings()}
            if rings:
                amine_cycle_sizes.setdefault(exterior_count, set()).update(rings)
        return cls(
            reaction_id=reaction_id,
            amine_role=by_kind["amine"][0],
            aldehyde_role=by_kind["aldehyde"][0],
            isocyanide_role=by_kind["isocyanide"][0],
            registry_path=resolved,
            registry_sha256=observed,
            training_assignments_path=assignments,
            training_assignments_sha256=assignments_sha256,
            minimum_role_exterior_atoms=tuple(exterior_support),
            maximum_role_exterior_atoms=tuple(exterior_ceiling),
            minimum_ester_side_carbons=_lower_quantile(
                [value[0] for value in ester_sides], morphology_quantile
            ),
            minimum_ester_long_side_carbons=_lower_quantile(
                [value[1] for value in ester_sides], morphology_quantile
            ),
            minimum_amine_exterior_nitrogens=_lower_quantile(
                amine_exterior_nitrogens, morphology_quantile
            ),
            maximum_amine_exterior_nitrogens=_lower_quantile(
                amine_exterior_nitrogens, 1.0 - morphology_quantile
            ),
            morphology_quantile=float(morphology_quantile),
            measured_training_product_ids=tuple(sorted(measured_training_product_ids)),
            amine_cycle_sizes_by_exterior_count=tuple(
                (count, tuple(sorted(sizes))) for count, sizes in sorted(amine_cycle_sizes.items())
            ),
        )

    def minimum_exterior_atoms(self, role: str) -> int:
        support = dict(self.minimum_role_exterior_atoms)
        try:
            return int(support[role])
        except KeyError as error:
            raise UgiEsterChemotypeError(
                f"chemotype has no morphology support for role {role!r}"
            ) from error

    def maximum_exterior_atoms(self, role: str) -> int:
        support = dict(self.maximum_role_exterior_atoms)
        try:
            return int(support[role])
        except KeyError as error:
            raise UgiEsterChemotypeError(
                f"chemotype has no morphology support for role {role!r}"
            ) from error

    @property
    def minimum_constructive_aldehyde_exterior_atoms(self) -> int:
        """Smallest exterior containing both carbon arms and the two ester oxygens."""

        return int(self.minimum_ester_side_carbons + self.minimum_ester_long_side_carbons + 2)

    def minimum_topology_exterior_atoms(self, role: str) -> int:
        """Strengthen the measured floor only when exact chemotype feasibility requires it."""

        measured = self.minimum_exterior_atoms(role)
        if role != self.aldehyde_role:
            return measured
        return max(measured, self.minimum_constructive_aldehyde_exterior_atoms)

    def allowed_amine_cycle_sizes(self, exterior_count: int) -> tuple[int, ...] | None:
        """Return measured ring sizes, or no override outside the measured size support."""

        return dict(self.amine_cycle_sizes_by_exterior_count).get(int(exterior_count))

    def to_mapping(self) -> dict[str, Any]:
        measured_product_payload = "\n".join(self.measured_training_product_ids).encode("utf-8")
        return {
            "reaction_id": self.reaction_id,
            "amine_role": self.amine_role,
            "aldehyde_role": self.aldehyde_role,
            "isocyanide_role": self.isocyanide_role,
            "registry_path": str(self.registry_path),
            "registry_sha256": self.registry_sha256,
            "training_assignments_path": str(self.training_assignments_path),
            "training_assignments_sha256": self.training_assignments_sha256,
            "requested_chemotype": "canonical_ester_bearing_hydrophobic_ugi_lipid",
            "descriptor_equivalent_motif": "C(=O)-O-C",
            "morphology_reference": (
                "unique source-adjudicated measured role components in the Ugi train fold"
            ),
            "morphology_quantile": self.morphology_quantile,
            "minimum_role_exterior_atoms": dict(self.minimum_role_exterior_atoms),
            "maximum_role_exterior_atoms": dict(self.maximum_role_exterior_atoms),
            "minimum_ester_side_carbons": self.minimum_ester_side_carbons,
            "minimum_ester_long_side_carbons": self.minimum_ester_long_side_carbons,
            "minimum_constructive_aldehyde_exterior_atoms": (
                self.minimum_constructive_aldehyde_exterior_atoms
            ),
            "minimum_amine_exterior_nitrogens": self.minimum_amine_exterior_nitrogens,
            "maximum_amine_exterior_nitrogens": self.maximum_amine_exterior_nitrogens,
            "amine_cycle_sizes_by_exterior_count": {
                str(count): list(sizes) for count, sizes in self.amine_cycle_sizes_by_exterior_count
            },
            "measured_training_product_count": len(self.measured_training_product_ids),
            "measured_training_product_ids_sha256": hashlib.sha256(
                measured_product_payload
            ).hexdigest(),
            "morphology_program_sampling": (
                "occurrence_weighted_joint_count_only_programs_from_measured_training_products"
                if self.measured_training_product_ids
                else "factorized_training_prior_conditioned_on_chemotype"
            ),
            "amine_exterior_elements": ["C", "N"],
            "aldehyde_non_motif_exterior_elements": ["C"],
            "isocyanide_exterior_elements": ["C"],
            "generated_exterior_formal_charge": 0,
            "required_by_ugi_transform": False,
            "component_identity_conditioning": False,
        }


__all__ = ["UgiEsterChemotypeError", "UgiEsterChemotypePolicy"]
