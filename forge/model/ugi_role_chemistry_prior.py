"""Train-fold-only soft local-chemistry priors for Ugi terminal decoding.

The prior stores aggregated atom- and bond-state counts, never component identities,
component graphs, or fragment tokens.  It therefore calibrates the Transformer's local
chemistry readout without turning open-ended generation into catalogue selection.
"""

from __future__ import annotations

import csv
import gzip
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from forge.core.hashing import sha256_file, sha256_json
from forge.model.defog_feasibility import AtomState

BOND_STATE_BY_NAME = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
DEFAULT_ROLES = (
    "amine_head",
    "oxoester_aldehyde_body_tail",
    "isocyanide_tail",
)


class UgiRoleChemistryPriorError(ValueError):
    """The empirical chemistry-prior inputs or policy are malformed."""


AtomCount = tuple[str, int, int, int, str, int, int, int, int]
BondCount = tuple[str, int, int, str, str, int, int]


def _open_csv(path: Path) -> Iterable[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise UgiRoleChemistryPriorError(f"CSV has no header: {path}")
        yield from reader


def _checked_sha256(path: Path, expected: str | None, label: str) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise UgiRoleChemistryPriorError(f"{label} is missing: {resolved}")
    observed = str(sha256_file(resolved))
    if expected is not None and observed != expected:
        raise UgiRoleChemistryPriorError(
            f"{label} changed: expected {expected}, found {observed}"
        )
    return observed


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true"}


def _depth_bucket(depth: int, maximum_depth_bucket: int) -> int:
    if depth < 1:
        raise UgiRoleChemistryPriorError("exterior atom has nonpositive distance to the core")
    return min(depth, maximum_depth_bucket)


def _degree_bucket(degree: int) -> int:
    if degree < 1:
        raise UgiRoleChemistryPriorError("exterior atom has zero heavy-atom degree")
    return min(degree, 4)


def _atom_key(atom: Chem.Atom) -> tuple[str, int, int, int]:
    return (
        atom.GetSymbol(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        atom.GetNumExplicitHs(),
    )


@dataclass(frozen=True)
class UgiRoleChemistryPrior:
    """Aggregated local state frequencies from unique measured train-fold components."""

    reaction_id: str
    roles: tuple[str, ...]
    maximum_depth_bucket: int
    smoothing: float
    minimum_context_count: int
    assignments_path: Path
    assignments_sha256: str
    semantic_atoms_path: Path
    semantic_atoms_sha256: str
    semantic_bonds_path: Path
    semantic_bonds_sha256: str
    representative_components_by_role: tuple[tuple[str, int], ...]
    representative_pairs_sha256: str
    atom_counts: tuple[AtomCount, ...]
    bond_counts: tuple[BondCount, ...]

    @classmethod
    def from_training_data(
        cls,
        *,
        assignments_path: Path,
        semantic_atoms_path: Path,
        semantic_bonds_path: Path,
        reaction_id: str = "ugi_3cr_agile",
        roles: Sequence[str] = DEFAULT_ROLES,
        maximum_depth_bucket: int = 6,
        smoothing: float = 1.0,
        minimum_context_count: int = 4,
        expected_assignments_sha256: str | None = None,
        expected_semantic_atoms_sha256: str | None = None,
        expected_semantic_bonds_sha256: str | None = None,
    ) -> UgiRoleChemistryPrior:
        """Fit aggregated statistics without retaining any component constitution."""

        role_tuple = tuple(str(role) for role in roles)
        if (
            len(role_tuple) != len(set(role_tuple))
            or not role_tuple
            or maximum_depth_bucket < 2
            or not math.isfinite(smoothing)
            or smoothing <= 0
            or minimum_context_count < 1
        ):
            raise UgiRoleChemistryPriorError("invalid role-chemistry prior policy")
        assignments = assignments_path.resolve()
        semantic_atoms = semantic_atoms_path.resolve()
        semantic_bonds = semantic_bonds_path.resolve()
        assignments_sha = _checked_sha256(
            assignments, expected_assignments_sha256, "Ugi assignments"
        )
        semantic_atoms_sha = _checked_sha256(
            semantic_atoms, expected_semantic_atoms_sha256, "Ugi semantic atoms"
        )
        semantic_bonds_sha = _checked_sha256(
            semantic_bonds, expected_semantic_bonds_sha256, "Ugi semantic bonds"
        )

        products: dict[str, str] = {}
        component_products: dict[str, dict[str, set[str]]] = {
            role: defaultdict(set) for role in role_tuple
        }
        required_assignment_columns = {
            "product_id",
            "canonical_product_smiles",
            "primary_product_fold",
            "is_source_adjudicated_measured_product",
            *(f"{role}_smiles" for role in role_tuple),
        }
        for row in _open_csv(assignments):
            if not required_assignment_columns.issubset(row):
                raise UgiRoleChemistryPriorError("Ugi assignments schema changed")
            if row["primary_product_fold"] != "train" or not _truthy(
                row["is_source_adjudicated_measured_product"]
            ):
                continue
            product_id = row["product_id"].strip()
            product_smiles = row["canonical_product_smiles"].strip()
            if not product_id or not product_smiles:
                raise UgiRoleChemistryPriorError("measured train row lacks product identity")
            previous = products.setdefault(product_id, product_smiles)
            if previous != product_smiles:
                raise UgiRoleChemistryPriorError("one product ID maps to multiple constitutions")
            for role in role_tuple:
                component_smiles = row[f"{role}_smiles"].strip()
                if not component_smiles:
                    raise UgiRoleChemistryPriorError(
                        f"measured train row lacks the {role!r} component"
                    )
                component_products[role][component_smiles].add(product_id)
        if not products or any(not values for values in component_products.values()):
            raise UgiRoleChemistryPriorError(
                "measured Ugi train fold does not cover every precursor role"
            )

        # One deterministic product represents each unique role component.  This removes the
        # Cartesian-product replication in the virtual library while preserving node-level local
        # chemistry statistics inside each selected component.
        representative_pairs = tuple(
            sorted(
                (role, min(product_ids))
                for role, by_component in component_products.items()
                for product_ids in by_component.values()
            )
        )
        selected_roles_by_product: dict[str, set[str]] = defaultdict(set)
        for role, product_id in representative_pairs:
            selected_roles_by_product[product_id].add(role)
        molecules: dict[str, Chem.Mol] = {}
        for product_id in selected_roles_by_product:
            molecule = Chem.MolFromSmiles(products[product_id])
            if molecule is None:
                raise UgiRoleChemistryPriorError(
                    f"measured train product no longer parses: {product_id}"
                )
            molecules[product_id] = molecule

        atom_counts: Counter[tuple[Any, ...]] = Counter()
        atom_metadata: dict[tuple[str, int], tuple[str, int, str, int]] = {}
        observed_pairs: set[tuple[str, str]] = set()
        required_atom_columns = {
            "product_id",
            "product_atom_index",
            "origin_role",
            "is_ugi_core",
            "distance_to_nearest_core",
        }
        for row in _open_csv(semantic_atoms):
            if not required_atom_columns.issubset(row):
                raise UgiRoleChemistryPriorError("Ugi semantic-atom schema changed")
            product_id = row["product_id"]
            role = row["origin_role"]
            if role not in selected_roles_by_product.get(product_id, set()) or _truthy(
                row["is_ugi_core"]
            ):
                continue
            atom_index = int(row["product_atom_index"])
            molecule = molecules[product_id]
            if atom_index < 0 or atom_index >= molecule.GetNumAtoms():
                raise UgiRoleChemistryPriorError("semantic atom index is outside its product")
            atom = molecule.GetAtomWithIdx(atom_index)
            depth = _depth_bucket(int(row["distance_to_nearest_core"]), maximum_depth_bucket)
            degree = _degree_bucket(atom.GetDegree())
            in_ring = int(atom.IsInRing())
            state = _atom_key(atom)
            atom_counts[(role, depth, degree, in_ring, *state)] += 1
            atom_metadata[(product_id, atom_index)] = (role, depth, state[0], in_ring)
            observed_pairs.add((role, product_id))
        expected_pair_counts = Counter(role for role, _ in representative_pairs)
        if not set(representative_pairs).issubset(observed_pairs):
            raise UgiRoleChemistryPriorError(
                "semantic atoms do not cover every representative train component"
            )

        bond_counts: Counter[tuple[Any, ...]] = Counter()
        required_bond_columns = {
            "product_id",
            "begin_atom_index",
            "end_atom_index",
            "bond_type",
            "is_in_ring",
        }
        for row in _open_csv(semantic_bonds):
            if not required_bond_columns.issubset(row):
                raise UgiRoleChemistryPriorError("Ugi semantic-bond schema changed")
            product_id = row["product_id"]
            left = atom_metadata.get((product_id, int(row["begin_atom_index"])))
            right = atom_metadata.get((product_id, int(row["end_atom_index"])))
            if left is None or right is None or left[0] != right[0]:
                continue
            try:
                bond_state = BOND_STATE_BY_NAME[row["bond_type"]]
            except KeyError as error:
                raise UgiRoleChemistryPriorError(
                    f"unsupported semantic bond type: {row['bond_type']!r}"
                ) from error
            symbols = tuple(sorted((left[2], right[2])))
            bond_counts[
                (left[0], min(left[1], right[1]), int(_truthy(row["is_in_ring"])), *symbols, bond_state)
            ] += 1
        if not atom_counts or not bond_counts:
            raise UgiRoleChemistryPriorError("Ugi local chemistry statistics are empty")

        return cls(
            reaction_id=reaction_id,
            roles=role_tuple,
            maximum_depth_bucket=maximum_depth_bucket,
            smoothing=float(smoothing),
            minimum_context_count=minimum_context_count,
            assignments_path=assignments,
            assignments_sha256=assignments_sha,
            semantic_atoms_path=semantic_atoms,
            semantic_atoms_sha256=semantic_atoms_sha,
            semantic_bonds_path=semantic_bonds,
            semantic_bonds_sha256=semantic_bonds_sha,
            representative_components_by_role=tuple(sorted(expected_pair_counts.items())),
            representative_pairs_sha256=str(sha256_json(representative_pairs)),
            atom_counts=tuple((*key, count) for key, count in sorted(atom_counts.items())),
            bond_counts=tuple((*key, count) for key, count in sorted(bond_counts.items())),
        )

    def _atom_distribution(
        self, role: str, depth: int, degree: int, in_ring: bool
    ) -> Counter[tuple[str, int, int, int]]:
        depth = min(int(depth), self.maximum_depth_bucket)
        degree = _degree_bucket(int(degree))
        entries = [
            (entry[:4], entry[4:8], entry[8])
            for entry in self.atom_counts
            if entry[0] == role
        ]
        selectors = (
            lambda context: context == (role, depth, degree, int(in_ring)),
            lambda context: context[0] == role
            and context[2:] == (degree, int(in_ring)),
            lambda context: context[:2] == (role, depth),
            lambda context: context[0] == role,
        )
        for selector in selectors:
            counts: Counter[tuple[str, int, int, int]] = Counter()
            for context, state, count in entries:
                if selector(context):
                    counts[state] += int(count)
            if sum(counts.values()) >= self.minimum_context_count:
                return counts
        return Counter()

    def atom_log_bias(
        self,
        *,
        role: str,
        depth: int,
        degree: int,
        in_ring: bool,
        atom_vocabulary: Sequence[AtomState],
    ) -> np.ndarray:
        """Return smoothed log probabilities aligned to the model atom vocabulary."""

        counts = self._atom_distribution(role, depth, degree, in_ring)
        if not counts:
            return np.zeros(len(atom_vocabulary), dtype=np.float64)
        total = float(sum(counts.values()))
        denominator = total + self.smoothing * len(atom_vocabulary)
        values = np.asarray(
            [
                math.log((counts.get(atom.key(), 0) + self.smoothing) / denominator)
                for atom in atom_vocabulary
            ],
            dtype=np.float64,
        )
        return values - values.max()

    def _bond_distribution(
        self,
        role: str,
        depth: int,
        in_ring: bool,
        symbols: tuple[str, str],
    ) -> Counter[int]:
        depth = min(int(depth), self.maximum_depth_bucket)
        symbols = tuple(sorted(symbols))
        entries = [
            (entry[:5], int(entry[5]), int(entry[6]))
            for entry in self.bond_counts
            if entry[0] == role
        ]
        selectors = (
            lambda context: context == (role, depth, int(in_ring), *symbols),
            lambda context: context[0] == role
            and context[2:] == (int(in_ring), *symbols),
            lambda context: context[0] == role and context[3:] == symbols,
            lambda context: context[0] == role,
        )
        for selector in selectors:
            counts: Counter[int] = Counter()
            for context, state, count in entries:
                if selector(context):
                    counts[state] += count
            if sum(counts.values()) >= self.minimum_context_count:
                return counts
        return Counter()

    def bond_log_bias(
        self,
        *,
        role: str,
        depth: int,
        in_ring: bool,
        symbols: tuple[str, str],
        bond_classes: int,
    ) -> np.ndarray:
        """Return smoothed log probabilities aligned to bond-state indices."""

        counts = self._bond_distribution(role, depth, in_ring, symbols)
        if not counts:
            return np.zeros(bond_classes, dtype=np.float64)
        total = float(sum(counts.values()))
        denominator = total + self.smoothing * bond_classes
        values = np.asarray(
            [
                math.log((counts.get(state, 0) + self.smoothing) / denominator)
                for state in range(bond_classes)
            ],
            dtype=np.float64,
        )
        return values - values.max()

    def to_mapping(self) -> dict[str, Any]:
        """Return an auditable identity-free policy receipt."""

        statistics = {
            "atom_counts": [list(value) for value in self.atom_counts],
            "bond_counts": [list(value) for value in self.bond_counts],
        }
        return {
            "schema_version": "forge.ugi_role_chemistry_prior.v1",
            "reaction_id": self.reaction_id,
            "roles": list(self.roles),
            "maximum_depth_bucket": self.maximum_depth_bucket,
            "smoothing": self.smoothing,
            "minimum_context_count": self.minimum_context_count,
            "inputs": {
                "assignments": {
                    "path": str(self.assignments_path),
                    "sha256": self.assignments_sha256,
                },
                "semantic_atoms": {
                    "path": str(self.semantic_atoms_path),
                    "sha256": self.semantic_atoms_sha256,
                },
                "semantic_bonds": {
                    "path": str(self.semantic_bonds_path),
                    "sha256": self.semantic_bonds_sha256,
                },
            },
            "training_reference": (
                "unique source-adjudicated role components from the measured Ugi train fold"
            ),
            "representative_components_by_role": dict(
                self.representative_components_by_role
            ),
            "representative_pairs_sha256": self.representative_pairs_sha256,
            "statistics_sha256": str(sha256_json(statistics)),
            "atom_context_rows": len(self.atom_counts),
            "bond_context_rows": len(self.bond_counts),
            "component_identity_conditioning": False,
            "component_graph_conditioning": False,
            "fragment_vocabulary_conditioning": False,
            "hard_support_changed": False,
        }


__all__ = ["UgiRoleChemistryPrior", "UgiRoleChemistryPriorError"]
