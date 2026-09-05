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
from collections.abc import Iterable, Mapping, Sequence
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
BondTerminalCount = tuple[str, int, int, str, str, int, int]
UnsaturationPatternFrequency = tuple[str, tuple[int, ...], tuple[int, ...], int]
HeadArrangementSignature = tuple[Any, ...]
HeadTopologySignature = tuple[Any, ...]


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
        raise UgiRoleChemistryPriorError(f"{label} changed: expected {expected}, found {observed}")
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


def _ring_atom_bucket(count: int) -> int:
    if count <= 0:
        return 0
    if count <= 4:
        return 1
    if count <= 6:
        return 2
    return 3


def _head_arrangement_signatures(
    nodes: Sequence[int],
    *,
    symbols_by_node: Mapping[int, str],
    depths_by_node: Mapping[int, int],
    neighbors: Mapping[int, Iterable[int]] | Sequence[set[int]],
    ring_nodes: frozenset[int] | set[int],
) -> tuple[HeadArrangementSignature, HeadArrangementSignature, HeadArrangementSignature]:
    """Encode whole-head topology and heteroatom placement without component identity."""

    ordered = tuple(sorted(int(node) for node in nodes))
    if not ordered or len(ordered) != len(set(ordered)):
        raise UgiRoleChemistryPriorError("amine-head arrangement nodes are invalid")
    node_set = set(ordered)
    symbols = {node: str(symbols_by_node[node]) for node in ordered}
    depths = {node: int(depths_by_node[node]) for node in ordered}
    if any(depth < 1 for depth in depths.values()):
        raise UgiRoleChemistryPriorError("amine-head arrangement depth is invalid")
    induced_neighbors = {
        node: {int(neighbor) for neighbor in neighbors[node] if int(neighbor) in node_set}
        for node in ordered
    }
    edges = tuple(
        (left, right)
        for left in ordered
        for right in sorted(induced_neighbors[left])
        if left < right
    )
    remaining = set(ordered)
    components = 0
    while remaining:
        components += 1
        frontier = [remaining.pop()]
        for node in frontier:
            unseen = induced_neighbors[node] & remaining
            remaining.difference_update(unseen)
            frontier.extend(sorted(unseen))
    cycle_rank = max(0, len(edges) - len(ordered) + components)
    hetero_nodes = tuple(node for node in ordered if symbols[node] != "C")
    nitrogen_atoms = sum(symbols[node] == "N" for node in ordered)
    oxygen_atoms = sum(symbols[node] == "O" for node in ordered)
    ring_node_set = node_set & set(ring_nodes)
    ring_heteroatoms = sum(node in ring_node_set for node in hetero_nodes)
    pair_counts = Counter(
        tuple(sorted((symbols[left], symbols[right])))
        for left, right in edges
        if symbols[left] != "C" and symbols[right] != "C"
    )
    terminal_heteroatoms = sum(len(induced_neighbors[node]) <= 1 for node in hetero_nodes)
    boundary_heteroatoms = sum(
        any(int(neighbor) not in node_set for neighbor in neighbors[node]) for node in hetero_nodes
    )
    boundary_symbol_pairs = tuple(
        sorted(
            (
                symbols[node],
                str(symbols_by_node.get(int(neighbor), "UNKNOWN")),
            )
            for node in ordered
            for neighbor in neighbors[node]
            if int(neighbor) not in node_set
        )
    )
    degree_histogram = tuple(
        sum(min(len(induced_neighbors[node]), 4) == degree for node in ordered)
        for degree in range(5)
    )
    hetero_features = tuple(
        sorted(
            (
                symbols[node],
                min(depths[node], 6),
                min(len(induced_neighbors[node]), 4),
                int(node in ring_node_set),
                min(sum(int(neighbor) not in node_set for neighbor in neighbors[node]), 2),
            )
            for node in hetero_nodes
        )
    )
    exact = (
        len(ordered),
        nitrogen_atoms,
        oxygen_atoms,
        len(ring_node_set),
        ring_heteroatoms,
        cycle_rank,
        pair_counts[("N", "N")],
        pair_counts[("N", "O")],
        pair_counts[("O", "O")],
        terminal_heteroatoms,
        boundary_heteroatoms,
        boundary_symbol_pairs,
        degree_histogram,
        hetero_features,
    )
    coarse = (
        len(ordered),
        nitrogen_atoms,
        oxygen_atoms,
        _ring_atom_bucket(len(ring_node_set)),
        min(ring_heteroatoms, 3),
        min(cycle_rank, 2),
        min(pair_counts[("N", "N")], 2),
        min(pair_counts[("N", "O")], 2),
        min(pair_counts[("O", "O")], 2),
        min(terminal_heteroatoms, 3),
        min(boundary_heteroatoms, 2),
        boundary_symbol_pairs,
        tuple(
            sorted((symbol, degree, in_ring) for symbol, _, degree, in_ring, _ in hetero_features)
        ),
    )
    basic = (
        len(ordered),
        nitrogen_atoms,
        oxygen_atoms,
        int(bool(ring_node_set)),
        int(sum(pair_counts.values()) > 0),
        min(terminal_heteroatoms, 3),
        min(boundary_heteroatoms, 2),
        tuple(sorted(pair for pair in boundary_symbol_pairs if pair[0] != "C")),
    )
    return exact, coarse, basic


def _head_topology_signatures(
    nodes: Sequence[int],
    *,
    depths_by_node: Mapping[int, int],
    neighbors: Mapping[int, Iterable[int]] | Sequence[set[int]],
    ring_nodes: frozenset[int] | set[int],
) -> tuple[HeadTopologySignature, HeadTopologySignature]:
    """Encode rooted head shape without atom identities or a stored component graph."""

    ordered = tuple(sorted(int(node) for node in nodes))
    if not ordered or len(ordered) != len(set(ordered)):
        raise UgiRoleChemistryPriorError("amine-head topology nodes are invalid")
    node_set = set(ordered)
    depths = {node: int(depths_by_node[node]) for node in ordered}
    if any(depth < 1 for depth in depths.values()):
        raise UgiRoleChemistryPriorError("amine-head topology depth is invalid")
    induced_neighbors = {
        node: {int(neighbor) for neighbor in neighbors[node] if int(neighbor) in node_set}
        for node in ordered
    }
    edges = tuple(
        (left, right)
        for left in ordered
        for right in sorted(induced_neighbors[left])
        if left < right
    )
    components = 0
    remaining = set(ordered)
    while remaining:
        components += 1
        frontier = [remaining.pop()]
        for node in frontier:
            unseen = induced_neighbors[node] & remaining
            remaining.difference_update(unseen)
            frontier.extend(sorted(unseen))
    cycle_rank = max(0, len(edges) - len(ordered) + components)
    ring_set = node_set & set(ring_nodes)
    boundary_nodes = tuple(
        node
        for node in ordered
        if any(int(neighbor) not in node_set for neighbor in neighbors[node])
    )
    degree_histogram = tuple(
        sum(min(len(induced_neighbors[node]), 4) == degree for node in ordered)
        for degree in range(5)
    )
    branch_depths = tuple(
        sorted(min(depths[node], 6) for node in ordered if len(induced_neighbors[node]) >= 3)
    )
    terminal_depths = tuple(
        sorted(min(depths[node], 6) for node in ordered if len(induced_neighbors[node]) <= 1)
    )
    exact = (
        len(ordered),
        cycle_rank,
        len(ring_set),
        degree_histogram,
        tuple(sorted(min(depths[node], 6) for node in ordered)),
        tuple(sorted(len(induced_neighbors[node]) for node in boundary_nodes)),
        branch_depths,
        terminal_depths,
    )
    coarse = (
        len(ordered),
        min(cycle_rank, 2),
        _ring_atom_bucket(len(ring_set)),
        degree_histogram,
        min(max(depths.values()), 6),
        tuple(sorted(len(induced_neighbors[node]) for node in boundary_nodes)),
        len(branch_depths),
        len(terminal_depths),
    )
    return exact, coarse


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
    # Atom contexts deliberately pool distal positions because the atom vocabulary is broad and
    # sparse.  Bond-order placement is a much smaller categorical problem and needs finer depth
    # resolution to distinguish measured distal unsaturation from near-core C=C/C#C placement.
    maximum_bond_depth_bucket: int | None = None
    # A second identity-free coordinate measures bond position from the nearest distal carbon
    # terminus of the precursor-derived role.  Core depth alone does not transfer a measured
    # terminal or internal unsaturation pattern across tails of different lengths.
    bond_terminal_counts: tuple[BondTerminalCount, ...] = ()
    amine_head_exact_signatures: frozenset[HeadArrangementSignature] = frozenset()
    amine_head_coarse_signatures: frozenset[HeadArrangementSignature] = frozenset()
    amine_head_basic_signatures: frozenset[HeadArrangementSignature] = frozenset()
    amine_head_topology_exact_signatures: frozenset[HeadTopologySignature] = frozenset()
    amine_head_topology_coarse_signatures: frozenset[HeadTopologySignature] = frozenset()
    role_unsaturation_count_support: tuple[tuple[str, int, int], ...] = ()
    # Counts are over unique measured train-fold components, not virtual-library product
    # multiplicities.  They calibrate the count-level draw before positional assignments are
    # enumerated, preventing long tails from receiving more mass merely because they contain more
    # possible unsaturation positions.
    role_unsaturation_count_frequencies: tuple[tuple[str, int, int, int], ...] = ()
    # A role-level pattern keeps the terminal offsets of all C=C and C#C bonds together.  This
    # captures the small measured distinction between a terminal alkyne, one internal alkene and
    # a paired internal-diene pattern without storing the originating component or graph.
    role_unsaturation_pattern_frequencies: tuple[UnsaturationPatternFrequency, ...] = ()

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
        maximum_bond_depth_bucket: int | None = None,
        smoothing: float = 1.0,
        minimum_context_count: int = 4,
        expected_assignments_sha256: str | None = None,
        expected_semantic_atoms_sha256: str | None = None,
        expected_semantic_bonds_sha256: str | None = None,
    ) -> UgiRoleChemistryPrior:
        """Fit aggregated statistics without retaining any component constitution."""

        role_tuple = tuple(str(role) for role in roles)
        effective_bond_depth_bucket = (
            maximum_depth_bucket if maximum_bond_depth_bucket is None else maximum_bond_depth_bucket
        )
        if (
            len(role_tuple) != len(set(role_tuple))
            or not role_tuple
            or maximum_depth_bucket < 2
            or isinstance(effective_bond_depth_bucket, bool)
            or not isinstance(effective_bond_depth_bucket, int)
            or effective_bond_depth_bucket < 2
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
        bond_depth_by_atom: dict[tuple[str, int], int] = {}
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
            raw_depth = int(row["distance_to_nearest_core"])
            depth = _depth_bucket(raw_depth, maximum_depth_bucket)
            degree = _degree_bucket(atom.GetDegree())
            in_ring = int(atom.IsInRing())
            state = _atom_key(atom)
            atom_counts[(role, depth, degree, in_ring, *state)] += 1
            atom_metadata[(product_id, atom_index)] = (role, depth, state[0], in_ring)
            bond_depth_by_atom[(product_id, atom_index)] = _depth_bucket(
                raw_depth, effective_bond_depth_bucket
            )
            observed_pairs.add((role, product_id))
        expected_pair_counts = Counter(role for role, _ in representative_pairs)
        if not set(representative_pairs).issubset(observed_pairs):
            raise UgiRoleChemistryPriorError(
                "semantic atoms do not cover every representative train component"
            )

        # Compute distances within each precursor-derived role to a genuine terminal carbon.
        # Molecular degree (rather than role-induced degree) excludes the near-core attachment
        # atom, while the carbon requirement excludes the terminal ester oxygen.  The resulting
        # coordinate is invariant to atom numbering and stores no component constitution.
        terminal_distance_by_atom: dict[tuple[str, int], int] = {}
        for role, product_id in representative_pairs:
            molecule = molecules[product_id]
            nodes = {
                atom_index
                for (candidate_product, atom_index), metadata in atom_metadata.items()
                if candidate_product == product_id and metadata[0] == role
            }
            neighbors = {node: set() for node in nodes}
            for bond in molecule.GetBonds():
                left = bond.GetBeginAtomIdx()
                right = bond.GetEndAtomIdx()
                if left in nodes and right in nodes:
                    neighbors[left].add(right)
                    neighbors[right].add(left)
            frontier = [
                node
                for node in sorted(nodes)
                if molecule.GetAtomWithIdx(node).GetSymbol() == "C"
                and molecule.GetAtomWithIdx(node).GetDegree() == 1
            ]
            distances = {node: 0 for node in frontier}
            cursor = 0
            while cursor < len(frontier):
                node = frontier[cursor]
                cursor += 1
                for neighbor in sorted(neighbors[node]):
                    if neighbor in distances:
                        continue
                    distances[neighbor] = distances[node] + 1
                    frontier.append(neighbor)
            for node, distance in distances.items():
                terminal_distance_by_atom[(product_id, node)] = min(
                    int(distance), effective_bond_depth_bucket
                )

        bond_counts: Counter[tuple[Any, ...]] = Counter()
        bond_terminal_counts: Counter[tuple[Any, ...]] = Counter()
        unsaturation_counts: Counter[tuple[str, str, int]] = Counter()
        unsaturation_terminal_offsets: dict[tuple[str, str, int], list[int]] = defaultdict(list)
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
            bond_depth = min(
                bond_depth_by_atom[(product_id, int(row["begin_atom_index"]))],
                bond_depth_by_atom[(product_id, int(row["end_atom_index"]))],
            )
            bond_counts[
                (
                    left[0],
                    bond_depth,
                    int(_truthy(row["is_in_ring"])),
                    *symbols,
                    bond_state,
                )
            ] += 1
            left_terminal_distance = terminal_distance_by_atom.get(
                (product_id, int(row["begin_atom_index"]))
            )
            right_terminal_distance = terminal_distance_by_atom.get(
                (product_id, int(row["end_atom_index"]))
            )
            if left_terminal_distance is not None and right_terminal_distance is not None:
                bond_terminal_counts[
                    (
                        left[0],
                        min(left_terminal_distance, right_terminal_distance),
                        int(_truthy(row["is_in_ring"])),
                        *symbols,
                        bond_state,
                    )
                ] += 1
            if symbols == ("C", "C") and bond_state in {1, 2}:
                unsaturation_counts[(left[0], product_id, bond_state)] += 1
                if left_terminal_distance is None or right_terminal_distance is None:
                    raise UgiRoleChemistryPriorError(
                        "measured role unsaturation has no distal-carbon terminal coordinate"
                    )
                unsaturation_terminal_offsets[(left[0], product_id, bond_state)].append(
                    min(left_terminal_distance, right_terminal_distance)
                )
        if not atom_counts or not bond_counts:
            raise UgiRoleChemistryPriorError("Ugi local chemistry statistics are empty")

        amine_role = "amine_head"
        exact_signatures: set[HeadArrangementSignature] = set()
        coarse_signatures: set[HeadArrangementSignature] = set()
        basic_signatures: set[HeadArrangementSignature] = set()
        topology_exact_signatures: set[HeadTopologySignature] = set()
        topology_coarse_signatures: set[HeadTopologySignature] = set()
        for role, product_id in representative_pairs:
            if role != amine_role:
                continue
            molecule = molecules[product_id]
            nodes = tuple(
                sorted(
                    atom_index
                    for (candidate_product, atom_index), metadata in atom_metadata.items()
                    if candidate_product == product_id and metadata[0] == amine_role
                )
            )
            signatures = _head_arrangement_signatures(
                nodes,
                symbols_by_node={
                    node: molecule.GetAtomWithIdx(node).GetSymbol()
                    for node in range(molecule.GetNumAtoms())
                },
                depths_by_node={node: atom_metadata[(product_id, node)][1] for node in nodes},
                neighbors={
                    node: {
                        neighbor.GetIdx()
                        for neighbor in molecule.GetAtomWithIdx(node).GetNeighbors()
                    }
                    for node in range(molecule.GetNumAtoms())
                },
                ring_nodes={node for node in nodes if molecule.GetAtomWithIdx(node).IsInRing()},
            )
            exact_signatures.add(signatures[0])
            coarse_signatures.add(signatures[1])
            basic_signatures.add(signatures[2])
            topology_signatures = _head_topology_signatures(
                nodes,
                depths_by_node={node: atom_metadata[(product_id, node)][1] for node in nodes},
                neighbors={
                    node: {
                        neighbor.GetIdx()
                        for neighbor in molecule.GetAtomWithIdx(node).GetNeighbors()
                    }
                    for node in range(molecule.GetNumAtoms())
                },
                ring_nodes={node for node in nodes if molecule.GetAtomWithIdx(node).IsInRing()},
            )
            topology_exact_signatures.add(topology_signatures[0])
            topology_coarse_signatures.add(topology_signatures[1])
        if (
            not exact_signatures
            or not coarse_signatures
            or not basic_signatures
            or not topology_exact_signatures
            or not topology_coarse_signatures
        ):
            raise UgiRoleChemistryPriorError("measured Ugi amine-head arrangements are empty")

        role_unsaturation_count_support = tuple(
            sorted(
                {
                    (
                        role,
                        int(unsaturation_counts[(role, product_id, 1)]),
                        int(unsaturation_counts[(role, product_id, 2)]),
                    )
                    for role, product_id in representative_pairs
                }
            )
        )
        role_unsaturation_count_frequencies = tuple(
            (*key, count)
            for key, count in sorted(
                Counter(
                    (
                        role,
                        int(unsaturation_counts[(role, product_id, 1)]),
                        int(unsaturation_counts[(role, product_id, 2)]),
                    )
                    for role, product_id in representative_pairs
                ).items()
            )
        )
        role_unsaturation_pattern_frequencies = tuple(
            (*key, count)
            for key, count in sorted(
                Counter(
                    (
                        role,
                        tuple(
                            sorted(
                                unsaturation_terminal_offsets.get((role, product_id, 1), ())
                            )
                        ),
                        tuple(
                            sorted(
                                unsaturation_terminal_offsets.get((role, product_id, 2), ())
                            )
                        ),
                    )
                    for role, product_id in representative_pairs
                ).items()
            )
        )

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
            maximum_bond_depth_bucket=effective_bond_depth_bucket,
            bond_terminal_counts=tuple(
                (*key, count) for key, count in sorted(bond_terminal_counts.items())
            ),
            amine_head_exact_signatures=frozenset(exact_signatures),
            amine_head_coarse_signatures=frozenset(coarse_signatures),
            amine_head_basic_signatures=frozenset(basic_signatures),
            amine_head_topology_exact_signatures=frozenset(topology_exact_signatures),
            amine_head_topology_coarse_signatures=frozenset(topology_coarse_signatures),
            role_unsaturation_count_support=role_unsaturation_count_support,
            role_unsaturation_count_frequencies=role_unsaturation_count_frequencies,
            role_unsaturation_pattern_frequencies=role_unsaturation_pattern_frequencies,
        )

    def _atom_distribution(
        self, role: str, depth: int, degree: int, in_ring: bool
    ) -> Counter[tuple[str, int, int, int]]:
        depth = min(int(depth), self.maximum_depth_bucket)
        degree = _degree_bucket(int(degree))
        entries = [
            (entry[:4], entry[4:8], entry[8]) for entry in self.atom_counts if entry[0] == role
        ]
        selectors = (
            lambda context: context == (role, depth, degree, int(in_ring)),
            lambda context: context[0] == role and context[2:] == (degree, int(in_ring)),
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

    def atom_support_tiers(
        self,
        *,
        role: str,
        depth: int,
        degree: int,
        in_ring: bool,
        atom_vocabulary: Sequence[AtomState],
    ) -> np.ndarray:
        """Return novelty-neutral evidence specificity for every atom state.

        A state receives the most specific tier at which it was observed in a sufficiently
        populated measured-training context.  Counts determine whether a context is supported,
        but never make a common supported state preferable to a rarer supported state at the same
        specificity.  This distinction prevents the local prior from acting as a mode-seeking
        component-frequency surrogate.
        """

        depth = min(int(depth), self.maximum_depth_bucket)
        degree = _degree_bucket(int(degree))
        entries = [
            (entry[:4], entry[4:8], int(entry[8])) for entry in self.atom_counts if entry[0] == role
        ]
        selectors = (
            lambda context: context == (role, depth, degree, int(in_ring)),
            lambda context: context[0] == role and context[2:] == (degree, int(in_ring)),
            lambda context: context[:2] == (role, depth),
            lambda context: context[0] == role,
        )
        tiers = np.zeros(len(atom_vocabulary), dtype=np.float64)
        for tier, selector in zip((4.0, 3.0, 2.0, 1.0), selectors, strict=True):
            counts: Counter[tuple[str, int, int, int]] = Counter()
            for context, state, count in entries:
                if selector(context):
                    counts[state] += count
            if sum(counts.values()) < self.minimum_context_count:
                continue
            for index, atom in enumerate(atom_vocabulary):
                if tiers[index] == 0 and counts[atom.key()] > 0:
                    tiers[index] = tier
        return tiers

    def _bond_distribution(
        self,
        role: str,
        depth: int,
        in_ring: bool,
        symbols: tuple[str, str],
    ) -> Counter[int]:
        depth = min(
            int(depth),
            self.maximum_depth_bucket
            if self.maximum_bond_depth_bucket is None
            else self.maximum_bond_depth_bucket,
        )
        symbols = tuple(sorted(symbols))
        entries = [
            (entry[:5], int(entry[5]), int(entry[6]))
            for entry in self.bond_counts
            if entry[0] == role
        ]
        selectors = (
            lambda context: context == (role, depth, int(in_ring), *symbols),
            lambda context: context[0] == role and context[2:] == (int(in_ring), *symbols),
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

    def bond_support_tiers(
        self,
        *,
        role: str,
        depth: int,
        in_ring: bool,
        symbols: tuple[str, str],
        bond_classes: int,
    ) -> np.ndarray:
        """Return novelty-neutral evidence specificity for each local bond state."""

        depth = min(
            int(depth),
            self.maximum_depth_bucket
            if self.maximum_bond_depth_bucket is None
            else self.maximum_bond_depth_bucket,
        )
        symbols = tuple(sorted(symbols))
        entries = [
            (entry[:5], int(entry[5]), int(entry[6]))
            for entry in self.bond_counts
            if entry[0] == role
        ]
        selectors = (
            lambda context: context == (role, depth, int(in_ring), *symbols),
            lambda context: context[0] == role and context[2:] == (int(in_ring), *symbols),
            lambda context: context[0] == role and context[3:] == symbols,
            lambda context: context[0] == role,
        )
        tiers = np.zeros(bond_classes, dtype=np.float64)
        for tier, selector in zip((4.0, 3.0, 2.0, 1.0), selectors, strict=True):
            counts: Counter[int] = Counter()
            for context, state, count in entries:
                if selector(context):
                    counts[state] += count
            if sum(counts.values()) < self.minimum_context_count:
                continue
            for state in range(bond_classes):
                if tiers[state] == 0 and counts[state] > 0:
                    tiers[state] = tier
        return tiers

    def bond_terminal_support_tiers(
        self,
        *,
        role: str,
        terminal_offset: int,
        in_ring: bool,
        symbols: tuple[str, str],
        bond_classes: int,
    ) -> np.ndarray:
        """Return support specificity indexed from the nearest distal carbon terminus."""

        maximum = (
            self.maximum_depth_bucket
            if self.maximum_bond_depth_bucket is None
            else self.maximum_bond_depth_bucket
        )
        offset = min(int(terminal_offset), maximum)
        symbols = tuple(sorted(symbols))
        entries = [
            (entry[:5], int(entry[5]), int(entry[6]))
            for entry in self.bond_terminal_counts
            if entry[0] == role
        ]
        selectors = (
            lambda context: context == (role, offset, int(in_ring), *symbols),
            lambda context: context[0] == role and context[2:] == (int(in_ring), *symbols),
            lambda context: context[0] == role and context[3:] == symbols,
            lambda context: context[0] == role,
        )
        tiers = np.zeros(bond_classes, dtype=np.float64)
        for tier, selector in zip((4.0, 3.0, 2.0, 1.0), selectors, strict=True):
            counts: Counter[int] = Counter()
            for context, state, count in entries:
                if selector(context):
                    counts[state] += count
            if sum(counts.values()) < self.minimum_context_count:
                continue
            for state in range(bond_classes):
                if tiers[state] == 0 and counts[state] > 0:
                    tiers[state] = tier
        return tiers

    def edge_symbol_support_tier(
        self,
        *,
        role: str,
        depth: int,
        in_ring: bool,
        symbols: tuple[str, str],
    ) -> float:
        """Return the strongest measured context supporting an adjacent symbol pair.

        Unlike ``bond_support_tiers``, this score is independent of bond order and can therefore
        rank a complete amine atom assignment before bond decoding.  It is the missing local check
        for uncommon N--N or O--N placement in otherwise valid heads.
        """

        depth = min(int(depth), self.maximum_depth_bucket)
        symbols = tuple(sorted(symbols))
        selectors = (
            lambda context: context == (role, depth, int(in_ring), *symbols),
            lambda context: context[0] == role and context[2:] == (int(in_ring), *symbols),
            lambda context: context[0] == role and context[3:] == symbols,
        )
        for tier, selector in zip((4.0, 3.0, 2.0), selectors, strict=True):
            count = sum(
                int(entry[6])
                for entry in self.bond_counts
                if entry[0] == role and selector(entry[:5])
            )
            if count >= self.minimum_context_count:
                return tier
        return 0.0

    def amine_head_arrangement_support_tier(
        self,
        *,
        nodes: Sequence[int],
        symbols_by_node: Mapping[int, str],
        depths_by_node: Mapping[int, int],
        neighbors: Mapping[int, Iterable[int]] | Sequence[set[int]],
        ring_nodes: frozenset[int] | set[int],
    ) -> float:
        """Return novelty-neutral support for a completed amine-head arrangement."""

        exact, coarse, basic = _head_arrangement_signatures(
            nodes,
            symbols_by_node=symbols_by_node,
            depths_by_node=depths_by_node,
            neighbors=neighbors,
            ring_nodes=ring_nodes,
        )
        if exact in self.amine_head_exact_signatures:
            return 4.0
        if coarse in self.amine_head_coarse_signatures:
            return 3.0
        if basic in self.amine_head_basic_signatures:
            return 2.0
        return 0.0

    def amine_head_arrangement_similarity(
        self,
        *,
        nodes: Sequence[int],
        symbols_by_node: Mapping[int, str],
        depths_by_node: Mapping[int, int],
        neighbors: Mapping[int, Iterable[int]] | Sequence[set[int]],
        ring_nodes: frozenset[int] | set[int],
    ) -> float:
        """Return graded proximity to measured whole-head morphology, without identity lookup."""

        _, _, candidate = _head_arrangement_signatures(
            nodes,
            symbols_by_node=symbols_by_node,
            depths_by_node=depths_by_node,
            neighbors=neighbors,
            ring_nodes=ring_nodes,
        )

        def multiset_distance(left: Sequence[Any], right: Sequence[Any]) -> float:
            left_counts = Counter(left)
            right_counts = Counter(right)
            denominator = max(sum(left_counts.values()), sum(right_counts.values()), 1)
            return (
                sum(
                    abs(left_counts[value] - right_counts[value])
                    for value in left_counts.keys() | right_counts.keys()
                )
                / denominator
            )

        distances = []
        for reference in self.amine_head_basic_signatures:
            scale = max(int(candidate[0]), int(reference[0]), 1)
            distance = (
                abs(int(candidate[0]) - int(reference[0])) / scale
                + abs(int(candidate[1]) - int(reference[1]))
                + abs(int(candidate[2]) - int(reference[2]))
                + float(candidate[3] != reference[3])
                + float(candidate[4] != reference[4])
                + abs(int(candidate[5]) - int(reference[5])) / 3.0
                + abs(int(candidate[6]) - int(reference[6])) / 2.0
                + multiset_distance(candidate[7], reference[7])
            )
            distances.append(float(distance))
        if not distances:
            raise UgiRoleChemistryPriorError("measured amine-head support is empty")
        return 1.0 / (1.0 + min(distances))

    def amine_head_topology_support_tier(
        self,
        *,
        nodes: Sequence[int],
        depths_by_node: Mapping[int, int],
        neighbors: Mapping[int, Iterable[int]] | Sequence[set[int]],
        ring_nodes: frozenset[int] | set[int],
    ) -> float:
        """Return novelty-neutral support for one rooted head topology."""

        exact, coarse = _head_topology_signatures(
            nodes,
            depths_by_node=depths_by_node,
            neighbors=neighbors,
            ring_nodes=ring_nodes,
        )
        if exact in self.amine_head_topology_exact_signatures:
            return 2.0
        if coarse in self.amine_head_topology_coarse_signatures:
            return 1.0
        return 0.0

    def amine_head_arrangement_signatures(
        self,
        *,
        nodes: Sequence[int],
        symbols_by_node: Mapping[int, str],
        depths_by_node: Mapping[int, int],
        neighbors: Mapping[int, Iterable[int]] | Sequence[set[int]],
        ring_nodes: frozenset[int] | set[int],
    ) -> tuple[HeadArrangementSignature, HeadArrangementSignature, HeadArrangementSignature]:
        """Return identity-free exact, coarse and basic arrangement-class labels."""

        return _head_arrangement_signatures(
            nodes,
            symbols_by_node=symbols_by_node,
            depths_by_node=depths_by_node,
            neighbors=neighbors,
            ring_nodes=ring_nodes,
        )

    def supported_unsaturation_counts(self, role: str) -> tuple[tuple[int, int], ...]:
        """Return measured double/triple-bond count pairs without component identities."""

        return tuple(
            (double_count, triple_count)
            for candidate_role, double_count, triple_count in self.role_unsaturation_count_support
            if candidate_role == role
        )

    def unsaturation_count_frequencies(self, role: str) -> tuple[tuple[int, int, int], ...]:
        """Return count-level frequencies over unique measured train-fold role components."""

        frequencies = tuple(
            (double_count, triple_count, count)
            for candidate_role, double_count, triple_count, count in (
                self.role_unsaturation_count_frequencies
            )
            if candidate_role == role
        )
        if frequencies:
            return frequencies
        # Directly constructed test/legacy priors predate count frequencies.  Uniform support is
        # the only identity-free backward-compatible interpretation; production-built priors always
        # carry explicit measured frequencies.
        return tuple((*value, 1) for value in self.supported_unsaturation_counts(role))

    def unsaturation_pattern_frequencies(
        self, role: str
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...], int], ...]:
        """Return complete measured terminal-offset patterns for one role.

        Each row is ``(double_offsets, triple_offsets, unique_component_count)``.  Offsets are
        anonymous integers and the source component identity is irreversibly aggregated away.
        """

        return tuple(
            (double_offsets, triple_offsets, count)
            for candidate_role, double_offsets, triple_offsets, count in (
                self.role_unsaturation_pattern_frequencies
            )
            if candidate_role == role
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return an auditable identity-free policy receipt."""

        statistics = {
            "atom_counts": [list(value) for value in self.atom_counts],
            "bond_counts": [list(value) for value in self.bond_counts],
            "bond_terminal_counts": [list(value) for value in self.bond_terminal_counts],
            "amine_head_exact_signatures": sorted(self.amine_head_exact_signatures),
            "amine_head_coarse_signatures": sorted(self.amine_head_coarse_signatures),
            "amine_head_basic_signatures": sorted(self.amine_head_basic_signatures),
            "amine_head_topology_exact_signatures": sorted(
                self.amine_head_topology_exact_signatures
            ),
            "amine_head_topology_coarse_signatures": sorted(
                self.amine_head_topology_coarse_signatures
            ),
            "role_unsaturation_count_support": [
                list(value) for value in self.role_unsaturation_count_support
            ],
            "role_unsaturation_count_frequencies": [
                list(value) for value in self.role_unsaturation_count_frequencies
            ],
            "role_unsaturation_pattern_frequencies": [
                [role, list(double_offsets), list(triple_offsets), count]
                for role, double_offsets, triple_offsets, count in (
                    self.role_unsaturation_pattern_frequencies
                )
            ],
        }
        return {
            "schema_version": "forge.ugi_role_chemistry_prior.v4",
            "reaction_id": self.reaction_id,
            "roles": list(self.roles),
            "maximum_depth_bucket": self.maximum_depth_bucket,
            "maximum_bond_depth_bucket": (
                self.maximum_depth_bucket
                if self.maximum_bond_depth_bucket is None
                else self.maximum_bond_depth_bucket
            ),
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
            "representative_components_by_role": dict(self.representative_components_by_role),
            "representative_pairs_sha256": self.representative_pairs_sha256,
            "statistics_sha256": str(sha256_json(statistics)),
            "atom_context_rows": len(self.atom_counts),
            "bond_context_rows": len(self.bond_counts),
            "bond_terminal_context_rows": len(self.bond_terminal_counts),
            "bond_terminal_coordinate": (
                "shortest role-induced distance to a full-molecule degree-one carbon terminus"
            ),
            "amine_head_arrangement_support": {
                "exact_signatures": len(self.amine_head_exact_signatures),
                "coarse_signatures": len(self.amine_head_coarse_signatures),
                "basic_signatures": len(self.amine_head_basic_signatures),
                "frequency_weighting": False,
            },
            "amine_head_topology_support": {
                "exact_signatures": len(self.amine_head_topology_exact_signatures),
                "coarse_signatures": len(self.amine_head_topology_coarse_signatures),
                "frequency_weighting": False,
            },
            "role_unsaturation_count_support": {
                role: [
                    [double_count, triple_count]
                    for candidate_role, double_count, triple_count in (
                        self.role_unsaturation_count_support
                    )
                    if candidate_role == role
                ]
                for role in self.roles
            },
            "role_unsaturation_count_frequencies": {
                role: [
                    [double_count, triple_count, count]
                    for candidate_role, double_count, triple_count, count in (
                        self.role_unsaturation_count_frequencies
                    )
                    if candidate_role == role
                ]
                for role in self.roles
            },
            "role_unsaturation_pattern_frequencies": {
                role: [
                    [list(double_offsets), list(triple_offsets), count]
                    for candidate_role, double_offsets, triple_offsets, count in (
                        self.role_unsaturation_pattern_frequencies
                    )
                    if candidate_role == role
                ]
                for role in self.roles
            },
            "component_identity_conditioning": False,
            "component_graph_conditioning": False,
            "fragment_vocabulary_conditioning": False,
            "hard_support_changed": False,
            "support_tier_semantics": (
                "context evidence specificity without within-tier frequency preference"
            ),
        }


__all__ = ["UgiRoleChemistryPrior", "UgiRoleChemistryPriorError"]
