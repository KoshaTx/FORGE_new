"""Molecular endpoint evaluation for Phase 1 whole-lipid generation."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from forge.design.audit.lipid_morphology_audit import molecule_morphology
from forge.design.flow.defog_feasibility import AtomState, graph_to_molecule
from forge.design.flow.lipid_context import (
    LIPID_REGION_NAMES,
    assign_lipid_regions,
    select_lipid_polar_root,
)
from forge.design.flow.phase1_flow import Phase1FlowError, TrainingGraphRecord
from forge.design.flow.sparse_topology_feasibility import INDEX_TO_DENSE_BOND

try:
    from rdkit import Chem
    from rdkit import rdBase as rd_base  # noqa: N813
except ModuleNotFoundError:  # pragma: no cover - required by the chem extra
    Chem = None
    rd_base = None


def training_record_edges(record: TrainingGraphRecord) -> np.ndarray:
    """Reconstruct the dense adjacency labels used only for endpoint evaluation."""

    edges = np.zeros((record.node_count, record.node_count), dtype=np.int64)
    for child in range(1, record.node_count):
        parent = int(record.parents[child])
        bond = INDEX_TO_DENSE_BOND[int(record.parent_bonds[child])]
        edges[child, parent] = edges[parent, child] = bond
    for left, right, bond_index in zip(
        record.closure_left,
        record.closure_right,
        record.closure_bonds,
        strict=True,
    ):
        bond = INDEX_TO_DENSE_BOND[int(bond_index)]
        edges[int(left), int(right)] = edges[int(right), int(left)] = bond
    return edges


def _connected(edges: np.ndarray) -> bool:
    if edges.shape[0] == 0:
        return False
    visited = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in np.flatnonzero(edges[node]):
            index = int(neighbor)
            if index not in visited:
                visited.add(index)
                queue.append(index)
    return len(visited) == edges.shape[0]


def _probability(values: Sequence[int], support: int) -> np.ndarray:
    counts = np.bincount(np.asarray(values, dtype=np.int64), minlength=support).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return np.zeros(support, dtype=np.float64)
    return counts / total


def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)

    def divergence(first: np.ndarray, second: np.ndarray) -> float:
        mask = first > 0
        return float(np.sum(first[mask] * np.log(first[mask] / second[mask])))

    return 0.5 * divergence(left, midpoint) + 0.5 * divergence(right, midpoint)


def _canonical_nonisomeric(smiles: str) -> str:
    if Chem is None:
        raise Phase1FlowError("Phase 1 product evaluation requires RDKit")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise Phase1FlowError(f"training record has invalid canonical SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _graph_statistics(
    graphs: Sequence[tuple[np.ndarray, np.ndarray]],
    atom_vocabulary: Sequence[AtomState],
) -> dict[str, Any]:
    degrees: list[int] = []
    cycle_ranks: list[int] = []
    branch_atoms = 0
    atom_count = 0
    encoded_aromatic_atoms = 0
    charged_atoms = 0
    heteroatoms = 0
    atom_states: list[int] = []
    elements: Counter[str] = Counter()
    maximum_degrees: list[int] = []
    bond_counts: Counter[int] = Counter()
    for nodes, edges in graphs:
        local_degrees = np.count_nonzero(edges, axis=1).astype(np.int64)
        degrees.extend(int(value) for value in local_degrees)
        maximum_degrees.append(int(local_degrees.max(initial=0)))
        branch_atoms += int(np.count_nonzero(local_degrees >= 3))
        atom_count += len(nodes)
        edge_count = int(np.count_nonzero(np.triu(edges, 1)))
        bond_counts.update(int(value) for value in edges[np.triu_indices(len(nodes), 1)] if value)
        components = 1 if _connected(edges) else 0
        cycle_ranks.append(max(0, edge_count - len(nodes) + components))
        for node in nodes:
            index = int(node)
            state = atom_vocabulary[index]
            atom_states.append(index)
            elements[state.symbol] += 1
            encoded_aromatic_atoms += int(state.aromatic)
            charged_atoms += int(state.formal_charge != 0)
            heteroatoms += int(state.symbol not in {"C", "H"})
    degree_support = max(degrees, default=0) + 1
    cycle_support = max(cycle_ranks, default=0) + 1
    bond_count = sum(bond_counts.values())
    return {
        "atoms": atom_count,
        "bonds": bond_count,
        "branch_atom_fraction": branch_atoms / max(1, atom_count),
        "nonaromatic_unsaturated_bond_fraction": (bond_counts[2] + bond_counts[3])
        / max(1, bond_count),
        "double_bond_fraction": bond_counts[2] / max(1, bond_count),
        "triple_bond_fraction": bond_counts[3] / max(1, bond_count),
        "aromatic_bond_fraction": bond_counts[4] / max(1, bond_count),
        "encoded_aromatic_atom_fraction": encoded_aromatic_atoms / max(1, atom_count),
        "charged_atom_fraction": charged_atoms / max(1, atom_count),
        "heteroatom_fraction": heteroatoms / max(1, atom_count),
        "mean_maximum_degree": float(np.mean(maximum_degrees)) if maximum_degrees else 0.0,
        "mean_cycle_rank": float(np.mean(cycle_ranks)) if cycle_ranks else 0.0,
        "degrees": degrees,
        "cycle_ranks": cycle_ranks,
        "atom_states": atom_states,
        "element_fractions": {
            element: count / max(1, atom_count) for element, count in sorted(elements.items())
        },
        "degree_support": degree_support,
        "cycle_support": cycle_support,
    }


def _lipid_region_statistics(molecules: Sequence[Any]) -> dict[str, dict[str, float | int]]:
    atom_counts: Counter[int] = Counter()
    branch_counts: Counter[int] = Counter()
    heteroatom_counts: Counter[int] = Counter()
    ring_atom_counts: Counter[int] = Counter()
    unsaturated_atom_counts: Counter[int] = Counter()
    for molecule in molecules:
        regions = assign_lipid_regions(molecule, select_lipid_polar_root(molecule))
        for atom in molecule.GetAtoms():
            index = atom.GetIdx()
            region = int(regions[index])
            atom_counts[region] += 1
            branch_counts[region] += int(atom.GetDegree() >= 3)
            heteroatom_counts[region] += int(atom.GetSymbol() not in {"C", "H"})
            ring_atom_counts[region] += int(atom.IsInRing())
            unsaturated_atom_counts[region] += int(
                any(
                    not bond.GetIsAromatic()
                    and bond.GetBondType() in {Chem.BondType.DOUBLE, Chem.BondType.TRIPLE}
                    for bond in atom.GetBonds()
                )
            )
    total_atoms = sum(atom_counts.values())
    return {
        name: {
            "atoms": atom_counts[index],
            "atom_fraction": atom_counts[index] / max(1, total_atoms),
            "branch_atom_fraction": branch_counts[index] / max(1, atom_counts[index]),
            "heteroatom_fraction": heteroatom_counts[index] / max(1, atom_counts[index]),
            "ring_atom_fraction": ring_atom_counts[index] / max(1, atom_counts[index]),
            "nonaromatic_unsaturated_atom_fraction": unsaturated_atom_counts[index]
            / max(1, atom_counts[index]),
        }
        for index, name in enumerate(LIPID_REGION_NAMES)
    }


def _morphology_statistics(molecules: Sequence[Any]) -> dict[str, Any]:
    rows = [molecule_morphology(molecule) for molecule in molecules]
    tail_components = [component for row in rows for component in row["tail_components"]]
    oxygen_environments: Counter[str] = Counter()
    for row in rows:
        oxygen_environments.update(row["oxygen_environments"])
    oxygen_count = sum(oxygen_environments.values())
    return {
        "mean_maximum_root_distance": (
            float(np.mean([row["maximum_root_distance"] for row in rows])) if rows else 0.0
        ),
        "mean_tail_components_per_molecule": (
            float(np.mean([len(row["tail_components"]) for row in rows])) if rows else 0.0
        ),
        "path_like_tail_component_fraction": sum(
            component["path_like"] for component in tail_components
        )
        / max(1, len(tail_components)),
        "tail_junction_atoms_per_component": sum(
            component["junction_atoms"] for component in tail_components
        )
        / max(1, len(tail_components)),
        "mean_tail_component_atoms": (
            float(np.mean([component["atoms"] for component in tail_components]))
            if tail_components
            else 0.0
        ),
        "mean_tail_component_diameter": (
            float(np.mean([component["diameter"] for component in tail_components]))
            if tail_components
            else 0.0
        ),
        "oxygen_environment_fractions": {
            name: oxygen_environments[name] / max(1, oxygen_count)
            for name in (
                "anionic",
                "carbonyl",
                "ring",
                "single_bond_bridge",
                "terminal_single",
                "other",
            )
        },
    }


def _rdkit_statistics(
    molecules: Sequence[Any],
    *,
    macrocycle_minimum_ring_size: int,
) -> dict[str, Any]:
    if macrocycle_minimum_ring_size < 3:
        raise Phase1FlowError("macrocycle threshold must be at least three")
    aromatic_atoms = 0
    atom_count = 0
    ring_sizes: list[int] = []
    maximum_ring_sizes: list[int] = []
    macrocycle_molecules = 0
    for molecule in molecules:
        local_sizes = [len(ring) for ring in molecule.GetRingInfo().AtomRings()]
        ring_sizes.extend(local_sizes)
        maximum = max(local_sizes, default=0)
        maximum_ring_sizes.append(maximum)
        macrocycle_molecules += int(maximum >= macrocycle_minimum_ring_size)
        aromatic_atoms += sum(atom.GetIsAromatic() for atom in molecule.GetAtoms())
        atom_count += molecule.GetNumAtoms()
    return {
        "aromatic_atom_fraction": aromatic_atoms / max(1, atom_count),
        "rings_per_molecule": len(ring_sizes) / max(1, len(molecules)),
        "mean_ring_size": float(np.mean(ring_sizes)) if ring_sizes else 0.0,
        "mean_maximum_ring_size": (
            float(np.mean(maximum_ring_sizes)) if maximum_ring_sizes else 0.0
        ),
        "macrocycle_molecule_fraction": macrocycle_molecules / max(1, len(molecules)),
        "ring_sizes": ring_sizes,
    }


def evaluate_product_samples(
    samples: Sequence[tuple[np.ndarray, np.ndarray]],
    train_records: Sequence[TrainingGraphRecord],
    atom_vocabulary: Sequence[AtomState],
    *,
    macrocycle_minimum_ring_size: int = 9,
) -> dict[str, Any]:
    """Evaluate generated endpoints against the frozen broad-lipid training fold."""

    if not samples:
        raise Phase1FlowError("product evaluation requires at least one generated sample")
    if not train_records:
        raise Phase1FlowError("product evaluation requires the frozen training records")
    if Chem is None or rd_base is None:
        raise Phase1FlowError("Phase 1 product evaluation requires RDKit")

    train_smiles: set[str] = set()
    reference_molecules = []
    for record in train_records:
        smiles = _canonical_nonisomeric(record.canonical_smiles)
        train_smiles.add(smiles)
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:  # pragma: no cover - guarded by canonicalization
            raise Phase1FlowError(f"canonical training SMILES became invalid: {smiles}")
        reference_molecules.append(molecule)
    train_graphs = [(record.node_states, training_record_edges(record)) for record in train_records]
    valid_smiles: list[str] = []
    generated_molecules = []
    connected = 0
    invalid_reasons: Counter[str] = Counter()
    for nodes, edges in samples:
        connected += int(_connected(edges))
        try:
            with rd_base.BlockLogs():
                molecule = graph_to_molecule(nodes, edges, atom_vocabulary)
                smiles = Chem.MolToSmiles(
                    molecule,
                    canonical=True,
                    isomericSmiles=False,
                )
        except (ValueError, RuntimeError) as exc:
            invalid_reasons[type(exc).__name__] += 1
            continue
        valid_smiles.append(smiles)
        generated_molecules.append(molecule)

    generated = _graph_statistics(samples, atom_vocabulary)
    reference = _graph_statistics(train_graphs, atom_vocabulary)
    generated_rdkit = _rdkit_statistics(
        generated_molecules,
        macrocycle_minimum_ring_size=macrocycle_minimum_ring_size,
    )
    reference_rdkit = _rdkit_statistics(
        reference_molecules,
        macrocycle_minimum_ring_size=macrocycle_minimum_ring_size,
    )
    generated["rdkit"] = {
        key: value for key, value in generated_rdkit.items() if key != "ring_sizes"
    }
    generated["lipid_regions"] = _lipid_region_statistics(generated_molecules)
    generated["morphology"] = _morphology_statistics(generated_molecules)
    reference["rdkit"] = {
        key: value for key, value in reference_rdkit.items() if key != "ring_sizes"
    }
    reference["lipid_regions"] = _lipid_region_statistics(reference_molecules)
    reference["morphology"] = _morphology_statistics(reference_molecules)
    degree_support = max(generated.pop("degree_support"), reference.pop("degree_support"))
    cycle_support = max(generated.pop("cycle_support"), reference.pop("cycle_support"))
    generated_degree = _probability(generated.pop("degrees"), degree_support)
    reference_degree = _probability(reference.pop("degrees"), degree_support)
    generated_cycle = _probability(generated.pop("cycle_ranks"), cycle_support)
    reference_cycle = _probability(reference.pop("cycle_ranks"), cycle_support)
    ring_support = (
        max(
            max(generated_rdkit["ring_sizes"], default=0),
            max(reference_rdkit["ring_sizes"], default=0),
        )
        + 1
    )
    generated_ring_sizes = _probability(generated_rdkit["ring_sizes"], ring_support)
    reference_ring_sizes = _probability(reference_rdkit["ring_sizes"], ring_support)
    generated_states = _probability(generated.pop("atom_states"), len(atom_vocabulary))
    reference_states = _probability(reference.pop("atom_states"), len(atom_vocabulary))
    valid_count = len(valid_smiles)
    unique_count = len(set(valid_smiles))
    memorized_count = sum(smiles in train_smiles for smiles in valid_smiles)
    return {
        "samples": len(samples),
        "validity": valid_count / len(samples),
        "connectedness": connected / len(samples),
        "valid_and_unique_fraction": unique_count / len(samples),
        "uniqueness_among_valid": unique_count / max(1, valid_count),
        "exact_train_memorization_among_valid": memorized_count / max(1, valid_count),
        "exact_train_novelty_among_valid": 1.0 - memorized_count / max(1, valid_count),
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "macrocycle_minimum_ring_size": macrocycle_minimum_ring_size,
        "generated": generated,
        "r0_train_reference": reference,
        "distribution_fidelity": {
            "degree_jensen_shannon": _jensen_shannon(
                generated_degree,
                reference_degree,
            ),
            "cycle_rank_jensen_shannon": _jensen_shannon(
                generated_cycle,
                reference_cycle,
            ),
            "ring_size_jensen_shannon_among_rings": _jensen_shannon(
                generated_ring_sizes,
                reference_ring_sizes,
            ),
            "atom_state_jensen_shannon": _jensen_shannon(
                generated_states,
                reference_states,
            ),
            "absolute_branch_atom_fraction_error": abs(
                generated["branch_atom_fraction"] - reference["branch_atom_fraction"]
            ),
            "absolute_nonaromatic_unsaturated_bond_fraction_error": abs(
                generated["nonaromatic_unsaturated_bond_fraction"]
                - reference["nonaromatic_unsaturated_bond_fraction"]
            ),
            "absolute_rdkit_aromatic_atom_fraction_error": abs(
                generated["rdkit"]["aromatic_atom_fraction"]
                - reference["rdkit"]["aromatic_atom_fraction"]
            ),
            "absolute_charged_atom_fraction_error": abs(
                generated["charged_atom_fraction"] - reference["charged_atom_fraction"]
            ),
            "absolute_heteroatom_fraction_error": abs(
                generated["heteroatom_fraction"] - reference["heteroatom_fraction"]
            ),
            "absolute_mean_cycle_rank_error": abs(
                generated["mean_cycle_rank"] - reference["mean_cycle_rank"]
            ),
            "absolute_mean_ring_size_error": abs(
                generated["rdkit"]["mean_ring_size"] - reference["rdkit"]["mean_ring_size"]
            ),
            "absolute_macrocycle_molecule_fraction_error": abs(
                generated["rdkit"]["macrocycle_molecule_fraction"]
                - reference["rdkit"]["macrocycle_molecule_fraction"]
            ),
            "regional_absolute_errors": {
                name: {
                    metric: abs(
                        generated["lipid_regions"][name][metric]
                        - reference["lipid_regions"][name][metric]
                    )
                    for metric in (
                        "atom_fraction",
                        "branch_atom_fraction",
                        "heteroatom_fraction",
                        "ring_atom_fraction",
                        "nonaromatic_unsaturated_atom_fraction",
                    )
                }
                for name in LIPID_REGION_NAMES
            },
            "morphology_absolute_errors": {
                metric: abs(generated["morphology"][metric] - reference["morphology"][metric])
                for metric in (
                    "mean_maximum_root_distance",
                    "mean_tail_components_per_molecule",
                    "path_like_tail_component_fraction",
                    "tail_junction_atoms_per_component",
                    "mean_tail_component_atoms",
                    "mean_tail_component_diameter",
                )
            },
            "oxygen_environment_fraction_absolute_errors": {
                name: abs(
                    generated["morphology"]["oxygen_environment_fractions"][name]
                    - reference["morphology"]["oxygen_environment_fractions"][name]
                )
                for name in generated["morphology"]["oxygen_environment_fractions"]
            },
        },
    }
