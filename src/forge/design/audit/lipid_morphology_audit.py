"""Corpus-grounded morphology diagnostics for whole-lipid generation."""

from __future__ import annotations

import csv
import gzip
import json
import os
import tempfile
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from forge.design.flow.defog_feasibility import FeasibilityError, sha256_file
from forge.design.flow.lipid_context import (
    LIPID_REGION_NAMES,
    assign_lipid_regions,
    rooted_distances,
    select_lipid_polar_root,
)


def _connected_components(molecule: Chem.Mol, selected: set[int]) -> list[set[int]]:
    components = []
    remaining = set(selected)
    while remaining:
        start = min(remaining)
        component = {start}
        queue = deque([start])
        remaining.remove(start)
        while queue:
            index = queue.popleft()
            for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors():
                candidate = neighbor.GetIdx()
                if candidate in remaining:
                    remaining.remove(candidate)
                    component.add(candidate)
                    queue.append(candidate)
        components.append(component)
    return components


def _component_diameter(molecule: Chem.Mol, component: set[int]) -> int:
    maximum = 0
    for start in component:
        distances = {start: 0}
        queue = deque([start])
        while queue:
            index = queue.popleft()
            for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors():
                candidate = neighbor.GetIdx()
                if candidate in component and candidate not in distances:
                    distances[candidate] = distances[index] + 1
                    queue.append(candidate)
        maximum = max(maximum, max(distances.values(), default=0))
    return maximum


def _oxygen_environment(atom: Chem.Atom) -> str:
    if atom.GetFormalCharge() < 0:
        return "anionic"
    bonds = list(atom.GetBonds())
    if atom.IsInRing():
        return "ring"
    if len(bonds) == 1 and bonds[0].GetBondType() == Chem.BondType.DOUBLE:
        return "carbonyl"
    if len(bonds) == 1 and bonds[0].GetBondType() == Chem.BondType.SINGLE:
        return "terminal_single"
    if len(bonds) == 2 and all(bond.GetBondType() == Chem.BondType.SINGLE for bond in bonds):
        return "single_bond_bridge"
    return "other"


def molecule_morphology(molecule: Chem.Mol) -> dict[str, Any]:
    """Measure lipid organization without using named fragments or component IDs."""

    if molecule.GetNumAtoms() == 0 or len(Chem.GetMolFrags(molecule)) != 1:
        raise FeasibilityError("lipid morphology requires one nonempty connected molecule")
    root = select_lipid_polar_root(molecule)
    distances = np.asarray(rooted_distances(molecule, root), dtype=np.int64)
    regions = assign_lipid_regions(molecule, root)
    degrees = np.asarray([atom.GetDegree() for atom in molecule.GetAtoms()], dtype=np.int64)

    region_rows: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(LIPID_REGION_NAMES):
        selected = np.flatnonzero(regions == index)
        region_rows[name] = {
            "atoms": int(selected.size),
            "branch_atoms": int(np.count_nonzero(degrees[selected] >= 3)),
            "terminal_atoms": int(np.count_nonzero(degrees[selected] == 1)),
            "ring_atoms": int(
                sum(molecule.GetAtomWithIdx(int(atom)).IsInRing() for atom in selected)
            ),
            "heteroatoms": int(
                sum(
                    molecule.GetAtomWithIdx(int(atom)).GetSymbol() not in {"C", "H"}
                    for atom in selected
                )
            ),
            "maximum_root_distance": int(distances[selected].max(initial=0)),
            "components": len(_connected_components(molecule, {int(atom) for atom in selected})),
        }

    transition_counts: Counter[str] = Counter()
    lateral_edges = 0
    unsaturated_bonds = 0
    for bond in molecule.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        if distances[left] == distances[right]:
            lateral_edges += 1
            first, second = sorted((int(regions[left]), int(regions[right])))
        elif distances[left] < distances[right]:
            first, second = int(regions[left]), int(regions[right])
        else:
            first, second = int(regions[right]), int(regions[left])
        transition_counts[f"{LIPID_REGION_NAMES[first]}->{LIPID_REGION_NAMES[second]}"] += 1
        unsaturated_bonds += int(
            not bond.GetIsAromatic()
            and bond.GetBondType() in {Chem.BondType.DOUBLE, Chem.BondType.TRIPLE}
        )

    tail_atoms = {
        index
        for index, region in enumerate(regions.tolist())
        if region == len(LIPID_REGION_NAMES) - 1
    }
    tail_components = _connected_components(molecule, tail_atoms)
    tail_component_rows = []
    for component in tail_components:
        internal_degrees = {
            index: sum(
                neighbor.GetIdx() in component
                for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors()
            )
            for index in component
        }
        internal_edges = sum(internal_degrees.values()) // 2
        cycle_rank = max(0, internal_edges - len(component) + 1)
        tail_component_rows.append(
            {
                "atoms": len(component),
                "diameter": _component_diameter(molecule, component),
                "junction_atoms": sum(value >= 3 for value in internal_degrees.values()),
                "terminal_atoms": sum(value <= 1 for value in internal_degrees.values()),
                "cycle_rank": cycle_rank,
                "path_like": cycle_rank == 0 and max(internal_degrees.values(), default=0) <= 2,
            }
        )

    oxygen_environments: Counter[str] = Counter()
    oxygen_root_distances = []
    nitrogen_root_distances = []
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() == "O":
            oxygen_environments[_oxygen_environment(atom)] += 1
            oxygen_root_distances.append(int(distances[atom.GetIdx()]))
        elif atom.GetSymbol() == "N":
            nitrogen_root_distances.append(int(distances[atom.GetIdx()]))

    return {
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
        "root_index": root,
        "maximum_root_distance": int(distances.max(initial=0)),
        "cycle_rank": max(0, molecule.GetNumBonds() - molecule.GetNumAtoms() + 1),
        "branch_atoms": int(np.count_nonzero(degrees >= 3)),
        "unsaturated_bonds": unsaturated_bonds,
        "lateral_edges": lateral_edges,
        "regions": region_rows,
        "region_transitions": dict(sorted(transition_counts.items())),
        "tail_components": tail_component_rows,
        "oxygen_environments": dict(sorted(oxygen_environments.items())),
        "oxygen_root_distances": oxygen_root_distances,
        "nitrogen_root_distances": nitrogen_root_distances,
    }


def _quantiles(values: Sequence[int | float]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("minimum", "q10", "median", "q90", "maximum", "mean")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "q10": float(np.quantile(array, 0.1)),
        "median": float(np.quantile(array, 0.5)),
        "q90": float(np.quantile(array, 0.9)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def audit_r0_morphology(
    r0_path: Path,
    assignments_path: Path,
    *,
    fold: str = "R0_train",
) -> dict[str, Any]:
    assignments: dict[str, Mapping[str, str]] = {}
    with assignments_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            assignments[row["r0_structure_id"]] = row

    molecule_rows = []
    source_counts: Counter[str] = Counter()
    with gzip.open(r0_path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            assignment = assignments.get(row["r0_structure_id"])
            if (
                assignment is None
                or assignment["source_study_fold"] != fold
                or row["r0_pretraining_eligible"] != "True"
            ):
                continue
            molecule = Chem.MolFromSmiles(row["canonical_constitutional_smiles"])
            if molecule is None:
                raise FeasibilityError(f"invalid R0 molecule: {row['r0_structure_id']}")
            morphology = molecule_morphology(molecule)
            morphology["structure_id"] = row["r0_structure_id"]
            morphology["source_study_group_id"] = assignment["source_study_group_id"]
            molecule_rows.append(morphology)
            source_counts[assignment["source_study_group_id"]] += 1

    if not molecule_rows:
        raise FeasibilityError("R0 morphology audit selected no molecules")

    region_totals = {name: Counter() for name in LIPID_REGION_NAMES}
    transition_totals: Counter[str] = Counter()
    oxygen_totals: Counter[str] = Counter()
    tail_component_sizes = []
    tail_component_diameters = []
    tail_component_junctions = []
    tail_component_path_like = []
    tail_components_per_molecule = []
    oxygen_distances = []
    nitrogen_distances = []
    for row in molecule_rows:
        for name in LIPID_REGION_NAMES:
            region_totals[name].update(row["regions"][name])
        transition_totals.update(row["region_transitions"])
        oxygen_totals.update(row["oxygen_environments"])
        tail_components_per_molecule.append(len(row["tail_components"]))
        for component in row["tail_components"]:
            tail_component_sizes.append(component["atoms"])
            tail_component_diameters.append(component["diameter"])
            tail_component_junctions.append(component["junction_atoms"])
            tail_component_path_like.append(int(component["path_like"]))
        oxygen_distances.extend(row["oxygen_root_distances"])
        nitrogen_distances.extend(row["nitrogen_root_distances"])

    total_atoms = sum(row["heavy_atoms"] for row in molecule_rows)
    result = {
        "schema_version": "phase1_r0_lipid_morphology_audit.v1",
        "status": "complete",
        "fold": fold,
        "molecules": len(molecule_rows),
        "inputs": {
            "r0_constitutional": {
                "path": str(r0_path),
                "sha256": sha256_file(r0_path),
            },
            "assignments": {
                "path": str(assignments_path),
                "sha256": sha256_file(assignments_path),
            },
        },
        "source_study_balance": {
            "groups": len(source_counts),
            "rows_per_group": _quantiles(list(source_counts.values())),
            "largest_groups": dict(source_counts.most_common(20)),
        },
        "molecule_topology": {
            "heavy_atoms": _quantiles([row["heavy_atoms"] for row in molecule_rows]),
            "maximum_root_distance": _quantiles(
                [row["maximum_root_distance"] for row in molecule_rows]
            ),
            "cycle_rank": _quantiles([row["cycle_rank"] for row in molecule_rows]),
            "branch_atoms": _quantiles([row["branch_atoms"] for row in molecule_rows]),
            "unsaturated_bonds": _quantiles([row["unsaturated_bonds"] for row in molecule_rows]),
        },
        "regions": {
            name: {
                "atom_fraction": region_totals[name]["atoms"] / max(1, total_atoms),
                "branch_atom_fraction": region_totals[name]["branch_atoms"]
                / max(1, region_totals[name]["atoms"]),
                "terminal_atom_fraction": region_totals[name]["terminal_atoms"]
                / max(1, region_totals[name]["atoms"]),
                "ring_atom_fraction": region_totals[name]["ring_atoms"]
                / max(1, region_totals[name]["atoms"]),
                "heteroatom_fraction": region_totals[name]["heteroatoms"]
                / max(1, region_totals[name]["atoms"]),
                "components_per_molecule": region_totals[name]["components"] / len(molecule_rows),
            }
            for name in LIPID_REGION_NAMES
        },
        "region_transitions": dict(sorted(transition_totals.items())),
        "tail_morphology": {
            "components_per_molecule": _quantiles(tail_components_per_molecule),
            "component_atoms": _quantiles(tail_component_sizes),
            "component_diameter": _quantiles(tail_component_diameters),
            "component_junction_atoms": _quantiles(tail_component_junctions),
            "path_like_component_fraction": sum(tail_component_path_like)
            / max(1, len(tail_component_path_like)),
        },
        "functional_group_context": {
            "oxygen_environments": dict(sorted(oxygen_totals.items())),
            "oxygen_root_distance": _quantiles(oxygen_distances),
            "nitrogen_root_distance": _quantiles(nitrogen_distances),
        },
        "interpretation_limits": [
            "Regions are generic deterministic structural annotations, not reaction components.",
            "Terminal single-bond oxygen includes alcohol-like and acid-like environments.",
            "Source-study row counts measure corpus imbalance, not biological importance.",
        ],
    }
    return result


def write_audit(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
