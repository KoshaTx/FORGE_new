#!/usr/bin/env python3
"""Compare generated Ugi tail-branch geometry with the weighted training measure."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from forge.corpus.r1_prime_audit import sha256_file
from forge.potency.annotations import ROLE_NAMES
from experiments.phase1.product_l1.training.ugi_joint_sparse_training import _training_weights
from experiments.phase1.product_l1.training.ugi_training_cache import load_ugi_training_cache

REPO = Path(__file__).resolve().parents[3]
ROLES = ("oxoester_aldehyde_body_tail", "isocyanide_tail")
REFERENCE_FOLDS = ("train", "calibration")
SAMPLING = {
    "mode": "source_stratified_role_family_raked",
    "source_mass": {
        "current_phase1_union": 0.5,
        "expanded_exact_forward_enumeration": 0.5,
    },
    "uniform_row_mixture": 0.5,
}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _handle_atom(molecule: Chem.Mol, role: str) -> int:
    if role == "isocyanide_tail":
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum() != 6 or atom.GetFormalCharge() != -1:
                continue
            if any(
                bond.GetBondType() == Chem.BondType.TRIPLE
                and bond.GetOtherAtom(atom).GetAtomicNum() == 7
                for bond in atom.GetBonds()
            ):
                return atom.GetIdx()
    elif role == "oxoester_aldehyde_body_tail":
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum() != 6 or atom.GetTotalNumHs() <= 0:
                continue
            if any(
                bond.GetBondType() == Chem.BondType.DOUBLE
                and bond.GetOtherAtom(atom).GetAtomicNum() == 8
                for bond in atom.GetBonds()
            ):
                return atom.GetIdx()
    raise ValueError(f"cannot identify {role} reactive handle")


def component_branch_geometry(smiles: str, role: str) -> dict[str, Any]:
    """Describe carbon branching outward from a precursor's Ugi handle.

    The first-branch arm ratio distinguishes a short side twig from a balanced
    hydrophobic fork. It is a descriptive statistic, not a generation rule.
    """

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid component SMILES: {smiles}")
    root = _handle_atom(molecule, role)
    distances = [math.inf] * molecule.GetNumAtoms()
    distances[root] = 0
    queue = deque([root])
    while queue:
        source = queue.popleft()
        for neighbor in molecule.GetAtomWithIdx(source).GetNeighbors():
            target = neighbor.GetIdx()
            if distances[target] > distances[source] + 1:
                distances[target] = distances[source] + 1
                queue.append(target)

    carbon_neighbors = {
        atom.GetIdx(): tuple(
            neighbor.GetIdx() for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 6
        )
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 6
    }
    branch_atoms = tuple(
        atom_index
        for atom_index, neighbors in carbon_neighbors.items()
        if len(neighbors) >= 3 and not molecule.GetAtomWithIdx(atom_index).IsInRing()
    )
    adjacent_branch_edges = sum(
        target in branch_atoms
        for source in branch_atoms
        for target in carbon_neighbors[source]
        if source < target
    )
    events = []
    for branch_atom in sorted(branch_atoms, key=distances.__getitem__):
        distal = tuple(
            neighbor
            for neighbor in carbon_neighbors[branch_atom]
            if distances[neighbor] > distances[branch_atom]
        )
        if len(distal) < 2:
            continue
        arm_lengths = []
        for start in distal:
            # In an acyclic tail this equals the usual longest outward arm.
            # Ring-containing distal groups require a visited shortest-path
            # traversal; an unrestricted DFS can walk a carbon cycle forever.
            local_distances = {start: 1}
            queue = deque([start])
            while queue:
                atom_index = queue.popleft()
                for neighbor in carbon_neighbors.get(atom_index, ()):
                    if neighbor == branch_atom:
                        continue
                    if distances[neighbor] <= distances[branch_atom]:
                        continue
                    if neighbor not in local_distances:
                        local_distances[neighbor] = local_distances[atom_index] + 1
                        queue.append(neighbor)
            arm_lengths.append(max(local_distances.values()))
        arm_lengths.sort(reverse=True)
        events.append(
            {
                "branch_depth_from_handle": int(distances[branch_atom]),
                "longer_distal_arm_carbons": int(arm_lengths[0]),
                "shorter_distal_arm_carbons": int(arm_lengths[1]),
                "distal_arm_balance": float(arm_lengths[1] / arm_lengths[0]),
            }
        )
    return {
        "carbon_branch_atoms": len(branch_atoms),
        "adjacent_carbon_branch_edges": int(adjacent_branch_edges),
        "first_outward_branch": events[0] if events else None,
    }


def _cached_component_branch_geometry(
    cache: dict[tuple[str, str], dict[str, Any]],
    smiles: str,
    role: str,
) -> dict[str, Any]:
    key = (smiles, role)
    if key not in cache:
        cache[key] = component_branch_geometry(smiles, role)
    return cache[key]


def _weighted_quantiles(
    values: Sequence[float], weights: Sequence[float], quantiles: Sequence[float]
) -> dict[str, float]:
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    cumulative = np.cumsum(np.asarray(weights, dtype=np.float64)[order])
    cumulative /= cumulative[-1]
    return {
        str(quantile): float(
            sorted_values[min(int(np.searchsorted(cumulative, quantile)), len(values) - 1)]
        )
        for quantile in quantiles
    }


def summarize_geometry(weighted: Sequence[tuple[Mapping[str, Any], float]]) -> dict[str, Any]:
    total_mass = sum(weight for _, weight in weighted)
    branched = [
        (geometry, weight)
        for geometry, weight in weighted
        if geometry["first_outward_branch"] is not None
    ]
    branch_mass = sum(weight for _, weight in branched)
    output: dict[str, Any] = {
        "component_rows": len(weighted),
        "first_outward_branch_fraction": branch_mass / total_mass,
        "mean_carbon_branch_atoms": sum(
            geometry["carbon_branch_atoms"] * weight for geometry, weight in weighted
        )
        / total_mass,
        "adjacent_branch_component_fraction": sum(
            (geometry["adjacent_carbon_branch_edges"] > 0) * weight for geometry, weight in weighted
        )
        / total_mass,
    }
    if not branched:
        return output
    fields = (
        "branch_depth_from_handle",
        "longer_distal_arm_carbons",
        "shorter_distal_arm_carbons",
        "distal_arm_balance",
    )
    for field in fields:
        values = [geometry["first_outward_branch"][field] for geometry, _ in branched]
        weights = [weight for _, weight in branched]
        output[field] = {
            "mean": sum(value * weight for value, weight in zip(values, weights, strict=True))
            / branch_mass,
            "quantiles": _weighted_quantiles(values, weights, (0.1, 0.25, 0.5, 0.75, 0.9)),
        }
    output["first_branch_shape_fractions"] = {
        "twig_like_balance_at_most_0.25": sum(
            (geometry["first_outward_branch"]["distal_arm_balance"] <= 0.25) * weight
            for geometry, weight in branched
        )
        / branch_mass,
        "balanced_fork_at_least_0.67": sum(
            (geometry["first_outward_branch"]["distal_arm_balance"] >= 0.67) * weight
            for geometry, weight in branched
        )
        / branch_mass,
    }
    return output


def branch_geometry_signature(geometry: Mapping[str, Any]) -> str:
    """Serialize branch-event geometry without a hand-weighted scalar score."""

    first = geometry["first_outward_branch"]
    fields = [
        f"branch_atoms={int(geometry['carbon_branch_atoms'])}",
        f"adjacent_branch_edges={int(geometry['adjacent_carbon_branch_edges'])}",
    ]
    if first is None:
        fields.append("first_branch=none")
    else:
        fields.extend(
            (
                f"first_depth={int(first['branch_depth_from_handle'])}",
                f"long_arm={int(first['longer_distal_arm_carbons'])}",
                f"short_arm={int(first['shorter_distal_arm_carbons'])}",
            )
        )
    return "|".join(fields)


def _weighted_signature_mass(
    weighted: Sequence[tuple[Mapping[str, Any], float]],
) -> dict[str, float]:
    mass: Counter[str] = Counter()
    for geometry, weight in weighted:
        mass[branch_geometry_signature(geometry)] += float(weight)
    total = sum(mass.values())
    if total <= 0:
        raise ValueError("branch signature measure has no mass")
    return {key: value / total for key, value in sorted(mass.items())}


def _jensen_shannon_divergence_bits(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left).union(right)
    value = 0.0
    for key in keys:
        p = float(left.get(key, 0.0))
        q = float(right.get(key, 0.0))
        midpoint = 0.5 * (p + q)
        if p > 0:
            value += 0.5 * p * math.log2(p / midpoint)
        if q > 0:
            value += 0.5 * q * math.log2(q / midpoint)
    return value


def _generated_components(sample: Mapping[str, Any], role: str) -> list[str]:
    return [
        str(row["component_smiles_by_role"][role])
        for row in sample["samples"]
        if row.get("valid") is True
        and row.get("component_reconstruction_valid") is True
        and (row.get("l1_forward_verification") or {}).get("exact_product_reconstructed") is True
    ]


def _eligible_generated_rows(sample: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in sample["samples"]
        if row.get("valid") is True
        and row.get("component_reconstruction_valid") is True
        and (row.get("l1_forward_verification") or {}).get("exact_product_reconstructed") is True
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO / "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt",
    )
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/ugi_branch_arm_geometry_audit_v1.json",
    )
    args = parser.parse_args()
    corpus, joint_records_by_fold = load_ugi_training_cache(args.cache)
    assignments = tuple(row for fold in REFERENCE_FOLDS for row in corpus.assignments_by_fold[fold])
    joint_records = tuple(row for fold in REFERENCE_FOLDS for row in joint_records_by_fold[fold])
    if len(assignments) != len(joint_records):
        raise RuntimeError("training assignment and joint-record counts differ")
    if any(
        str(assignment["product_id"]) != str(record.product_id)
        for assignment, record in zip(assignments, joint_records, strict=True)
    ):
        raise RuntimeError("training assignment and joint-record order differs")
    weights = _training_weights(assignments, SAMPLING)
    geometry_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def geometry(smiles: str, role: str) -> dict[str, Any]:
        return _cached_component_branch_geometry(geometry_cache, smiles, role)

    arms: dict[str, Any] = {"weighted_training_measure": {}}
    for role in ROLES:
        arms["weighted_training_measure"][role] = summarize_geometry(
            [
                (geometry(str(row[f"{role}_smiles"]), role), float(weight))
                for row, weight in zip(assignments, weights, strict=True)
            ]
        )
    inputs: dict[str, Any] = {
        "training_cache": {"path": str(args.cache), "sha256": sha256_file(args.cache)}
    }
    for specification in args.sample:
        name, raw_path = specification.split("=", 1)
        path = Path(raw_path)
        sample = json.loads(path.read_text())
        eligible_rows = _eligible_generated_rows(sample)
        arms[name] = {"unconditional": {}, "program_matched_training_reference": {}}
        for role in ROLES:
            role_index = ROLE_NAMES.index(role)
            components = [str(row["component_smiles_by_role"][role]) for row in eligible_rows]
            arms[name]["unconditional"][role] = summarize_geometry(
                [(geometry(smiles, role), 1.0) for smiles in components]
            )
            generated_key_counts: dict[tuple[int, int], int] = {}
            for row in eligible_rows:
                program = row["program"]
                key = (
                    int(program["node_counts"][role_index]),
                    int(program["junction_budgets"][role_index]),
                )
                generated_key_counts[key] = generated_key_counts.get(key, 0) + 1
            training_groups: dict[tuple[int, int], list[tuple[dict[str, Any], float]]] = {}
            for assignment, record, weight in zip(assignments, joint_records, weights, strict=True):
                key = (
                    int(record.program.node_counts[role_index]),
                    int(record.program.junction_budgets[role_index]),
                )
                training_groups.setdefault(key, []).append(
                    (
                        geometry(str(assignment[f"{role}_smiles"]), role),
                        float(weight),
                    )
                )
            matched: list[tuple[dict[str, Any], float]] = []
            missing = 0
            matched_keys: set[tuple[int, int]] = set()
            for key, count in generated_key_counts.items():
                group = training_groups.get(key)
                if not group:
                    missing += count
                    continue
                matched_keys.add(key)
                group_mass = sum(weight for _, weight in group)
                target_key_mass = count / len(eligible_rows)
                matched.extend(
                    (record, target_key_mass * weight / group_mass) for record, weight in group
                )
            generated_matched_geometry = [
                (
                    geometry(str(row["component_smiles_by_role"][role]), role),
                    1.0,
                )
                for row in eligible_rows
                if (
                    int(row["program"]["node_counts"][role_index]),
                    int(row["program"]["junction_budgets"][role_index]),
                )
                in matched_keys
            ]
            generated_signature_mass = _weighted_signature_mass(generated_matched_geometry)
            training_signature_mass = _weighted_signature_mass(matched)
            arms[name]["program_matched_training_reference"][role] = {
                "matching_coordinates": ["role_node_count", "role_junction_budget"],
                "generated_rows": len(eligible_rows),
                "exactly_matched_generated_rows": len(eligible_rows) - missing,
                "exact_match_fraction": (len(eligible_rows) - missing) / len(eligible_rows),
                "generated_summary_on_exactly_matched_programs": summarize_geometry(
                    generated_matched_geometry
                ),
                "summary_on_exactly_matched_program_mass": summarize_geometry(matched),
                "branch_geometry_signature": {
                    "fields": [
                        "carbon_branch_atoms",
                        "adjacent_carbon_branch_edges",
                        "first_branch_depth_from_handle",
                        "first_branch_longer_distal_arm_carbons",
                        "first_branch_shorter_distal_arm_carbons",
                    ],
                    "generated_mass": generated_signature_mass,
                    "training_reference_mass": training_signature_mass,
                    "jensen_shannon_divergence_bits": _jensen_shannon_divergence_bits(
                        generated_signature_mass, training_signature_mass
                    ),
                },
            }
        inputs[name] = {"path": str(path), "sha256": sha256_file(path)}
    output = {
        "schema_version": "phase1_ugi_branch_arm_geometry_audit.v1",
        "status": "complete_descriptive_nonselecting_audit",
        "reference_folds": list(REFERENCE_FOLDS),
        "heldout_fold_used": False,
        "definition": {
            "orientation": "outward from the exact precursor Ugi handle",
            "branch_atom": "acyclic carbon with at least three carbon neighbors",
            "arm_length": (
                "maximum shortest-path carbon distance within each distal component after removing "
                "the first outward branch atom"
            ),
            "arm_balance": "shorter divided by longer distal-component arm length",
            "thresholds_are_descriptive_not_generation_constraints": True,
        },
        "inputs": inputs,
        "arms": arms,
    }
    _atomic_json(args.output, output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "reference_folds": list(REFERENCE_FOLDS),
                "arms": {
                    name: {
                        role: {
                            "exact_match_fraction": value["program_matched_training_reference"][
                                role
                            ]["exact_match_fraction"],
                            "adjacent_branch_component_fraction": value[
                                "program_matched_training_reference"
                            ][role]["generated_summary_on_exactly_matched_programs"][
                                "adjacent_branch_component_fraction"
                            ],
                        }
                        for role in ROLES
                    }
                    for name, value in arms.items()
                    if name != "weighted_training_measure"
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
