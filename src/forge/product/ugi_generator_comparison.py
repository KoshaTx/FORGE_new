"""Matched, source-stratified evaluation of Phase 1 Ugi generator arms."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from forge.bio.ugi_semantic_annotations import ROLE_NAMES
from forge.product.defog_feasibility import sha256_file
from forge.product.ugi_morphology_program import (
    attached_tree_matches_program,
    preorder_attached_forest_to_parents,
)


class UgiGeneratorComparisonError(RuntimeError):
    """Raised when two generator outputs are not exactly comparable."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise UgiGeneratorComparisonError(f"JSON root is not an object: {path}")
    return value


def _canonical_program(value: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(item) for item in value[key])
        for key in (
            "node_counts",
            "junction_budgets",
            "cycle_ranks",
            "attachment_counts",
        )
    )


def _motifs(molecule: Chem.Mol) -> dict[str, bool]:
    oxygen_oxygen = False
    nitrogen_nitrogen = False
    triple_bond = False
    for bond in molecule.GetBonds():
        pair = {
            bond.GetBeginAtom().GetAtomicNum(),
            bond.GetEndAtom().GetAtomicNum(),
        }
        oxygen_oxygen |= pair == {8}
        nitrogen_nitrogen |= pair == {7}
        triple_bond |= bond.GetBondType() == Chem.BondType.TRIPLE
    cumulated_double_bonds = any(
        atom.GetAtomicNum() == 6
        and sum(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom.GetBonds()) >= 2
        for atom in molecule.GetAtoms()
    )
    return {
        "oxygen_oxygen_bond": oxygen_oxygen,
        "nitrogen_nitrogen_bond": nitrogen_nitrogen,
        "triple_bond": triple_bond,
        "cumulated_carbon_double_bonds": cumulated_double_bonds,
    }


def _largest_adjacent_branch_component(
    offspring: np.ndarray,
    *,
    attachment_count: int,
) -> int:
    parents = preorder_attached_forest_to_parents(
        offspring,
        attachment_count=attachment_count,
    )
    branch_nodes = {index for index, value in enumerate(offspring) if value >= 2}
    adjacency = {index: set() for index in branch_nodes}
    for child, parent in enumerate(parents.tolist()):
        if child in branch_nodes and parent in branch_nodes:
            adjacency[child].add(parent)
            adjacency[parent].add(child)
    remaining = set(branch_nodes)
    largest = 0
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        size = 1
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in sorted(adjacency[node] & remaining):
                remaining.remove(neighbor)
                queue.append(neighbor)
                size += 1
        largest = max(largest, size)
    return largest


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["valid"]]
    motif_names = tuple(next(iter(valid))["motifs"]) if valid else ()
    by_role: dict[str, Any] = {}
    for role in ROLE_NAMES:
        branch_counts = [row["branch_atoms_by_role"][role] for row in valid]
        branch_runs = [row["adjacent_branch_component_by_role"][role] for row in valid]
        by_role[role] = {
            "mean_branch_atoms": float(np.mean(branch_counts)) if branch_counts else None,
            "maximum_adjacent_branch_component": max(branch_runs, default=None),
        }
    return {
        "samples": len(rows),
        "valid_fraction": len(valid) / len(rows) if rows else None,
        "unique_valid_fraction": (
            len({row["canonical_smiles"] for row in valid}) / len(valid) if valid else None
        ),
        "exact_program_fraction": (
            sum(bool(row["exact_program_match"]) for row in rows) / len(rows) if rows else None
        ),
        "motif_fractions": {
            name: sum(bool(row["motifs"][name]) for row in valid) / len(valid)
            for name in motif_names
        },
        "morphology_by_role": by_role,
    }


def _evaluate_arm(
    arm: dict[str, Any],
    probe_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples = arm.get("samples")
    if not isinstance(samples, list) or len(samples) != len(probe_rows):
        raise UgiGeneratorComparisonError("arm and probe sample counts differ")
    evaluated = []
    for index, (sample, probe) in enumerate(zip(samples, probe_rows, strict=True)):
        if _canonical_program(sample["program"]) != _canonical_program(probe["program"]):
            raise UgiGeneratorComparisonError(f"program mismatch at sample {index}")
        molecule = Chem.MolFromSmiles(str(sample.get("smiles") or ""))
        valid = bool(sample.get("valid")) and molecule is not None
        branch_atoms_by_role: dict[str, int] = {}
        adjacent_by_role: dict[str, int] = {}
        exact_program = True
        for role_index, role in enumerate(ROLE_NAMES):
            offspring = np.asarray(sample["offspring_by_role"][role], dtype=np.int64)
            program = probe["program"]
            matches = attached_tree_matches_program(
                offspring,
                node_count=int(program["node_counts"][role_index]),
                junction_budget=int(program["junction_budgets"][role_index]),
                attachment_count=int(program["attachment_counts"][role_index]),
            )
            exact_program &= matches
            branch_atoms_by_role[role] = int(np.count_nonzero(offspring >= 2))
            adjacent_by_role[role] = _largest_adjacent_branch_component(
                offspring,
                attachment_count=int(program["attachment_counts"][role_index]),
            )
        evaluated.append(
            {
                "index": index,
                "source_stratum": probe["source_stratum"],
                "branch_class": probe["branch_class"],
                "valid": valid,
                "canonical_smiles": (Chem.MolToSmiles(molecule, canonical=True) if valid else None),
                "exact_program_match": exact_program,
                "motifs": _motifs(molecule) if valid else {},
                "branch_atoms_by_role": branch_atoms_by_role,
                "adjacent_branch_component_by_role": adjacent_by_role,
            }
        )
    by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_branch: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        by_source[row["source_stratum"]].append(row)
        by_branch[row["branch_class"]].append(row)
    return (
        {
            "overall": _summarize_rows(evaluated),
            "by_source_stratum": {
                key: _summarize_rows(value) for key, value in sorted(by_source.items())
            },
            "by_branch_class": {
                key: _summarize_rows(value) for key, value in sorted(by_branch.items())
            },
            "reported_reference_comparison": arm.get("reference_comparison"),
            "terminal_tree_repairs": int(
                arm.get("sampling", {})
                .get("morphology", arm.get("sampling", {}))
                .get("terminal_tree_repairs", 0)
            ),
        },
        evaluated,
    )


def _reference_motif_fractions(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    molecules = 0
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            molecule = Chem.MolFromSmiles(row["product_smiles"])
            if molecule is None:
                raise UgiGeneratorComparisonError(f"invalid reference product: {row['product_id']}")
            molecules += 1
            for name, present in _motifs(molecule).items():
                counts[name] += int(present)
    return {
        "molecules": molecules,
        "motif_fractions": {name: count / molecules for name, count in sorted(counts.items())},
    }


def compare_ugi_generator_arms(
    *,
    joint_result_path: Path,
    staged_result_path: Path,
    probe_path: Path,
    reference_products_path: Path,
) -> dict[str, Any]:
    """Compare matched joint and staged arms without making a paper-level claim."""

    joint = _read_json(joint_result_path)
    staged = _read_json(staged_result_path)
    probe = _read_json(probe_path)
    probe_rows = probe.get("samples")
    if not isinstance(probe_rows, list) or not probe_rows:
        raise UgiGeneratorComparisonError("probe has no samples")
    joint_summary, _ = _evaluate_arm(joint, probe_rows)
    staged_summary, _ = _evaluate_arm(staged, probe_rows)
    reference = _reference_motif_fractions(reference_products_path)
    unsupported = {
        name for name, fraction in reference["motif_fractions"].items() if fraction == 0.0
    }
    joint_motifs = joint_summary["overall"]["motif_fractions"]
    staged_motifs = staged_summary["overall"]["motif_fractions"]
    gates = {
        "matched_program_count": len(joint["samples"]) == len(staged["samples"]) == len(probe_rows),
        "joint_exact_program_support": joint_summary["overall"]["exact_program_fraction"] == 1.0,
        "staged_exact_program_support": staged_summary["overall"]["exact_program_fraction"] == 1.0,
        "joint_no_terminal_tree_repair": joint_summary["terminal_tree_repairs"] == 0,
        "staged_no_terminal_tree_repair": staged_summary["terminal_tree_repairs"] == 0,
        "joint_no_unsupported_reference_motif": all(
            joint_motifs[name] == 0.0 for name in unsupported
        ),
        "staged_no_unsupported_reference_motif": all(
            staged_motifs[name] == 0.0 for name in unsupported
        ),
    }
    return {
        "schema_version": "phase1_ugi_matched_generator_comparison.v1",
        "status": "pass" if all(gates.values()) else "fail",
        "selection_status": "insufficient_for_final_selection_smoke_only",
        "inputs": {
            "joint_result": {
                "path": str(joint_result_path),
                "sha256": sha256_file(joint_result_path),
            },
            "staged_result": {
                "path": str(staged_result_path),
                "sha256": sha256_file(staged_result_path),
            },
            "probe": {"path": str(probe_path), "sha256": sha256_file(probe_path)},
            "reference_products": {
                "path": str(reference_products_path),
                "sha256": sha256_file(reference_products_path),
            },
        },
        "probe_strata": dict(
            sorted(
                Counter(
                    f"{row['source_stratum']}|{row['branch_class']}" for row in probe_rows
                ).items()
            )
        ),
        "reference": reference,
        "unsupported_reference_motifs": sorted(unsupported),
        "arms": {"joint": joint_summary, "staged": staged_summary},
        "gates": gates,
        "interpretation_policy": {
            "validity_alone_selects_architecture": False,
            "smoke_run_supports_paper_level_superiority_claim": False,
            "formal_selection_requires_matched_training_and_larger_held_evaluation": True,
        },
    }
