"""Role-specific branching audit for the frozen Ugi generator sample.

The audit measures carbon-skeleton branching in exact precursor components.
It is nonselecting and does not label unusual branching as infeasible.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.data.r1_prime_audit import sha256_file
from forge.potency.ugi_semantic_annotations import ROLE_NAMES

CONFIG_SCHEMA_VERSION = "phase1_ugi_postselection_branching_audit_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_postselection_branching_audit.v1"


class UgiPostselectionBranchingError(ValueError):
    """Raised when branching-audit inputs violate their frozen contract."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise UgiPostselectionBranchingError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise UgiPostselectionBranchingError(f"{label} must be a JSON object")
    return value


def _component_molecule(smiles: Any, *, label: str) -> Chem.Mol:
    if not isinstance(smiles, str) or not smiles:
        raise UgiPostselectionBranchingError(f"{label} must be non-empty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise UgiPostselectionBranchingError(f"{label} must be one valid connected graph")
    return molecule


def carbon_branch_metrics(smiles: str) -> dict[str, int | None]:
    """Return carbon-skeleton branch counts without fragment assumptions."""

    molecule = _component_molecule(smiles, label="component")
    branch_atoms = {
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 6
        and sum(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()) >= 3
    }
    adjacency: dict[int, set[int]] = {index: set() for index in branch_atoms}
    adjacent_edges = 0
    for bond in molecule.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        if left in branch_atoms and right in branch_atoms:
            adjacency[left].add(right)
            adjacency[right].add(left)
            adjacent_edges += 1
    maximum_run = 0
    unseen = set(branch_atoms)
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        maximum_run = max(maximum_run, size)
    minimum_distance: int | None = None
    branch_list = sorted(branch_atoms)
    for offset, left in enumerate(branch_list):
        distances = Chem.GetDistanceMatrix(molecule)[left]
        for right in branch_list[offset + 1 :]:
            distance = int(distances[right])
            minimum_distance = (
                distance if minimum_distance is None else min(minimum_distance, distance)
            )
    return {
        "carbon_branch_atoms": len(branch_atoms),
        "adjacent_carbon_branch_edges": adjacent_edges,
        "maximum_adjacent_branch_run": maximum_run,
        "minimum_branch_graph_distance": minimum_distance,
    }


def _summarize(rows: Iterable[Mapping[str, int | None]]) -> dict[str, Any]:
    values = list(rows)
    if not values:
        raise UgiPostselectionBranchingError("cannot summarize an empty branching cohort")
    count = len(values)
    branch_distribution = Counter(int(row["carbon_branch_atoms"] or 0) for row in values)
    adjacent = sum(int(row["adjacent_carbon_branch_edges"] or 0) > 0 for row in values)
    long_runs = sum(int(row["maximum_adjacent_branch_run"] or 0) >= 3 for row in values)
    distances = [
        int(row["minimum_branch_graph_distance"])
        for row in values
        if row["minimum_branch_graph_distance"] is not None
    ]
    return {
        "components": count,
        "carbon_branch_atom_count_distribution": {
            str(key): value for key, value in sorted(branch_distribution.items())
        },
        "components_with_any_carbon_branch": sum(
            key > 0 for key in [int(row["carbon_branch_atoms"] or 0) for row in values]
        ),
        "components_with_adjacent_carbon_branch_atoms": adjacent,
        "adjacent_carbon_branch_fraction": adjacent / count,
        "components_with_branch_run_at_least_three": long_runs,
        "minimum_observed_branch_graph_distance": min(distances) if distances else None,
        "maximum_observed_branch_run": max(
            int(row["maximum_adjacent_branch_run"] or 0) for row in values
        ),
    }


def _read_reference(path: Path, folds: Sequence[str]) -> dict[str, list[dict[str, int | None]]]:
    output = {role: [] for role in ROLE_NAMES}
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise UgiPostselectionBranchingError("reference assignments have no header")
            for row in reader:
                if row.get("primary_product_fold") not in folds:
                    continue
                for role in ROLE_NAMES:
                    output[role].append(carbon_branch_metrics(row[f"{role}_smiles"]))
    except (OSError, csv.Error, KeyError) as exc:
        raise UgiPostselectionBranchingError("invalid reference assignments") from exc
    return output


def build_postselection_branching_audit(
    config_path: Path,
    production_manifest_path: Path,
    selected_sample_path: Path,
    reference_assignments_path: Path,
) -> dict[str, Any]:
    """Audit role-specific branch spacing in the selected sample and reference."""

    config = _load_json(config_path, label="branching audit config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiPostselectionBranchingError("unsupported branching audit config schema")
    input_paths = {
        "production_manifest": production_manifest_path,
        "selected_sample": selected_sample_path,
        "selection_reference_assignments": reference_assignments_path,
    }
    for name, path in input_paths.items():
        expected = config["inputs"][name]["sha256"]
        observed = sha256_file(path)
        if observed != expected:
            raise UgiPostselectionBranchingError(
                f"{name} hash mismatch: expected {expected}, observed {observed}"
            )
    production = _load_json(production_manifest_path, label="production manifest")
    selected = _load_json(selected_sample_path, label="selected sample")
    if selected.get("status") != "complete":
        raise UgiPostselectionBranchingError("selected sample is not complete")
    generated = {role: [] for role in ROLE_NAMES}
    valid_products = 0
    flagged_products: list[str] = []
    tail_roles = tuple(config["decision_policy"]["tail_roles"])
    for row in selected.get("samples", []):
        if not row.get("component_reconstruction_valid"):
            continue
        components = row.get("component_smiles_by_role")
        if not isinstance(components, dict) or set(components) != set(ROLE_NAMES):
            raise UgiPostselectionBranchingError("generated component roles are incomplete")
        valid_products += 1
        metrics_by_role = {}
        for role in ROLE_NAMES:
            metrics = carbon_branch_metrics(components[role])
            generated[role].append(metrics)
            metrics_by_role[role] = metrics
        if any(metrics_by_role[role]["adjacent_carbon_branch_edges"] for role in tail_roles):
            flagged_products.append(str(row["structure_id"]))
    if valid_products != selected["statistics"]["valid_molecules"]:
        raise UgiPostselectionBranchingError("generated component denominator mismatch")
    folds = tuple(config["reference_policy"]["folds"])
    reference = _read_reference(reference_assignments_path, folds)
    generated_summary = {role: _summarize(generated[role]) for role in ROLE_NAMES}
    reference_summary = {role: _summarize(reference[role]) for role in ROLE_NAMES}
    tail_rate_exceeds = {
        role: generated_summary[role]["adjacent_carbon_branch_fraction"]
        > reference_summary[role]["adjacent_carbon_branch_fraction"]
        for role in tail_roles
    }
    challenger_required = bool(
        config["decision_policy"][
            "challenger_required_if_generated_tail_adjacent_branch_rate_exceeds_reference"
        ]
        and any(tail_rate_exceeds.values())
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_nonselecting_postselection_audit",
        "task": "role_specific_carbon_skeleton_branching",
        "scope": {
            "can_reopen_frozen_checkpoint": False,
            "synthesis_feasibility_assessed": False,
            "biological_activity_assessed": False,
            "challenger_retraining_decision_only": True,
        },
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in input_paths.items()
        },
        "production_generator": production.get("selected_model"),
        "summary": {
            "generated_products_with_components": valid_products,
            "selection_reference_products": len(reference[ROLE_NAMES[0]]),
            "generated": generated_summary,
            "selection_reference": reference_summary,
            "products_with_adjacent_tail_carbon_branch_atoms": len(flagged_products),
            "products_with_adjacent_tail_carbon_branch_fraction": len(flagged_products)
            / valid_products,
            "tail_adjacent_branch_rate_exceeds_reference": tail_rate_exceeds,
            "challenger_required_before_candidate_lock": challenger_required,
        },
        "flagged_product_ids": flagged_products,
        "safe_claim": (
            "The frozen generator remains a reproducible baseline, but adjacent carbon-skeleton "
            "branch points occur in generated tail-bearing components despite being absent from "
            "the selection-visible reference. A separately versioned challenger sampling or "
            "training policy is required before candidate lock."
            if challenger_required
            else "Generated tail branching does not exceed the selection-visible reference."
        ),
        "limitations": [
            "Carbon-skeleton branch metrics do not establish synthesis infeasibility or biological quality.",
            "The audit is post-selection and cannot reopen the frozen checkpoint.",
            "The reference is the selection-visible Ugi training and calibration corpus, not all lipids.",
        ],
    }
