#!/usr/bin/env python3
"""Build the exact occurrence reference implied by production training weights."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from forge.data.r1_prime_audit import sha256_file
from forge.potency.ugi_semantic_annotations import ROLE_NAMES
from forge.product.ugi_adapter_features import ORIGIN_TO_INDEX, PORT_TO_INDEX
from forge.product.ugi_joint_sparse_training import _training_weights
from forge.product.ugi_tail_chemotype_audit import component_chemotype_metrics
from forge.product.ugi_training_cache import load_ugi_training_cache

REPO = Path(__file__).resolve().parents[1]
FEATURE_FIELDS = {
    "has_adjacent_carbon_branches": "adjacent_carbon_branch_edges",
    "has_carbon_branch": "carbon_branch_atoms",
    "has_carbon_carbon_double_bond": "carbon_carbon_double_bonds",
    "has_carbon_carbon_triple_bond": "carbon_carbon_triple_bonds",
    "has_ester_like_carbonyl": "ester_like_carbonyl_count",
    "has_ether_oxygen": "ether_oxygen_count",
    "has_ring": "ring_count",
}
MAXIMUM_DISTANCE_BIN = 8


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def _effective_count(masses: dict[str, float]) -> float:
    total = sum(masses.values())
    return math.exp(
        -sum((mass / total) * math.log(mass / total) for mass in masses.values() if mass > 0)
    )


def _distance_to_role_port(condition: Any, role: str) -> np.ndarray:
    """Recompute graph distances without depending on mutable validator policy."""

    port = np.flatnonzero(condition.port_states == PORT_TO_INDEX[role])
    if port.size != 1:
        raise RuntimeError(f"{condition.structure_id} does not have one {role} port")
    adjacency = [set() for _ in range(condition.node_count)]
    parents = condition.parents
    for child in range(1, condition.node_count):
        parent = int(parents[child])
        adjacency[parent].add(child)
        adjacency[child].add(parent)
    for left, right in zip(condition.closure_left, condition.closure_right, strict=True):
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    distances = np.full(condition.node_count, -1, dtype=np.int64)
    source = int(port[0])
    distances[source] = 0
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if distances[neighbor] < 0:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    if np.any(distances < 0):
        raise RuntimeError(f"{condition.structure_id} is disconnected")
    return distances


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO / "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            REPO / "results/phase1/ugi_joint_sparse_production_refit_full/checkpoint_step_1000.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/ugi_balanced_terminal_reference_v1.json",
    )
    args = parser.parse_args()

    import torch

    corpus, records_by_fold = load_ugi_training_cache(args.cache)
    assignments = tuple(
        row
        for fold in ("train", "calibration", "heldout")
        for row in corpus.assignments_by_fold[fold]
    )
    sampling = {
        "mode": "source_stratified_role_family_raked",
        "source_mass": {
            "current_phase1_union": 0.5,
            "expanded_exact_forward_enumeration": 0.5,
        },
        "uniform_row_mixture": 0.5,
    }
    weights = _training_weights(assignments, sampling)
    if weights.shape != (len(assignments),) or not np.isclose(weights.sum(), 1.0):
        raise RuntimeError("production training weights are malformed")
    unique_smiles = {row[f"{role}_smiles"] for row in assignments for role in ROLE_NAMES}
    metrics = {smiles: component_chemotype_metrics(smiles) for smiles in unique_smiles}
    role_reference: dict[str, Any] = {}
    for role in ROLE_NAMES:
        masses: dict[str, float] = {}
        feature_mass = {name: 0.0 for name in FEATURE_FIELDS}
        for row, weight in zip(assignments, weights, strict=True):
            smiles = row[f"{role}_smiles"]
            masses[smiles] = masses.get(smiles, 0.0) + float(weight)
            for name, field in FEATURE_FIELDS.items():
                feature_mass[name] += float(weight) * (metrics[smiles][field] > 0)
        role_reference[role] = {
            "unique_components": len(masses),
            "effective_weighted_component_count": _effective_count(masses),
            "feature_occurrence_fractions": feature_mass,
        }

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    chemistry_records = tuple(
        record
        for fold in ("train", "calibration", "heldout")
        for record in corpus.records_by_fold[fold]
    )
    joint_records = tuple(
        record for fold in ("train", "calibration", "heldout") for record in records_by_fold[fold]
    )
    if tuple(record.product_id for record in chemistry_records) != tuple(
        record.product_id for record in joint_records
    ):
        raise RuntimeError("chemistry and joint-cache records are not aligned")
    atom_classes = len(corpus.atom_vocabulary)
    atom_by_role_distance = {
        role: {
            str(distance): {
                "atom_state_mass": [0.0] * atom_classes,
                "atom_mass": 0.0,
            }
            for distance in range(MAXIMUM_DISTANCE_BIN + 1)
        }
        for role in ROLE_NAMES
    }
    for record, weight in zip(chemistry_records, weights, strict=True):
        for role in ROLE_NAMES:
            own_port_distance = _distance_to_role_port(record.condition, role)
            nodes = np.flatnonzero(
                (record.condition.origin_states == ORIGIN_TO_INDEX[role])
                & ~record.condition.fixed_atom_mask
            )
            for node in nodes:
                distance = min(int(own_port_distance[int(node)]), MAXIMUM_DISTANCE_BIN)
                state = int(record.target.atom_states[int(node)])
                bucket = atom_by_role_distance[role][str(distance)]
                bucket["atom_state_mass"][state] += float(weight)
                bucket["atom_mass"] += float(weight)
    output = {
        "schema_version": "phase1_ugi_balanced_terminal_reference.v1",
        "status": "complete",
        "interpretation": "Exact all-fold product occurrence measure used by the production refit; this is not the raw row-frequency distribution.",
        "inputs": {
            "training_cache": {
                "path": str(args.cache.relative_to(REPO)),
                "sha256": sha256_file(args.cache),
            },
            "checkpoint": {
                "path": str(args.checkpoint.relative_to(REPO)),
                "sha256": sha256_file(args.checkpoint),
            },
        },
        "sampling": sampling,
        "products": len(assignments),
        "minimum_row_weight": float(weights.min()),
        "maximum_row_weight": float(weights.max()),
        "effective_weighted_product_count": float(
            math.exp(-sum(weight * math.log(weight) for weight in weights if weight > 0))
        ),
        "role_component_reference": role_reference,
        "role_atom_source_marginals": np.asarray(
            checkpoint["source_marginals"]["atoms"], dtype=np.float64
        ).tolist(),
        "role_distance_atom_reference": {
            "maximum_distance_bin": MAXIMUM_DISTANCE_BIN,
            "distance_bin_policy": f"exact_0_to_{MAXIMUM_DISTANCE_BIN - 1}_then_ge_{MAXIMUM_DISTANCE_BIN}",
            "by_role_distance": atom_by_role_distance,
        },
    }
    _atomic_json(args.output, output)
    print(json.dumps({"output": str(args.output), "roles": role_reference}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
