"""Corpus-wide audit of Ugi adapter coordinates and core-rooted serializations."""

from __future__ import annotations

import csv
import gzip
import math
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, rdBase

from forge.model.defog_feasibility import sha256_file
from forge.model.ugi_adapter_features import (
    CORE_POSITION_TO_INDEX,
    ORIGIN_STATES,
    adapter_canonical_tree,
    ugi_adapter_features,
)
from forge.potency.annotations import ROLE_NAMES

TRAVERSALS = (
    "breadth_first",
    "breadth_first_tree_preorder",
    "depth_first_preorder",
)
_IMPLEMENTATION_PATHS = {
    "ugi_adapter_feature_audit": Path(__file__).resolve(),
    "ugi_adapter_features": Path(__file__).with_name("ugi_adapter_features.py").resolve(),
}


class UgiAdapterFeatureAuditError(RuntimeError):
    """Raised when Ugi adapter features violate the frozen chemistry semantics."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "q00": float(array.min()),
        "q10": float(np.quantile(array, 0.10)),
        "q50": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
        "q100": float(array.max()),
    }


def _origin_runs(labels: Sequence[int]) -> tuple[int, float]:
    counts = Counter(int(value) for value in labels)
    runs: Counter[int] = Counter()
    largest: Counter[int] = Counter()
    current = int(labels[0])
    length = 1
    transitions = 0
    for raw in labels[1:]:
        value = int(raw)
        if value == current:
            length += 1
            continue
        runs[current] += 1
        largest[current] = max(largest[current], length)
        transitions += 1
        current = value
        length = 1
    runs[current] += 1
    largest[current] = max(largest[current], length)
    mean_largest_fraction = float(np.mean([largest[origin] / counts[origin] for origin in counts]))
    return transitions, mean_largest_fraction


def _root_distances(molecule: Chem.Mol, root: int) -> list[int]:
    distances = [-1] * molecule.GetNumAtoms()
    distances[root] = 0
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for neighbor in molecule.GetAtomWithIdx(node).GetNeighbors():
            index = neighbor.GetIdx()
            if distances[index] >= 0:
                continue
            distances[index] = distances[node] + 1
            queue.append(index)
    if min(distances) < 0:
        raise UgiAdapterFeatureAuditError("Ugi product graph is disconnected")
    return distances


def _train_product_ids(assignments: Sequence[Mapping[str, str]]) -> set[str]:
    return {
        str(row["product_id"])
        for row in assignments
        if all(row[f"held_{role}_fold"] == "train" for role in ROLE_NAMES)
    }


def audit_ugi_adapter_features(
    products_path: Path,
    atoms_path: Path,
    assignments_path: Path,
) -> dict[str, Any]:
    """Audit feature exactness and compare serialization coordinates on train roles."""

    for path in (products_path, atoms_path, assignments_path):
        if not path.is_file():
            raise UgiAdapterFeatureAuditError(f"required input is absent: {path}")
    products = _read_csv(products_path)
    atom_rows = _read_csv(atoms_path)
    assignments = _read_csv(assignments_path)
    atoms_by_product: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in atom_rows:
        atoms_by_product[row["product_id"]].append(row)
    train_ids = _train_product_ids(assignments)
    if not train_ids:
        raise UgiAdapterFeatureAuditError("component-training partition is empty")

    counts: Counter[str] = Counter()
    metrics: defaultdict[str, list[float]] = defaultdict(list)
    traversal_metrics = {traversal: defaultdict(list) for traversal in TRAVERSALS}
    origin_names = {index: name for index, name in enumerate(ORIGIN_STATES)}
    for product in products:
        product_id = product["product_id"]
        rows = atoms_by_product.get(product_id, [])
        features = ugi_adapter_features(product, rows)
        molecule = Chem.MolFromSmiles(product["product_smiles"])
        if molecule is None or molecule.GetNumAtoms() != features.node_count:
            raise UgiAdapterFeatureAuditError(f"invalid semantic product: {product_id}")
        core_candidates = np.flatnonzero(features.core_membership).tolist()
        if len(core_candidates) != 5:
            raise UgiAdapterFeatureAuditError(f"wrong Ugi core size: {product_id}")
        counts["products"] += 1
        counts["training_partition_products"] += int(product_id in train_ids)
        counts["exact_five_atom_cores"] += 1
        counts["exact_three_role_anchors"] += int(np.count_nonzero(features.port_states) == 3)
        counts["orthogonal_core_origin_products"] += int(
            len(set(features.origin_states[features.core_membership].tolist())) > 1
        )
        for role_index, role in enumerate(ROLE_NAMES):
            selected = features.origin_states == ORIGIN_STATES.index(role)
            metrics[f"{role}_atom_count"].append(float(np.count_nonzero(selected)))
            metrics[f"{role}_maximum_own_port_distance"].append(
                float(features.distance_to_own_port[selected].max(initial=0))
            )

        if product_id not in train_ids:
            continue
        for traversal in TRAVERSALS:
            order, _ = adapter_canonical_tree(
                molecule,
                core_candidates=core_candidates,
                tree_traversal=traversal,
            )
            origins = features.origin_states[np.asarray(order, dtype=np.int64)].tolist()
            transitions, largest_fraction = _origin_runs(origins)
            traversal_metrics[traversal]["origin_transition_fraction"].append(
                transitions / max(1, len(origins) - 1)
            )
            traversal_metrics[traversal]["mean_largest_origin_run_fraction"].append(
                largest_fraction
            )
            traversal_metrics[traversal]["root_is_core"].append(
                float(features.core_membership[order[0]])
            )

            if traversal == "depth_first_preorder":
                root_distances = _root_distances(molecule, order[0])
                mixed_bins = 0
                depth_to_origins: defaultdict[int, set[int]] = defaultdict(set)
                for atom_index, distance in enumerate(root_distances):
                    depth_to_origins[distance].add(int(features.origin_states[atom_index]))
                for origins_at_depth in depth_to_origins.values():
                    mixed_bins += int(len(origins_at_depth) > 1)
                metrics["root_depth_bins"].append(float(len(depth_to_origins)))
                metrics["root_depth_bins_with_multiple_origins"].append(float(mixed_bins))
                counts["products_where_root_depth_conflates_origins"] += int(mixed_bins > 0)

    product_count = int(counts["products"])
    train_count = int(counts["training_partition_products"])
    if product_count != len(products) or set(atoms_by_product) != {
        row["product_id"] for row in products
    }:
        raise UgiAdapterFeatureAuditError("semantic ledgers do not align by product")
    traversal_summary = {
        traversal: {
            key: _quantiles(values) for key, values in sorted(traversal_metrics[traversal].items())
        }
        for traversal in TRAVERSALS
    }
    transition_means = {
        traversal: traversal_summary[traversal]["origin_transition_fraction"]["mean"]
        for traversal in TRAVERSALS
    }
    minimum_transition = min(transition_means.values())
    winners = [
        traversal
        for traversal, value in transition_means.items()
        if math.isclose(value, minimum_transition, abs_tol=1e-12, rel_tol=0.0)
    ]
    gates = {
        "all_products_have_exact_features": product_count == len(products),
        "all_products_have_five_atom_core": counts["exact_five_atom_cores"] == product_count,
        "all_products_have_three_role_anchors": counts["exact_three_role_anchors"] == product_count,
        "core_membership_is_not_an_exclusive_origin": counts["orthogonal_core_origin_products"]
        == product_count,
        "all_serialization_roots_are_core_atoms": all(
            traversal_summary[traversal]["root_is_core"]["mean"] == 1.0 for traversal in TRAVERSALS
        ),
        "root_depth_is_empirically_insufficient": counts[
            "products_where_root_depth_conflates_origins"
        ]
        == train_count,
    }
    return {
        "schema_version": "phase1_ugi_adapter_feature_audit.v1",
        "status": "pass" if all(gates.values()) else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "semantic_products": {
                "path": str(products_path),
                "sha256": sha256_file(products_path),
            },
            "semantic_atoms": {
                "path": str(atoms_path),
                "sha256": sha256_file(atoms_path),
            },
            "assignments": {
                "path": str(assignments_path),
                "sha256": sha256_file(assignments_path),
            },
        },
        "implementation": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in sorted(_IMPLEMENTATION_PATHS.items())
        },
        "software": {"rdkit": rdBase.rdkitVersion, "numpy": np.__version__},
        "counts": dict(sorted(counts.items())),
        "feature_metrics": {key: _quantiles(values) for key, values in sorted(metrics.items())},
        "serialization_metrics": traversal_summary,
        "lowest_origin_transition_fraction": (winners[0] if len(winners) == 1 else "tie"),
        "origin_state_names": origin_names,
        "core_position_states": CORE_POSITION_TO_INDEX,
        "gates": gates,
        "decision_policy": {
            "root_role": "serialization_only",
            "primary_semantics": "origin_core_port_and_post_topology_graph_distances",
            "selection_partition": "all_three_component_roles_in_training_fold",
            "all_port_distance_vector": "ablation_only",
        },
    }
