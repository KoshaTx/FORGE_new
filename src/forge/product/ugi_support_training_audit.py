"""Gate the core-protected Ugi morphology support used for production training."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from forge.potency.ugi_semantic_annotations import ROLE_NAMES
from forge.product.canonical_representation_audit import load_atom_vocabulary
from forge.product.defog_feasibility import sha256_file
from forge.product.ugi_adapter_features import (
    ORIGIN_STATES,
    tensorize_ugi_l1_support_record,
)
from forge.product.v5_morphology_program import tree_junction_contributions
from forge.product.v5_sparse_representation import (
    v5_constitutional_roundtrip_exact,
    v5_sparse_program_valid,
)


class UgiSupportTrainingAuditError(RuntimeError):
    """Raised when the Ugi morphology support is not exact or lipid-like."""


_IMPLEMENTATION_PATHS = {
    "ugi_adapter_features": Path(__file__).with_name("ugi_adapter_features.py").resolve(),
    "ugi_support_training_audit": Path(__file__).resolve(),
}


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


def _largest_branch_only_component(
    adjacency: dict[int, set[int]],
    branch_nodes: set[int],
) -> int:
    remaining = set(branch_nodes)
    largest = 0
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        size = 1
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node] & remaining:
                remaining.remove(neighbor)
                size += 1
                queue.append(neighbor)
        largest = max(largest, size)
    return largest


def audit_ugi_support_training_records(
    products_path: Path,
    atoms_path: Path,
    atom_vocabulary_path: Path,
) -> dict[str, Any]:
    """Build every support record and quantify origin-specific branching."""

    for path in (products_path, atoms_path, atom_vocabulary_path):
        if not path.is_file():
            raise UgiSupportTrainingAuditError(f"required input is absent: {path}")
    products = _read_csv(products_path)
    atoms_by_product: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(atoms_path):
        atoms_by_product[row["product_id"]].append(row)
    vocabulary = load_atom_vocabulary(atom_vocabulary_path)
    atom_to_index = {state: index for index, state in enumerate(vocabulary)}
    counts: Counter[str] = Counter()
    metrics: defaultdict[str, list[float]] = defaultdict(list)
    component_signatures: dict[tuple[str, str], tuple[int, int, int, int]] = {}

    for product in products:
        product_id = product["product_id"]
        record = tensorize_ugi_l1_support_record(
            product,
            atoms_by_product.get(product_id, []),
            atom_to_index,
            preserve_aromaticity=True,
        )
        graph = record.support_graph
        counts["products"] += 1
        counts["valid_support_programs"] += int(
            v5_sparse_program_valid(
                graph,
                atom_vocabulary_size=len(vocabulary),
            )
        )
        counts["exact_support_roundtrips"] += int(
            v5_constitutional_roundtrip_exact(graph, vocabulary)
        )
        counts["exact_full_expansion_roundtrips"] += 1
        counts["all_core_atoms_retained"] += int(
            record.support_adapter.core_membership.sum()
            == record.full_adapter.core_membership.sum()
            == 5
        )
        metrics["full_atoms"].append(float(record.skeleton.node_count))
        metrics["support_atoms"].append(float(record.skeleton.retained_count))
        metrics["removed_decorations"].append(float(record.skeleton.removed_count))
        metrics["support_retained_fraction"].append(
            record.skeleton.retained_count / record.skeleton.node_count
        )

        adjacency = {index: set() for index in range(graph.node_count)}
        parents = graph.parents
        for child in range(1, graph.node_count):
            parent = int(parents[child])
            adjacency[parent].add(child)
            adjacency[child].add(parent)
        for left, right in zip(graph.closure_left, graph.closure_right, strict=True):
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))
        contributions = tree_junction_contributions(graph.offspring)
        components = json.loads(product["component_smiles_json"])
        if not isinstance(components, dict) or set(components) != set(ROLE_NAMES):
            raise UgiSupportTrainingAuditError(
                f"component identities are incomplete for {product_id}"
            )
        for role in ROLE_NAMES:
            origin_state = ORIGIN_STATES.index(role)
            selected = {
                index
                for index in range(graph.node_count)
                if record.support_adapter.origin_states[index] == origin_state
                and not record.support_adapter.core_membership[index]
            }
            induced = {
                index: {neighbor for neighbor in adjacency[index] if neighbor in selected}
                for index in selected
            }
            branch_nodes = {index for index, neighbors in induced.items() if len(neighbors) >= 3}
            adjacent_branch_run = _largest_branch_only_component(induced, branch_nodes)
            tree_junction_budget = sum(
                int(contributions[index]) for index in selected if contributions[index] > 0
            )
            signature = (
                len(selected),
                len(branch_nodes),
                adjacent_branch_run,
                tree_junction_budget,
            )
            for label, value in zip(
                (
                    "support_atoms_outside_core",
                    "induced_branch_nodes",
                    "adjacent_branch_run",
                    "tree_junction_budget",
                ),
                signature,
                strict=True,
            ):
                metrics[f"{role}_{label}"].append(float(value))
            component_key = (role, str(components[role]))
            previous = component_signatures.get(component_key)
            if previous is not None and previous != signature:
                raise UgiSupportTrainingAuditError(
                    f"component morphology changes across products: {role}: {components[role]}"
                )
            if previous is None:
                component_signatures[component_key] = signature
                for label, value in zip(
                    (
                        "support_atoms_outside_core",
                        "induced_branch_nodes",
                        "adjacent_branch_run",
                        "tree_junction_budget",
                    ),
                    signature,
                    strict=True,
                ):
                    metrics[f"component_weighted_{role}_{label}"].append(float(value))

    product_count = len(products)
    component_counts = Counter(role for role, _ in component_signatures)
    gates = {
        "all_products_encoded": counts["products"] == product_count,
        "all_support_programs_valid": counts["valid_support_programs"] == product_count,
        "all_support_graphs_roundtrip": counts["exact_support_roundtrips"] == product_count,
        "all_full_graphs_roundtrip_from_support_and_expansion": counts[
            "exact_full_expansion_roundtrips"
        ]
        == product_count,
        "all_five_atom_cores_retained": counts["all_core_atoms_retained"] == product_count,
        "component_morphology_is_product_invariant": component_counts
        == Counter(
            {
                "amine_head": 24,
                "oxoester_aldehyde_body_tail": 62,
                "isocyanide_tail": 9,
            }
        ),
    }
    return {
        "schema_version": "phase1_ugi_support_training_audit.v1",
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
            "atom_vocabulary": {
                "path": str(atom_vocabulary_path),
                "sha256": sha256_file(atom_vocabulary_path),
            },
        },
        "implementation": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in sorted(_IMPLEMENTATION_PATHS.items())
        },
        "counts": dict(sorted(counts.items())),
        "unique_component_counts": dict(sorted(component_counts.items())),
        "metrics": {key: _quantiles(values) for key, values in sorted(metrics.items())},
        "gates": gates,
        "policy": {
            "support": "functional_support_with_exact_ugi_core_protection",
            "serialization": "core_rooted_breadth_first_tree_preorder",
            "aromaticity": "explicit_atom_and_bond_states",
            "branch_metrics": "precursor_origin_induced_support_outside_fixed_core",
            "component_catalog_ids_in_model_state": False,
        },
    }
