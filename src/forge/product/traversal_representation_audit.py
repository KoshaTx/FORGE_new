"""Compare canonical breadth-first and depth-first sparse lipid encodings."""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, rdBase

from forge.product.canonical_representation_audit import (
    _random_nonidentity_order,
    _row_seed,
    load_atom_vocabulary,
    sparse_record_signature,
)
from forge.product.defog_feasibility import sha256_file
from forge.product.lipid_context import rooted_distances
from forge.product.phase1_tree_topology_flow import (
    offspring_to_parents,
    preorder_offspring_to_parents,
    preorder_record_offspring_counts,
    record_offspring_counts,
)
from forge.product.v5_sparse_representation import (
    V5SparseGraphRecord,
    canonical_constitutional_molecule,
    canonical_tree,
    tensorize_v5_sparse_molecule,
    v5_constitutional_roundtrip_exact,
    v5_sparse_program_valid,
)

_IMPLEMENTATION_PATHS = {
    "canonical_representation_audit": Path(__file__)
    .with_name("canonical_representation_audit.py")
    .resolve(),
    "offspring_tree_decoder": Path(__file__).with_name("phase1_tree_topology_flow.py").resolve(),
    "traversal_representation_audit": Path(__file__).resolve(),
    "v5_sparse_representation": Path(__file__).with_name("v5_sparse_representation.py").resolve(),
}


class TraversalRepresentationAuditError(RuntimeError):
    """Raised when the traversal-comparison contract cannot be evaluated."""


TRAVERSALS = (
    "breadth_first",
    "breadth_first_tree_preorder",
    "depth_first_preorder",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise TraversalRepresentationAuditError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TraversalRepresentationAuditError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TraversalRepresentationAuditError(f"{label} must be a JSON object")
    return payload


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "q00": float(array.min()),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "q50": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "q100": float(array.max()),
    }


def _label_contiguity(labels: Sequence[int], target_label: int | None = None) -> dict[str, float]:
    if not labels:
        raise TraversalRepresentationAuditError("cannot measure an empty label sequence")
    counts = Counter(int(label) for label in labels)
    runs: Counter[int] = Counter()
    largest: Counter[int] = Counter()
    current = int(labels[0])
    length = 1
    transitions = 0
    for raw_label in labels[1:]:
        label = int(raw_label)
        if label == current:
            length += 1
            continue
        runs[current] += 1
        largest[current] = max(largest[current], length)
        current = label
        length = 1
        transitions += 1
    runs[current] += 1
    largest[current] = max(largest[current], length)
    present = sorted(counts)
    result = {
        "transition_fraction": transitions / max(1, len(labels) - 1),
        "mean_runs_per_present_label": float(np.mean([runs[label] for label in present])),
        "mean_largest_run_fraction": float(
            np.mean([largest[label] / counts[label] for label in present])
        ),
    }
    if target_label is not None:
        target_count = counts.get(target_label, 0)
        result["target_runs"] = float(runs.get(target_label, 0))
        result["target_largest_run_fraction"] = (
            largest.get(target_label, 0) / target_count if target_count else 1.0
        )
    return result


def _tree_distance(parents: np.ndarray, left: int, right: int) -> int:
    ancestors: dict[int, int] = {}
    node = left
    distance = 0
    while True:
        ancestors[node] = distance
        if node == 0:
            break
        node = int(parents[node])
        distance += 1
    node = right
    distance = 0
    while node not in ancestors:
        node = int(parents[node])
        distance += 1
    return distance + ancestors[node]


def _offspring(record: V5SparseGraphRecord, traversal: str) -> np.ndarray:
    if record.tree_traversal != traversal:
        raise TraversalRepresentationAuditError(
            f"record traversal mismatch: {record.tree_traversal} != {traversal}"
        )
    offspring = record.offspring
    if traversal == "breadth_first":
        restored = offspring_to_parents(offspring)
        extracted = record_offspring_counts(record)
    elif traversal in {"breadth_first_tree_preorder", "depth_first_preorder"}:
        restored = preorder_offspring_to_parents(offspring)
        extracted = preorder_record_offspring_counts(record)
    else:  # pragma: no cover - guarded by the frozen traversal set
        raise TraversalRepresentationAuditError(f"unsupported traversal: {traversal}")
    if not np.array_equal(restored, record.parents) or not np.array_equal(
        extracted,
        offspring,
    ):
        raise TraversalRepresentationAuditError(
            f"offspring word does not restore {traversal} parents: {record.structure_id}"
        )
    return offspring


def _sibling_symmetry_ties(
    molecule: Chem.Mol,
    *,
    root_strategy: str,
    traversal: str,
) -> tuple[int, int, list[int]]:
    order, parents = canonical_tree(
        molecule,
        root_strategy=root_strategy,
        tree_traversal=traversal,
    )
    symmetry_ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=False))
    children: defaultdict[int, list[int]] = defaultdict(list)
    for child in order[1:]:
        children[parents[child]].append(child)
    tied = 0
    for values in children.values():
        ranks = [symmetry_ranks[child] for child in values]
        tied += int(len(ranks) != len(set(ranks)))
    return tied, len(children), order


def _new_metrics() -> defaultdict[str, list[float]]:
    return defaultdict(list)


def _append_record_metrics(
    metrics: defaultdict[str, list[float]],
    record: V5SparseGraphRecord,
    offspring: np.ndarray,
    graph_root_distances: Sequence[int],
) -> None:
    if record.region_states is None:
        raise TraversalRepresentationAuditError("traversal audit requires region states")
    contiguity = _label_contiguity(record.region_states.tolist(), target_label=2)
    for key, value in contiguity.items():
        metrics[f"region_{key}"].append(value)
    metrics["maximum_children"].append(float(offspring.max(initial=0)))
    metrics["tree_leaf_fraction"].append(float(np.mean(offspring == 0)))
    parents = record.parents
    tree_root_distances = np.zeros(record.node_count, dtype=np.int64)
    for child in range(1, record.node_count):
        tree_root_distances[child] = tree_root_distances[int(parents[child])] + 1
    graph_root_distance_array = np.asarray(graph_root_distances, dtype=np.int64)
    if graph_root_distance_array.shape != (record.node_count,):
        raise TraversalRepresentationAuditError("root-distance vector has the wrong shape")
    root_distance_stretch = tree_root_distances - graph_root_distance_array
    if np.any(root_distance_stretch < 0):
        raise TraversalRepresentationAuditError("tree path is shorter than a graph shortest path")
    metrics["maximum_tree_root_distance"].append(float(tree_root_distances.max()))
    metrics["maximum_graph_root_distance"].append(float(graph_root_distance_array.max()))
    metrics["mean_root_distance_stretch"].append(float(root_distance_stretch.mean()))
    metrics["maximum_root_distance_stretch"].append(float(root_distance_stretch.max()))
    if record.closure_count:
        for left, right in zip(record.closure_left, record.closure_right, strict=True):
            span = abs(int(left) - int(right))
            metrics["closure_sequence_span"].append(float(span))
            metrics["closure_normalized_sequence_span"].append(span / max(1, record.node_count - 1))
            metrics["closure_tree_distance"].append(
                float(_tree_distance(record.parents, int(left), int(right)))
            )


def _bigram_cross_entropy(
    train_counts: np.ndarray,
    heldout_counts: np.ndarray,
    *,
    alpha: float,
) -> float:
    if alpha <= 0:
        raise TraversalRepresentationAuditError("bigram smoothing must be positive")
    vocabulary_size = train_counts.shape[1]
    probabilities = (train_counts + alpha) / (
        train_counts.sum(axis=1, keepdims=True) + alpha * vocabulary_size
    )
    tokens = float(heldout_counts.sum())
    if tokens == 0:
        raise TraversalRepresentationAuditError("heldout split has no bigram tokens")
    return float(-(heldout_counts * np.log(probabilities)).sum() / tokens)


def _update_bigram(
    counts: np.ndarray,
    offspring: np.ndarray,
    regions: np.ndarray,
    *,
    maximum_children: int,
) -> None:
    if int(offspring.max(initial=0)) > maximum_children:
        raise TraversalRepresentationAuditError("offspring count exceeds declared token support")
    child_states = maximum_children + 1
    tokens = regions.astype(np.int64) * child_states + offspring
    previous = counts.shape[1]
    for token in tokens:
        counts[previous, int(token)] += 1
        previous = int(token)


def _origin_sequences(
    product_rows: Sequence[Mapping[str, str]],
    atom_rows: Sequence[Mapping[str, str]],
    *,
    root_strategy: str,
    allowed_constitutional_smiles: set[str],
) -> dict[str, list[list[int]]]:
    origin_names = sorted({str(row["origin_role"]) for row in atom_rows})
    origin_to_index = {name: index for index, name in enumerate(origin_names)}
    atoms_by_product: defaultdict[str, dict[int, int]] = defaultdict(dict)
    for row in atom_rows:
        atoms_by_product[str(row["product_id"])][int(row["product_atom_index"])] = origin_to_index[
            str(row["origin_role"])
        ]
    sequences = {traversal: [] for traversal in TRAVERSALS}
    for row in product_rows:
        product_id = str(row["product_id"])
        smiles = str(row["product_smiles"])
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise TraversalRepresentationAuditError(f"invalid Ugi product: {product_id}")
        molecule = canonical_constitutional_molecule(molecule, product_id)
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        if canonical not in allowed_constitutional_smiles:
            continue
        if canonical != smiles:
            raise TraversalRepresentationAuditError(
                f"Ugi product atom identity is not canonical output order: {product_id}"
            )
        origin_by_atom = atoms_by_product[product_id]
        if sorted(origin_by_atom) != list(range(molecule.GetNumAtoms())):
            raise TraversalRepresentationAuditError(
                f"Ugi origin rows do not cover the product graph: {product_id}"
            )
        for traversal in TRAVERSALS:
            order, _ = canonical_tree(
                molecule,
                root_strategy=root_strategy,
                tree_traversal=traversal,
            )
            sequences[traversal].append([origin_by_atom[index] for index in order])
    return sequences


def _summarize_metrics(metrics: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    return {key: _quantiles(values) for key, values in sorted(metrics.items())}


def _comparison_winner(values: Mapping[str, float], *, higher_is_better: bool) -> str:
    target = max(values.values()) if higher_is_better else min(values.values())
    winners = [
        name
        for name, value in values.items()
        if math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12)
    ]
    return winners[0] if len(winners) == 1 else "tie"


def audit_traversal_representations(
    r0_path: Path,
    assignments_path: Path,
    atom_vocabulary_path: Path,
    config_path: Path,
    ugi_products_path: Path,
    ugi_atoms_path: Path,
    *,
    seed: int,
    permutations_per_record: int,
    bigram_alpha: float = 0.5,
    maximum_failure_examples: int = 20,
) -> dict[str, Any]:
    """Measure the two exact offspring encodings on broad and Ugi data."""

    if seed < 0 or permutations_per_record < 1 or bigram_alpha <= 0:
        raise TraversalRepresentationAuditError("invalid traversal-audit numeric policy")
    paths = {
        "r0_constitutional": r0_path,
        "r0_assignments": assignments_path,
        "atom_vocabulary": atom_vocabulary_path,
        "product_config": config_path,
        "ugi_semantic_products": ugi_products_path,
        "ugi_semantic_atoms": ugi_atoms_path,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise TraversalRepresentationAuditError(f"{label} not found: {path}")

    config = _load_json(config_path, "product config")
    model = config.get("model")
    if not isinstance(model, dict):
        raise TraversalRepresentationAuditError("product config is missing model settings")
    preserve_aromaticity = bool(model.get("preserve_aromaticity", False))
    root_strategy = str(model.get("root_strategy", "canonical"))
    region_scheme = str(model.get("region_scheme", "none"))
    maximum_children = int(model["degree_prior_maximum_degree"])
    region_classes = int(model["region_classes"])
    vocabulary = load_atom_vocabulary(atom_vocabulary_path)
    atom_to_index = {state: index for index, state in enumerate(vocabulary)}
    rows = _read_csv(r0_path)
    assignments = {
        str(row["r0_structure_id"]): str(row["source_study_fold"])
        for row in _read_csv(assignments_path)
    }
    if set(assignments) != {str(row["r0_structure_id"]) for row in rows}:
        raise TraversalRepresentationAuditError("source-study folds do not cover R0 exactly")
    train_constitutions: set[str] = set()
    for row in rows:
        if assignments[str(row["r0_structure_id"])] != "R0_train":
            continue
        molecule = Chem.MolFromSmiles(str(row["canonical_isomeric_smiles"]))
        if molecule is None:  # pragma: no cover - guarded below for all rows
            raise TraversalRepresentationAuditError("invalid R0 molecule in training fold")
        train_constitutions.add(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False))

    token_classes = region_classes * (maximum_children + 1)
    counts = {traversal: Counter() for traversal in TRAVERSALS}
    metrics = {traversal: _new_metrics() for traversal in TRAVERSALS}
    bigrams = {
        traversal: {
            fold: np.zeros((token_classes + 1, token_classes), dtype=np.int64)
            for fold in ("R0_train", "R0_cal")
        }
        for traversal in TRAVERSALS
    }
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    for row in rows:
        structure_id = str(row["r0_structure_id"])
        smiles = str(row["canonical_isomeric_smiles"])
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise TraversalRepresentationAuditError(f"invalid R0 molecule: {structure_id}")
        permutation_rng = np.random.default_rng(_row_seed(seed, structure_id))
        permutation_orders = [
            _random_nonidentity_order(molecule.GetNumAtoms(), permutation_rng)
            for _ in range(permutations_per_record)
        ]
        normalized = canonical_constitutional_molecule(molecule, structure_id)
        fold = assignments[structure_id]
        for traversal in TRAVERSALS:
            record = tensorize_v5_sparse_molecule(
                molecule,
                atom_to_index,
                structure_id=structure_id,
                canonical_smiles=smiles,
                preserve_aromaticity=preserve_aromaticity,
                root_strategy=root_strategy,
                region_scheme=region_scheme,
                tree_traversal=traversal,
            )
            signature = sparse_record_signature(record)
            sparse_valid = v5_sparse_program_valid(
                record,
                atom_vocabulary_size=len(vocabulary),
                region_classes=region_classes,
                maximum_children=maximum_children,
            )
            constitutional_exact = v5_constitutional_roundtrip_exact(record, vocabulary)
            offspring = _offspring(record, traversal)
            counts[traversal]["records"] += 1
            counts[traversal]["valid_sparse_programs"] += int(sparse_valid)
            counts[traversal]["exact_constitutional_roundtrips"] += int(constitutional_exact)
            counts[traversal]["exact_offspring_parent_decodes"] += 1
            tied_nodes, parent_nodes, canonical_order = _sibling_symmetry_ties(
                normalized,
                root_strategy=root_strategy,
                traversal=traversal,
            )
            if fold == "R0_cal":
                distances_by_old_index = rooted_distances(normalized, canonical_order[0])
                _append_record_metrics(
                    metrics[traversal],
                    record,
                    offspring,
                    [distances_by_old_index[index] for index in canonical_order],
                )
            counts[traversal]["sibling_symmetry_tie_nodes"] += tied_nodes
            counts[traversal]["parent_nodes"] += parent_nodes
            if fold in bigrams[traversal]:
                if record.region_states is None:
                    raise TraversalRepresentationAuditError("region states are required")
                _update_bigram(
                    bigrams[traversal][fold],
                    offspring,
                    record.region_states,
                    maximum_children=maximum_children,
                )
            for permutation_index, order in enumerate(permutation_orders):
                permuted = tensorize_v5_sparse_molecule(
                    Chem.RenumberAtoms(molecule, order),
                    atom_to_index,
                    structure_id=structure_id,
                    canonical_smiles=smiles,
                    preserve_aromaticity=preserve_aromaticity,
                    root_strategy=root_strategy,
                    region_scheme=region_scheme,
                    tree_traversal=traversal,
                )
                exact = sparse_record_signature(permuted) == signature
                counts[traversal]["permutation_encodings"] += 1
                counts[traversal]["exact_permutation_encodings"] += int(exact)
                if not exact and len(failures) < maximum_failure_examples:
                    failures.append(
                        {
                            "structure_id": structure_id,
                            "traversal": traversal,
                            "permutation_index": permutation_index,
                            "base_signature": signature,
                            "permuted_signature": sparse_record_signature(permuted),
                        }
                    )

    origin_sequences = _origin_sequences(
        _read_csv(ugi_products_path),
        _read_csv(ugi_atoms_path),
        root_strategy=root_strategy,
        allowed_constitutional_smiles=train_constitutions,
    )
    origin_product_count = len(origin_sequences[TRAVERSALS[0]])
    if origin_product_count == 0:
        raise TraversalRepresentationAuditError(
            "no exact Ugi semantic products lie in the R0 training fold"
        )
    origin_metrics = {traversal: _new_metrics() for traversal in TRAVERSALS}
    for traversal, sequences in origin_sequences.items():
        for sequence in sequences:
            values = _label_contiguity(sequence)
            for key, value in values.items():
                origin_metrics[traversal][f"origin_{key}"].append(value)

    traversal_results: dict[str, Any] = {}
    for traversal in TRAVERSALS:
        traversal_counts = counts[traversal]
        records = traversal_counts["records"]
        permutations = traversal_counts["permutation_encodings"]
        acceptance = {
            "every_record_is_a_valid_sparse_program": (
                traversal_counts["valid_sparse_programs"] == records
            ),
            "every_record_constitutionally_roundtrips": (
                traversal_counts["exact_constitutional_roundtrips"] == records
            ),
            "all_atom_permutations_serialize_exactly": (
                traversal_counts["exact_permutation_encodings"] == permutations
            ),
            "offspring_word_decodes_exact_parents": (
                traversal_counts["exact_offspring_parent_decodes"] == records
            ),
        }
        traversal_results[traversal] = {
            "status": "pass" if all(acceptance.values()) else "fail",
            "counts": {key: int(value) for key, value in sorted(traversal_counts.items())},
            "acceptance": acceptance,
            "r0_metrics": _summarize_metrics(metrics[traversal]),
            "ugi_origin_metrics": _summarize_metrics(origin_metrics[traversal]),
            "fixed_bigram_model": {
                "token": "joint_region_and_offspring_count",
                "train_fold": "R0_train",
                "evaluation_fold": "R0_cal",
                "alpha": bigram_alpha,
                "evaluation_cross_entropy_nats_per_token": _bigram_cross_entropy(
                    bigrams[traversal]["R0_train"],
                    bigrams[traversal]["R0_cal"],
                    alpha=bigram_alpha,
                ),
            },
        }

    comparisons = {
        "lower_region_transition_fraction": _comparison_winner(
            {
                traversal: traversal_results[traversal]["r0_metrics"]["region_transition_fraction"][
                    "mean"
                ]
                for traversal in TRAVERSALS
            },
            higher_is_better=False,
        ),
        "higher_tail_largest_run_fraction": _comparison_winner(
            {
                traversal: traversal_results[traversal]["r0_metrics"][
                    "region_target_largest_run_fraction"
                ]["mean"]
                for traversal in TRAVERSALS
            },
            higher_is_better=True,
        ),
        "lower_ugi_origin_transition_fraction": _comparison_winner(
            {
                traversal: traversal_results[traversal]["ugi_origin_metrics"][
                    "origin_transition_fraction"
                ]["mean"]
                for traversal in TRAVERSALS
            },
            higher_is_better=False,
        ),
        "lower_closure_normalized_sequence_span": _comparison_winner(
            {
                traversal: traversal_results[traversal]["r0_metrics"][
                    "closure_normalized_sequence_span"
                ]["mean"]
                for traversal in TRAVERSALS
            },
            higher_is_better=False,
        ),
        "lower_bigram_cross_entropy": _comparison_winner(
            {
                traversal: traversal_results[traversal]["fixed_bigram_model"][
                    "evaluation_cross_entropy_nats_per_token"
                ]
                for traversal in TRAVERSALS
            },
            higher_is_better=False,
        ),
        "lower_mean_root_distance_stretch": _comparison_winner(
            {
                traversal: traversal_results[traversal]["r0_metrics"]["mean_root_distance_stretch"][
                    "mean"
                ]
                for traversal in TRAVERSALS
            },
            higher_is_better=False,
        ),
    }
    status = (
        "pass"
        if all(result["status"] == "pass" for result in traversal_results.values())
        else "fail"
    )
    return {
        "schema_version": "phase1_v5_traversal_representation_audit.v2",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "implementation": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in sorted(_IMPLEMENTATION_PATHS.items())
        },
        "policy": {
            "seed": seed,
            "permutations_per_record": permutations_per_record,
            "root_strategy": root_strategy,
            "region_scheme": region_scheme,
            "preserve_aromaticity": preserve_aromaticity,
            "maximum_children": maximum_children,
            "tree_branch_budget_degree": "tree_degree_only",
            "closure_degree": "separate_from_tree_branch_budget",
            "selection_metrics_fold": "R0_cal",
            "r0_heldout_use": "exact_invariance_checks_only_not_architecture_selection",
            "ugi_origin_metrics_fold": "R0_train",
            "ugi_origin_product_count": origin_product_count,
            "selection": "measurement_only_no_traversal_winner_is_precommitted",
        },
        "environment": {
            "rdkit_version": rdBase.rdkitVersion,
            "numpy_version": np.__version__,
        },
        "traversals": traversal_results,
        "metric_winners": comparisons,
        "failure_examples": failures,
        "runtime": {"elapsed_seconds": time.monotonic() - started},
        "decision": (
            "all_encodings_pass_review_metric_tradeoffs_before_freeze"
            if status == "pass"
            else "do_not_train_v5_until_representation_failure_is_resolved"
        ),
    }


def write_audit(path: Path, result: Mapping[str, Any]) -> None:
    """Write the traversal comparison atomically."""

    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
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
