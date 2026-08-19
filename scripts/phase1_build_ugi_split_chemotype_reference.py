#!/usr/bin/env python3
"""Build a fold-specific, source-balanced Ugi component chemotype reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from forge.data.r1_prime_audit import sha256_file
from forge.potency.ugi_semantic_annotations import ROLE_NAMES
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
SAMPLING = {
    "mode": "source_stratified_role_family_raked",
    "source_mass": {
        "current_phase1_union": 0.5,
        "expanded_exact_forward_enumeration": 0.5,
    },
    "uniform_row_mixture": 0.5,
}


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


def build_split_reference(cache_path: Path, fold: str) -> dict[str, Any]:
    """Compute the exact weighted chemotype measure for one frozen structural fold."""

    corpus, _ = load_ugi_training_cache(cache_path)
    if fold not in corpus.assignments_by_fold:
        raise ValueError(f"unknown Ugi structural fold: {fold}")
    assignments = tuple(corpus.assignments_by_fold[fold])
    weights = _training_weights(assignments, SAMPLING)
    if len(weights) != len(assignments) or not math.isclose(float(weights.sum()), 1.0):
        raise RuntimeError("split training weights are malformed")
    unique_smiles = {row[f"{role}_smiles"] for row in assignments for role in ROLE_NAMES}
    metrics = {smiles: component_chemotype_metrics(smiles) for smiles in unique_smiles}
    role_reference: dict[str, Any] = {}
    for role in ROLE_NAMES:
        masses: dict[str, float] = {}
        feature_mass = {name: 0.0 for name in FEATURE_FIELDS}
        for row, weight in zip(assignments, weights, strict=True):
            smiles = str(row[f"{role}_smiles"])
            masses[smiles] = masses.get(smiles, 0.0) + float(weight)
            for name, field in FEATURE_FIELDS.items():
                feature_mass[name] += float(weight) * (metrics[smiles][field] > 0)
        role_reference[role] = {
            "unique_components": len(masses),
            "effective_weighted_component_count": _effective_count(masses),
            "feature_occurrence_fractions": feature_mass,
        }
    product_ids = sorted(str(row["product_id"]) for row in assignments)
    return {
        "schema_version": "phase1_ugi_split_chemotype_reference.v1",
        "status": "complete",
        "fold": fold,
        "input": {
            "path": str(cache_path.relative_to(REPO)),
            "sha256": sha256_file(cache_path),
        },
        "sampling": SAMPLING,
        "products": len(assignments),
        "product_id_sha256": hashlib.sha256("\n".join(product_ids).encode()).hexdigest(),
        "minimum_row_weight": float(weights.min()),
        "maximum_row_weight": float(weights.max()),
        "effective_weighted_product_count": math.exp(
            -sum(float(weight) * math.log(float(weight)) for weight in weights if weight > 0)
        ),
        "role_component_reference": role_reference,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO / "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt",
    )
    parser.add_argument("--fold", choices=("train", "calibration", "heldout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = build_split_reference(args.cache, args.fold)
    _atomic_json(args.output, output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fold": args.fold,
                "products": output["products"],
                "role_component_reference": output["role_component_reference"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
