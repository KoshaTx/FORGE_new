#!/usr/bin/env python3
"""Compare canonical BFS and preorder sparse lipid representations."""

from __future__ import annotations

import argparse
from pathlib import Path

from forge.design.audit.traversal_representation_audit import (
    audit_traversal_representations,
    write_audit,
)

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r0",
        type=Path,
        default=REPO / "results/m0_03/r0_constitutional.csv.gz",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=REPO / "data/splits/m0_03_constitutional/r0_fold_assignments.csv",
    )
    parser.add_argument(
        "--atom-vocabulary",
        type=Path,
        default=REPO / "results/phase1/product_v3_atom_vocabulary.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_product_pretrain_v3.json",
    )
    parser.add_argument(
        "--ugi-products",
        type=Path,
        default=REPO / "results/m0_07/ugi_semantic_products.csv.gz",
    )
    parser.add_argument(
        "--ugi-atoms",
        type=Path,
        default=REPO / "results/m0_07/ugi_semantic_atoms.csv.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/v5_tree_traversal_comparison.json",
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--permutations-per-record", type=int, default=1)
    args = parser.parse_args()

    result = audit_traversal_representations(
        args.r0.resolve(),
        args.assignments.resolve(),
        args.atom_vocabulary.resolve(),
        args.config.resolve(),
        args.ugi_products.resolve(),
        args.ugi_atoms.resolve(),
        seed=args.seed,
        permutations_per_record=args.permutations_per_record,
    )
    write_audit(args.output.resolve(), result)
    print(args.output)
    if result["status"] != "pass":
        raise RuntimeError("tree traversal representation comparison failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
