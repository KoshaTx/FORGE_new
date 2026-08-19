#!/usr/bin/env python3
"""Audit the V5 canonical tree-plus-closure representation under atom permutations."""

from __future__ import annotations

import argparse
from pathlib import Path

from forge.product.canonical_representation_audit import (
    audit_canonical_representation,
    require_pass,
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
        "--output",
        type=Path,
        default=REPO / "results/phase1/v5_canonical_representation_audit.json",
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--permutations-per-record", type=int, default=2)
    parser.add_argument("--tree-traversal", default="breadth_first_tree_preorder")
    args = parser.parse_args()

    result = audit_canonical_representation(
        args.r0.resolve(),
        args.atom_vocabulary.resolve(),
        args.config.resolve(),
        seed=args.seed,
        permutations_per_record=args.permutations_per_record,
        tree_traversal=args.tree_traversal,
    )
    write_audit(args.output.resolve(), result)
    print(args.output)
    require_pass(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
