#!/usr/bin/env python3
"""Audit exact lipid support skeletons on broad R0 and Ugi product corpora."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.archive.phase1.design_audits.lipid_support_skeleton_audit import (
    audit_lipid_support_skeletons,
    write_audit_artifacts,
)

REPO = Path(__file__).resolve().parents[3]


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
        "--ugi-l1",
        type=Path,
        default=REPO / "data/splits/phase1/ugi_l1_assignments.csv.gz",
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
        default=REPO / "results/phase1/lipid_support_skeleton_audit.json",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPO / "results/phase1/lipid_support_skeleton_ledger.csv.gz",
    )
    parser.add_argument(
        "--origin-ledger",
        type=Path,
        default=REPO / "results/phase1/lipid_support_skeleton_ugi_origins.csv.gz",
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--permutation-records", type=int, default=256)
    args = parser.parse_args()

    result, ledger, origin_ledger = audit_lipid_support_skeletons(
        args.r0.resolve(),
        args.assignments.resolve(),
        args.ugi_l1.resolve(),
        args.ugi_products.resolve(),
        args.ugi_atoms.resolve(),
        seed=args.seed,
        permutation_records=args.permutation_records,
    )
    write_audit_artifacts(
        args.output.resolve(),
        args.ledger.resolve(),
        args.origin_ledger.resolve(),
        result,
        ledger,
        origin_ledger,
    )
    print(args.output)
    if result["status"] != "pass":
        raise RuntimeError("lipid support skeleton audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
