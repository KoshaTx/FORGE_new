#!/usr/bin/env python3
"""Audit the global morphology of the frozen R0 whole-lipid training fold."""

from __future__ import annotations

import argparse
from pathlib import Path

from forge.design.audit.lipid_morphology_audit import audit_r0_morphology, write_audit

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
        "--output",
        type=Path,
        default=REPO / "results/phase1/r0_lipid_morphology_audit.json",
    )
    args = parser.parse_args()
    result = audit_r0_morphology(
        args.r0.resolve(),
        args.assignments.resolve(),
    )
    write_audit(args.output.resolve(), result)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
