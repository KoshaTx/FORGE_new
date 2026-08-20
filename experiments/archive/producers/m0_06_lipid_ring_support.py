#!/usr/bin/env python3
"""Audit lipid-native ring topology and freeze the primary Ugi support gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.archive.phase1.design_audits.ring_support_audit import (
    RingSupportAuditError,
    run_ring_support_audit,
)

REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/m0_06_lipid_ring_support.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/m0_06_ring_support/result.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_ring_support_audit(args.config, args.output, REPO)
    except RingSupportAuditError as exc:
        print(f"M0-06 ring-support audit failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "r0_cyclic_records": result["broad_r0"]["cyclic_records"],
                "r0_macrocycle_records": result["broad_r0"]["macrocycle_records"],
                "agile_ring_size_counts": result["reconciled_agile_ugi"]["product"][
                    "ring_size_instance_counts"
                ],
                "primary_ugi_automatic_ring_gate_supported": result["decision"][
                    "primary_ugi_automatic_ring_gate_supported"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
