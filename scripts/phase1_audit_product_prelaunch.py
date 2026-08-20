#!/usr/bin/env python3
"""Audit every R1 product before Phase 1 GPU training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.design.audit.phase1_prelaunch_audit import (
    Phase1PrelaunchAuditError,
    run_prelaunch_audit,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_product_prelaunch_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/product_prelaunch_audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_prelaunch_audit(args.config, args.output, REPO)
    except Phase1PrelaunchAuditError as exc:
        print(f"Phase 1 prelaunch audit failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "r1_rows": result["r1_audit"]["rows"],
                "maximum_heavy_atoms": result["r1_audit"]["maximum_heavy_atoms"],
                "maximum_closures": result["r1_audit"]["maximum_closures"],
                "failure_counts": result["r1_audit"]["failure_counts"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
