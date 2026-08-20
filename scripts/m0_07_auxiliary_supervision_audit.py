#!/usr/bin/env python3
"""Audit auxiliary Ugi and related 3CR biological supervision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.potency.oracle.oracle_auxiliary import (
    AuxiliarySupervisionError,
    run_auxiliary_supervision_audit,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/bio/m0_07_auxiliary_supervision.json",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/m0_07")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_auxiliary_supervision_audit(args.config, args.output_dir, args.repo_root)
    except AuxiliarySupervisionError as exc:
        print(f"M0-07 auxiliary supervision audit failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": str(args.output_dir),
                "studies": result["studies"],
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
