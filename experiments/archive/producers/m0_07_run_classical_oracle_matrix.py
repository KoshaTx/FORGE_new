#!/usr/bin/env python3
"""Run the deterministic classical lane of the M0-07 AGILE oracle matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.potency.oracle.oracle_classical import (
    OracleClassicalError,
    run_classical_oracle_matrix,
)

REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/bio/m0_07_oracle_classical.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/m0_07",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_classical_oracle_matrix(
            args.config,
            args.output_dir,
            args.repo_root,
        )
    except OracleClassicalError as exc:
        print(f"M0-07 classical oracle matrix failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": str(args.output_dir),
                "summary": result["summary"],
                "classical_lane_leaders": result["classical_lane_leaders"],
                "applicability": result["applicability"],
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
