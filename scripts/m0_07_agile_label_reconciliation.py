#!/usr/bin/env python3
"""Build the blocking AGILE/LANTERN assay and structure reconciliation gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.potency.oracle.agile_reconciliation import (
    AgileReconciliationError,
    run_agile_reconciliation,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/bio/m0_07_agile_label_reconciliation.json",
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
        result = run_agile_reconciliation(args.config, args.output_dir, args.repo_root)
    except AgileReconciliationError as exc:
        print(f"M0-07 AGILE reconciliation failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": str(args.output_dir),
                "summary": result["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
