#!/usr/bin/env python3
"""Reconcile AGILE-derived provenance and freeze constitutional R0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.corpus.r0_reconciliation import (
    R0ReconciliationError,
    build_r0_reconciliation,
    write_r0_reconciliation,
)

REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/corpus/m0_03_r0_reconciliation.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/m0_03",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result, corpus, ledger = build_r0_reconciliation(args.config, REPO)
        write_r0_reconciliation(result, corpus, ledger, args.output_dir)
    except R0ReconciliationError as exc:
        print(f"M0-03 R0 reconciliation failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                **result["summary"],
                "artifacts": result["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
