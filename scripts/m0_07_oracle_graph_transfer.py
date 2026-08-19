#!/usr/bin/env python3
"""Run label-free R0 encoder transfer across the frozen AGILE split matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.potency.oracle_graph_transfer import (
    OracleGraphTransferError,
    aggregate_transfer_fits,
    run_oracle_graph_transfer,
    run_transfer_fit_workers,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/m0_07")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--fits-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.aggregate_only and args.fits_only:
        print("--aggregate-only and --fits-only are mutually exclusive", file=sys.stderr)
        return 2
    try:
        if args.aggregate_only:
            result = aggregate_transfer_fits(
                args.config,
                args.output_dir,
                args.repo_root,
            )
        elif args.fits_only:
            result = {
                "status": "completed_transfer_fit_workers",
                "worker_summary": run_transfer_fit_workers(
                    args.config,
                    args.repo_root,
                ),
            }
        else:
            result = run_oracle_graph_transfer(
                args.config,
                args.output_dir,
                args.repo_root,
            )
    except (OracleGraphTransferError, RuntimeError, KeyError) as exc:
        print(f"M0-07 graph transfer failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "worker_summary": result.get("worker_summary"),
                "combined_transfer_leader": result.get("summary", {}).get(
                    "combined_transfer_leader"
                ),
                "oracle_model_frozen": result.get("decision", {}).get("oracle_model_frozen"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
