#!/usr/bin/env python3
"""Run the leakage-safe M0-07 supervised graph oracle matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.bio.oracle_graph_matrix import (
    OracleGraphMatrixError,
    aggregate_graph_fits,
    run_graph_fit_workers,
    run_oracle_graph_matrix,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/bio/m0_07_oracle_graph_matrix.json",
    )
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
            result = aggregate_graph_fits(
                args.config,
                args.output_dir,
                args.repo_root,
            )
        elif args.fits_only:
            result = {"worker_summary": run_graph_fit_workers(args.config, args.repo_root)}
        else:
            result = run_oracle_graph_matrix(
                args.config,
                args.output_dir,
                args.repo_root,
            )
    except OracleGraphMatrixError as exc:
        print(f"M0-07 graph matrix failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result.get("status", "completed_fit_workers"),
                "worker_summary": result.get("worker_summary"),
                "summary": result.get("summary"),
                "combined_architecture_leader": result.get("combined_architecture_leader"),
                "oracle_model_frozen": result.get("decision", {}).get("oracle_model_frozen"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
