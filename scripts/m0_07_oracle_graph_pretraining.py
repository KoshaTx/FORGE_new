#!/usr/bin/env python3
"""Profile or explicitly run label-free R0 graph-encoder pretraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.potency.oracle.oracle_graph_pretraining import (
    OracleGraphPretrainingError,
    run_graph_pretraining,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/bio/m0_07_oracle_graph_pretraining.json",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/m0_07")
    parser.add_argument(
        "--full-training",
        action="store_true",
        help="run the full configured CPU pretraining; default is a bounded one-epoch profile",
    )
    parser.add_argument(
        "--profile-maximum-graphs",
        type=int,
        default=None,
        help="override both bounded profile partition limits",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.profile_maximum_graphs is not None and args.profile_maximum_graphs < 1:
        print("--profile-maximum-graphs must be positive", file=sys.stderr)
        return 2
    try:
        result = run_graph_pretraining(
            args.config,
            args.output_dir,
            REPO,
            profile_only=not args.full_training,
            profile_maximum_graphs=args.profile_maximum_graphs,
        )
    except OracleGraphPretrainingError as exc:
        print(f"M0-07 graph pretraining failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "mode": result["mode"],
                "trained_partition_counts": result["trained_partition_counts"],
                "best_epoch": result["best_epoch"],
                "timing_seconds": result["timing_seconds"],
                "checkpoint": result["checkpoint"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
