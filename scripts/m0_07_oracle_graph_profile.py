#!/usr/bin/env python3
"""Run the M0-07 train-only sparse graph runtime gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.potency.oracle_graph_profile import (
    OracleGraphProfileError,
    run_oracle_graph_profile,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/bio/m0_07_oracle_graph_profile.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/m0_07/oracle_graph_profile_result.json",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_oracle_graph_profile(
            args.config,
            args.output,
            args.repo_root,
        )
    except OracleGraphProfileError as exc:
        print(f"M0-07 graph profile failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "partition": result["partition"],
                "architectures": [
                    {
                        "architecture": profile["architecture"],
                        "parameters": profile["parameters"],
                        "median_epoch_wall_seconds": profile["epoch_wall_seconds"]["median"],
                        "median_examples_per_second": profile["examples_per_second"]["median"],
                    }
                    for profile in result["architecture_profiles"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
