#!/usr/bin/env python3
"""Freeze the hash-pinned M0-08 endpoint decision package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.archive.m0.endpoint_decision import EndpointDecisionError, freeze_endpoint_decision

REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/bio/m0_08_endpoint_decision.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/m0_08/result.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = freeze_endpoint_decision(args.config, args.output, REPO)
    except EndpointDecisionError as exc:
        print(f"M0-08 endpoint freeze failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "leading_option": result["decision"]["current_lowest_risk_option"],
                "endpoint_locked": result["decision"]["endpoint_locked"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
