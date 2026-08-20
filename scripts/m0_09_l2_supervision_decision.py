#!/usr/bin/env python3
"""Freeze the M0-09 L2 supervision and architecture decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.route.sources.l2_supervision_decision import (
    L2SupervisionDecisionError,
    build_l2_supervision_decision,
    write_l2_supervision_decision,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/m0_09_l2_supervision_decision.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/m0_09",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_l2_supervision_decision(args.config, REPO)
        output = write_l2_supervision_decision(result, args.output_dir)
    except L2SupervisionDecisionError as exc:
        print(f"M0-09 L2 supervision decision failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": str(output),
                **result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
