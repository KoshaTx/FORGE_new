#!/usr/bin/env python3
"""Build exact AGILE component routes and project them across every LNPDB lipid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.route.assessment.route_awareness import (
    RouteAwarenessError,
    build_route_awareness,
    write_route_awareness,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/m0_09_agile_component_routes.json",
    )
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=REPO / "data/vendor",
    )
    parser.add_argument(
        "--m0-results-dir",
        type=Path,
        default=REPO / "results/m0_09",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/m0_09",
    )
    parser.add_argument(
        "--generated-utc",
        help="optional timezone-aware ISO-8601 timestamp for reproducible reruns",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result, route_artifact, lipid_rows = build_route_awareness(
            args.config,
            args.vendor_dir,
            args.m0_results_dir,
            generated_utc=args.generated_utc,
            seed=args.seed,
        )
        result = write_route_awareness(result, route_artifact, lipid_rows, args.output_dir)
    except RouteAwarenessError as exc:
        print(f"M0-09 route-awareness map failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output": str(args.output_dir / "route_awareness_result.json"),
                "lnpdb_lipids": result["summary"]["lnpdb"]["unique_lipids"],
                "route_records": result["summary"]["route_records"],
                "execution_closed_lipids": result["summary"]["lnpdb"]["execution_closed_lipids"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
