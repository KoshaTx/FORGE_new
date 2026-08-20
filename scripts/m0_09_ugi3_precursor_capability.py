#!/usr/bin/env python3
"""Build the bounded M0-09 Ugi-3 precursor capability audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.route.evidence.ugi3_capability import (
    Ugi3CapabilityError,
    build_ugi3_precursor_capability,
    write_ugi3_precursor_capability,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/m0_09_ugi3_precursor_capability.json",
    )
    parser.add_argument(
        "--building-block-pool",
        type=Path,
        default=REPO / "data/vendor/building_block_pool_v1.json",
    )
    parser.add_argument(
        "--agile-component-routes",
        type=Path,
        default=REPO / "results/m0_09/agile_component_routes.json",
    )
    parser.add_argument(
        "--component-ledger",
        type=Path,
        default=REPO / "results/m0_09/component_source_ledger.csv",
    )
    parser.add_argument(
        "--paper-route-reviews",
        type=Path,
        default=REPO / "results/m0_09/lnpdb_paper_route_reviews.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/m0_09/ugi3_precursor_capability.json",
    )
    parser.add_argument(
        "--generated-utc",
        help="optional timezone-aware ISO-8601 timestamp for reproducible reruns",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_ugi3_precursor_capability(
            args.config,
            args.building_block_pool,
            args.agile_component_routes,
            args.component_ledger,
            args.paper_route_reviews,
            generated_utc=args.generated_utc,
        )
        write_ugi3_precursor_capability(result, args.output)
    except Ugi3CapabilityError as exc:
        print(f"M0-09 Ugi-3 precursor capability audit failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), **result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
