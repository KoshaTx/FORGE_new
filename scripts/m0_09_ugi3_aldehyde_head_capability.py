#!/usr/bin/env python3
"""Build the bounded M0-09 Ugi-3 aldehyde and amine-head capability audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.route.evidence.ugi3_aldehyde_head_capability import (
    Ugi3AldehydeHeadCapabilityError,
    build_ugi3_aldehyde_head_capability,
    write_ugi3_aldehyde_head_capability,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/m0_09_ugi3_aldehyde_head_capability.json",
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
        "--qualified-reactions",
        type=Path,
        default=REPO / "data/vendor/qualified_reactions_v1.json",
    )
    parser.add_argument(
        "--precursor-capability",
        type=Path,
        default=REPO / "results/m0_09/ugi3_precursor_capability.json",
    )
    parser.add_argument(
        "--agile-head-procurement",
        type=Path,
        default=REPO / "configs/route/m0_09_ugi3_agile_head_procurement.json",
    )
    parser.add_argument(
        "--assembly-qualification",
        type=Path,
        default=REPO / "results/m0_09/ugi3_assembly_qualification.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/m0_09/ugi3_aldehyde_head_capability.json",
    )
    parser.add_argument(
        "--generated-utc",
        help="optional timezone-aware ISO-8601 timestamp for reproducible reruns",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_ugi3_aldehyde_head_capability(
            args.config,
            args.building_block_pool,
            args.agile_component_routes,
            args.qualified_reactions,
            args.precursor_capability,
            args.agile_head_procurement,
            args.assembly_qualification,
            generated_utc=args.generated_utc,
        )
        write_ugi3_aldehyde_head_capability(result, args.output)
    except Ugi3AldehydeHeadCapabilityError as exc:
        print(f"M0-09 Ugi-3 aldehyde/head capability audit failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), **result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
