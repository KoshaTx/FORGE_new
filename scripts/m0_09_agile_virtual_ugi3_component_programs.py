#!/usr/bin/env python3
"""Build recursive component programs for the AGILE virtual Ugi library."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.route.ugi3_virtual_programs import (
    Ugi3VirtualProgramError,
    build_ugi3_virtual_component_programs,
    write_ugi3_virtual_component_programs,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO
        / "configs/route/m0_09_agile_virtual_ugi3_component_programs.json",
    )
    parser.add_argument(
        "--component-ledger",
        type=Path,
        default=REPO
        / "results/m0_09/agile_virtual_ugi3_component_ledger.csv.gz",
    )
    parser.add_argument(
        "--virtual-capability",
        type=Path,
        default=REPO
        / "results/m0_09/agile_virtual_ugi3_capability.json",
    )
    parser.add_argument(
        "--agile-component-routes",
        type=Path,
        default=REPO / "results/m0_09/agile_component_routes.json",
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
        result, ledger = build_ugi3_virtual_component_programs(
            args.config,
            args.component_ledger,
            args.virtual_capability,
            args.agile_component_routes,
        )
        write_ugi3_virtual_component_programs(
            result,
            ledger,
            args.output_dir,
        )
    except Ugi3VirtualProgramError as exc:
        print(
            f"M0-09 AGILE virtual component programs failed: {exc}",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "output": str(
                    args.output_dir
                    / "agile_virtual_ugi3_component_programs.json"
                ),
                **result["summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
