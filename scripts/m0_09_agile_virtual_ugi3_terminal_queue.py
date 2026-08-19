#!/usr/bin/env python3
"""Build the unresolved terminal queue for AGILE virtual Ugi routes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.route.ugi3_virtual_terminal_queue import (
    Ugi3VirtualTerminalQueueError,
    build_ugi3_virtual_terminal_queue,
    write_ugi3_virtual_terminal_queue,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO
        / "configs/route/m0_09_agile_virtual_ugi3_terminal_queue.json",
    )
    parser.add_argument(
        "--component-ledger",
        type=Path,
        default=REPO
        / "results/m0_09/agile_virtual_ugi3_component_ledger.csv.gz",
    )
    parser.add_argument(
        "--component-program-ledger",
        type=Path,
        default=REPO
        / "results/m0_09/agile_virtual_ugi3_component_program_ledger.csv.gz",
    )
    parser.add_argument(
        "--agile-component-routes",
        type=Path,
        default=REPO / "results/m0_09/agile_component_routes.json",
    )
    parser.add_argument(
        "--terminal-procurement",
        type=Path,
        default=REPO
        / "configs/route/m0_09_ugi3_virtual_terminal_procurement.json",
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
        result, ledger = build_ugi3_virtual_terminal_queue(
            args.config,
            args.component_ledger,
            args.component_program_ledger,
            args.agile_component_routes,
            args.terminal_procurement,
        )
        write_ugi3_virtual_terminal_queue(
            result,
            ledger,
            args.output_dir,
        )
    except Ugi3VirtualTerminalQueueError as exc:
        print(
            f"M0-09 AGILE virtual terminal queue failed: {exc}",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "output": str(
                    args.output_dir
                    / "agile_virtual_ugi3_terminal_queue.json"
                ),
                **result["summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
