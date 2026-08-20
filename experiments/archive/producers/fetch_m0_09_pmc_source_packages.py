#!/usr/bin/env python3
"""Acquire and hash machine-readable sources for the 26 PMC-linked LNPDB papers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.synthesis.sources.pmc_sources import (
    PmcSourceError,
    acquire_pmc_queue,
    write_acquisition_result,
)

REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-queue",
        type=Path,
        default=REPO / "results/m0_09/source_queue.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO / "data/source_cache/m0_09_pmc",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/m0_09/pmc_source_acquisition.json",
    )
    parser.add_argument(
        "--generated-utc",
        help="optional timezone-aware ISO-8601 timestamp for reproducible reruns",
    )
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = acquire_pmc_queue(
            args.source_queue,
            args.cache_dir,
            generated_utc=args.generated_utc,
            workers=args.workers,
        )
        write_acquisition_result(result, args.output)
    except PmcSourceError as exc:
        print(f"M0-09 PMC source acquisition failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output": str(args.output),
                **result["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
