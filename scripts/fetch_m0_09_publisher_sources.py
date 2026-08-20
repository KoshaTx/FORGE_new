#!/usr/bin/env python3
"""Acquire publisher-hosted supplementary assets for M0-09 route review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.route.sources.publisher_sources import (
    PublisherSourceError,
    acquire_publisher_sources,
    write_publisher_acquisition_result,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/m0_09_publisher_sources.json",
    )
    parser.add_argument(
        "--source-queue",
        type=Path,
        default=REPO / "results/m0_09/source_queue.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO / "data/source_cache/m0_09_publishers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/m0_09/publisher_source_acquisition.json",
    )
    parser.add_argument(
        "--generated-utc",
        help="optional timezone-aware ISO-8601 timestamp for reproducible reruns",
    )
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = acquire_publisher_sources(
            args.config,
            args.source_queue,
            args.cache_dir,
            generated_utc=args.generated_utc,
            workers=args.workers,
        )
        write_publisher_acquisition_result(result, args.output)
    except PublisherSourceError as exc:
        print(f"M0-09 publisher-source acquisition failed: {exc}", file=sys.stderr)
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
    return 0 if result["summary"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
