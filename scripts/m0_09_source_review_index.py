#!/usr/bin/env python3
"""Build the hash-verified M0-09 source review index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.route.audit.source_review import (
    SourceReviewError,
    build_source_review_index,
    write_source_review_index,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-queue",
        type=Path,
        default=REPO / "results/m0_09/source_queue.csv",
    )
    parser.add_argument(
        "--pmc-acquisition",
        type=Path,
        default=REPO / "results/m0_09/pmc_source_acquisition.json",
    )
    parser.add_argument(
        "--component-ledger",
        type=Path,
        default=REPO / "results/m0_09/component_source_ledger.csv",
    )
    parser.add_argument(
        "--publisher-acquisition",
        type=Path,
        default=REPO / "results/m0_09/publisher_source_acquisition.json",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPO / "data/source_cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/m0_09/source_review_index.json",
    )
    parser.add_argument(
        "--generated-utc",
        help="optional timezone-aware ISO-8601 timestamp for reproducible reruns",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_source_review_index(
            args.source_queue,
            args.component_ledger,
            args.pmc_acquisition,
            args.publisher_acquisition,
            args.cache_root,
            generated_utc=args.generated_utc,
        )
        write_source_review_index(result, args.output)
    except SourceReviewError as exc:
        print(f"M0-09 source review index failed: {exc}", file=sys.stderr)
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
