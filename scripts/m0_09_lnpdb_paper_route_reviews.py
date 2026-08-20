#!/usr/bin/env python3
"""Validate and aggregate manually reviewed LNPDB subcomponent routes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.route.audit.paper_reviews import (
    PaperReviewError,
    build_paper_route_reviews,
    write_paper_route_reviews,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/m0_09_lnpdb_paper_reviews.json",
    )
    parser.add_argument(
        "--source-review-index",
        type=Path,
        default=REPO / "results/m0_09/source_review_index.json",
    )
    parser.add_argument(
        "--component-ledger",
        type=Path,
        default=REPO / "results/m0_09/component_source_ledger.csv",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPO / "data/source_cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/m0_09/lnpdb_paper_route_reviews.json",
    )
    parser.add_argument(
        "--generated-utc",
        help="optional timezone-aware ISO-8601 timestamp for reproducible reruns",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_paper_route_reviews(
            args.config,
            args.source_review_index,
            args.component_ledger,
            args.cache_root,
            generated_utc=args.generated_utc,
        )
        write_paper_route_reviews(result, args.output)
    except PaperReviewError as exc:
        print(f"M0-09 LNPDB paper route review failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), **result["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
