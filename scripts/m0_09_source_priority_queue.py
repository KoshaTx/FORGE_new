#!/usr/bin/env python3
"""Build the M0-09 evidence-bounded source-paper priority overlay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.route.source_priority import (
    SourcePriorityError,
    build_source_priority,
    write_source_priority,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/m0_09_source_priority.json",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/m0_09")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result, rows = build_source_priority(args.config, args.repo_root)
        write_source_priority(result, rows, args.output_dir)
    except SourcePriorityError as exc:
        print(f"M0-09 source-priority queue failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": str(args.output_dir / "source_priority_result.json"),
                "sources": result["summary"]["source_records"],
                "active_next_reviews": result["summary"]["active_next_review_sources"],
                "sources_with_missing_axes": result["summary"][
                    "sources_with_missing_priority_axes"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
