#!/usr/bin/env python3
"""Score two completed independent M0-05 chemist-review files."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from forge.corpus.decomposition_precision_review import score_reviews
from forge.corpus.r1_prime_audit import AuditError

REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reviewer_annotations",
        type=Path,
        nargs=2,
        help="two completed copies of the blinded review_packet.csv",
    )
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=REPO / "results/m0_05/answer_key.csv.gz",
        help="separate frozen M0-05 answer key",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/m0_05/human_review_scoring.json",
        help="new scoring output; existing results are never overwritten",
    )
    parser.add_argument(
        "--generated-utc",
        default=None,
        help="explicit ISO timestamp for reproducible tests; defaults to current UTC",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generated_utc = args.generated_utc or datetime.now(UTC).isoformat()
    try:
        result = score_reviews(
            args.answer_key,
            tuple(args.reviewer_annotations),
            args.output,
            generated_utc,
        )
    except AuditError as exc:
        print(f"M0-05 review scoring failed: {exc}", file=sys.stderr)
        return 1
    summary = {
        "output": str(args.output),
        "status": result["status"],
        "review_rows": result["review_rows"],
        "reviewers": sorted(result["per_reviewer"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
