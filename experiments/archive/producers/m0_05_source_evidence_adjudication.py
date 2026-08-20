#!/usr/bin/env python3
"""Build the automated, source-grounded M0-05 chemistry admission gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.corpus.source_evidence_adjudication import (
    SourceEvidenceAdjudicationError,
    run_source_evidence_adjudication,
)

REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/corpus/m0_05_source_evidence_adjudication.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/m0_05_source_adjudication",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_source_evidence_adjudication(
            args.config,
            args.output_dir,
            args.repo_root,
        )
    except SourceEvidenceAdjudicationError as exc:
        print(f"M0-05 source adjudication failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": str(args.output_dir),
                "summary": result["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
