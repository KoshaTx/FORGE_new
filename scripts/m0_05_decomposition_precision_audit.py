#!/usr/bin/env python3
"""Build the blinded M0-05 decomposition-precision review packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.data.decomposition_precision_audit import run_audit
from forge.data.r1_prime_audit import AuditError

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/corpus/m0_05_decomposition_precision_audit.json",
        help="hash-pinned M0-05 packet configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/m0_05",
        help="new output directory; existing results are never overwritten",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO,
        help="repository root against which config input paths are resolved",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_audit(
            args.config,
            args.output_dir,
            args.repo_root,
        )
    except AuditError as exc:
        print(f"M0-05 packet generation failed: {exc}", file=sys.stderr)
        return 1
    summary = {
        "output_dir": str(args.output_dir),
        "status": result["status"],
        "packet_rows": result["packet_summary"]["rows"],
        "frames": result["packet_summary"]["frames"],
        "human_review": result["human_review"]["status"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
