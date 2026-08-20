#!/usr/bin/env python3
"""Run the non-circular M0-04 R1-prime anchoring audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from forge.corpus.r1_prime_audit import AuditError, run_audit

REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/corpus/m0_04_r1_prime_audit_constitutional.json",
        help="hash-pinned M0-04 audit configuration",
    )
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=REPO / "data/vendor",
        help="directory containing hash-pinned source assets",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=REPO / "data/splits/m0_03_constitutional",
        help="frozen constitutional M0-03 split bundle",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/m0_04_constitutional",
        help="new output directory; existing results are never overwritten",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="total CPU worker budget for scheme processes and fingerprint-query threads",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO / "checkpoints/m0_04_constitutional",
        help="ignored directory for resumable completed-scheme checkpoints",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_audit(
            args.config,
            args.vendor_dir,
            args.split_dir,
            args.output_dir,
            args.workers,
            args.checkpoint_dir,
        )
    except AuditError as exc:
        print(f"M0-04 audit failed: {exc}", file=sys.stderr)
        return 1

    summary = {
        "output_dir": str(args.output_dir),
        "historical_control": result["control"]["historical_existing_r1_non_agile_full_r0"],
        "current_control": result["control"]["current_existing_r1_non_agile_full_r0"],
        "schemes": {
            scheme: {
                "r1_prime_products": details["r1_prime_enumeration"]["globally_unique_products"],
                "all_heldout_recovery": details["r1_prime_exact_recovery"]["all_heldout"],
                "non_agile_heldout_recovery": details["r1_prime_exact_recovery"][
                    "non_agile_heldout"
                ],
            }
            for scheme, details in result["schemes"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
