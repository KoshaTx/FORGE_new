#!/usr/bin/env python3
"""Build the hash-verified M0-09 L2 supervision inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.route.supervision_inventory import InventoryError, build_inventory, write_inventory

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPO / "configs/route/m0_09_supervision_sources.json",
        help="curated source ledger",
    )
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=REPO / "data/vendor",
        help="directory containing hash-pinned source assets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/m0_09/result.json",
        help="result JSON path",
    )
    parser.add_argument(
        "--generated-utc",
        help="optional timezone-aware ISO-8601 timestamp for reproducible reruns",
    )
    parser.add_argument("--seed", type=int, default=0, help="recorded seed; no randomness is used")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_inventory(
            args.ledger,
            args.vendor_dir,
            generated_utc=args.generated_utc,
            seed=args.seed,
        )
        write_inventory(result, args.output)
    except InventoryError as exc:
        print(f"M0-09 inventory failed: {exc}", file=sys.stderr)
        return 1

    summary = {
        "output": str(args.output),
        "uspto_large_corpus_reactions": result["inventory"]["uspto_pretraining"]["large_corpus"][
            "reaction_records"
        ],
        "uspto_benchmark_reactions": result["inventory"]["uspto_pretraining"][
            "class_labeled_benchmark"
        ]["reaction_records"],
        "lipid_specific_reactions": result["lipid_specific_combined"][
            "distinct_upstream_reaction_instances"
        ],
        "terminal_precursor_scaffolds": result["lipid_specific_combined"][
            "unique_terminal_precursor_scaffolds"
        ],
        "explicit_failed_syntheses": result["lipid_specific_combined"]["explicit_failed_syntheses"],
        "recommended_architecture": result["viability_decision"]["recommended_architecture"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
