#!/usr/bin/env python3
"""Build the bounded M0-09 LNPDB head-transfer candidate census."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.route.lnpdb_head_transfer import (
    LnpdbHeadTransferError,
    build_lnpdb_head_transfer,
    write_lnpdb_head_transfer,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/m0_09_lnpdb_head_transfer.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/m0_09",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result, ledger = build_lnpdb_head_transfer(args.config, REPO)
        write_lnpdb_head_transfer(result, ledger, args.output_dir)
    except LnpdbHeadTransferError as exc:
        print(f"M0-09 LNPDB head-transfer census failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "head_rows": result["summary"]["head_rows"],
                "disposition_counts": result["summary"]["disposition_counts"],
                "transfer_candidate_ring_size_counts": result["summary"][
                    "transfer_candidate_ring_size_counts"
                ],
                "automatically_admitted_candidates": result["summary"][
                    "automatically_admitted_candidates"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
