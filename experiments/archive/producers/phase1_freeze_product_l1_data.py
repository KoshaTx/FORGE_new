#!/usr/bin/env python3
"""Freeze and verify the Phase 1 product plus exact Ugi-L1 data contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.corpus.phase1_data import Phase1DataError, freeze_phase1_data_contract

REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_product_l1_data.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = freeze_phase1_data_contract(args.config, REPO)
    except Phase1DataError as exc:
        print(f"Phase 1 data contract failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
