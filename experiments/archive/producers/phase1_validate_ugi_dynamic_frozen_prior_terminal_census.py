#!/usr/bin/env python3
"""Validate a completed frozen-prior terminal census without mutating it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.archive.phase1.design_audits.ugi_dynamic_frozen_prior_terminal_census_validator import (
    validate_completed_census_from_config,
    write_validation_receipt,
)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPO / "configs/model/phase1_ugi_dynamic_frozen_prior_terminal_census_validator_v1.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only post-aggregation validation of the frozen dynamic terminal census. "
            "This command cannot launch, resume, or aggregate census shards."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--census-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    result = validate_completed_census_from_config(REPO, args.config, args.census_dir)
    write_validation_receipt(args.receipt, result, census_dir=args.census_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "result_sha256": result["result_sha256"],
                "aggregate_result_sha256": result["aggregate_result_sha256"],
                "counts": result["counts"],
                "receipt": str(args.receipt.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
