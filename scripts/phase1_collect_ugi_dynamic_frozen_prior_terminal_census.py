#!/usr/bin/env python3
"""Plan, execute one shard, or aggregate the frozen-prior terminal census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.design.flow.ugi_dynamic_frozen_prior_terminal_census import (
    aggregate_completed_census,
    census_plan,
    run_census_shard,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/model/phase1_ugi_dynamic_frozen_prior_terminal_census_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibration-only native terminal census. The CLI intentionally executes at most "
            "one shard per invocation; it has no full-run shortcut."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--shard-index", type=int)
    action.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    if args.plan:
        result = census_plan(REPO, args.config)
    elif args.aggregate:
        result = aggregate_completed_census(REPO, args.config, args.output_dir)
    else:
        result = run_census_shard(
            REPO,
            args.config,
            args.output_dir,
            shard_index=args.shard_index,
        )
    summary = {
        "status": result["status"],
        "result_sha256": result.get("result_sha256"),
        "shard_index": result.get("shard_index"),
        "counts": result.get("counts"),
        "design": result.get("design"),
        "resumed_completed_shard": result.get("resumed_completed_shard"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
