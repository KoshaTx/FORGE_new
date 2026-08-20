#!/usr/bin/env python3
"""Plan, execute or aggregate the matched production candidate arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from forge.design.sampling.ugi_production_candidate_generation import (
    aggregate_production_candidate_generation,
    production_candidate_plan,
    run_all_production_candidate_shards,
    run_one_production_candidate_shard,
)

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/phase1_ugi_production_candidate_generation_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_production_candidate_generation_v1"),
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--plan", action="store_true")
    actions.add_argument("--run-all", action="store_true")
    actions.add_argument("--run-shard", type=int)
    actions.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    if args.plan:
        result: Any = production_candidate_plan(REPO, config)
    elif args.run_all:
        receipts = run_all_production_candidate_shards(REPO, config, output)
        result = {
            "status": "production_candidate_shards_complete",
            "shards": len(receipts),
            "resumed": sum(bool(receipt["resumed"]) for receipt in receipts),
        }
    elif args.run_shard is not None:
        result = run_one_production_candidate_shard(
            REPO, config, output, shard_index=args.run_shard
        )
    else:
        result = aggregate_production_candidate_generation(REPO, config, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
