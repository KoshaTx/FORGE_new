#!/usr/bin/env python3
"""Plan, run or aggregate fresh unused-program morphology confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.phase1.product_l1.sampling.ugi_morphology_proposal_confirmation import (
    aggregate_confirmation,
    confirmation_plan,
    run_all_confirmation_shards,
)

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/phase1_ugi_morphology_proposal_confirmation_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_morphology_proposal_confirmation_v1"),
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--plan", action="store_true")
    actions.add_argument("--run-all", action="store_true")
    actions.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    if args.plan:
        result: Any = confirmation_plan(REPO, config)
    elif args.run_all:
        receipts = run_all_confirmation_shards(REPO, config, output)
        result = {
            "status": "confirmation_shards_complete",
            "shards": len(receipts),
            "resumed": sum(bool(receipt["resumed"]) for receipt in receipts),
        }
    else:
        result = aggregate_confirmation(REPO, config, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
