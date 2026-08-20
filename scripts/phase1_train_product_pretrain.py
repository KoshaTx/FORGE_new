#!/usr/bin/env python3
"""Run Phase 1 sparse whole-lipid product pretraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.design.flow.phase1_flow import Phase1FlowError, train_product_pretrain

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_product_pretrain.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overfit-only", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = json.loads(args.config.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"Phase 1 product pretraining failed: invalid config {args.config}: {exc}",
            file=sys.stderr,
        )
        return 1
    output_dir = args.output_dir
    try:
        if output_dir is None:
            output_dir = (
                REPO / "results/phase1/product_pretrain_smoke"
                if args.smoke
                else REPO / config["outputs"]["default_full_directory"]
            )
        result = train_product_pretrain(
            args.config,
            REPO,
            output_dir,
            smoke=args.smoke,
            overfit_only=args.overfit_only,
            resume_checkpoint=args.resume_checkpoint,
            overwrite=args.overwrite,
        )
    except (KeyError, TypeError, Phase1FlowError) as exc:
        print(f"Phase 1 product pretraining failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "mode": result["mode"],
                "overfit_gate": result["overfit_gate"],
                "training": result.get("training"),
                "checkpoint": result.get("checkpoint"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
