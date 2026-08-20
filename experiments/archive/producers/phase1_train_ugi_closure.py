#!/usr/bin/env python3
"""Train the sparse feasible Ugi closure-edge scorer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.phase1.product_l1.training.ugi_closure_training import (
    UgiClosureTrainingError,
    train_ugi_closure_scorer,
)

REPO = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_closure.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or (
        REPO / "results/phase1" / ("ugi_closure_smoke" if args.smoke else "ugi_closure_full")
    )
    try:
        result = train_ugi_closure_scorer(
            args.config.resolve(),
            REPO,
            output_dir.resolve(),
            smoke=args.smoke,
            overwrite=args.overwrite,
        )
    except (KeyError, TypeError, UgiClosureTrainingError) as exc:
        print(f"Ugi closure training failed: {exc}", file=sys.stderr)
        return 1
    final = result["evaluations"][-1]
    final_summary = {"step": final["step"]}
    for split in ("train", "calibration_all", "calibration_novel"):
        metrics = final[split]
        final_summary[split] = {
            key: metrics[key] for key in ("components", "mean_nll", "exact_set_fraction")
        }
    payload = {
        "status": result["status"],
        "mode": result["mode"],
        "corpus": result["corpus"],
        "checkpoint": result["checkpoint"],
        "selection": result.get("selection"),
        "heldout_selected": result.get("heldout_selected"),
        "last_evaluation": final_summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
