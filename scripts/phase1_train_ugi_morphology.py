#!/usr/bin/env python3
"""Train the Ugi-first core-anchored Phase 1 morphology flow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.product.ugi_morphology_training import (
    UgiMorphologyTrainingError,
    train_ugi_morphology,
)

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_morphology.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or (
        REPO / "results/phase1" / ("ugi_morphology_smoke" if args.smoke else "ugi_morphology_full")
    )
    try:
        result = train_ugi_morphology(
            args.config.resolve(),
            REPO,
            output_dir.resolve(),
            smoke=args.smoke,
            overwrite=args.overwrite,
        )
    except (KeyError, TypeError, UgiMorphologyTrainingError) as exc:
        print(f"Ugi morphology training failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "mode": result["mode"],
                "optimization": result["optimization"],
                "checkpoint": result["checkpoint"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
