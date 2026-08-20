#!/usr/bin/env python3
"""Train the bounded Ugi topology-conditioned chemistry flow."""

from __future__ import annotations

import argparse
from pathlib import Path

from forge.design.training.ugi_chemistry_training import train_ugi_chemistry

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_chemistry.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi_chemistry_smoke",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = train_ugi_chemistry(
        args.config,
        REPO,
        args.output_dir,
        smoke=args.smoke,
        overwrite=args.overwrite,
    )
    print(args.output_dir / "result.json")
    print(result["checkpoint"]["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
