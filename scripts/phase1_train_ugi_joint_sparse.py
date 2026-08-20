#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from forge.design.training.ugi_joint_sparse_training import train_ugi_joint_sparse

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_joint_sparse.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = train_ugi_joint_sparse(
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

