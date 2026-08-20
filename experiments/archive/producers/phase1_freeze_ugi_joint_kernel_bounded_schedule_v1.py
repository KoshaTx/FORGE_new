#!/usr/bin/env python3
"""Materialize the nonexecuting joint-kernel bounded schedule."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from experiments.phase1.synthesis_guidance.schedule.ugi_joint_kernel_bounded_schedule import (
    build_joint_kernel_bounded_schedule,
)

REPO = Path(__file__).resolve().parents[3]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/phase1_ugi_joint_kernel_bounded_schedule_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_joint_kernel_bounded_schedule_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")
    result = build_joint_kernel_bounded_schedule(REPO, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        (temporary_path / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary_path, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
