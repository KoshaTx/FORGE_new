#!/usr/bin/env python3
"""Materialize the non-executing HeLa potency-guidance readiness receipt."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from experiments.phase1.hela_potency.readiness import (
    build_hela_potency_diagnostic_readiness,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/phase1_ugi_hela_potency_guidance_readiness_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_hela_potency_guidance_readiness_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    repo = args.repo.resolve()
    config = args.config if args.config.is_absolute() else repo / args.config
    output = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")
    result = build_hela_potency_diagnostic_readiness(repo, config)
    payload = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        (temporary_path / "readiness.json").write_bytes(payload)
        os.replace(temporary_path, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
