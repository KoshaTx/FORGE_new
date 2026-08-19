#!/usr/bin/env python3
"""Validate and materialize the blocked nonzero-guidance runner plan."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.product.ugi_nonzero_guidance_runner import build_nonzero_guidance_runner_plan


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/phase1_ugi_nonzero_guidance_runner_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_nonzero_guidance_runner_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    repo = args.repo.resolve()
    config = args.config if args.config.is_absolute() else repo / args.config
    output = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = build_nonzero_guidance_runner_plan(repo, config)
    payload = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        (temporary_path / "plan.json").write_bytes(payload)
        os.replace(temporary_path, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
