#!/usr/bin/env python3
"""Render the nonselecting supported-interpolative HeLa preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.bio.ugi_hela_diagnostic_preview import build_preview, write_preview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bio/phase1_ugi_hela_supported_diagnostic_preview_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_hela_supported_diagnostic_preview_v1"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = args.config if args.config.is_absolute() else repo / args.config
    output = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    result, artifacts = build_preview(repo, config)
    write_preview(output, artifacts)
    print(json.dumps({"output_dir": str(output), "result": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
