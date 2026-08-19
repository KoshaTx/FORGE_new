#!/usr/bin/env python3
"""Materialize the read-only generated-role applicability census."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.potency.ugi_generated_role_applicability_census import (
    build_generated_role_applicability_census,
)

REPO = Path(__file__).resolve().parents[1]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/model/phase1_ugi_generated_role_applicability_census_readiness_v1.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_generated_role_applicability_census_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")
    result, ledger = build_generated_role_applicability_census(REPO, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        (temporary_path / "ledger.csv.gz").write_bytes(ledger)
        (temporary_path / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary_path, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
