#!/usr/bin/env python3
"""Materialize the nonselecting Ugi dynamic-controller analysis."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.potency.controller import build_dynamic_controller_analysis

REPO = Path(__file__).resolve().parents[3]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bio/phase1_ugi_dynamic_controller_analysis_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_dynamic_controller_analysis_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")
    result, terminal_ledger, oof_ledger = build_dynamic_controller_analysis(REPO, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        (temporary_path / "terminal_support.csv.gz").write_bytes(terminal_ledger)
        (temporary_path / "oof_predictions.csv.gz").write_bytes(oof_ledger)
        (temporary_path / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary_path, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
