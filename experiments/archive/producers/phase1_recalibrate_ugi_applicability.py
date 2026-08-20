#!/usr/bin/env python3
"""Run the nested, nonselecting Ugi applicability recalibration."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.potency.applicability.ugi_applicability_recalibration import build_applicability_recalibration

REPO = Path(__file__).resolve().parents[3]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bio/phase1_ugi_applicability_recalibration_v2.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_applicability_recalibration_v2"),
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")
    result, ledger = build_applicability_recalibration(REPO, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        (temporary_path / "nested_authority_ledger.csv.gz").write_bytes(ledger)
        (temporary_path / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary_path, output)
    print(output / "result.json")


if __name__ == "__main__":
    main()
