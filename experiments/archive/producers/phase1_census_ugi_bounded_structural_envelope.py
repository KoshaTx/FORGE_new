#!/usr/bin/env python3
"""Apply the frozen bounded envelope to the existing exact-L1 terminal pool."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.potency.applicability.ugi_bounded_structural_envelope import (
    build_bounded_structural_envelope_census,
)

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/phase1_ugi_bounded_structural_envelope_census_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_bounded_structural_envelope_census_v1"),
    )
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")
    result, ledger = build_bounded_structural_envelope_census(REPO, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        (temporary_path / "ledger.csv.gz").write_bytes(ledger)
        (temporary_path / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary_path, output)


if __name__ == "__main__":
    main()
