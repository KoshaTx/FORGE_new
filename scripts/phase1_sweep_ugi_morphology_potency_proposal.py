#!/usr/bin/env python3
"""Sweep a nested morphology-level potency proposal under frozen safeguards."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.bio.ugi_morphology_potency_proposal_sweep import (
    build_morphology_potency_proposal_sweep,
)

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bio/phase1_ugi_morphology_potency_proposal_sweep_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_morphology_potency_proposal_sweep_v1"),
    )
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")
    result, ledger = build_morphology_potency_proposal_sweep(REPO, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        path = Path(temporary)
        (path / "proposal_ledger.csv.gz").write_bytes(ledger)
        (path / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(path, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
