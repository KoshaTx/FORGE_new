#!/usr/bin/env python3
"""Freeze the Ugi morphology proposal-strength development sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.potency.ugi_morphology_proposal_strength_sweep import (
    build_morphology_proposal_strength_sweep,
)

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bio/phase1_ugi_morphology_proposal_strength_sweep_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase1/ugi_morphology_proposal_strength_sweep_v1"),
    )
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output if args.output.is_absolute() else REPO / args.output
    result, ledger = build_morphology_proposal_strength_sweep(REPO, config)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "sweep_ledger.csv.gz").write_bytes(ledger)
    print(json.dumps(result["development_selected_challenger"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
