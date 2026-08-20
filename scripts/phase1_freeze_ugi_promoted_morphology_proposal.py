#!/usr/bin/env python3
"""Freeze the promoted applicability morphology proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.design.guidance.ugi_promoted_morphology_proposal import (
    build_promoted_morphology_proposal,
)

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/phase1_ugi_promoted_morphology_proposal_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase1/ugi_promoted_morphology_proposal_v1"),
    )
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output if args.output.is_absolute() else REPO / args.output
    result, ledger = build_promoted_morphology_proposal(REPO, config)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "proposal_ledger.jsonl.gz").write_bytes(ledger)
    print(json.dumps(result["proposal"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
