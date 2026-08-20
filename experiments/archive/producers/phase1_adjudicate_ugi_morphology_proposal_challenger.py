#!/usr/bin/env python3
"""Adjudicate the morphology proposal-strength challenger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.phase1.hela_potency.morphology.ugi_morphology_proposal_challenger_adjudication import (
    build_morphology_proposal_challenger_adjudication,
)

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bio/phase1_ugi_morphology_proposal_challenger_adjudication_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/phase1/ugi_morphology_proposal_challenger_adjudication_v1/result.json"
        ),
    )
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output if args.output.is_absolute() else REPO / args.output
    result = build_morphology_proposal_challenger_adjudication(REPO, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
