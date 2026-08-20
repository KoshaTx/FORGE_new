#!/usr/bin/env python3
"""Freeze the morphology proposal-strength challenger schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.phase1.synthesis_guidance.schedule.ugi_morphology_proposal_challenger_schedule import (
    build_morphology_proposal_challenger_schedule,
)

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/phase1_ugi_morphology_proposal_challenger_schedule_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase1/ugi_morphology_proposal_challenger_schedule_v1"),
    )
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output if args.output.is_absolute() else REPO / args.output
    result, schedule = build_morphology_proposal_challenger_schedule(REPO, config)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "schedule.json").write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["proposal"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
