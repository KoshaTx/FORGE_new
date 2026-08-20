#!/usr/bin/env python3
"""Adjudicate the single frozen decision-aligned Ugi potency challenger."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.potency.morphology.ugi_high_potency_challenger_adjudication import (
    adjudicate_high_potency_challenger,
)

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_high_potency_challenger_adjudication_v1"),
    )
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")
    result = adjudicate_high_potency_challenger(
        signal_path=REPO
        / "results/phase1/ugi_morphology_high_potency_challenger_v1_retry1/result.json",
        proposal_path=REPO / "results/phase1/ugi_morphology_high_potency_proposal_v1/result.json",
        generation_path=REPO
        / "results/phase1/ugi_high_potency_challenger_terminal_generation_v1/result.json",
        ranking_path=REPO
        / "results/phase1/ugi_high_potency_challenger_continuous_ranking_v1/result.json",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        path = Path(temporary)
        (path / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(path, output)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
