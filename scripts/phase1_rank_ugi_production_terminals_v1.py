#!/usr/bin/env python3
"""Classify and rank the frozen two-arm production terminal pools."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.bio.ugi_production_terminal_ranking import build_production_terminal_ranking

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bio/phase1_ugi_production_terminal_ranking_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_production_terminal_ranking_v1"),
    )
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")
    result, ledger = build_production_terminal_ranking(REPO, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        path = Path(temporary)
        (path / "terminal_ranking.csv.gz").write_bytes(ledger)
        (path / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(path, output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "matched_oracle_budgets_per_arm": result["matched_oracle_budgets_per_arm"],
                "arms": result["arms"],
                "next_gate": result["next_gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
