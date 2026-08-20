#!/usr/bin/env python3
"""Rank exact-new continuously interpolative matched terminals."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.potency.ranking.ugi_continuous_novelty_matched_ranking import (
    build_continuous_novelty_matched_ranking,
)

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bio/phase1_ugi_continuous_novelty_matched_ranking_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_continuous_novelty_matched_ranking_v1"),
    )
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")
    result, ledger = build_continuous_novelty_matched_ranking(REPO, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        path = Path(temporary)
        (path / "terminal_ranking.csv.gz").write_bytes(ledger)
        (path / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(path, output)
    print(json.dumps(result["interpretation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
