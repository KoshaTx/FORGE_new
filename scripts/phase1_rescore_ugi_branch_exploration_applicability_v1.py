#!/usr/bin/env python3
"""Apply the frozen HeLa policy to the corrected branch-exploration pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.potency.applicability.ugi_branch_exploration_applicability_v1 import (
    build_branch_exploration_applicability,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bio/phase1_ugi_branch_exploration_applicability_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_branch_exploration_applicability_v1"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = (repo / args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    result, ledger = build_branch_exploration_applicability(repo, repo / args.config)
    (output / "terminal_rescoring.csv.gz").write_bytes(ledger)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
