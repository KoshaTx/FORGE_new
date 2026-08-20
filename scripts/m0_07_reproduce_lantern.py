#!/usr/bin/env python3
"""Reproduce the released LANTERN AGILE checkpoint as an audit-only result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.potency.oracle.oracle_lantern import run_lantern_reproduction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bio/m0_07_lantern_reproduction.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/m0_07"))
    arguments = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    result = run_lantern_reproduction(
        config_path=repo_root / arguments.config,
        output_dir=repo_root / arguments.output_dir,
        repo_root=repo_root,
    )
    summary = {
        "status": result["status"],
        "test": result["metrics"]["test"],
        "selection_eligible": result["policy"]["selection_eligible"],
        "leakage_status": result["reproduction_contract"]["preprocessing"]["leakage_status"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
