#!/usr/bin/env python3
"""Refit and hash the frozen M0-07 production oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.potency.oracle_production import run_oracle_production_refit


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bio/m0_07_oracle_production.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    result = run_oracle_production_refit(
        config_path=args.config.resolve(),
        repo_root=args.repo_root.resolve(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_model": result["selected_model"]["candidate_id"],
                "checkpoint": result["checkpoint"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
