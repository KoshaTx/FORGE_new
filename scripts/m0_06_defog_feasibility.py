#!/usr/bin/env python3
"""Run the bounded M0-06 DeFoG-style product-prior feasibility gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.design.flow.defog_feasibility import FeasibilityError, run_feasibility

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/m0_06_defog_feasibility.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/m0_06",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_feasibility(args.config, REPO, args.output_dir)
    except FeasibilityError as exc:
        print(f"M0-06 feasibility gate failed: {exc}", file=sys.stderr)
        return 1
    summary = {
        "status": result["status"],
        "support_profile_matches": result["support_audit"]["profile_matches"],
        "recommendation": result["decision"]["recommendation"],
        "runs": {
            run["name"]: {
                "training_graphs_per_second": run["training"]["graphs_per_second"],
                "endpoint_validity": run["endpoints"]["endpoint_validity"],
                "bond_recall": run["heldout_reconstruction"]["metrics"]["bond_recall"],
            }
            for run in result["runs"]
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
