#!/usr/bin/env python3
"""Run the bounded sparse-topology follow-up to the M0-06 DeFoG gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.design.flow.sparse_topology_feasibility import (
    FeasibilityError,
    run_sparse_feasibility,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/m0_06_sparse_topology_feasibility.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/m0_06_sparse",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_sparse_feasibility(args.config, REPO, args.output_dir)
    except FeasibilityError as exc:
        print(f"M0-06 sparse feasibility failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "recommendation": result["decision"]["recommendation"],
                "roundtrip_fraction": result["representation_audit"]["roundtrip_fraction"],
                "constitutional_roundtrip_fraction": result["representation_audit"][
                    "constitutional_roundtrip_fraction"
                ],
                "aromatic_constitutional_roundtrip_fraction": result["representation_audit"][
                    "aromatic_constitutional_roundtrip_fraction"
                ],
                "runs": {
                    run["name"]: {
                        "training_graphs_per_second": run["training"]["graphs_per_second"],
                        "endpoint_validity": run["endpoints"]["endpoint_validity"],
                        "connectedness": run["endpoints"]["connectedness"],
                    }
                    for run in result["runs"]
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
