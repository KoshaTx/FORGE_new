#!/usr/bin/env python3
"""Freeze the completed R0 transfer lane without rerunning any model fit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.potency.oracle_graph_transfer_decision import (
    OracleGraphTransferDecisionError,
    run_transfer_decision,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/bio/m0_07_oracle_graph_transfer_decision.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/m0_07/oracle_graph_transfer_decision.json",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_transfer_decision(args.config, args.output, args.repo_root)
    except OracleGraphTransferDecisionError as exc:
        print(f"M0-07 transfer adjudication failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_model": result["selected_cross_representation_model"]["candidate_id"],
                "transfer_candidates": result["transfer_candidates"],
                "guidance_oracle_authorized": result["decision"]["guidance_oracle_authorized"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
