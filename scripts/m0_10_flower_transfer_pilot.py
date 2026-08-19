#!/usr/bin/env python3
"""Audit and freeze the bounded M0-10 FlowER transfer-pilot disposition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.verify.flower_transfer_audit import (
    FlowerTransferAuditError,
    run_flower_transfer_audit,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/verify/m0_10_flower_transfer_pilot.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/m0_10/result.json",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_flower_transfer_audit(args.config, args.output, args.repo_root)
    except FlowerTransferAuditError as exc:
        print(f"M0-10 FlowER audit failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "blockers": result["pilot_readiness"]["blockers"],
                "flowER_role": result["demotion_rule"]["flowER_role"],
                "deterministic_verifier_load_bearing": result["demotion_rule"][
                    "deterministic_atom_mapped_verifier_load_bearing"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
