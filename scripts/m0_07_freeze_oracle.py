#!/usr/bin/env python3
"""Freeze the M0-07 oracle architecture and applicability policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.potency.oracle_freeze import freeze_oracle


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bio/m0_07_oracle_freeze.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/m0_07/oracle_freeze_result.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    result = freeze_oracle(
        config_path=args.config.resolve(),
        output_path=args.output.resolve(),
        repo_root=args.repo_root.resolve(),
    )
    selected = result["selected_model"]
    summary = {
        "status": result["status"],
        "candidate_id": selected["candidate_id"],
        "selection_calibration_r2": selected["equal_endpoint_mean_equal_scheme_calibration_r2"],
        "post_selection_test_r2": selected["equal_endpoint_mean_equal_scheme_test_r2"],
        "any_guidance_domain_authorized": result["applicability_policy"][
            "any_guidance_domain_authorized"
        ],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
