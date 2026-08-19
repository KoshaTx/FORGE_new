#!/usr/bin/env python3
"""Select a future campaign oracle from the immutable M0-07 matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.potency.oracle_campaign_selection import run_campaign_selection


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bio/phase1_oracle_campaign_selection.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase1/oracle_campaign_selection.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    result = run_campaign_selection(
        config_path=args.config,
        output_path=args.output,
        repo_root=args.repo_root,
    )
    selected = result["selected_model"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "endpoint": selected["endpoint"],
                "candidate_id": selected["candidate_id"],
                "calibration_r2": selected["calibration_r2"],
                "post_selection_test_r2": selected["post_selection_test_r2"],
                "same_as_m0_shared_winner": result["comparison_with_immutable_m0_freeze"][
                    "same_candidate"
                ],
                "guidance_authorized": result["guidance_policy"]["authorized"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
