#!/usr/bin/env python3
"""Run or validate the one-shot blinded Ugi-3 route-saturation holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.route.ugi3_route_saturation_blinded_execution import (
    run_one_shot_blinded_holdout,
    validate_completed_blinded_holdout,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/route/phase1_ugi3_route_saturation_blinded_execution_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--validate-completed",
        type=Path,
        help=(
            "Privately recompute and validate an existing seal while returning no "
            "identity-bearing records"
        ),
    )
    args = parser.parse_args()
    if args.validate_completed is not None:
        result = validate_completed_blinded_holdout(
            REPO,
            args.validate_completed,
            config_path=args.config,
        )
    else:
        result = run_one_shot_blinded_holdout(REPO, args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
