#!/usr/bin/env python3
"""Run the read-only stepwise route-value contrast audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.archive.phase1.synthesis_value_audits.ugi3_stepwise_route_value_contrast import (
    build_stepwise_route_value_contrast,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/phase1_ugi3_stepwise_route_value_contrast_v2.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    payload = json.loads(config.read_text())
    output = root / payload["output"]
    if output.exists():
        raise RuntimeError("write-once stepwise contrast output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = build_stepwise_route_value_contrast(root, config)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(json.dumps(result["retry_authorization"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
