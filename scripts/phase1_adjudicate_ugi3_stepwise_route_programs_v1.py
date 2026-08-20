#!/usr/bin/env python3
"""Run the frozen stepwise route-program adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.route.evidence.ugi3_stepwise_route_program_adjudication import (
    build_stepwise_route_program_adjudication,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/route/phase1_ugi3_stepwise_route_program_adjudication_v1.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    payload = json.loads(config.read_text())
    output = root / payload["output_directory"]
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("write-once stepwise output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)
    result, ledger = build_stepwise_route_program_adjudication(root, config)
    (output / "stepwise_route_ledger.jsonl.gz").write_bytes(ledger)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
