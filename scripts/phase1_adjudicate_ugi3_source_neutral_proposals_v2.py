#!/usr/bin/env python3
"""Run the frozen proposal-first source-neutral adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.route.ugi3_source_neutral_proposal_adjudication_v2 import (
    build_source_neutral_adjudication_v2,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/route/phase1_ugi3_source_neutral_proposal_adjudication_v2.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    payload = json.loads(config.read_text())
    output = root / payload["output_directory"]
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("write-once v2 adjudication output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)
    result, ledger = build_source_neutral_adjudication_v2(root, config)
    (output / "adjudication_ledger.jsonl.gz").write_bytes(ledger)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
