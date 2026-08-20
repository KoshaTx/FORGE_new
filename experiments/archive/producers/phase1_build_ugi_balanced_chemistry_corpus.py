#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.corpus.ugi_balanced_chemistry_corpus import build_balanced_ugi_chemistry_corpus

REPO = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_balanced_chemistry_corpus.json",
    )
    args = parser.parse_args()
    result = build_balanced_ugi_chemistry_corpus(args.config, REPO)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

