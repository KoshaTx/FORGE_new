#!/usr/bin/env python3
"""Enumerate the exact family-split Phase 1 Ugi product corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forge.corpus.ugi_expanded_enumeration import (  # noqa: E402
    ExpandedEnumerationError,
    enumerate_expanded_ugi_corpus,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_expanded_enumeration.json",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO)
    args = parser.parse_args()
    try:
        result = enumerate_expanded_ugi_corpus(args.config, args.repo_root)
    except ExpandedEnumerationError as exc:
        print(f"Phase 1 Ugi enumeration failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
