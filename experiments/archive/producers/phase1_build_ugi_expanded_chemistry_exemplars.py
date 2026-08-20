#!/usr/bin/env python3
"""Build fold-clean exact chemistry exemplars for expanded Ugi components."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forge.corpus.ugi_expanded_exemplars import (  # noqa: E402
    ExpandedExemplarError,
    build_expanded_ugi_chemistry_exemplars,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_expanded_chemistry_exemplars.json",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO)
    args = parser.parse_args()
    try:
        result = build_expanded_ugi_chemistry_exemplars(
            args.config.resolve(),
            args.repo_root.resolve(),
        )
    except ExpandedExemplarError as exc:
        print(f"Phase 1 expanded chemistry exemplar build failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
