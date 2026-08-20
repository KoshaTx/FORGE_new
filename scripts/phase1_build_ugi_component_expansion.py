#!/usr/bin/env python3
"""Build the provenance-aware Phase 1 Ugi component expansion registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forge.design.corpus.ugi_component_expansion import (  # noqa: E402
    ComponentExpansionError,
    build_ugi_component_expansion,
    write_ugi_component_expansion,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_component_expansion.json",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO)
    args = parser.parse_args()
    try:
        result, rows = build_ugi_component_expansion(args.config, args.repo_root)
        result = write_ugi_component_expansion(args.config, args.repo_root, result, rows)
    except ComponentExpansionError as exc:
        print(f"Phase 1 Ugi component expansion failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
