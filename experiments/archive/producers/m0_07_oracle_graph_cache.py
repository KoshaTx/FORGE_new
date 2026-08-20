#!/usr/bin/env python3
"""Build the deterministic M0-07 graph tensor cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.potency.oracle.oracle_graph_cache import (
    OracleGraphCacheError,
    build_oracle_graph_cache,
)

REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/bio/m0_07_oracle_graph_cache.json",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/m0_07")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_oracle_graph_cache(
            args.config,
            args.output_dir,
            args.repo_root,
        )
    except OracleGraphCacheError as exc:
        print(f"M0-07 graph tensor cache failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": result["summary"],
                "collections": result["collections"],
                "artifacts": result["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
