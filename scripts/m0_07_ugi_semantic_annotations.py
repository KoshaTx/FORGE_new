#!/usr/bin/env python3
"""Build exact-source Ugi semantic annotations for whole-graph supervision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.potency.audit.ugi_semantic_annotations import (
    UgiSemanticAnnotationError,
    build_ugi_semantic_annotations,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/bio/m0_07_ugi_semantic_annotations.json",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/m0_07")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_ugi_semantic_annotations(
            args.config,
            args.output_dir,
            args.repo_root,
        )
    except UgiSemanticAnnotationError as exc:
        print(f"M0-07 Ugi semantic annotation failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": result["summary"],
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
