#!/usr/bin/env python3
"""Create or validate the frozen, leakage-safe M0-03 R0 split bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.data.r0_splits import (
    SplitError,
    build_split_bundle,
    load_frozen_r0_splits,
    verify_source_inputs,
    write_frozen_split_bundle,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/corpus/m0_03_r0_splits_constitutional.json",
        help="frozen split configuration",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO,
        help="repository root containing hash-pinned source assets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "data/splits/m0_03_constitutional",
        help="frozen split bundle directory",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=REPO / "results/m0_03/constitutional_split_result.json",
        help="M0-03 result JSON",
    )
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help="rebuild in memory and require byte-identical frozen artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.output_dir.exists() and not args.reproduce:
            _, inputs = verify_source_inputs(args.config, args.source_root)
            frozen = load_frozen_r0_splits(args.output_dir)
            summary = {
                "status": "validated_frozen_bundle",
                "output_dir": str(args.output_dir),
                "source_inputs_verified": len(inputs),
                "row_count": len(frozen.assignments),
                "fold_counts": {
                    scheme: details["fold_counts"]
                    for scheme, details in frozen.manifest["schemes"].items()
                },
            }
        else:
            assignment_payload, manifest, result = build_split_bundle(
                args.config,
                args.source_root,
            )
            write_frozen_split_bundle(
                assignment_payload, manifest, result, args.output_dir, args.result
            )
            status = "reproduced_frozen_bundle" if args.reproduce else "created_frozen_bundle"
            summary = {
                "status": status,
                "output_dir": str(args.output_dir),
                "result": str(args.result),
                "row_count": manifest["row_count"],
                "fold_counts": {
                    scheme: details["fold_counts"]
                    for scheme, details in manifest["schemes"].items()
                },
            }
    except SplitError as exc:
        print(f"M0-03 split freeze failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
