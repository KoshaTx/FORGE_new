#!/usr/bin/env python3
"""Run the M0-04 per-platform reaction-registry coverage addendum."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.corpus.source_platform_registry_audit import (
    SourcePlatformAuditError,
    build_source_platform_audit,
    write_source_platform_audit,
)

REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/corpus/m0_04_source_platform_registry_audit.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/m0_04",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result, artifacts = build_source_platform_audit(
            args.config,
            REPO / "data/vendor/r0_observed_real_structures.csv",
            REPO / "data/splits/m0_03/r0_fold_assignments.csv",
            (
                REPO / "data/vendor/qualified_reaction_families_v1.json",
                REPO / "data/vendor/qualified_reactions_v1.json",
            ),
            REPO / "configs/corpus/m0_04_r1_prime_audit.json",
            REPO / "results/m0_04/result.json",
            REPO / "results/m0_04/component_pool.csv.gz",
        )
        write_source_platform_audit(result, artifacts, args.output_dir)
    except SourcePlatformAuditError as exc:
        print(f"M0-04 source-platform audit failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
