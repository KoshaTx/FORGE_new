#!/usr/bin/env python3
"""Qualify Ugi-3 reactive-site semantics on all 1,200 measured AGILE products."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.synthesis.evidence.ugi3_assembly_qualification import (
    Ugi3AssemblyQualificationError,
    build_ugi3_assembly_qualification,
    write_ugi3_assembly_qualification,
)

REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/assembly/m0_09_ugi3_assembly_qualification.json",
    )
    parser.add_argument(
        "--agile-measured-library",
        type=Path,
        default=REPO / "data/vendor/AGILE_smiles_with_value_group.csv",
    )
    parser.add_argument(
        "--qualified-reactions",
        type=Path,
        default=REPO / "data/vendor/qualified_reactions_v1.json",
    )
    parser.add_argument(
        "--ugi-variant",
        type=Path,
        default=REPO / "configs/assembly/ugi_variant.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/m0_09/ugi3_assembly_qualification.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_ugi3_assembly_qualification(
            args.config,
            args.agile_measured_library,
            args.qualified_reactions,
            args.ugi_variant,
        )
        write_ugi3_assembly_qualification(result, args.output)
    except Ugi3AssemblyQualificationError as exc:
        print(f"M0-09 Ugi-3 assembly qualification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), **result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
