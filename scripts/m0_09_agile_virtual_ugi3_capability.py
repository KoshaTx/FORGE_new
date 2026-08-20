#!/usr/bin/env python3
"""Audit all 12,276 AGILE virtual products and their unique Ugi components."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.route.evidence.ugi3_virtual_capability import (
    Ugi3VirtualCapabilityError,
    build_ugi3_virtual_capability,
    write_ugi3_virtual_capability,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/m0_09_agile_virtual_ugi3_capability.json",
    )
    parser.add_argument(
        "--virtual-smiles",
        type=Path,
        default=REPO / "data/derived/agile_virtual12k_smiles.csv.gz",
    )
    parser.add_argument(
        "--virtual-manifest",
        type=Path,
        default=REPO
        / "data/derived/agile_virtual12k_smiles.manifest.json",
    )
    parser.add_argument(
        "--qualified-reactions",
        type=Path,
        default=REPO / "data/vendor/qualified_reactions_v1.json",
    )
    parser.add_argument(
        "--assembly-qualification",
        type=Path,
        default=REPO / "results/m0_09/ugi3_assembly_qualification.json",
    )
    parser.add_argument(
        "--precursor-capability",
        type=Path,
        default=REPO / "results/m0_09/ugi3_precursor_capability.json",
    )
    parser.add_argument(
        "--aldehyde-head-capability",
        type=Path,
        default=REPO / "results/m0_09/ugi3_aldehyde_head_capability.json",
    )
    parser.add_argument(
        "--agile-component-routes",
        type=Path,
        default=REPO / "results/m0_09/agile_component_routes.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/m0_09",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result, artifacts = build_ugi3_virtual_capability(
            args.config,
            args.virtual_smiles,
            args.virtual_manifest,
            args.qualified_reactions,
            args.assembly_qualification,
            args.precursor_capability,
            args.aldehyde_head_capability,
            args.agile_component_routes,
        )
        write_ugi3_virtual_capability(
            result,
            artifacts,
            args.output_dir,
        )
    except Ugi3VirtualCapabilityError as exc:
        print(f"M0-09 AGILE virtual Ugi capability failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": str(
                    args.output_dir / "agile_virtual_ugi3_capability.json"
                ),
                **result["summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
