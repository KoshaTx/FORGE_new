#!/usr/bin/env python3
"""Extract a compact, deterministic SMILES-only AGILE virtual candidate table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.corpus.agile_virtual import (
    AgileVirtualExtractionError,
    extract_agile_virtual_smiles,
    write_agile_virtual_smiles,
)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = Path(
    "/Users/rmaganti/Desktop/thesis_projects_ML/diffusion_project/"
    "lipid_diffusion/data/agile_virtual/candidate_set_smiles_plus_features.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/corpus/m0_09_agile_virtual_smiles.json",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "data/derived/agile_virtual12k_smiles.csv.gz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "data/derived/agile_virtual12k_smiles.manifest.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest, payload = extract_agile_virtual_smiles(
            args.config,
            args.source,
        )
        write_agile_virtual_smiles(
            manifest,
            payload,
            args.output,
            args.manifest,
        )
    except AgileVirtualExtractionError as exc:
        print(f"M0-09 AGILE virtual extraction failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest": str(args.manifest),
                **manifest["summary"],
                "output_bytes": len(payload),
                "output_sha256": manifest["output"]["sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
