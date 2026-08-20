#!/usr/bin/env python3
"""Build the fixed M0-09 cross-platform hydrophobic-motif transfer pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.synthesis.assessment.hydrophobic_motif_transfer import (
    HydrophobicMotifTransferError,
    build_hydrophobic_motif_transfer,
    write_hydrophobic_motif_transfer,
)

REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO
        / "configs/route/m0_09_hydrophobic_motif_transfer.json",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO,
        help="Root used to resolve hash-pinned source assets in the config.",
    )
    parser.add_argument(
        "--component-ledger",
        type=Path,
        default=REPO
        / "results/m0_09/agile_virtual_ugi3_component_ledger.csv.gz",
    )
    parser.add_argument(
        "--reaction-registry",
        type=Path,
        default=REPO / "data/vendor/qualified_reactions_v1.json",
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
        result, ledger = build_hydrophobic_motif_transfer(
            args.config,
            args.source_root,
            args.component_ledger,
            args.reaction_registry,
        )
        write_hydrophobic_motif_transfer(
            result,
            ledger,
            args.output_dir,
        )
    except HydrophobicMotifTransferError as exc:
        print(
            f"M0-09 hydrophobic-motif transfer pilot failed: {exc}",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "output": str(
                    args.output_dir / "hydrophobic_motif_transfer.json"
                ),
                **result["summary"],
                **result["decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
