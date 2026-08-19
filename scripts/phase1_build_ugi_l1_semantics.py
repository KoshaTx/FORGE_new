#!/usr/bin/env python3
"""Build exact atom-origin supervision for the Phase 1 Ugi product union."""

from __future__ import annotations

import argparse
from pathlib import Path

from forge.product.ugi_l1_origin_annotations import build_phase1_ugi_l1_semantics

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_l1_semantics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi_l1_semantics",
    )
    args = parser.parse_args()
    result = build_phase1_ugi_l1_semantics(
        args.config.resolve(),
        args.output_dir.resolve(),
        REPO,
    )
    print(args.output_dir / "ugi_l1_semantics_result.json")
    if result["status"] != "pass":
        raise RuntimeError("Ugi L1 semantic annotation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
