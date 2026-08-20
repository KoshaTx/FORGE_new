#!/usr/bin/env python3
"""Build the conservative Ugi-3 dossier-coverage census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.route.audit.ugi3_dossier_coverage import build_ugi3_dossier_coverage

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/phase1_ugi3_dossier_coverage.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi3_dossier_coverage",
    )
    args = parser.parse_args()
    result, components, products = build_ugi3_dossier_coverage(
        args.config,
        REPO / "results/m0_09/agile_virtual_ugi3_component_program_ledger.csv.gz",
        REPO / "results/m0_09/agile_virtual_ugi3_product_ledger.csv.gz",
        REPO / "configs/route/m0_09_ugi3_virtual_terminal_procurement.json",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "component_dossier_ledger.csv.gz").write_bytes(components)
    (args.output_dir / "product_dossier_ledger.csv.gz").write_bytes(products)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
