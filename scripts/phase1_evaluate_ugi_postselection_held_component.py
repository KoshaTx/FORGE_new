#!/usr/bin/env python3
"""Evaluate the frozen Ugi product/L1 generator on held-family programs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.product.ugi_postselection_held_component import (
    evaluate_postselection_held_component_stress,
)

REPO = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/model/phase1_ugi_product_l1_postselection_held_component_stress_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase1/ugi_product_l1_postselection_held_component_stress_v1.json"),
    )
    args = parser.parse_args()
    result = evaluate_postselection_held_component_stress(
        config_path=_resolve(args.config),
        output_path=_resolve(args.output),
        repository=REPO,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(_resolve(args.output)),
                "overall": result["generation"]["overall"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
