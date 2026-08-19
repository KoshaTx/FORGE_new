#!/usr/bin/env python3
"""Compare matched source-balanced joint and staged Ugi generator samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.product.ugi_generator_comparison import compare_ugi_generator_arms


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-result", type=Path, required=True)
    parser.add_argument("--staged-result", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--reference-products", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    result = compare_ugi_generator_arms(
        joint_result_path=args.joint_result,
        staged_result_path=args.staged_result,
        probe_path=args.probe,
        reference_products_path=args.reference_products,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
