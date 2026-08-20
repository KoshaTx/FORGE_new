#!/usr/bin/env python3
"""Requalify immutable fresh-pool v5 under the current value source."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from experiments.archive.phase1.synthesis_value_coverage.ugi3_fresh_pool_route_coverage_v6 import (
    build_fresh_pool_route_coverage_v6,
)

REPO = Path(__file__).resolve().parents[3]


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/phase1_ugi3_fresh_pool_route_coverage_v6.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi3_fresh_pool_route_coverage_v6",
    )
    arguments = parser.parse_args()
    config = arguments.config if arguments.config.is_absolute() else REPO / arguments.config
    output = (
        arguments.output_dir if arguments.output_dir.is_absolute() else REPO / arguments.output_dir
    )
    result, components, products = build_fresh_pool_route_coverage_v6(REPO, config)
    _write(output / "component_synthesis_values.json.gz", components)
    _write(output / "product_synthesis_values.json.gz", products)
    _write(output / "result.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
