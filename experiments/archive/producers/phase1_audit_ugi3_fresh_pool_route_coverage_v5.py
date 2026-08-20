#!/usr/bin/env python3
"""Apply one hardened exact C16 route to immutable fresh-pool coverage v4."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from experiments.archive.phase1.synthesis_value_coverage.ugi3_fresh_pool_route_coverage_v5 import (
    build_fresh_pool_route_coverage_v5,
)

REPO = Path(__file__).resolve().parents[3]


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/phase1_ugi3_fresh_pool_route_coverage_v5.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi3_fresh_pool_route_coverage_v5",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    result, component_ledger, product_ledger = build_fresh_pool_route_coverage_v5(REPO, config_path)
    _write_atomic(output_dir / "component_synthesis_values.json.gz", component_ledger)
    _write_atomic(output_dir / "product_synthesis_values.json.gz", product_ledger)
    _write_atomic(
        output_dir / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
