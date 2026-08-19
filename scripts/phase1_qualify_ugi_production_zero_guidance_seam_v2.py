#!/usr/bin/env python3
"""Qualify the selected-v2 production guidance seam at lambda zero."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.product.ugi_production_zero_guidance_seam_v2 import (
    run_production_zero_guidance_seam_v2,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/model/phase1_ugi_production_zero_guidance_seam_v2.json"
DEFAULT_OUTPUT = REPO / "results/phase1/ugi_production_zero_guidance_seam_v2"


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_output(output: Path, values: dict[str, object]) -> None:
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        root = Path(temporary)
        for name, value in values.items():
            (root / name).write_bytes(_canonical_bytes(value))
        os.replace(root, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    cache_root = output.parent / f".{output.name}.cache"
    if cache_root.exists():
        raise SystemExit(f"refusing to reuse seam cache directory: {cache_root}")
    result, run = run_production_zero_guidance_seam_v2(
        REPO,
        args.config,
        cache_root,
    )
    _atomic_output(output, {"result.json": result, "run.json": run})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
