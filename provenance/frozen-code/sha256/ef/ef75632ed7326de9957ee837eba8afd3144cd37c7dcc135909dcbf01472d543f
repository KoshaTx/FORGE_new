#!/usr/bin/env python3
"""Qualify the equivalence-bound selected-v3 production seam at lambda zero."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.product.ugi_production_zero_guidance_seam_v3 import (
    run_production_zero_guidance_seam_v3,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/model/phase1_ugi_production_zero_guidance_seam_v3.json"
DEFAULT_OUTPUT = REPO / "results/phase1/ugi_production_zero_guidance_seam_v3"
EXPECTED_FILENAMES = {"result.json", "run.json", "support_audits.json"}


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_output(output: Path, values: dict[str, object]) -> None:
    if set(values) != EXPECTED_FILENAMES:
        raise SystemExit(
            "seam v3 must persist exactly result.json, run.json and support_audits.json"
        )
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        root = Path(temporary)
        for name, value in values.items():
            (root / name).write_bytes(_canonical_bytes(value))
        if {path.name for path in root.iterdir()} != EXPECTED_FILENAMES:
            raise SystemExit("temporary seam-v3 artifact set changed before publication")
        os.replace(root, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    cache_root = output.parent / f".{output.name}.cache"
    if cache_root.exists():
        raise SystemExit(f"refusing to reuse seam-v3 cache directory: {cache_root}")
    result, run, support_audits = run_production_zero_guidance_seam_v3(
        REPO,
        args.config,
        cache_root,
        output,
    )
    _atomic_output(
        output,
        {
            "result.json": result,
            "run.json": run,
            "support_audits.json": support_audits,
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
