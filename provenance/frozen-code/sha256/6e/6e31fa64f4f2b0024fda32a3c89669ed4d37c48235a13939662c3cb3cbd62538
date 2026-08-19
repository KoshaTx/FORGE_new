#!/usr/bin/env python3
"""Run the single reviewed selected-v3 synthesis-only qualification at lambda 0.25."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.product.ugi_production_synthesis_guidance_seam_v4 import (
    RERUN_TOKEN,
    REVIEW_TOKEN,
    run_production_synthesis_guidance_seam_v4,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/model/phase1_ugi_production_synthesis_guidance_seam_v4.json"
DEFAULT_OUTPUT = REPO / "results/phase1/ugi_production_synthesis_guidance_seam_v4_retry1"
EXPECTED_FILENAMES = {
    "diagnostics.json",
    "result.json",
    "run.json",
    "support_audits.json",
}


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_output(output: Path, values: dict[str, object]) -> None:
    if set(values) != EXPECTED_FILENAMES:
        raise SystemExit(
            "seam v4 must persist exactly result.json, run.json, "
            "support_audits.json and diagnostics.json"
        )
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        root = Path(temporary)
        for name, value in values.items():
            (root / name).write_bytes(_canonical_bytes(value))
        if {path.name for path in root.iterdir()} != EXPECTED_FILENAMES:
            raise SystemExit("temporary seam-v4 artifact set changed before publication")
        os.replace(root, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--review-token",
        required=True,
        help="Explicit one-run review token; the config alone cannot authorize execution.",
    )
    parser.add_argument(
        "--rerun-token",
        required=True,
        help="Explicit token for the one operationally reauthorized same-seed rerun.",
    )
    args = parser.parse_args()
    if args.review_token != REVIEW_TOKEN:
        raise SystemExit("incorrect review token; refusing nonzero synthesis guidance")
    if args.rerun_token != RERUN_TOKEN:
        raise SystemExit("incorrect rerun token; refusing operational retry")

    output = args.output_dir.resolve()
    cache_root = output.parent / f".{output.name}.cache"
    if cache_root.exists():
        raise SystemExit(f"refusing to reuse seam-v4 cache directory: {cache_root}")
    result, run, support_audits, diagnostics = run_production_synthesis_guidance_seam_v4(
        REPO,
        args.config,
        cache_root,
        output,
        review_token=args.review_token,
        rerun_token=args.rerun_token,
    )
    _atomic_output(
        output,
        {
            "result.json": result,
            "run.json": run,
            "support_audits.json": support_audits,
            "diagnostics.json": diagnostics,
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
