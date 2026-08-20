#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.design.guidance.ugi_graded_route_readiness_guidance import (
    run_graded_route_readiness_guidance,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/model/phase1_ugi_graded_route_readiness_guidance_v1.json"
DEFAULT_OUTPUT = REPO / "results/phase1/ugi_graded_route_readiness_guidance_v1_retry3"
EXPECTED_FILES = {"result.json", "run.json", "support_audits.json", "diagnostics.json"}


def _bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _publish(output: Path, values: dict[str, object]) -> None:
    if set(values) != EXPECTED_FILES:
        raise SystemExit("graded-readiness output set changed")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temp:
        root = Path(temp)
        for name, value in values.items():
            (root / name).write_bytes(_bytes(value))
        os.replace(root, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-token", required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    cache = output.parent / f".{output.name}.cache"
    if cache.exists():
        raise SystemExit(f"refusing to reuse cache: {cache}")
    result, run, support, diagnostics = run_graded_route_readiness_guidance(
        REPO,
        args.config,
        cache,
        output,
        review_token=args.review_token,
    )
    _publish(
        output,
        {
            "result.json": result,
            "run.json": run,
            "support_audits.json": support,
            "diagnostics.json": diagnostics,
        },
    )
    print(json.dumps(result["execution"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
