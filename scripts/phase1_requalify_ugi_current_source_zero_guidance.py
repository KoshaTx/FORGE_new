#!/usr/bin/env python3
"""Freeze the current L3 source and replay locked zero-guidance route values."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.design.guidance.ugi_current_source_zero_guidance_requalification import (
    build_current_source_zero_guidance_requalification,
)

REPO = Path(__file__).resolve().parents[1]


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO
        / "configs/model/phase1_ugi_current_source_zero_guidance_requalification_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO
        / "results/phase1/ugi_current_source_zero_guidance_requalification_v1/result.json",
    )
    args = parser.parse_args()
    result = build_current_source_zero_guidance_requalification(REPO, args.config)
    _atomic_write(args.output, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())


if __name__ == "__main__":
    main()
