#!/usr/bin/env python3
"""Requalify immutable guidance prerequisites against fresh-pool v6."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.design.guidance.ugi_source_guidance_prerequisite_requalification import (
    build_source_guidance_prerequisite_requalification,
)

REPO = Path(__file__).resolve().parents[1]


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
        default=(
            REPO / "configs/model/phase1_ugi_source_guidance_prerequisite_requalification_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO / "results/phase1/ugi_source_guidance_prerequisite_requalification_v1/result.json"
        ),
    )
    arguments = parser.parse_args()
    config = arguments.config if arguments.config.is_absolute() else REPO / arguments.config
    output = arguments.output if arguments.output.is_absolute() else REPO / arguments.output
    result = build_source_guidance_prerequisite_requalification(REPO, config)
    _write(output, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(result["adjudication"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
