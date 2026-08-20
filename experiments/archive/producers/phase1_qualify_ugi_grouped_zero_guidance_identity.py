#!/usr/bin/env python3
"""Run the selected-model grouped lambda-zero identity qualification."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from experiments.phase1.synthesis_guidance.guidance.ugi_grouped_zero_guidance_identity import (
    build_grouped_zero_guidance_identity,
)

REPO = Path(__file__).resolve().parents[3]


def _atomic_write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite qualification output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite qualification output: {path}") from error
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(REPO / "configs/model/phase1_ugi_grouped_zero_guidance_identity_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(REPO / "results/phase1/ugi_grouped_zero_guidance_identity_v1/result.json"),
    )
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output if args.output.is_absolute() else REPO / args.output
    result = build_grouped_zero_guidance_identity(REPO, config)
    _atomic_write_once(
        output,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )


if __name__ == "__main__":
    main()
