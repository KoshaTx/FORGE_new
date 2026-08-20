#!/usr/bin/env python3
"""Run the nonselecting Ugi distributional-applicability audit."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.potency.applicability.ugi_distributional_applicability import (
    build_distributional_applicability_audit,
)

REPO = Path(__file__).resolve().parents[1]


def _write_once(path: Path, payload: bytes) -> None:
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
            raise FileExistsError(f"refusing to overwrite audit output: {path}") from error
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/bio/phase1_ugi_distributional_applicability_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi_distributional_applicability_v1",
    )
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    if output.exists():
        raise FileExistsError(f"refusing to reuse audit output directory: {output}")
    result, generated, heldout = build_distributional_applicability_audit(REPO, config)
    output.mkdir(parents=True, exist_ok=False)
    _write_once(output / "generated_applicability.csv.gz", generated)
    _write_once(output / "heldout_applicability.csv.gz", heldout)
    _write_once(
        output / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )


if __name__ == "__main__":
    main()
