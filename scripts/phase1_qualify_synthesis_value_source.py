#!/usr/bin/env python3
"""Freeze the behavior-preserving synthesis-value source qualification."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.value.synthesis.synthesis_source_qualification import build_synthesis_source_qualification

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
        default=REPO / "configs/route/phase1_synthesis_value_source_qualification_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/synthesis_value_source_qualification_v1/result.json",
    )
    arguments = parser.parse_args()
    config = arguments.config if arguments.config.is_absolute() else REPO / arguments.config
    output = arguments.output if arguments.output.is_absolute() else REPO / arguments.output
    result = build_synthesis_source_qualification(REPO, config)
    _write(output, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
