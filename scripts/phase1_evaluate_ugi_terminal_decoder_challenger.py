#!/usr/bin/env python3
"""Evaluate matched argmax and stochastic Ugi terminal decoders."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.design.sampling.ugi_terminal_decoder_challenger import build_terminal_decoder_evaluation

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
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_terminal_decoder_evaluation(REPO, args.config.resolve())
    _atomic_write(
        args.output.resolve(),
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(result["adjudication"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
