#!/usr/bin/env python3
"""Audit the fresh v2 Ugi product-plus-L1 pool."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.design.audit.ugi_fresh_pool_audit import build_fresh_pool_audit

REPO = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output if args.output.is_absolute() else REPO / args.output
    result = build_fresh_pool_audit(REPO, config.resolve())
    _atomic_json(output.resolve(), result)
    print(
        json.dumps({"summary": result["summary"], "adjudication": result["adjudication"]}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
