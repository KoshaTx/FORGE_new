#!/usr/bin/env python3
"""Run the frozen nonselecting Ugi chemistry-bias attribution audit."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from forge.data.r1_prime_audit import sha256_file
from forge.product.ugi_chemistry_bias_attribution import build_chemistry_bias_attribution

REPO = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
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
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output if args.output.is_absolute() else REPO / args.output
    result = build_chemistry_bias_attribution(REPO, config)
    _atomic_json(output, result)
    print(output)
    print(sha256_file(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
