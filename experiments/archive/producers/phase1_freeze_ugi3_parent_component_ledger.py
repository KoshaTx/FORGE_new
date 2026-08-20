#!/usr/bin/env python3
"""Freeze the final prereveal R0/without-C18 parent component ledger."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.synthesis.sources.ugi3_frozen_parent_component_ledger import (
    Ugi3FrozenParentComponentLedgerError,
    build_frozen_parent_component_ledger,
)

REPO = Path(__file__).resolve().parents[3]


def _write_frozen(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise Ugi3FrozenParentComponentLedgerError(
                f"frozen output already exists with different bytes: {path}"
            )
        return
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/phase1_ugi3_frozen_parent_component_ledger_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi3_frozen_parent_component_ledger_v1",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    result, ledger = build_frozen_parent_component_ledger(REPO, config_path)
    result_payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    _write_frozen(output_dir / "final_component_ledger.json.gz", ledger)
    _write_frozen(output_dir / "result.json", result_payload)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
