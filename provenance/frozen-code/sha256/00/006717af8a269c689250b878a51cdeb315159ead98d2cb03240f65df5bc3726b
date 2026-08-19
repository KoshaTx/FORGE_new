#!/usr/bin/env python3
"""Freeze expanded full-corpus Ugi support and the branch exploration schedule."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from forge.product.ugi_full_corpus_branch_exploration_schedule import (
    build_full_corpus_branch_exploration_schedule,
)

REPO = Path(__file__).resolve().parents[1]


def _atomic_bytes(path: Path, payload: bytes) -> None:
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


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/phase1_ugi_full_corpus_branch_exploration_schedule_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_full_corpus_branch_exploration_schedule_v1"),
    )
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    result, support_ledger, schedule = build_full_corpus_branch_exploration_schedule(REPO, config)
    _atomic_bytes(output / "support_ledger.jsonl.gz", support_ledger)
    _atomic_json(output / "schedule.json", schedule)
    _atomic_json(output / "result.json", result)
    print(json.dumps({"support": result["support"], "schedule": result["schedule"]}, indent=2))


if __name__ == "__main__":
    main()
