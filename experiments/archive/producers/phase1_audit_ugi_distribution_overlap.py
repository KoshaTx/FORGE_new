#!/usr/bin/env python3
"""Run the directional Ugi distribution-overlap audit."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from experiments.archive.phase1.ugi_distribution_overlap import build_ugi_distribution_overlap_audit


def _atomic(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/phase1_ugi_distribution_overlap_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_distribution_overlap_v1"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    result, ledger = build_ugi_distribution_overlap_audit(repo, repo / args.config)
    output = repo / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _atomic(output / "generated_overlap_ledger.csv.gz", ledger)
    _atomic(output / "result.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())


if __name__ == "__main__":
    main()
