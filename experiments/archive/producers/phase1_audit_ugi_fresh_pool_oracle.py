#!/usr/bin/env python3
"""Run the nonselecting HeLa-oracle audit on the frozen fresh Ugi pool."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from experiments.archive.phase1.potency_audits.ugi_fresh_pool_oracle_audit import build_fresh_pool_oracle_audit

REPO = Path(__file__).resolve().parents[3]


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
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/bio/phase1_ugi_fresh_pool_oracle_audit_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi_fresh_pool_oracle_audit_v1",
    )
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    result, ledger = build_fresh_pool_oracle_audit(REPO, config)
    _atomic_write(output / "oracle_predictions.csv.gz", ledger)
    _atomic_write(
        output / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
