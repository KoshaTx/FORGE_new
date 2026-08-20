#!/usr/bin/env python3
"""Execute or finalize bounded joint-program lambda-zero qualification."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from experiments.phase1.synthesis_guidance.guidance.ugi_joint_kernel_lambda_zero_identity import (
    finalize_all_seed_qualification,
    run_seed_qualification,
)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "configs/model/phase1_ugi_joint_kernel_lambda_zero_identity_v1.json"
DEFAULT_OUTPUT = REPO / "results/phase1/ugi_joint_kernel_lambda_zero_identity_v1"


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


def _relative_or_absolute(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seed", type=int)
    mode.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    config = _relative_or_absolute(args.config)
    output = _relative_or_absolute(args.output_dir)
    started = time.perf_counter()
    if args.finalize:
        result, ledger = finalize_all_seed_qualification(
            REPO,
            config,
            output / "seeds",
        )
        result_path = output / "result.json"
        ledger_path = output / "ledger.csv.gz"
    else:
        seed = int(args.seed)
        result, ledger = run_seed_qualification(REPO, config, seed)
        result_path = output / "seeds" / str(seed) / "result.json"
        ledger_path = output / "seeds" / str(seed) / "ledger.csv.gz"
    _atomic_write_once(ledger_path, ledger)
    _atomic_write_once(
        result_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "status": result["status"],
                "result": str(result_path),
                "ledger": str(ledger_path),
                "elapsed_wall_seconds": elapsed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
