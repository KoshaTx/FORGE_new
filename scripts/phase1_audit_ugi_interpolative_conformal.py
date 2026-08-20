#!/usr/bin/env python3
"""Run the additive interpolative-bin conformal audit."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.potency.applicability.ugi_interpolative_conformal import build_interpolative_conformal_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bio/phase1_ugi_interpolative_conformal_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_interpolative_conformal_v1"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    result = build_interpolative_conformal_audit(repo, repo / args.config)
    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix="result.", suffix=".json", dir=output_dir)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_dir / "result.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == "__main__":
    main()
