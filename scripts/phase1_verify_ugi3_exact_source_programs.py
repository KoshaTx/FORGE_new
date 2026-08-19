#!/usr/bin/env python3
"""Run the conservative exact-source Ugi upstream forward-verification gate."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.route.ugi3_exact_source_forward_verification import (
    build_exact_source_forward_verification,
)

REPO = Path(__file__).resolve().parents[1]


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(REPO / "configs/route/phase1_ugi3_exact_source_forward_verification.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi3_exact_source_forward_verification",
    )
    args = parser.parse_args()
    result, ledger = build_exact_source_forward_verification(
        args.config,
        REPO / "results/m0_09/agile_virtual_ugi3_component_program_ledger.csv.gz",
        REPO / "results/m0_09/agile_component_routes.json",
        REPO / "configs/route/phase1_ugi3_upstream_qualified_reactions_v1.json",
        REPO / "data/vendor/qualified_reaction_families_v1.json",
    )
    _write_atomic(args.output_dir / "step_verification_ledger.csv.gz", ledger)
    _write_atomic(
        args.output_dir / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
