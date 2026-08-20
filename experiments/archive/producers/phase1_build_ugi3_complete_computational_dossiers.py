#!/usr/bin/env python3
"""Build complete exact-source Ugi computational dossier ledgers."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.synthesis.evidence.ugi3_complete_computational_dossiers import (
    build_complete_computational_dossiers,
)

REPO = Path(__file__).resolve().parents[3]


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        default=REPO / "configs/route/phase1_ugi3_complete_computational_dossiers.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi3_complete_computational_dossiers",
    )
    args = parser.parse_args()
    result, components, products = build_complete_computational_dossiers(
        args.config,
        REPO / "results/m0_09/agile_virtual_ugi3_component_program_ledger.csv.gz",
        REPO / "results/m0_09/agile_virtual_ugi3_product_ledger.csv.gz",
        REPO / "results/phase1/ugi3_dossier_coverage/component_dossier_ledger.csv.gz",
        REPO / "results/phase1/ugi3_dossier_coverage/product_dossier_ledger.csv.gz",
        REPO / "results/phase1/ugi3_dossier_coverage/result.json",
        REPO
        / "results/phase1/ugi3_exact_source_forward_verification/step_verification_ledger.csv.gz",
        REPO / "results/phase1/ugi3_exact_source_forward_verification/result.json",
        REPO / "configs/route/m0_09_ugi3_virtual_terminal_procurement.json",
    )
    _write_atomic(args.output_dir / "component_dossier_ledger.csv.gz", components)
    _write_atomic(args.output_dir / "product_dossier_ledger.csv.gz", products)
    _write_atomic(
        args.output_dir / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
