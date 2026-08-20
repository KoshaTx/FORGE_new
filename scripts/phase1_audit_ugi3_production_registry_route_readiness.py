#!/usr/bin/env python3
"""Audit route readiness for the full Ugi production component registry."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.route.assessment.ugi3_production_registry_route_readiness import (
    build_production_registry_route_readiness,
)

REPO = Path(__file__).resolve().parents[1]


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
        default=REPO / "configs/route/phase1_ugi3_production_registry_route_readiness.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi3_production_registry_route_readiness",
    )
    args = parser.parse_args()
    result, components, gaps = build_production_registry_route_readiness(
        args.config,
        REPO / "results/phase1/ugi_component_expansion/component_registry.csv.gz",
        REPO / "results/phase1/ugi_component_expansion/result.json",
        REPO
        / "results/phase1/ugi3_complete_computational_dossiers/component_dossier_ledger.csv.gz",
        REPO / "results/phase1/ugi3_complete_computational_dossiers/result.json",
        REPO / "results/m0_09/hydrophobic_motif_transfer_ledger.csv.gz",
        REPO / "configs/route/phase1_ugi3_upstream_qualified_reactions_v1.json",
        REPO
        / "configs/route/variants/ugi3_upstream_primary_alcohol_oxidation_exact_source_v1.json",
    )
    _write_atomic(args.output_dir / "component_readiness_ledger.csv.gz", components)
    _write_atomic(args.output_dir / "recurring_gap_ledger.csv.gz", gaps)
    _write_atomic(
        args.output_dir / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
