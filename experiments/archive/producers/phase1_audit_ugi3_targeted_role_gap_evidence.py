#!/usr/bin/env python3
"""Audit one exact head terminal and one homologue-only isocyanide gap."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.synthesis.evidence.ugi3_targeted_role_gap_evidence import (
    build_targeted_role_gap_evidence_audit,
)

REPO = Path(__file__).resolve().parents[3]
INPUT_PATHS = {
    "agile_supplement": REPO / "data/vendor/agile_supplementary_information.pdf",
    "audit_source": REPO / "src/forge/route/ugi3_targeted_role_gap_evidence.py",
    "evidence_pack": REPO / "configs/route/phase1_ugi3_targeted_role_gap_evidence_v1.json",
    "paper_reviews": REPO / "configs/route/m0_09_lnpdb_paper_reviews.json",
    "product_impact_ledger": REPO
    / "results/phase1/ugi3_targeted_aldehyde_evidence_audit_v1/product_closure_impact_ledger.csv.gz",
    "qualifier_source": REPO / "scripts/phase1_audit_ugi3_targeted_role_gap_evidence.py",
    "readiness_ledger": REPO
    / "results/phase1/ugi3_production_registry_route_readiness/component_readiness_ledger.csv.gz",
    "targeted_aldehyde_result": REPO
    / "results/phase1/ugi3_targeted_aldehyde_evidence_audit_v1/result.json",
    "targeted_overlay_result": REPO
    / "results/phase1/ugi3_targeted_exact_overlay_diagnostic_v1/result.json",
}


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
        default=REPO / "configs/route/phase1_ugi3_targeted_role_gap_evidence_audit_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi3_targeted_role_gap_evidence_audit_v1",
    )
    args = parser.parse_args()
    result, ledger = build_targeted_role_gap_evidence_audit(
        config_path=args.config,
        input_paths=INPUT_PATHS,
    )
    _write_atomic(args.output_dir / "product_impact_ledger.csv.gz", ledger)
    _write_atomic(
        args.output_dir / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
