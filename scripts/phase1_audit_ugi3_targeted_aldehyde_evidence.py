#!/usr/bin/env python3
"""Audit targeted exact evidence for high-impact Ugi aldehydes."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.route.evidence.ugi3_targeted_aldehyde_evidence import (
    build_targeted_aldehyde_evidence_audit,
)

REPO = Path(__file__).resolve().parents[1]

INPUT_PATHS = {
    "audit_source": REPO / "src/forge/route/ugi3_targeted_aldehyde_evidence.py",
    "cli_source": REPO / "scripts/phase1_audit_ugi3_targeted_aldehyde_evidence.py",
    "evidence_pack": REPO / "configs/route/phase1_ugi3_targeted_aldehyde_evidence_v1.json",
    "kovalerchik_source": REPO
    / "data/source_cache/phase1_targeted_l2/KOVALERCHIK_2022/marinedrugs-20-00265-v2.pdf",
    "mo_source": REPO
    / "data/source_cache/phase1_targeted_l2/TYPHONOSIDES/supporting_information.pdf",
    "busta_source": REPO
    / "data/source_cache/phase1_targeted_l2/BUSTA_2016/BustaEtAl_2016_Phytochem.pdf",
    "route_readiness_ledger": REPO
    / "results/phase1/ugi3_production_registry_route_readiness/component_readiness_ledger.csv.gz",
    "route_readiness_result": REPO
    / "results/phase1/ugi3_production_registry_route_readiness/result.json",
    "product_gap_ledger": REPO
    / "results/phase1/ugi3_l2_coverage_priority_audit_v1/product_gap_ledger.csv.gz",
    "priority_result": REPO / "results/phase1/ugi3_l2_coverage_priority_audit_v1/result.json",
    "terminal_procurement": REPO / "configs/route/m0_09_ugi3_virtual_terminal_procurement.json",
    "upstream_registry": REPO / "configs/route/phase1_ugi3_upstream_qualified_reactions_v1.json",
    "oxidation_variant": REPO
    / "configs/route/variants/ugi3_upstream_primary_alcohol_oxidation_exact_source_v1.json",
    "qualified_forward_source": REPO / "src/forge/route/qualified_forward.py",
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
        default=REPO / "configs/route/phase1_ugi3_targeted_aldehyde_evidence_audit_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi3_targeted_aldehyde_evidence_audit_v1",
    )
    args = parser.parse_args()
    result, route_ledger, product_ledger = build_targeted_aldehyde_evidence_audit(
        config_path=args.config,
        input_paths=INPUT_PATHS,
    )
    _write_atomic(args.output_dir / "route_verification_ledger.csv.gz", route_ledger)
    _write_atomic(args.output_dir / "product_closure_impact_ledger.csv.gz", product_ledger)
    _write_atomic(
        args.output_dir / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
