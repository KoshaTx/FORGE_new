#!/usr/bin/env python3
"""Qualify the targeted exact-evidence overlay across the Ugi registry."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.route.ugi3_targeted_exact_overlay_diagnostic import (
    build_targeted_exact_overlay_diagnostic,
)

REPO = Path(__file__).resolve().parents[1]

HYBRID_INPUT_PATHS = {
    "component_dossier": REPO
    / "results/phase1/ugi3_complete_computational_dossiers/component_dossier_ledger.csv.gz",
    "component_dossier_result": REPO
    / "results/phase1/ugi3_complete_computational_dossiers/result.json",
    "component_program": REPO / "results/m0_09/agile_virtual_ugi3_component_program_ledger.csv.gz",
    "component_program_result": REPO / "results/m0_09/agile_virtual_ugi3_component_programs.json",
    "exact_adapter_source": REPO / "src/forge/route/ugi3_exact_evidence_source.py",
    "exact_evidence_ledger": REPO
    / "results/phase1/ugi3_exact_evidence_source/assessment_ledger.json.gz",
    "exact_evidence_result": REPO / "results/phase1/ugi3_exact_evidence_source/result.json",
    "hybrid_adapter_source": REPO / "src/forge/route/ugi3_hybrid_search.py",
    "l1_variant": REPO / "configs/assembly/ugi_variant.yaml",
    "oxidation_variant": REPO
    / "configs/route/variants/ugi3_upstream_primary_alcohol_oxidation_exact_source_v1.json",
    "planner_cache_source": REPO / "src/forge/route/planner_cache.py",
    "planner_contract_source": REPO / "src/forge/route/planner.py",
    "qualified_forward_source": REPO / "src/forge/route/qualified_forward.py",
    "readiness_config": REPO / "configs/route/phase1_ugi3_production_registry_route_readiness.json",
    "readiness_ledger": REPO
    / "results/phase1/ugi3_production_registry_route_readiness/component_readiness_ledger.csv.gz",
    "readiness_result": REPO
    / "results/phase1/ugi3_production_registry_route_readiness/result.json",
    "step_ledger": REPO
    / "results/phase1/ugi3_exact_source_forward_verification/step_verification_ledger.csv.gz",
    "step_result": REPO / "results/phase1/ugi3_exact_source_forward_verification/result.json",
    "terminal_procurement": REPO / "configs/route/m0_09_ugi3_virtual_terminal_procurement.json",
    "transfer_config": REPO / "configs/route/m0_09_hydrophobic_motif_transfer.json",
    "transfer_ledger": REPO / "results/m0_09/hydrophobic_motif_transfer_ledger.csv.gz",
    "transfer_result": REPO / "results/m0_09/hydrophobic_motif_transfer.json",
    "upstream_registry": REPO / "configs/route/phase1_ugi3_upstream_qualified_reactions_v1.json",
}

TARGETED_AUDIT_INPUT_PATHS = {
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

INPUT_PATHS = {
    "diagnostic_source": REPO / "src/forge/route/ugi3_targeted_exact_overlay_diagnostic.py",
    "hybrid_config": REPO / "configs/route/phase1_ugi3_hybrid_search.json",
    "hybrid_ledger": REPO / "results/phase1/ugi3_hybrid_search/assessment_ledger.json.gz",
    "hybrid_result": REPO / "results/phase1/ugi3_hybrid_search/result.json",
    "overlay_source": REPO / "src/forge/route/ugi3_targeted_exact_overlay.py",
    "qualifier_source": REPO / "scripts/phase1_qualify_ugi3_targeted_exact_overlay.py",
    "targeted_audit_config": REPO
    / "configs/route/phase1_ugi3_targeted_aldehyde_evidence_audit_v1.json",
    "targeted_audit_result": REPO
    / "results/phase1/ugi3_targeted_aldehyde_evidence_audit_v1/result.json",
    "targeted_product_ledger": REPO
    / "results/phase1/ugi3_targeted_aldehyde_evidence_audit_v1/product_closure_impact_ledger.csv.gz",
    "targeted_route_ledger": REPO
    / "results/phase1/ugi3_targeted_aldehyde_evidence_audit_v1/route_verification_ledger.csv.gz",
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
        default=REPO / "configs/route/phase1_ugi3_targeted_exact_overlay_diagnostic_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi3_targeted_exact_overlay_diagnostic_v1",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="forge-targeted-overlay-cache-") as cache_dir:
        result, ledger = build_targeted_exact_overlay_diagnostic(
            config_path=args.config,
            input_paths=INPUT_PATHS,
            hybrid_input_paths=HYBRID_INPUT_PATHS,
            targeted_audit_input_paths=TARGETED_AUDIT_INPUT_PATHS,
            cache_root=Path(cache_dir),
        )
    _write_atomic(args.output_dir / "assessment_ledger.json.gz", ledger)
    _write_atomic(
        args.output_dir / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
