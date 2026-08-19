#!/usr/bin/env python3
"""Build typed pre-prospective Ugi component and product synthesis values."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.value.ugi3_synthesis_value_audit import build_ugi3_synthesis_value_audit

REPO = Path(__file__).resolve().parents[1]

INPUT_PATHS = {
    "audit_source": REPO / "src/forge/value/ugi3_synthesis_value_audit.py",
    "generated_component_provenance": REPO
    / "results/phase1/ugi_postselection_provenance_audit_v1/component_provenance_ledger.csv.gz",
    "generated_product_provenance": REPO
    / "results/phase1/ugi_postselection_provenance_audit_v1/product_provenance_ledger.csv.gz",
    "head_terminal_config": REPO
    / "configs/route/phase1_ugi3_high_leverage_head_terminal_audit_v1.json",
    "head_terminal_product_ledger": REPO
    / "results/phase1/ugi3_high_leverage_head_terminal_audit_v1/product_impact_ledger.csv.gz",
    "head_terminal_result": REPO
    / "results/phase1/ugi3_high_leverage_head_terminal_audit_v1/result.json",
    "hybrid_assessment_ledger": REPO
    / "results/phase1/ugi3_hybrid_search/assessment_ledger.json.gz",
    "latest_product_impact": REPO
    / "results/phase1/ugi3_high_leverage_head_terminal_audit_v1/product_impact_ledger.csv.gz",
    "production_generator": REPO / "results/phase1/ugi_product_l1_production_generator_v1.json",
    "qualifier_source": REPO / "scripts/phase1_build_ugi3_synthesis_values.py",
    "role_gap_config": REPO / "configs/route/phase1_ugi3_targeted_role_gap_evidence_audit_v1.json",
    "role_gap_product_ledger": REPO
    / "results/phase1/ugi3_targeted_role_gap_evidence_audit_v1/product_impact_ledger.csv.gz",
    "role_gap_result": REPO / "results/phase1/ugi3_targeted_role_gap_evidence_audit_v1/result.json",
    "targeted_audit_config": REPO
    / "configs/route/phase1_ugi3_targeted_aldehyde_evidence_audit_v1.json",
    "targeted_audit_result": REPO
    / "results/phase1/ugi3_targeted_aldehyde_evidence_audit_v1/result.json",
    "targeted_product_ledger": REPO
    / "results/phase1/ugi3_targeted_aldehyde_evidence_audit_v1/product_closure_impact_ledger.csv.gz",
    "targeted_route_ledger": REPO
    / "results/phase1/ugi3_targeted_aldehyde_evidence_audit_v1/route_verification_ledger.csv.gz",
    "value_source": REPO / "src/forge/value/synthesis.py",
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

ROLE_GAP_INPUT_PATHS = {
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

HEAD_TERMINAL_INPUT_PATHS = {
    "audit_source": REPO / "src/forge/route/ugi3_high_leverage_head_terminals.py",
    "evidence_pack": REPO / "configs/route/phase1_ugi3_high_leverage_head_terminals_v1.json",
    "prior_product_impact": REPO
    / "results/phase1/ugi3_targeted_role_gap_evidence_audit_v1/product_impact_ledger.csv.gz",
    "prior_result": REPO / "results/phase1/ugi3_targeted_role_gap_evidence_audit_v1/result.json",
    "qualifier_source": REPO / "scripts/phase1_audit_ugi3_high_leverage_head_terminals.py",
    "readiness_ledger": REPO
    / "results/phase1/ugi3_production_registry_route_readiness/component_readiness_ledger.csv.gz",
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
        default=REPO / "configs/route/phase1_ugi3_synthesis_value_audit_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi3_synthesis_value_audit_v1",
    )
    args = parser.parse_args()
    result, component_ledger, product_ledger = build_ugi3_synthesis_value_audit(
        config_path=args.config,
        input_paths=INPUT_PATHS,
        targeted_audit_input_paths=TARGETED_AUDIT_INPUT_PATHS,
        role_gap_input_paths=ROLE_GAP_INPUT_PATHS,
        head_terminal_input_paths=HEAD_TERMINAL_INPUT_PATHS,
    )
    _write_atomic(args.output_dir / "component_synthesis_values.json.gz", component_ledger)
    _write_atomic(args.output_dir / "product_synthesis_values.json.gz", product_ledger)
    _write_atomic(
        args.output_dir / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
