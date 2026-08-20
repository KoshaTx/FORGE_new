from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.data.r1_prime_audit import sha256_file
from forge.route.engine.planner import KnowledgeDisposition, KnowledgeResult, RouteTarget
from forge.route.terminals.ugi3_third_wave_head_terminals import (
    Ugi3ThirdWaveHeadTerminalError,
    build_third_wave_head_terminal_audit,
    load_third_wave_head_terminal_overlay,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi3_third_wave_head_terminal_audit_v1.json"
INPUTS = {
    "audit_source": REPO / "src/forge/route/ugi3_third_wave_head_terminals.py",
    "evidence_pack": REPO / "configs/route/phase1_ugi3_third_wave_head_terminals_v1.json",
    "prior_product_impact": REPO
    / "results/phase1/ugi3_second_wave_head_terminal_audit_v1/product_impact_ledger.csv.gz",
    "prior_result": REPO / "results/phase1/ugi3_second_wave_head_terminal_audit_v1/result.json",
    "qualifier_source": REPO / "scripts/phase1_audit_ugi3_third_wave_head_terminals.py",
    "readiness_ledger": REPO
    / "results/phase1/ugi3_production_registry_route_readiness/component_readiness_ledger.csv.gz",
    "supplier_page": REPO
    / "data/source_cache/phase1_ugi3_third_wave_head_terminals"
    / "chemimpex_03599_2026-08-01.html",
}


class MissingSource:
    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        return KnowledgeResult(
            disposition=KnowledgeDisposition.MISSING_KNOWLEDGE,
            evidence=(),
            detail=f"delegated missing target: {target.canonical_smiles}",
        )


def test_third_wave_audit_is_reproducible_and_archived() -> None:
    first, first_ledger = build_third_wave_head_terminal_audit(
        config_path=CONFIG,
        input_paths=INPUTS,
    )
    second, second_ledger = build_third_wave_head_terminal_audit(
        config_path=CONFIG,
        input_paths=INPUTS,
    )

    assert first == second
    assert first_ledger == second_ledger
    assert first["summary"]["registry_complete_components_after"] == 50
    assert first["summary"]["generated_products_complete_after"] == 45
    assert first["inputs"]["supplier_page"]["sha256"] == (
        "b8473f5749707bdb9adecca9a017132333d7aa14607ad3a7efc60006b0f9b518"
    )
    assert first["policy"]["production_synthesis_guidance"] is False


def test_supplier_archive_tampering_fails_closed(tmp_path: Path) -> None:
    source = INPUTS["supplier_page"].read_text()
    supplier_page = tmp_path / "supplier.html"
    supplier_page.write_text(source.replace("Ships Today", "Availability unassessed"))
    evidence = json.loads(INPUTS["evidence_pack"].read_text())
    evidence["records"][0]["source_asset"] = str(supplier_page)
    evidence["records"][0]["source_asset_sha256"] = sha256_file(supplier_page)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    inputs = {
        **INPUTS,
        "supplier_page": supplier_page,
        "evidence_pack": evidence_path,
    }
    config = json.loads(CONFIG.read_text())
    for label in ("supplier_page", "evidence_pack"):
        config["inputs"][label] = {
            "asset": str(inputs[label]),
            "expected_sha256": sha256_file(inputs[label]),
        }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    with pytest.raises(Ugi3ThirdWaveHeadTerminalError, match="required field"):
        build_third_wave_head_terminal_audit(
            config_path=config_path,
            input_paths=inputs,
        )


def test_third_wave_overlay_changes_only_cyclohexylamine() -> None:
    overlay, metadata = load_third_wave_head_terminal_overlay(
        base_source=MissingSource(),
        audit_config_path=CONFIG,
        audit_input_paths=INPUTS,
        stored_audit_result_path=REPO
        / "results/phase1/ugi3_third_wave_head_terminal_audit_v1/result.json",
        stored_product_ledger_path=REPO
        / "results/phase1/ugi3_third_wave_head_terminal_audit_v1/product_impact_ledger.csv.gz",
    )

    assert metadata["selected_exact_head_terminals"] == 1
    assert overlay.lookup(RouteTarget("amine_head", "NC1CCCCC1")).disposition is (
        KnowledgeDisposition.TERMINAL
    )
    assert overlay.lookup(RouteTarget("amine_head", "CCCN")).disposition is (
        KnowledgeDisposition.MISSING_KNOWLEDGE
    )
