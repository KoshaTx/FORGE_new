from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.data.r1_prime_audit import sha256_file
from forge.route.planner import KnowledgeDisposition, KnowledgeResult, RouteTarget
from forge.route.ugi3_second_wave_head_terminals import (
    Ugi3SecondWaveHeadTerminalError,
    build_second_wave_head_terminal_audit,
    load_second_wave_head_terminal_overlay,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi3_second_wave_head_terminal_audit_v1.json"
INPUTS = {
    "audit_source": REPO / "src/forge/route/ugi3_second_wave_head_terminals.py",
    "evidence_pack": REPO / "configs/route/phase1_ugi3_second_wave_head_terminals_v1.json",
    "prior_product_impact": REPO
    / "results/phase1/ugi3_high_leverage_head_terminal_audit_v1/product_impact_ledger.csv.gz",
    "prior_result": REPO / "results/phase1/ugi3_high_leverage_head_terminal_audit_v1/result.json",
    "qualifier_source": REPO / "scripts/phase1_audit_ugi3_second_wave_head_terminals.py",
    "readiness_ledger": REPO
    / "results/phase1/ugi3_production_registry_route_readiness/component_readiness_ledger.csv.gz",
}


def _tampered_inputs(tmp_path: Path, mutate) -> tuple[Path, dict[str, Path]]:
    evidence = json.loads(INPUTS["evidence_pack"].read_text())
    mutate(evidence)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    inputs = {**INPUTS, "evidence_pack": evidence_path}
    config = json.loads(CONFIG.read_text())
    config["inputs"]["evidence_pack"] = {
        "asset": str(evidence_path),
        "expected_sha256": sha256_file(evidence_path),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return config_path, inputs


def test_second_wave_audit_is_reproducible() -> None:
    first, first_ledger = build_second_wave_head_terminal_audit(
        config_path=CONFIG,
        input_paths=INPUTS,
    )
    second, second_ledger = build_second_wave_head_terminal_audit(
        config_path=CONFIG,
        input_paths=INPUTS,
    )

    assert first == second
    assert first_ledger == second_ledger
    assert first["summary"]["registry_complete_components_after"] == 49
    assert first["summary"]["generated_products_complete_after"] == 42
    assert first["policy"]["production_synthesis_guidance"] is False


def test_past_dated_listing_cannot_be_promoted(tmp_path: Path) -> None:
    config, inputs = _tampered_inputs(
        tmp_path,
        lambda evidence: evidence["records"][0].update({"availability": "past_dated_availability"}),
    )

    with pytest.raises(Ugi3SecondWaveHeadTerminalError, match="exact and current"):
        build_second_wave_head_terminal_audit(
            config_path=config,
            input_paths=inputs,
        )


class MissingSource:
    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        return KnowledgeResult(
            disposition=KnowledgeDisposition.MISSING_KNOWLEDGE,
            evidence=(),
            detail=f"delegated missing target: {target.canonical_smiles}",
        )


def test_second_wave_overlay_changes_only_propylamine() -> None:
    overlay, metadata = load_second_wave_head_terminal_overlay(
        base_source=MissingSource(),
        audit_config_path=CONFIG,
        audit_input_paths=INPUTS,
        stored_audit_result_path=REPO
        / "results/phase1/ugi3_second_wave_head_terminal_audit_v1/result.json",
        stored_product_ledger_path=REPO
        / "results/phase1/ugi3_second_wave_head_terminal_audit_v1/product_impact_ledger.csv.gz",
    )

    assert metadata["selected_exact_head_terminals"] == 1
    assert overlay.lookup(RouteTarget("amine_head", "CCCN")).disposition is (
        KnowledgeDisposition.TERMINAL
    )
    assert overlay.lookup(RouteTarget("amine_head", "NC1CCCCC1")).disposition is (
        KnowledgeDisposition.MISSING_KNOWLEDGE
    )
