from __future__ import annotations

from pathlib import Path

from experiments.phase1.synthesis_guidance.sources.coverage.ugi3_fresh_pool_route_coverage_v2 import (
    ALDEHYDE_ROLE,
    ALDEHYDE_SMILES,
    HEAD_ROLE,
    HEAD_SMILES,
    _terminal_value,
    build_fresh_pool_route_coverage_v2,
)
from forge.synthesis.engine.planner import RouteTarget

REPO = Path(__file__).resolve().parents[1]


def test_exact_terminal_value_closes_without_scalar_probability() -> None:
    value = _terminal_value(
        target=RouteTarget(role=HEAD_ROLE, canonical_smiles=HEAD_SMILES),
        source_path=REPO / "configs/route/m0_09_ugi3_agile_head_procurement.json",
        evidence_id="test-head-terminal",
        locator="test#head",
    )
    assert value.route_complete is True
    assert value.protection_burden.count is None
    assert value.purification_burden.count is None


def test_frozen_v2_delta_reproduces() -> None:
    result, component_ledger, product_ledger = build_fresh_pool_route_coverage_v2(
        REPO,
        REPO / "configs/route/phase1_ugi3_fresh_pool_route_coverage_v2.json",
    )
    summary = result["summary"]
    assert summary["changed_component_keys"] == sorted(
        [f"{HEAD_ROLE}\t{HEAD_SMILES}", f"{ALDEHYDE_ROLE}\t{ALDEHYDE_SMILES}"]
    )
    assert summary["product_outcomes_after"]["complete"] == 171
    assert summary["newly_complete_products"] == 24
    assert component_ledger
    assert product_ledger
