from __future__ import annotations

from pathlib import Path

from forge.value.ugi3_fresh_pool_route_coverage_v3 import (
    HEAD_ROLE,
    HEAD_SMILES,
    _find_octadecylamine_record,
    build_fresh_pool_route_coverage_v3,
)
from forge.value.ugi3_fresh_pool_route_coverage import _load_json

REPO = Path(__file__).resolve().parents[1]


def test_octadecylamine_record_is_exact_and_current() -> None:
    payload = _load_json(
        REPO / "configs/route/m0_09_ugi3_virtual_terminal_procurement.json",
        label="virtual terminal procurement",
    )
    record = _find_octadecylamine_record(
        payload, assessment_as_of_utc="2026-08-03T01:15:00Z"
    )
    assert record["identity"]["inchi_key"] == "REYJJPSVUYRZGE-UHFFFAOYSA-N"
    assert record["current_item_level_procurement_closed"] is True


def test_frozen_v3_refresh_reproduces() -> None:
    result, component_ledger, product_ledger = build_fresh_pool_route_coverage_v3(
        REPO,
        REPO / "configs/route/phase1_ugi3_fresh_pool_route_coverage_v3.json",
    )
    summary = result["summary"]
    assert summary["changed_component_keys"] == [f"{HEAD_ROLE}\t{HEAD_SMILES}"]
    assert summary["target_occurrences"] == {f"{HEAD_ROLE}\t{HEAD_SMILES}": 9}
    assert summary["product_outcomes_after"]["complete"] == 172
    assert summary["newly_complete_products"] == 1
    assert component_ledger
    assert product_ledger
