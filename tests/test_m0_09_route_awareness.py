from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest

from forge.route.route_awareness import (
    RouteAwarenessError,
    analyze_agile_measured,
    build_route_awareness,
    build_route_catalog,
    load_route_config,
    sha256_file,
    write_route_awareness,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/m0_09_agile_component_routes.json"
VENDOR = REPO / "data/vendor"
RESULTS = REPO / "results/m0_09"


def _require_vendored_inputs() -> None:
    required = (
        VENDOR / "AGILE_smiles_with_value_group.csv",
        VENDOR / "agile_supplementary_information.pdf",
        VENDOR / "lnpdb_fc7c389.csv",
        RESULTS / "component_source_ledger.csv",
        RESULTS / "source_queue.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.skip(f"route-awareness inputs are not vendored: {missing}")


def test_agile_measured_grid_and_route_catalog_are_exact() -> None:
    _require_vendored_inputs()
    config = load_route_config(CONFIG)
    summary, _ = analyze_agile_measured(VENDOR / "AGILE_smiles_with_value_group.csv", config)
    routes = build_route_catalog(config)

    assert summary["measured_products"] == 1200
    assert summary["complete_cartesian_product"] is True
    assert summary["component_counts"] == {
        "amine": 20,
        "aldehyde_ester": 12,
        "isocyanide": 5,
    }
    assert summary["exact_extracted_route_components"] == {
        "aldehyde_ester": 11,
        "isocyanide": 5,
    }
    assert len(routes) == 24
    assert sum(len(route["steps"]) == 1 for route in routes) == 2
    assert sum(len(route["steps"]) == 2 for route in routes) == 22
    assert {
        route["component_label"]
        for route in routes
        if route["route_family_id"] == "agile_additional_isocyanide_two_step"
    } == {"C11", "C17"}
    assert all(
        route["forward_verification_status"] == "not_run_missing_qualified_template"
        for route in routes
    )
    assert all(route["execution_closure_status"] == "unknown" for route in routes)


def test_b5_source_discrepancy_cannot_be_silently_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["unresolved_aldehyde_ester_components"][0]["must_not_infer_route"] = False
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    with pytest.raises(RouteAwarenessError, match="must prohibit inferred route closure"):
        load_route_config(path)


def test_every_lnpdb_lipid_is_mapped_with_calibrated_evidence(tmp_path: Path) -> None:
    _require_vendored_inputs()
    generated = "2026-07-28T12:00:00+00:00"
    result, route_artifact, lipid_rows = build_route_awareness(
        CONFIG,
        VENDOR,
        RESULTS,
        generated_utc=generated,
    )

    summary = result["summary"]["lnpdb"]
    assert summary["lnpdb_records"] == 19797
    assert summary["unique_lipids"] == 12837
    assert summary["unique_annotated_components"] == 726
    assert summary["lipids_with_source_linked_l1_assembly"] == 1200
    assert summary["agile_reference_route_status_counts"] == {
        "all_components_source_resolved": 1100,
        "contains_source_discrepancy": 100,
    }
    assert summary["agile_reference_identity_labels"][
        "source_label_reconciled_to_measured_reference_raw_conflict_preserved"
    ] == ["C2"]
    assert summary["forward_verified_routes"] == 0
    assert summary["procurement_closed_components"] == 0
    assert summary["execution_closed_lipids"] == 0
    assert all(row["execution_closure_status"] == "unknown" for row in lipid_rows)

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = write_route_awareness(result, route_artifact, lipid_rows, first_dir)
    second = write_route_awareness(result, route_artifact, lipid_rows, second_dir)
    assert first["outputs"] == second["outputs"]
    for name in (
        "agile_component_routes.json",
        "lnpdb_lipid_route_ledger.csv.gz",
        "route_awareness_result.json",
    ):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()


def test_committed_route_awareness_outputs_are_hash_linked() -> None:
    result_path = RESULTS / "route_awareness_result.json"
    ledger_path = RESULTS / "lnpdb_lipid_route_ledger.csv.gz"
    routes_path = RESULTS / "agile_component_routes.json"
    if not all(path.exists() for path in (result_path, ledger_path, routes_path)):
        pytest.fail("committed M0-09 route-awareness artifacts are missing")

    result = json.loads(result_path.read_text())
    assert result["schema_version"] == "m0_09_lnpdb_route_awareness.v1"
    assert result["decision"]["map_every_lnpdb_lipid"] is True
    assert result["decision"]["whole_graph_generation_preserved"] is True
    assert result["decision"]["model_built"] is False
    assert result["outputs"]["agile_component_routes"]["sha256"] == sha256_file(routes_path)
    assert result["outputs"]["lnpdb_lipid_route_ledger"]["sha256"] == sha256_file(ledger_path)

    with gzip.open(ledger_path, mode="rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12837
    assert all(row["execution_closure_status"] == "unknown" for row in rows)
