from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest

from forge.corpus.r1_prime_audit import sha256_file
from forge.synthesis.terminals.ugi3_virtual_terminal_queue import (
    Ugi3VirtualTerminalQueueError,
    build_ugi3_virtual_terminal_queue,
    write_ugi3_virtual_terminal_queue,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/m0_09_agile_virtual_ugi3_terminal_queue.json"
COMPONENT_LEDGER = REPO / "results/m0_09/agile_virtual_ugi3_component_ledger.csv.gz"
PROGRAM_LEDGER = REPO / "results/m0_09/agile_virtual_ugi3_component_program_ledger.csv.gz"
SOURCE_ROUTES = REPO / "results/m0_09/agile_component_routes.json"
PROCUREMENT = REPO / "configs/route/m0_09_ugi3_virtual_terminal_procurement.json"
RESULT = REPO / "results/m0_09/agile_virtual_ugi3_terminal_queue.json"


def _production_paths() -> tuple[Path, ...]:
    return CONFIG, COMPONENT_LEDGER, PROGRAM_LEDGER, SOURCE_ROUTES, PROCUREMENT


def test_rejects_program_ledger_hash_mismatch(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["inputs"]["component_program_ledger"]["expected_sha256"] = "0" * 64
    bad_config = tmp_path / "config.json"
    bad_config.write_text(json.dumps(config))

    with pytest.raises(Ugi3VirtualTerminalQueueError, match="hash mismatch"):
        build_ugi3_virtual_terminal_queue(
            bad_config,
            *_production_paths()[1:],
        )


def test_writer_is_deterministic_for_committed_payloads(
    tmp_path: Path,
) -> None:
    result = json.loads(RESULT.read_text())
    details = result["artifacts"]["agile_virtual_ugi3_terminal_queue.csv.gz"]
    ledger = (REPO / details["path"]).read_bytes()

    write_ugi3_virtual_terminal_queue(result, ledger, tmp_path)
    first = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    write_ugi3_virtual_terminal_queue(result, ledger, tmp_path)
    second = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    assert first == second
    assert not list(tmp_path.glob(".*.tmp"))


def test_committed_queue_is_deduplicated_and_unresolved() -> None:
    result = json.loads(RESULT.read_text())
    summary = result["summary"]

    assert summary["terminal_candidates"] == 38
    assert summary["unresolved_terminal_candidates"] == 3
    assert summary["proposed_route_leaf_candidates"] == 33
    assert summary["unresolved_head_candidates"] == 5
    assert summary["terminal_classes"] == {
        "diol": 4,
        "fatty_acid": 14,
        "primary_alcohol": 6,
        "primary_amine": 9,
        "unresolved_amine_head": 5,
    }
    assert summary["candidates_with_source_vendor_claim_only"] == 23
    assert summary["candidates_with_current_accepted_procurement"] == 35
    assert result["claims_boundary"]["source_vendor_claim_is_current_availability"] is False
    assert result["claims_boundary"]["dependency_incidence_is_unique_product_coverage"] is False

    details = result["artifacts"]["agile_virtual_ugi3_terminal_queue.csv.gz"]
    ledger_path = REPO / details["path"]
    assert ledger_path.stat().st_size == details["bytes"]
    assert sha256_file(ledger_path) == details["sha256"]
    with gzip.open(ledger_path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 38
    assert len({row["canonical_smiles"] for row in rows}) == 38
    assert sum(row["current_accepted_procurement"] == "true" for row in rows) == 35
    assert sum(row["procurement_status"] == "unresolved" for row in rows) == 3
    diols = [row for row in rows if row["terminal_class"] == "diol"]
    assert len(diols) == 4
    assert all(int(row["dependent_component_count"]) == 14 for row in diols)
    assert all(row["current_accepted_procurement"] == "true" for row in diols)
    fatty_acids = [row for row in rows if row["terminal_class"] == "fatty_acid"]
    assert len(fatty_acids) == 14
    assert all(row["current_accepted_procurement"] == "true" for row in fatty_acids)
    primary_alcohols = [row for row in rows if row["terminal_class"] == "primary_alcohol"]
    assert len(primary_alcohols) == 6
    assert all(row["current_accepted_procurement"] == "true" for row in primary_alcohols)
    primary_amines = [row for row in rows if row["terminal_class"] == "primary_amine"]
    assert sum(row["current_accepted_procurement"] == "true" for row in primary_amines) == 8
    unresolved_heads = [row for row in rows if row["terminal_class"] == "unresolved_amine_head"]
    assert sum(row["current_accepted_procurement"] == "true" for row in unresolved_heads) == 3
    assert {
        row["canonical_smiles"] for row in rows if row["procurement_status"] == "unresolved"
    } == {
        "CCCCCCCC/C=C\\CCCCCCCCN",
        "CN(C)N",
        "NC1CN2CCC1CC2",
    }
    oleylamine = next(
        row for row in primary_amines if row["canonical_smiles"] == "CCCCCCCC/C=C\\CCCCCCCCN"
    )
    assert oleylamine["current_accepted_procurement"] == "false"
    assert (
        oleylamine["next_action"]
        == "confirm_current_us_stock_or_shipping_for_high_purity_exact_item"
    )
