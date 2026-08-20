from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem

from forge.route.sources.lnpdb_head_transfer import (
    _ledger_row,
    build_lnpdb_head_transfer,
    sha256_file,
)

REPO = Path(__file__).resolve().parents[1]
AMINE_QUERY = Chem.MolFromSmarts("[NX3;H2,H1]")
RING_SUPPORT = {
    "definitions": {"macrocycle": "ring size at least 9"},
    "decision": {
        "primary_ugi_automatic_support": {
            "acyclic_products": True,
            "single_ring_sizes": [5, 6],
            "maximum_rings": 1,
            "fused_allowed": False,
            "spiro_allowed": False,
            "bridgehead_allowed": False,
            "macrocycle_allowed": False,
        }
    },
}


def _row(component_id: str, smiles: str) -> dict[str, str]:
    return {
        "component_id": component_id,
        "canonical_smiles": smiles,
        "parse_status": "parsed",
        "unique_lipid_count": "3",
        "source_count": "2",
        "source_ids_json": '["source-a","source-b"]',
        "source_pmids_json": '["1","2"]',
        "route_evidence_status": "not_assessed",
        "procurement_evidence_status": "not_assessed",
        "execution_closure_status": "not_assessed",
    }


@pytest.mark.parametrize(
    ("smiles", "disposition"),
    (
        ("NCCN1CCCCC1", "lnpdb_ugi_head_transfer_candidate"),
        ("NCC1CCCCCCCCCCC1", "outside_primary_ring_topology"),
        ("CN(C)C", "without_qualified_ugi_amine"),
    ),
)
def test_head_transfer_requires_ugi_handle_and_bounded_ring_topology(
    smiles: str,
    disposition: str,
) -> None:
    record = _ledger_row(
        _row("test-head", smiles),
        amine_query=AMINE_QUERY,
        allowed_sites=frozenset({1, 2}),
        multiplicity_semantics="symmetry_distinct_required_handle_matches",
        agile_heads=set(),
        ring_support=RING_SUPPORT,
    )

    assert record["disposition"] == disposition
    assert not record["automatic_candidate_lock_admission"]


def test_exact_agile_identity_is_provenance_not_admission() -> None:
    smiles = "NCCN1CCCCC1"
    record = _ledger_row(
        _row("known-head", smiles),
        amine_query=AMINE_QUERY,
        allowed_sites=frozenset({1, 2}),
        multiplicity_semantics="symmetry_distinct_required_handle_matches",
        agile_heads={Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)},
        ring_support=RING_SUPPORT,
    )

    assert record["disposition"] == "exact_agile_head"
    assert not record["automatic_candidate_lock_admission"]


@pytest.mark.needs_vendor
def test_full_lnpdb_head_transfer_contract() -> None:
    result, ledger = build_lnpdb_head_transfer(
        REPO / "configs/route/m0_09_lnpdb_head_transfer.json",
        REPO,
    )

    assert result["schema_version"] == "m0_09_lnpdb_head_transfer_result.v1"
    assert result["summary"]["head_rows"] == 408
    assert result["summary"]["parsed_head_rows"] == 396
    assert result["summary"]["unique_agile_heads"] == 20
    assert result["summary"]["disposition_counts"] == {
        "exact_agile_head": 20,
        "invalid_source_head_structure": 12,
        "lnpdb_ugi_head_transfer_candidate": 259,
        "outside_primary_ring_topology": 62,
        "without_qualified_ugi_amine": 55,
    }
    assert result["summary"]["transfer_candidate_ring_size_counts"] == {
        "[]": 130,
        "[5]": 22,
        "[6]": 107,
    }
    assert result["summary"]["automatically_admitted_candidates"] == 0
    assert len(ledger) == 408
    assert sum(row["disposition"] == "lnpdb_ugi_head_transfer_candidate" for row in ledger) == 259


@pytest.mark.needs_vendor
def test_frozen_lnpdb_head_transfer_artifacts_when_present() -> None:
    result_path = REPO / "results/m0_09/lnpdb_head_transfer.json"
    ledger_path = REPO / "results/m0_09/lnpdb_head_transfer_ledger.csv.gz"
    if not result_path.exists() or not ledger_path.exists():
        pytest.skip("LNPDB head-transfer artifacts have not been generated")
    result = json.loads(result_path.read_text())

    assert result["schema_version"] == "m0_09_lnpdb_head_transfer_result.v1"
    assert result["summary"]["automatically_admitted_candidates"] == 0
    for record in result["inputs"].values():
        path = REPO / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert sha256_file(path) == record["sha256"]
    config_path = REPO / result["configuration"]["path"]
    assert sha256_file(config_path) == result["configuration"]["sha256"]
