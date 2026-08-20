from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest

from forge.corpus.r0_reconciliation import (
    build_r0_reconciliation,
    sha256_file,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.needs_vendor
def test_constitutional_r0_reconciliation_contract() -> None:
    result, corpus_payload, ledger_payload = build_r0_reconciliation(
        REPO / "configs/corpus/m0_03_r0_reconciliation.json",
        REPO,
    )

    assert result["schema_version"] == "m0_03_r0_reconciliation_result.v1"
    assert result["summary"]["input_rows"] == 15_433
    assert result["summary"]["output_constitutions"] == 15_229
    assert result["summary"]["rows_collapsed"] == 204
    assert result["summary"]["input_duplicate_groups"] == 184
    assert result["summary"]["duplicate_group_size_distribution"] == {
        "2": 164,
        "3": 20,
    }
    assert result["summary"]["b4_mixture_rows_excluded_from_single_structure_weight"] == 100
    assert result["summary"]["b5_rows_corrected_to_pure_trans_provenance"] == 100
    assert result["summary"]["b4_constitutions_with_exact_b5_support"] == 100
    assert result["summary"]["mixture_only_constitutions"] == 0
    assert result["summary"]["model_graph_stereochemical_labels"] == 0
    assert len(corpus_payload) == result["artifacts"]["r0_constitutional.csv.gz"]["bytes"]
    assert len(ledger_payload) == result["artifacts"]["r0_reconciliation_ledger.csv.gz"]["bytes"]


@pytest.mark.needs_vendor
def test_frozen_constitutional_r0_has_unique_graphs_and_audited_b4_b5_actions() -> None:
    corpus_path = REPO / "results/m0_03/r0_constitutional.csv.gz"
    ledger_path = REPO / "results/m0_03/r0_reconciliation_ledger.csv.gz"
    if not corpus_path.exists() or not ledger_path.exists():
        pytest.skip("constitutional R0 artifacts have not been generated")
    with gzip.open(corpus_path, "rt", newline="") as handle:
        corpus = list(csv.DictReader(handle))
    with gzip.open(ledger_path, "rt", newline="") as handle:
        ledger = list(csv.DictReader(handle))

    constitutions = [row["canonical_constitutional_smiles"] for row in corpus]
    assert len(corpus) == 15_229
    assert len(constitutions) == len(set(constitutions))
    assert all(
        row["canonical_isomeric_smiles"] == constitution
        for row, constitution in zip(corpus, constitutions, strict=True)
    )
    assert all(row["r0_pretraining_eligible"] == "True" for row in corpus)

    b4 = [
        row
        for row in ledger
        if row["agile_action"] == "exclude_cis_trans_mixture_from_single_graph_oracle"
    ]
    b5 = [row for row in ledger if row["agile_action"] == "correct_B5_to_pure_trans_graph"]
    assert len(ledger) == 15_433
    assert len(b4) == 100
    assert len(b5) == 100
    assert all(row["single_compound_structure_evidence"] == "False" for row in b4)
    assert all(row["retained_as_structure_only_mixture_association"] == "True" for row in b4)
    assert all(row["reconciled_single_compound_isomeric_smiles"] == "" for row in b4)
    assert all(row["single_compound_structure_evidence"] == "True" for row in b5)
    assert all(row["reconciled_single_compound_isomeric_smiles"] for row in b5)
    assert {row["reconciled_r0_structure_id"] for row in b4} == {
        row["reconciled_r0_structure_id"] for row in b5
    }


@pytest.mark.needs_vendor
def test_frozen_r0_reconciliation_artifact_hashes() -> None:
    result_path = REPO / "results/m0_03/r0_reconciliation.json"
    if not result_path.exists():
        pytest.skip("R0 reconciliation result has not been generated")
    result = json.loads(result_path.read_text())

    for record in result["inputs"].values():
        path = REPO / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert sha256_file(path) == record["sha256"]
    config_path = REPO / result["configuration"]["path"]
    assert sha256_file(config_path) == result["configuration"]["sha256"]
    for name, record in result["artifacts"].items():
        path = REPO / "results/m0_03" / name
        assert path.stat().st_size == record["bytes"]
        assert sha256_file(path) == record["sha256"]
