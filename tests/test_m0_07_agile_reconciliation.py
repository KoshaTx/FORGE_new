from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import pytest

from forge.bio.agile_reconciliation import (
    AgileReconciliationError,
    _canonical,
    _load_config,
    _normalized_label,
    _render_csv_gzip,
    _source_matrix,
    _verify_input,
)

REPO = Path(__file__).resolve().parents[1]


def test_component_labels_normalize_both_orders() -> None:
    assert _normalized_label("A17B5C3") == "A17B5C3"
    assert _normalized_label("A17C3B5") == "A17B5C3"


def test_model_canonicalization_intentionally_removes_alkene_stereo() -> None:
    trans = "CCCCCCC/C=C/C(=O)OCCCCCC=O"
    cis = "CCCCCCC/C=C\\C(=O)OCCCCCC=O"

    assert _canonical(trans, isomeric=True, label="trans") != _canonical(
        cis, isomeric=True, label="cis"
    )
    assert _canonical(trans, isomeric=False, label="trans") == _canonical(
        cis, isomeric=False, label="cis"
    )


def test_gzip_rendering_is_byte_deterministic() -> None:
    rows = [{"a": "1", "b": "2"}]
    first = _render_csv_gzip(rows, ("a", "b"))
    second = _render_csv_gzip(rows, ("a", "b"))

    assert first == second
    assert gzip.decompress(first) == b"a,b\n1,2\n"


def test_hash_mismatch_fails_loudly(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("x\n")

    with pytest.raises(AgileReconciliationError, match="hash mismatch"):
        _verify_input(source, "0" * 64, "source")


def test_config_rejects_single_graph_mixture_training(tmp_path: Path) -> None:
    config = json.loads((REPO / "configs/bio/m0_07_agile_label_reconciliation.json").read_text())
    config["policy"]["mixture_handling"] = "train_as_single_graph"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    with pytest.raises(AgileReconciliationError, match="mixture_handling"):
        _load_config(path)


@pytest.mark.needs_vendor
def test_official_source_workbook_contains_both_1200_label_matrices() -> None:
    workbook = REPO / "data/vendor/agile_article_source_data.xlsx"
    if not workbook.exists():
        pytest.skip("AGILE official source workbook is not vendored")

    hela = _source_matrix(
        workbook,
        sheet_name="Figure 2",
        header_row=3,
        first_data_row=4,
        first_head_column=2,
    )
    raw = _source_matrix(
        workbook,
        sheet_name="Supplementary Figure 21",
        header_row=2,
        first_data_row=3,
        first_head_column=3,
    )

    assert len(hela) == len(raw) == 1200
    assert hela["A1B1C1"] == pytest.approx(4.05668942)
    assert raw["A1B1C1"] == pytest.approx(1.54475317)


@pytest.mark.needs_vendor
def test_frozen_reconciliation_contract_when_present() -> None:
    result_path = REPO / "results/m0_07/agile_label_reconciliation.json"
    if not result_path.exists():
        pytest.skip("M0-07 AGILE reconciliation result has not been generated")
    result = json.loads(result_path.read_text())

    assert result["schema_version"] == "m0_07_agile_reconciliation.v1"
    assert result["status"] == "completed_blocking_source_reconciliation"
    assert result["summary"]["nominal_experimental_measurements"] == 1200
    assert result["summary"]["curated_single_structure_records"] == 1100
    assert result["summary"]["excluded_mixture_measurements"] == 100
    assert result["summary"]["B5_pure_trans_graph_corrections"] == 100
    assert result["summary"]["unique_model_graphs"] == 1100
    assert all(result["adversarial_checks"].values())
    assert result["downstream_contract"]["raw_1200_row_file_allowed_for_oracle_fit"] is False

    for filename, metadata in result["artifacts"].items():
        path = REPO / "results/m0_07" / filename
        payload = path.read_bytes()
        assert len(payload) == metadata["bytes"]
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]

    with gzip.open(REPO / "results/m0_07/agile_oracle_curated.csv.gz", "rt") as handle:
        curated = list(csv.DictReader(handle))
    with gzip.open(REPO / "results/m0_07/agile_mixture_exclusions.csv.gz", "rt") as handle:
        excluded = list(csv.DictReader(handle))
    with gzip.open(REPO / "results/m0_07/agile_reconciliation_ledger.csv.gz", "rt") as handle:
        ledger = list(csv.DictReader(handle))

    assert len(curated) == 1100
    assert len(excluded) == 100
    assert len(ledger) == 1200
    assert not any("B4" in row["label"] for row in curated)
    assert all("B4" in row["label"] for row in excluded)
    assert all(row["structure_policy"] for row in curated)
    assert len({row["model_smiles"] for row in curated}) == 1100

    b5 = next(row for row in curated if row["label"] == "A1B5C1")
    assert "/" in b5["isomeric_smiles"]
    assert "\\" not in b5["isomeric_smiles"]

    raw = gzip.decompress(
        (REPO / "results/m0_07/agile_oracle_curated.csv.gz").read_bytes()
    ).decode()
    assert len(list(csv.DictReader(io.StringIO(raw)))) == 1100
