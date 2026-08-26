from __future__ import annotations

import csv
from pathlib import Path

import pytest

from forge.corpus.lnpdb import LNPDB_FIELDS, LNPDBDataError, load_lnpdb

REPO = Path(__file__).resolve().parents[1]
LNPDB = REPO / "data/vendor/lnpdb_fc7c389.csv"


def _source_row(**updates: str) -> dict[str, str]:
    row = {field: "NA" for field in LNPDB_FIELDS}
    row.update(
        {
            "Index": "1",
            "LNP_ID": "LNP_1",
            "Experiment_ID": "study",
            "Formulation_ID": "formulation",
            "IL_name": "lipid",
            "IL_SMILES": "CCN",
            "Model_type": "HeLa",
            "Experiment_value": "1.25",
            "Publication_PMID": "123",
            "IL_head_name": "A1",
            "IL_head_SMILES": "CN",
        }
    )
    row.update(updates)
    return row


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LNPDB_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_typed_catalogue_normalizes_missing_components_and_indexes_studies(
    tmp_path: Path,
) -> None:
    path = tmp_path / "lnpdb.csv"
    _write(path, [_source_row()])

    catalogue = load_lnpdb(path)
    row = catalogue.record("LNP_1")
    assert catalogue.study("study") == (row,)
    assert row.index == 1
    assert row.head.name == "A1"
    assert row.linker.name is None
    assert row.tail(1).smiles is None
    assert row.experiment_value == pytest.approx(1.25)


def test_typed_catalogue_rejects_duplicate_source_identity(tmp_path: Path) -> None:
    path = tmp_path / "lnpdb.csv"
    _write(path, [_source_row(), _source_row(Index="2")])

    with pytest.raises(LNPDBDataError, match="source record is duplicated"):
        load_lnpdb(path)


@pytest.mark.needs_vendor
def test_vendor_catalogue_exposes_shared_potency_and_reaction_studies() -> None:
    catalogue = load_lnpdb(LNPDB)

    assert len(catalogue.rows) == 19797
    assert len(catalogue.study("YX_2024")) == 2400
    assert len(catalogue.study("BL_2023")) == 773
    assert len(catalogue.study("LX_2024")) == 851
    agile = catalogue.study("YX_2024")[0]
    assert agile.head.name == "A1"
    assert agile.tail(1).name == "B1"
    assert agile.tail(2).name == "C1"
