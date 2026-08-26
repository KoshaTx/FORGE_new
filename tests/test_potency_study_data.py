from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest

from forge.potency.study_data import (
    LEDGER_FIELDS,
    PotencyStudyDataError,
    StudyEndpoint,
    build_potency_study_corpus,
    load_potency_study_corpus,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/bio/phase1_potency_study_corpus_v2.json"
CHECKED = REPO / "results/phase1/potency_study_corpus_v2"


def _build(output: Path) -> tuple[Path, Path]:
    ledger = output / "observations.csv.gz"
    result = output / "result.json"
    build_potency_study_corpus(CONFIG, REPO, ledger_path=ledger, result_path=result)
    return ledger, result


def test_config_prohibits_cross_study_raw_label_pooling(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["policy"]["raw_cross_study_label_pooling"] = True
    changed = tmp_path / "config.json"
    changed.write_text(json.dumps(config))

    with pytest.raises(PotencyStudyDataError, match="scientific policy"):
        build_potency_study_corpus(
            changed,
            REPO,
            ledger_path=tmp_path / "observations.csv.gz",
            result_path=tmp_path / "result.json",
        )


@pytest.mark.needs_vendor
def test_corpus_is_lnpdb_only_deterministic_and_row_preserving(tmp_path: Path) -> None:
    first_ledger, first_result = _build(tmp_path / "first")
    second_ledger, second_result = _build(tmp_path / "second")

    assert first_ledger.read_bytes() == second_ledger.read_bytes()
    assert first_result.read_bytes() == second_result.read_bytes()
    result = json.loads(first_result.read_text())
    assert set(result["inputs"]) == {"lnpdb"}
    assert result["artifact"]["rows"] == 3616
    assert result["studies"]["YX_2024"]["rows"] == 2200
    assert result["studies"]["JC_2023"]["rows"] == 288
    assert result["studies"]["LM_2019"]["rows"] == 1128
    assert result["source_accounting"] == {
        "selected_lnpdb_rows": 3816,
        "model_observations": 3616,
        "excluded_observations": 200,
    }
    assert all(result["gates"].values())


@pytest.mark.needs_vendor
def test_checked_in_corpus_matches_the_authenticated_builder(tmp_path: Path) -> None:
    ledger, result = _build(tmp_path / "rebuilt")

    assert ledger.read_bytes() == (CHECKED / "observations.csv.gz").read_bytes()
    assert result.read_bytes() == (CHECKED / "result.json").read_bytes()


@pytest.mark.needs_vendor
def test_lnpdb_values_are_preserved_and_mixtures_are_absent(tmp_path: Path) -> None:
    ledger, _ = _build(tmp_path / "run")
    with gzip.open(ledger, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == LEDGER_FIELDS
        rows = list(reader)

    hela = next(
        row
        for row in rows
        if row["study_id"] == "YX_2024"
        and row["source_lipid_name"] == "A1_B1_C1"
        and row["endpoint"] == "HeLa"
    )
    assert float(hela["label_value"]) == pytest.approx(-0.24556731)
    assert hela["label_semantics"] == "lnpdb_within_study_endpoint_zscore"
    assert "reaction_program_id" not in hela
    assert "component_smiles_json" not in hela

    assert not any(
        row["study_id"] == "YX_2024" and "_B4_" in f"_{row['source_lipid_name']}_" for row in rows
    )


@pytest.mark.needs_vendor
def test_typed_loader_requires_a_study_endpoint_selection(tmp_path: Path) -> None:
    ledger, _ = _build(tmp_path / "run")
    corpus = load_potency_study_corpus(ledger)

    assert len(corpus.records(StudyEndpoint("YX_2024", "HeLa"))) == 1100
    assert len(corpus.records(StudyEndpoint("JC_2023", "HeLa"))) == 288
    assert len(corpus.records(StudyEndpoint("LM_2019", "HeLa"))) == 1080
    assert {row.label_semantics for row in corpus.records(StudyEndpoint("YX_2024", "HeLa"))} == {
        "lnpdb_within_study_endpoint_zscore"
    }
