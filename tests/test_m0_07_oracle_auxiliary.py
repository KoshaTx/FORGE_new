from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import pytest

from forge.potency.oracle_auxiliary import (
    AuxiliarySupervisionError,
    _canonical,
    _load_config,
    _render_csv_gzip,
    filter_auxiliary_training_rows,
    run_auxiliary_supervision_audit,
)

REPO = Path(__file__).resolve().parents[1]


def test_auxiliary_config_prohibits_raw_label_pooling(tmp_path: Path) -> None:
    config = json.loads((REPO / "configs/bio/m0_07_auxiliary_supervision.json").read_text())
    config["policy"]["allow_cross_study_raw_label_pooling"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    with pytest.raises(AuxiliarySupervisionError, match="pooling"):
        _load_config(path)


def test_auxiliary_canonicalization_removes_alkene_stereo() -> None:
    cis = _canonical("CCCC/C=C\\CCCC", label="cis")
    trans = _canonical("CCCC/C=C/CCCC", label="trans")
    assert cis == trans


def test_auxiliary_gzip_is_byte_deterministic() -> None:
    row = {
        "study_id": "study",
        "pmid": "1",
        "source_record_id": "record",
        "source_lipid_name": "lipid",
        "model_type": "HeLa",
        "label_value": "0",
        "label_semantics": "zscore",
        "raw_source_product_smiles": "CC",
        "model_smiles": "CC",
        "model_identity_action": "source_product_retained",
        "chemistry_relation": "native",
        "component_ids_json": "{}",
        "component_smiles_json": "{}",
        "source_review_overrides_json": "{}",
        "forward_unique_product_count": "1",
        "forward_source_product_verified": "true",
        "forward_model_product_verified": "true",
        "paired_supervision_eligible": "true",
        "exact_agile_product_overlap": "false",
    }
    first = _render_csv_gzip([row])
    second = _render_csv_gzip([row])
    assert first == second
    assert len(list(csv.DictReader(io.StringIO(gzip.decompress(first).decode())))) == 1


def test_auxiliary_filter_removes_each_fold_leakage_class() -> None:
    def row(product: str, head: str, aldehyde: str, isocyanide: str) -> dict[str, str]:
        return {
            "model_smiles": product,
            "component_smiles_json": json.dumps(
                {
                    "head": head,
                    "aldehyde": aldehyde,
                    "isocyanide": isocyanide,
                }
            ),
        }

    rows = [
        row("CC", "CN", "CC=O", "[C-]#[N+]C"),
        row("c1ccccc1", "CN", "CCC=O", "[C-]#[N+]CC"),
        row("CCC", "NCC", "CCCC=O", "[C-]#[N+]CCC"),
        row("CCCC", "NCCC", "CCCCC=O", "[C-]#[N+]CCCC"),
        row("CCCCC", "NCCCC", "CCCCCC=O", "[C-]#[N+]CCCCC"),
    ]
    retained, reasons = filter_auxiliary_training_rows(
        rows,
        exact_test_products={"CC"},
        held_scaffolds={"c1ccccc1"},
        held_components={"head": {"CCN"}},
        held_component_pairs={
            ("aldehyde", "isocyanide"): {
                ("CCCCC=O", "[C-]#[N+]CCCC"),
            }
        },
    )

    assert [row["model_smiles"] for row in retained] == ["CCCCC"]
    assert reasons == {
        "exact_test_product": 1,
        "held_component": 1,
        "held_component_pair": 1,
        "held_scaffold": 1,
        "retained": 1,
    }


@pytest.mark.needs_vendor
def test_auxiliary_audit_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    config = REPO / "configs/bio/m0_07_auxiliary_supervision.json"

    run_auxiliary_supervision_audit(config, first, REPO)
    run_auxiliary_supervision_audit(config, second, REPO)

    for filename in (
        "oracle_auxiliary_supervision.json",
        "oracle_auxiliary_records.csv.gz",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


@pytest.mark.needs_vendor
def test_frozen_auxiliary_supervision_artifact_when_present() -> None:
    result_path = REPO / "results/m0_07/oracle_auxiliary_supervision.json"
    if not result_path.exists():
        pytest.skip("M0-07 auxiliary supervision result has not been generated")
    result = json.loads(result_path.read_text())

    assert result["schema_version"] == "m0_07_auxiliary_supervision.v1"
    assert result["status"] == "completed_auxiliary_supervision_audit"
    assert result["decision"]["raw_label_pooling_allowed"] is False
    assert (
        result["inputs"]["config"]["sha256"]
        == hashlib.sha256(
            (REPO / "configs/bio/m0_07_auxiliary_supervision.json").read_bytes()
        ).hexdigest()
    )
    assert result["studies"]["JC_2023"]["rows"] == 288
    assert result["studies"]["JC_2023"]["unique_products"] == 288
    assert result["studies"]["JC_2023"]["exact_agile_product_overlap"] == 32
    assert result["studies"]["JC_2023"]["model_identity_actions"] == {
        "source_component_forward_reconstruction": 18,
        "source_product_retained": 270,
    }
    assert result["studies"]["JC_2023"]["paired_supervision_eligible_rows"] == 288
    assert result["studies"]["JC_2023"]["source_review_identity_changes"] == 1
    assert result["studies"]["LM_2019"]["model_counts"] == {
        "BMDC": 36,
        "BMDM": 12,
        "HeLa": 1080,
    }
    assert result["studies"]["JC_2023"]["components"]["head"]["new_relative_to_agile_role"] == 0
    assert result["studies"]["JC_2023"]["components"]["aldehyde"]["new_relative_to_agile_role"] == 5
    assert (
        result["studies"]["JC_2023"]["components"]["isocyanide"]["new_relative_to_agile_role"] == 1
    )

    for filename, metadata in result["artifacts"].items():
        payload = (REPO / "results/m0_07" / filename).read_bytes()
        assert len(payload) == metadata["bytes"]
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]

    with gzip.open(
        REPO / "results/m0_07/oracle_auxiliary_records.csv.gz",
        "rt",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    jc_rows = [row for row in rows if row["study_id"] == "JC_2023"]
    a3_rows = [row for row in jc_rows if json.loads(row["component_ids_json"])["head"] == "A3"]
    assert len(a3_rows) == 18
    assert all(
        row["model_identity_action"] == "source_component_forward_reconstruction"
        and row["raw_source_product_smiles"] != row["model_smiles"]
        and row["forward_model_product_verified"] == "true"
        for row in a3_rows
    )
    component_by_id: dict[str, set[str]] = {}
    for row in jc_rows:
        ids = json.loads(row["component_ids_json"])
        structures = json.loads(row["component_smiles_json"])
        component_by_id.setdefault(ids["isocyanide"], set()).add(structures["isocyanide"])
    assert component_by_id["C2"] == {"[C-]#[N+]CCCCCCCCCCCCCCCCCC"}
    assert component_by_id["C3"] == {"[C-]#[N+]CCCCCCCCCCC"}
