from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from forge.bio.oracle_graph_corpus import (
    OracleGraphCorpusError,
    _load_json,
    run_oracle_graph_corpus_audit,
)

REPO = Path(__file__).resolve().parents[1]


def test_graph_corpus_config_freezes_leakage_policy() -> None:
    config = _load_json(
        REPO / "configs/bio/m0_07_oracle_graph_corpus.json",
        "config",
    )

    assert config["pretraining_policy"] == {
        "exclude_every_exact_oracle_constitution_globally": True,
        "deduplicate_retained_r0_by_constitution": True,
        "biological_labels_used": False,
        "virtual_candidates_used": False,
        "virtual_candidate_role": "applicability_only",
    }
    assert config["representation"]["stereochemistry"] == "excluded"
    assert config["representation"]["truncation_allowed"] is False


@pytest.mark.needs_vendor
def test_graph_corpus_audit_is_byte_deterministic(tmp_path: Path) -> None:
    config = REPO / "configs/bio/m0_07_oracle_graph_corpus.json"
    first = tmp_path / "first"
    second = tmp_path / "second"

    run_oracle_graph_corpus_audit(config, first, REPO)
    run_oracle_graph_corpus_audit(config, second, REPO)

    for filename in (
        "oracle_graph_corpus_result.json",
        "oracle_graph_r0_pretraining.csv.gz",
        "oracle_graph_r0_exclusions.csv.gz",
        "oracle_graph_isomer_collapse.csv.gz",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


@pytest.mark.needs_vendor
def test_frozen_graph_corpus_artifact_when_present() -> None:
    result_path = REPO / "results/m0_07/oracle_graph_corpus_result.json"
    if not result_path.exists():
        pytest.skip("M0-07 graph-corpus artifact has not been generated")
    result = json.loads(result_path.read_text())

    assert result["schema_version"] == "m0_07_oracle_graph_corpus.v2"
    assert result["summary"]["curated_oracle_unique_constitutions"] == 1100
    assert result["summary"]["r0_rows_matching_any_oracle_constitution"] == 1220
    assert result["summary"]["r0_unique_oracle_constitutions_matched"] == 1100
    assert result["summary"]["r0_unique_constitutions"] == 15229
    assert result["summary"]["r0_constitutional_duplicate_groups"] == 184
    assert result["summary"]["r0_rows_collapsed_before_oracle_exclusion"] == 204
    assert result["summary"]["r0_duplicate_group_size_distribution"] == {
        "2": 164,
        "3": 20,
    }
    assert result["summary"]["r0_oracle_overlap_group_size_distribution"] == {
        "1": 1000,
        "2": 80,
        "3": 20,
    }
    assert result["summary"]["r0_retained_unique_constitutions"] == 14129
    assert result["summary"]["post_exclusion_oracle_constitution_overlap"] == 0
    assert result["summary"]["post_exclusion_oracle_inchi_connectivity_overlap"] == 0
    assert result["summary"]["provenance_only_filter_remaining_oracle_rows"] == 20
    assert result["summary"]["provenance_only_filter_remaining_oracle_constitutions"] == 20
    assert result["summary"]["retained_inchi_connectivity_collapse_groups"] == 1
    assert result["decision"]["virtual_candidates_authorized_for_pretraining"] is False
    assert result["decision"]["manifest_kernel_profile_authorized_for_encoder_support"] is False
    assert result["graph_profiles"]["curated_oracle"]["records"] == 1100
    assert result["graph_profiles"]["r0_input"]["records"] == 15433
    assert result["graph_profiles"]["r0_pretraining"]["records"] == 14129
    assert result["graph_profiles"]["virtual_applicability"]["records"] == 12276
    retained_profile = result["graph_profiles"]["r0_pretraining"]
    assert retained_profile["atom_count_distribution"]["maximum"] == 282
    assert retained_profile["records_at_most_64_atoms"] == 9602
    assert retained_profile["records_at_most_96_atoms"] == 12779
    assert retained_profile["records_above_96_atoms"] == 1350
    assert retained_profile["feature_vocabulary"]["elements"]["F"] == 172
    assert retained_profile["feature_vocabulary"]["elements"]["Si"] == 1365
    assert retained_profile["feature_vocabulary"]["bond_types"]["TRIPLE"] == 2198
    assert set(retained_profile["feature_vocabulary"]["formal_charges"]) == {
        "-1",
        "0",
        "1",
    }
    assert result["r0_stereo_profile"]["potential_tetrahedral_records"] == 11478
    assert result["r0_stereo_profile"]["potential_double_bond_records"] == 4851
    assert result["r0_stereo_profile"]["potential_atom_or_bond_stereo_records"] == 11991
    assert result["r0_stereo_profile"]["assigned_atom_or_bond_modern_records"] == 4936
    manifest_audit = result["training_manifest_audit"]
    assert manifest_audit["profile_matches_pinned_r0"] is False
    assert manifest_audit["actual_pinned_r0"]["heavy_atoms"]["minimum"] == 14
    assert manifest_audit["actual_pinned_r0"]["heavy_atoms"]["median"] == 53
    assert manifest_audit["actual_pinned_r0"]["heavy_atoms"]["maximum"] == 282
    assert manifest_audit["actual_pinned_r0"]["net_charged_records"] == 509

    with gzip.open(
        REPO / "results/m0_07/oracle_graph_r0_pretraining.csv.gz",
        "rt",
        newline="",
    ) as handle:
        retained = list(csv.DictReader(handle))
    with gzip.open(
        REPO / "results/m0_07/oracle_graph_r0_exclusions.csv.gz",
        "rt",
        newline="",
    ) as handle:
        excluded = list(csv.DictReader(handle))
    assert len(retained) == 14129
    assert len(excluded) == 1220
    assert list(retained[0]) == [
        "graph_id",
        "constitutional_smiles",
        "atom_count",
        "undirected_bond_count",
        "directed_edge_count",
    ]
    assert not {
        "label",
        "hela",
        "raw",
        "observed_source_ids",
        "component_holdout_groups_json",
        "region_annotations_json",
    } & set(retained[0])
    assert not (
        {row["constitutional_smiles"] for row in retained}
        & {row["constitutional_smiles"] for row in excluded}
    )
    with gzip.open(
        REPO / "results/m0_07/oracle_graph_isomer_collapse.csv.gz",
        "rt",
        newline="",
    ) as handle:
        collapses = list(csv.DictReader(handle))
    assert len(collapses) == 184
    assert sum(row["group_scope"] == "retained" for row in collapses) == 84
    assert sum(row["group_scope"] == "oracle_excluded" for row in collapses) == 100

    for filename, metadata in result["artifacts"].items():
        payload = (REPO / "results/m0_07" / filename).read_bytes()
        assert len(payload) == metadata["bytes"]
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]


def test_graph_corpus_rejects_changed_seed(tmp_path: Path) -> None:
    config = _load_json(
        REPO / "configs/bio/m0_07_oracle_graph_corpus.json",
        "config",
    )
    config["seed"] = 1
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    with pytest.raises(OracleGraphCorpusError, match="seed"):
        run_oracle_graph_corpus_audit(path, tmp_path / "output", REPO)
