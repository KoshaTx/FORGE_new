from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path

import pytest

from experiments.archive.phase1.synthesis_value_coverage.ugi3_family_applicability_census import (
    EXTRAPOLATION_REVIEW,
    MISSING_METADATA,
    RECURRENT_BUCKET,
    Ugi3FamilyApplicabilityCensusError,
    build_family_applicability_census,
    classify_applicability_bucket,
    constitutional_applicability_axes,
)
from forge.corpus.r1_prime_audit import sha256_bytes, sha256_file
from forge.synthesis.terminals.ugi3_virtual_programs import ALDEHYDE_ROLE, ISOCYANIDE_ROLE

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi3_family_applicability_census_v1.json"


def _rows(payload: bytes) -> list[dict[str, str]]:
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
        with io.TextIOWrapper(compressed) as handle:
            return list(csv.DictReader(handle))


def test_constitutional_axes_are_handle_relative_and_position_preserving() -> None:
    aldehyde = constitutional_applicability_axes(ALDEHYDE_ROLE, "CC(C)CC=CC=O")
    assert aldehyde["reactive_handle_type"] == "aldehyde_carbonyl"
    assert aldehyde["candidate_carbon_count"] == 7
    assert aldehyde["carbon_carbon_double_bond_count"] == 1
    assert aldehyde["carbon_branch_point_count"] == 1
    unsaturation = json.loads(aldehyde["constitutional_unsaturation_positions_json"])
    branches = json.loads(aldehyde["branch_point_distances_from_reactive_center_json"])
    assert unsaturation == [
        {
            "bond_type": "double",
            "far_endpoint_distance": 2,
            "near_endpoint_distance": 1,
            "position_zone": "proximal",
        }
    ]
    assert branches == [4]

    isocyanide = constitutional_applicability_axes(ISOCYANIDE_ROLE, "[C-]#[N+]CCCC")
    assert isocyanide["reactive_handle_type"] == "isocyanide_carbon"
    assert isocyanide["candidate_carbon_count"] == 5
    assert isocyanide["maximum_reactive_center_to_carbon_distance_bonds"] == 5

    with pytest.raises(
        Ugi3FamilyApplicabilityCensusError,
        match="requires exactly one reactive handle",
    ):
        constitutional_applicability_axes(ALDEHYDE_ROLE, "CCCC")


def test_recurrence_classification_is_descriptive_and_keeps_missing_metadata() -> None:
    assert (
        classify_applicability_bucket(
            metadata_complete=True, bucket_component_count=3, recurrence_threshold=3
        )
        == RECURRENT_BUCKET
    )
    assert (
        classify_applicability_bucket(
            metadata_complete=True, bucket_component_count=2, recurrence_threshold=3
        )
        == EXTRAPOLATION_REVIEW
    )
    assert (
        classify_applicability_bucket(
            metadata_complete=False, bucket_component_count=20, recurrence_threshold=3
        )
        == MISSING_METADATA
    )


def test_final_projected_family_census_is_deterministic_and_nonqualifying() -> None:
    first = build_family_applicability_census(REPO, CONFIG)
    second = build_family_applicability_census(REPO, CONFIG)
    assert first == second
    result, ledger = first
    assert result["status"] == "complete_nonselecting_family_applicability_axis_census"
    assert result["artifacts"]["family_applicability_census.csv.gz"]["sha256"] == (
        sha256_bytes(ledger)
    )
    assert result["summary"]["family_projected_components"] == 209
    assert result["summary"]["one_gap_product_occurrences"] == 515
    assert result["summary"]["components_by_program_family"] == {
        "fatty_acid_diol_esterification_then_alcohol_oxidation": 92,
        "primary_alcohol_oxidation": 73,
        "primary_amine_formylation_then_formamide_dehydration": 44,
    }
    assert result["summary"]["components_by_projected_leaf_currency"] == {
        "all_current": 32,
        "no_current": 120,
        "partial_current": 57,
    }
    assert result["summary"]["missing_axis_metadata_components"] == 0
    assert result["summary"]["unique_applicability_buckets"] == 85
    assert result["summary"]["recurrent_interpolation_bucket_candidates"] == 25
    assert result["summary"]["sparse_extrapolation_review_buckets"] == 60
    assert result["summary"]["components_by_applicability_census_class"] == {
        RECURRENT_BUCKET: 136,
        EXTRAPOLATION_REVIEW: 73,
    }
    assert result["summary"]["one_gap_product_occurrences_by_applicability_census_class"] == {
        RECURRENT_BUCKET: 312,
        EXTRAPOLATION_REVIEW: 203,
    }
    assert result["adjudication"] == {
        "family_scope_qualification_performed": False,
        "recurrent_bucket_is_validated_interpolation": False,
        "sparse_bucket_is_chemical_incompatibility": False,
        "scalar_synthesis_value_defined": False,
        "synthesis_success_probability_defined": False,
        "synthesis_guidance_run": False,
        "prospective_candidate_selection_changed": False,
        "holdout_revealed": False,
    }

    rows = _rows(ledger)
    assert len(rows) == 209
    assert all(row["family_scope_qualified"] == "False" for row in rows)
    assert all(row["axis_metadata_status"] == "complete" for row in rows)
    assert {row["applicability_census_class"] for row in rows} <= {
        RECURRENT_BUCKET,
        EXTRAPOLATION_REVIEW,
        MISSING_METADATA,
    }
    for row in rows:
        unsaturations = json.loads(row["constitutional_unsaturation_positions_json"])
        assert len(unsaturations) == int(row["carbon_carbon_double_bond_count"]) + int(
            row["carbon_carbon_triple_bond_count"]
        )
        assert len(json.loads(row["branch_point_distances_from_reactive_center_json"])) == int(
            row["carbon_branch_point_count"]
        )
    assert not ({"scalar_value", "success_probability"} & set(rows[0]))


def test_config_pins_every_frozen_input() -> None:
    config = json.loads(CONFIG.read_text())
    for specification in config["inputs"].values():
        assert sha256_file(REPO / specification["path"]) == specification["sha256"]
