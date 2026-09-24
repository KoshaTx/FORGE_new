"""The full-corpus holdout projection cannot silently expand training eligibility."""

import copy
from pathlib import Path

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import sha256_file
from forge.corpus.compose_lipid_partition_projection import project_preparation_record
from forge.corpus.compose_lipid_partition_signatures import (
    FrozenGroupProjection,
    SourceSplitSignatures,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def signatures():
    path = ROOT / "data/vendor/compose_lipid_frozen_split_policy_v1.json"
    return SourceSplitSignatures.from_registry(ROOT, path, expected_sha256=sha256_file(path))


def record():
    source = {
        "target_id": "target",
        "primary_family": "family",
        "source_anchor": False,
        "heavy_atoms": 4,
        "constitution": "CCCO",
        "primary_metadata": {},
    }
    previous = {
        "target_id": "target",
        "family": "family",
        "source_anchor": False,
        "source_declared_heavy_atoms": 4,
        "component_instances": [["complete", "component", 1]],
        "disposition": "unresolved_partition_or_study",
        "training_admitted": False,
    }
    selected = {
        "formal_evaluation_families": ["family"],
        "reference_only_families": [],
        "selected_components": [],
        "selected_source_studies": ["pmid:123"],
        "selected_morphology_groups": [],
        "selected_combination_groups": [],
        "selected_calibration_morphology_groups": [],
        "selected_calibration_combination_groups": [],
    }
    return source, previous, selected


@pytest.mark.parametrize(
    "prior", ["protected", "eligible_for_program_preparation", "unresolved_partition_or_study"]
)
def test_projected_train_preserves_prior_disposition(signatures, prior):
    source, previous, selected = record()
    previous["disposition"] = prior
    before = copy.deepcopy((source, previous, selected))
    result = project_preparation_record(
        source,
        previous,
        signatures=signatures,
        projection=FrozenGroupProjection.from_selected_groups(selected),
        source_pmids=[],
    )
    assert result["corrected_projection"]["split"] == "train"
    assert result["disposition"] == prior
    assert result["newly_protected"] is False
    assert not result["training_admitted"]
    assert not result["full_universe_partition_qualified"]
    assert (source, previous, selected) == before


@pytest.mark.parametrize(
    "prior", ["eligible_for_program_preparation", "unresolved_partition_or_study"]
)
def test_new_global_component_holdout_protects_even_prior_eligible(signatures, prior):
    source, previous, selected = record()
    previous["disposition"] = prior
    selected["selected_components"] = ["component"]
    result = project_preparation_record(
        source,
        previous,
        signatures=signatures,
        projection=FrozenGroupProjection.from_selected_groups(selected),
        source_pmids=[],
    )
    assert result["disposition"] == "protected"
    assert result["newly_protected"] is True
    assert result["corrected_projection"]["test_panels"] == ["unseen_component_structure"]


def test_reported_source_missing_study_never_becomes_eligible(signatures):
    source, previous, selected = record()
    source["source_anchor"] = previous["source_anchor"] = True
    previous["disposition"] = "eligible_for_program_preparation"
    result = project_preparation_record(
        source,
        previous,
        signatures=signatures,
        projection=FrozenGroupProjection.from_selected_groups(selected),
        source_pmids=[],
    )
    assert result["source_study_identity_unresolved"] is True
    assert result["disposition"] == "unresolved_partition_or_study"


def test_source_study_holdout_is_preserved(signatures):
    source, previous, selected = record()
    result = project_preparation_record(
        source,
        previous,
        signatures=signatures,
        projection=FrozenGroupProjection.from_selected_groups(selected),
        source_pmids=[123],
    )
    assert result["disposition"] == "protected"
    assert "source_study_transfer_global" in result["corrected_projection"]["test_panels"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_id", "other"),
        ("family", "other"),
        ("source_anchor", True),
        ("source_declared_heavy_atoms", 3),
        ("training_admitted", True),
        ("disposition", "train"),
    ],
)
def test_join_or_scope_corruption_fails(signatures, field, value):
    source, previous, selected = record()
    previous[field] = value
    with pytest.raises(ComposeLipidError):
        project_preparation_record(
            source,
            previous,
            signatures=signatures,
            projection=FrozenGroupProjection.from_selected_groups(selected),
            source_pmids=[],
        )
