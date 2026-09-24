"""Frozen split descriptors preserve identity, quantity and global holdout scope."""

import copy
import json
from pathlib import Path

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import sha256_file
from forge.corpus.compose_lipid_partition_signatures import (
    FrozenGroupProjection,
    SourceSplitSignatures,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/vendor/compose_lipid_frozen_split_policy_v1.json"


@pytest.fixture(scope="module")
def source():
    return SourceSplitSignatures.from_registry(
        ROOT, REGISTRY, expected_sha256=sha256_file(REGISTRY)
    )


def descriptor(source, **changes):
    args = dict(
        family="family_a",
        smiles="CC(O)CO",
        metadata={"head_axis": "linear", "head_id": "alias"},
        source_anchor=False,
        instances=[["head", "head-id", 1], ["tail", "tail-id", 2]],
    )
    args.update(changes)
    return source.describe(**args)


def selection():
    return {
        "formal_evaluation_families": ["family_a", "family_b"],
        "reference_only_families": ["reference"],
        "selected_components": ["held-component"],
        "selected_source_studies": ["pmid:123"],
        "selected_morphology_groups": [["family_a", "held-morphology"]],
        "selected_combination_groups": [["family_a", "held-combination"]],
        "selected_calibration_morphology_groups": [["family_a", "calibration-morphology"]],
        "selected_calibration_combination_groups": [["family_a", "calibration-combination"]],
    }


def project(**changes):
    args = dict(
        family="family_a",
        component_ids=["ordinary"],
        source_pmids=[],
        morphology="ordinary",
        combination="ordinary",
    )
    args.update(changes)
    return FrozenGroupProjection.from_selected_groups(selection()).project(**args)


def test_source_order_and_smiles_order_invariant(source):
    baseline = descriptor(source)
    assert (
        descriptor(
            source, smiles="OCC(O)C", instances=[["tail", "tail-id", 2], ["head", "head-id", 1]]
        )
        == baseline
    )


def test_aliased_component_identifier_is_not_a_morphology_feature(source):
    assert descriptor(
        source, metadata={"head_axis": "linear", "head_id": "different"}
    ) == descriptor(source)


def test_complete_component_identity_changes_combination_only(source):
    original = descriptor(source)
    changed = descriptor(source, instances=[["head", "new-head", 1], ["tail", "tail-id", 2]])
    assert changed["combination_signature"] != original["combination_signature"]
    assert changed["morphology_group_signature"] == original["morphology_group_signature"]


def test_quantity_changes_component_multiset_and_regional_signature(source):
    original = descriptor(source)
    changed = descriptor(source, instances=[["head", "head-id", 1], ["tail", "tail-id", 1]])
    assert changed["combination_signature"] != original["combination_signature"]
    assert changed["regional_morphology_signature"] != original["regional_morphology_signature"]
    assert changed["regional_profile"] == original["regional_profile"]


def test_source_anchor_context_is_fixed_without_rewriting_input(source):
    metadata = {"head_axis": "new", "event_count": 3}
    before = copy.deepcopy(metadata)
    assert descriptor(source, source_anchor=True, metadata=metadata) == descriptor(
        source, source_anchor=True, metadata={}
    )
    assert metadata == before


def test_original_metadata_axis_and_ordered_events_matter(source):
    first = descriptor(source, metadata={"events": ["a", "b"]})
    second = descriptor(source, metadata={"events": ["b", "a"]})
    assert first["morphology_group_signature"] != second["morphology_group_signature"]
    assert descriptor(source, metadata={"custom_axis": "a"}) != descriptor(
        source, metadata={"custom_axis": "b"}
    )


@pytest.mark.parametrize("size", [96, 128, 214, 254])
def test_descriptors_do_not_truncate_molecular_size(source, size):
    result = source.profile("C" * size)
    assert result["heavy_atom_band"] == size // 8
    assert result["carbon_band"] == size // 6


def test_branch_and_charge_descriptors_have_source_semantics(source):
    assert source.profile("CC(C)C")["carbon_branch_points"] == 1
    assert source.profile("CCCC")["carbon_branch_points"] == 0
    assert source.profile("C[N+](C)(C)C")["formal_charge"] == 1


@pytest.mark.parametrize("smiles", ["", "this-is-not-smiles"])
def test_empty_and_invalid_molecular_graphs_fail(source, smiles):
    with pytest.raises(ComposeLipidError):
        source.profile(smiles)


@pytest.mark.parametrize(
    "instances",
    [[], [["head", "x", 0]], [["head", "x", True]], [["head", "x", 1], ["head", "x", 1]]],
)
def test_bad_component_quantities_cannot_create_a_signature(source, instances):
    with pytest.raises(ComposeLipidError):
        descriptor(source, instances=instances)


@pytest.mark.parametrize("family", ["family_a", "family_b"])
def test_component_and_study_protection_is_global(family):
    assert project(family=family, component_ids=["held-component"])["split"] == "test"
    assert project(family=family, source_pmids=[123])["split"] == "test"
    assert project(family=family, source_pmids=["pmid:123"])["split"] == "test"


@pytest.mark.parametrize(
    "key,value,fold",
    [
        ("morphology", "held-morphology", "test"),
        ("combination", "held-combination", "test"),
        ("morphology", "calibration-morphology", "calibration"),
        ("combination", "calibration-combination", "calibration"),
    ],
)
def test_morphology_and_combination_groups_are_family_scoped(key, value, fold):
    assert project(**{key: value})["split"] == fold
    assert project(family="family_b", **{key: value})["split"] == "train"


def test_new_test_calibration_collision_stays_protected():
    result = project(component_ids=["held-component"], morphology="calibration-morphology")
    assert result["split"] == "test"
    assert result["calibration_groups"] == ["regional_morphology"]
    assert not result["training_admitted"]


def test_train_projection_does_not_admit_training():
    result = project()
    assert result["split"] == "train"
    assert result["training_admitted"] is False


def test_reference_family_remains_reference():
    assert project(family="reference", component_ids=["held-component"])["split"] == "reference"


@pytest.mark.parametrize(
    "changes",
    [
        {"family": "unknown"},
        {"component_ids": []},
        {"morphology": ""},
        {"source_pmids": [None]},
        {"source_pmids": [True]},
        {"source_pmids": ["pmid:"]},
    ],
)
def test_unresolved_projection_inputs_fail(changes):
    with pytest.raises(ComposeLipidError):
        project(**changes)


def test_selected_group_overlap_rejected():
    selected = selection()
    selected["selected_calibration_morphology_groups"] = selected["selected_morphology_groups"]
    with pytest.raises(ComposeLipidError, match="overlap"):
        FrozenGroupProjection.from_selected_groups(selected)


def test_registry_authenticates_recovered_source_bytes(tmp_path):
    policy = json.loads(REGISTRY.read_text())
    policy["source_recovery"]["sha256"] = "0" * 64
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        SourceSplitSignatures.from_registry(ROOT, p, expected_sha256=sha256_file(p))
