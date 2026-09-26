"""Necessary compatibility cannot hide role, indexing or ordered-core differences."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.phase1.multireaction.component_feasibility_census_v2 import (
    FAMILIES,
    compatible_stages,
    layout_signature,
    paths,
    read,
    require_source_correspondence,
    retained_head_witness,
    role_aliases,
)
from forge.model.compose_lipid_layout import build_layout, summarize_layout


def shape():
    request = {
        "quantity": 2,
        "positions": [[2, 1], [3, 1, 1, 1]],
        "block_sizes": [2, 4],
        "morphology": [5, 1, 2, 3],
        "closures": 1,
        "allowed_ring_sizes": [6],
    }
    profile = {
        **request,
        "template": "core",
        "ring_sizes": [6],
        "head_status": "pass",
    }
    return profile, request


def test_source_shape_positive_and_same_total_different_blocks_refused():
    profile, request = shape()
    assert all(compatible_stages(profile, "core", request))
    profile["block_sizes"] = [3, 3]
    flags = compatible_stages(profile, "core", request)
    assert flags[:2] == [True, True] and not any(flags[2:])


@pytest.mark.parametrize(
    "field,value",
    [
        ("quantity", 1),
        ("template", "different"),
        ("positions", [[3, 1], [2, 1, 1, 1]]),
        ("morphology", [5, 2, 2, 3]),
        ("closures", 2),
        ("ring_sizes", [9]),
        ("head_status", "fail"),
    ],
)
def test_each_necessary_condition_is_preserved(field, value):
    profile, request = shape()
    profile[field] = value
    assert not compatible_stages(profile, "core", request)[-1]


def test_alias_mapping_is_from_saved_evidence_and_ambiguity_refuses():
    row = {
        "family": FAMILIES[0],
        "diagnostic": {"components": {"accepted": {"source_role": "source"}}},
    }
    assert role_aliases([row]) == {(FAMILIES[0], "accepted"): "source"}
    other = deepcopy(row)
    other["diagnostic"]["components"]["accepted"]["source_role"] = "wrong"
    with pytest.raises(ValueError, match="Ambiguous"):
        role_aliases([row, other])


def test_index_source_correspondence_is_exact():
    example = SimpleNamespace(
        family="family", record=SimpleNamespace(graph=SimpleNamespace(structure_id="record-2"))
    )
    require_source_correspondence("record-2", "family", example)
    with pytest.raises(ValueError, match="correspondence"):
        require_source_correspondence("record-1", "family", example)
    with pytest.raises(ValueError, match="correspondence"):
        require_source_correspondence("record-2", "other-family", example)


def test_sampled_template_matches_actual_source_summary():
    graph = SimpleNamespace(
        node_states=np.array([2, 3, 4, 5]),
        parents=np.array([0, 0, 0, 2]),
        parent_bonds=np.array([0, 1, 1, 1]),
        closure_left=np.array([], dtype=int),
        closure_right=np.array([], dtype=int),
        closure_bonds=np.array([], dtype=int),
    )
    record = SimpleNamespace(
        graph=graph,
        node_count=4,
        role_states=np.array([2, 2, 3, 3]),
        core_position_states=np.array([2, 1, 3, 1]),
        fixed_atom_mask=np.array([True, False, True, False]),
        program_id="program",
        program_state=1,
        program_depth=1,
        component_blocks=(
            SimpleNamespace(role="head", role_state=2, start=0, stop=2),
            SimpleNamespace(role="tail", role_state=3, start=2, stop=4),
        ),
    )
    example = SimpleNamespace(
        record=record,
        family="family",
        source_quantities={"head": 1, "tail": 1},
        introduced_roles=(),
    )
    bundle, options, rings = summarize_layout(example)
    layout = build_layout(bundle, options, rings, identity="request")
    reconstructed, roles = layout_signature(layout)
    assert reconstructed == bundle
    assert roles["head"]["positions"] == options[2]["positions"]
    assert roles["tail"]["morphology"] == options[3]["morphology"]


def test_retention_uses_actual_source_control_and_rejects_cross_role_rescue():
    files = paths()
    reaction = read(files["registry"])["reactions"][0]
    adjudication = read(files["adjudication"])
    contract = adjudication["families"]["ketone_ugi4"]
    positive = adjudication["source_controls"][0]["components"]["amine_head"]
    negative = next(
        r
        for r in adjudication["regression_controls"]
        if r["label"] == "cross_role_basic_nitrogen_cannot_rescue_head"
    )
    assert retained_head_witness(positive, reaction, contract)["status"] == "pass"
    assert (
        retained_head_witness(negative["components"]["amine_head"], reaction, contract)["status"]
        == "fail"
    )


def test_actual_saved_candidate_schema_and_role_correspondence():
    document = read(paths()["candidate_diagnostics"])
    assert "candidates" in document and "attempts" not in document
    aliases = role_aliases(document["candidates"])
    assert aliases[("ketone_ugi4", "amine_head")] == "amine_head"
    assert aliases[("a3_amine_aldehyde_alkyne", "amine_head")] == "amine_head"
