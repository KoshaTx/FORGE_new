"""Domain adjudication excludes out-of-domain inverses without choosing a target."""

import copy

import pytest

from forge.corpus.compose_lipid_ren_love_replay import load_contract, replay_record
from forge.corpus.compose_lipid_source_domain_inverse import qualify_inverse, qualify_saved_replay
from tests.test_compose_lipid_ren_love_replay import CONFIG, ROOT, source_item


@pytest.fixture(scope="module")
def contract():
    return load_contract(ROOT, CONFIG)


def test_reject_ethyl_bromide_but_keep_every_raw_candidate(contract):
    program = contract[2]["amine_alkylation"][2]["program"]
    candidates = [
        {"amine_head": "NCCN(CC)CC", "bromoester_arm": "CCCCOC(=O)CCCBr"},
        {"amine_head": "NCCN(CC)CC", "bromoester_arm": "CCBr"},
    ]
    before = copy.deepcopy(candidates)
    result = qualify_inverse(program, {"candidate_components": candidates, "complete_search": True})
    assert result["candidate_components"] == candidates[:1]
    assert result["unfiltered_candidate_components"] == before
    assert candidates == before
    assert (
        result["candidate_domain_audit"][1]["terminal_constraints"]["bromoester_arm"]["checks"][
            "element_count_O"
        ]
        is False
    )


def test_two_in_domain_candidates_remain_ambiguous(contract):
    program = contract[2]["amine_alkylation"][2]["program"]
    candidates = [
        {"amine_head": "NCCN(CC)CC", "bromoester_arm": "CCCCOC(=O)CCCBr"},
        {"amine_head": "NCCCN(CC)CC", "bromoester_arm": "CCCOC(=O)CCCBr"},
    ]
    result = qualify_inverse(program, {"candidate_components": candidates, "complete_search": True})
    assert result["candidate_components"] == candidates


def test_source_controls_and_other_failed_checks_are_preserved(contract):
    item, structures = source_item(contract, "ren")
    original = replay_record(item, structures, contract[2])
    program = contract[2]["amine_alkylation"][2]["program"]
    components = {r: structures[i] for r, i, _ in item["preparation"]["component_instances"]}
    result = qualify_saved_replay(program, original, components)
    assert result["computed_consistency_pass"]
    assert result["forward_layers"] == original["forward_layers"]
    assert result["inverse"]["stage_searches"] == original["inverse"]["stage_searches"]
    original["checks"]["complete_search"] = False
    assert not qualify_saved_replay(program, original, components)["computed_consistency_pass"]


def test_cannot_double_adjudicate(contract):
    program = contract[2]["amine_alkylation"][2]["program"]
    result = qualify_inverse(program, {"candidate_components": [], "complete_search": True})
    with pytest.raises(ValueError, match="already"):
        qualify_inverse(program, result)
