"""Documented stage endpoints, every event path and source quantity/identity gates."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import LibraryAssemblyError, constitutional_molecule
from forge.assembly.grouped_program import RegistryGroupedProgram
from forge.assembly.repeated_components import RepeatBounds
from forge.core.hashing import sha256_file
from forge.corpus.compose_lipid_grouped_replay import load_contract, replay_record

ROOT = Path(__file__).resolve().parents[1]
FAMILY = "epoxide_opening_o_acylation"
CONFIG = ROOT / "results/phase1/compose_lipid_han_db_source_v1/replay-config.json"


@pytest.fixture(scope="module")
def contract():
    return load_contract(ROOT, CONFIG)


@pytest.fixture
def source(contract):
    return json.loads(contract[1]["transform_controls"].read_text())["source_controls"][0]


def test_both_primary_source_controls_match_each_documented_stage(contract):
    for label in ("1-6-6", "1-10-8"):
        r = contract[-1][label]["replay"]
        assert r["computed_consistency_pass"]
        assert r["declared_quantities"] == {"amine_head": 1, "epoxide_tail": 2, "acyl_tail": 2}
        assert all(r["checks"].values())
        assert r["balance"]["reactants"] == r["balance"]["product_and_net_byproducts"]
        assert r["balance"]["reactants"]["Cl"] == 2
        assert not r["experimental_selectivity_qualified"]
        assert not r["within_stage_event_order_qualified"]


def test_distinct_event_paths_can_converge_at_a_source_stage_boundary(contract, source):
    parts = {**source["components"], "amine_head": "CNCCNCC"}
    target = "CN(CC(OC(=O)CCCCC)CCCC)CCN(CC)CC(OC(=O)CCCCC)CCCC"
    r = contract[2][FAMILY]["program"].replay(parts, target)
    assert r["computed_consistency_pass"]
    assert list(map(len, r["event_searches"][0]["layers"])) == [1, 2, 1]
    assert list(map(len, r["forward_layers"])) == [1, 1, 1]
    assert len(r["inverse"]["candidate_components"]) == 1


def test_source_ambiguity_and_nh_competitors_remain_failures(contract):
    r = contract[-1]["unprotected_head_NH_competitor"]["replay"]
    assert list(map(len, r["forward_layers"])) == [1, 2, 0]
    assert not r["checks"]["unique_each_completed_source_stage"]
    assert not r["checks"]["every_declared_event_replayed"]
    assert not r["computed_consistency_pass"]


@pytest.mark.parametrize("acyl", ["CCCCCC(=O)O", "CCCCCOC(=O)Cl", "CCCCNC(=O)Cl"])
def test_acid_chloroformate_carbamoyl_are_not_source_acyl_chlorides(contract, source, acyl):
    r = contract[2][FAMILY]["program"].replay(
        {**source["components"], "acyl_tail": acyl}, source["expected_product"]
    )
    assert not r["computed_consistency_pass"]
    assert r["forward_layers"][-1] == []


def test_wrong_epoxide_regioisomer_does_not_pass(contract, source):
    target = "CN(C)CCCN(C(COC(=O)CCCCC)CCCC)C(COC(=O)CCCCC)CCCC"
    r = contract[2][FAMILY]["program"].replay(source["components"], target)
    assert r["checks"]["full_element_hydrogen_charge_balance"]
    assert not r["checks"]["unique_forward_exact"]
    assert not r["computed_consistency_pass"]


def test_atom_order_invariance(contract, source):
    def reorder(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(
            Chem.RenumberAtoms(m, list(reversed(range(m.GetNumAtoms())))), canonical=False
        )

    r = contract[2][FAMILY]["program"].replay(
        {r: reorder(s) for r, s in source["components"].items()},
        reorder(source["expected_product"]),
    )
    assert r["computed_consistency_pass"]


@pytest.mark.parametrize(
    "bound", [{"maximum_outcomes": 1}, {"maximum_states": 2}, {"maximum_transitions": 1}]
)
def test_global_or_local_saturation_never_passes(contract, source, bound):
    p = replace(contract[2][FAMILY]["program"], bounds=RepeatBounds(**bound))
    r = p.replay(source["components"], source["expected_product"])
    assert not r["computed_consistency_pass"]
    assert not r["checks"]["complete_search"]


def prepared(source):
    parts = source["components"]
    structures = {f"id:{r}": s for r, s in parts.items()}
    return {
        "source": {"constitution": source["expected_product"]},
        "preparation": {
            "eligible_for_program_preparation": True,
            "constitution_id": hashlib.sha256(
                constitutional_molecule(source["expected_product"])[0].encode()
            ).hexdigest(),
            "component_instances": [[r, f"id:{r}", 1 if r == "amine_head" else 2] for r in parts],
        },
    }, structures


def test_current_global_ids_and_quantities_are_preserved(contract, source):
    item, structures = prepared(source)
    before = copy.deepcopy(item)
    r = replay_record(item, structures, contract[2][FAMILY])
    assert r["computed_consistency_pass"]
    assert r["verified_target_constitution_id"] == item["preparation"]["constitution_id"]
    assert item == before


@pytest.mark.parametrize("quantity", [1, 3, 0, True])
def test_reagent_excess_and_target_cannot_change_incorporated_counts(contract, source, quantity):
    item, structures = prepared(source)
    for c in item["preparation"]["component_instances"]:
        if c[0] == "epoxide_tail":
            c[2] = quantity
    if type(quantity) is not int or quantity < 1:
        with pytest.raises(ComposeLipidError):
            replay_record(item, structures, contract[2][FAMILY])
    else:
        assert not replay_record(item, structures, contract[2][FAMILY])["computed_consistency_pass"]


def test_two_tail_identities_cannot_replace_a_repeated_source_id(contract, source):
    item, structures = prepared(source)
    for c in item["preparation"]["component_instances"]:
        if c[0] == "acyl_tail":
            c[2] = 1
    structures["other"] = "CCCCCCCC(=O)Cl"
    item["preparation"]["component_instances"].append(["acyl_tail", "other", 1])
    assert (
        replay_record(item, structures, contract[2][FAMILY])["disposition"]
        == "unsupported_source_role_tuple"
    )


def test_protected_row_rejected_before_graph_decode(contract):
    with pytest.raises(ComposeLipidError, match="Protected"):
        replay_record(
            {
                "source": {"constitution": "invalid"},
                "preparation": {"eligible_for_program_preparation": False},
            },
            {},
            contract[2][FAMILY],
        )


def test_target_identity_cannot_change(contract, source):
    item, structures = prepared(source)
    item["preparation"]["constitution_id"] = "0" * 64
    with pytest.raises(ComposeLipidError, match="identity"):
        replay_record(item, structures, contract[2][FAMILY])


@pytest.mark.parametrize(
    "mutation",
    [
        "stage_order",
        "events_bool",
        "events_zero",
        "too_many_events",
        "duplicate_role",
        "unknown_role",
        "missing_constraint",
        "negative_inventory",
        "extra_key",
    ],
)
def test_grouped_contract_is_strict(contract, tmp_path, mutation):
    registry = json.loads(contract[1]["registry"].read_text())
    p = registry["grouped_programs"][0]
    if mutation == "stage_order":
        p["stages"].reverse()
    elif mutation == "events_bool":
        p["stages"][0]["events"] = True
    elif mutation == "events_zero":
        p["stages"][0]["events"] = 0
    elif mutation == "too_many_events":
        p["stages"][0]["events"] = 33
    elif mutation == "duplicate_role":
        p["stages"][1]["added_roles"] = ["epoxide_tail"]
    elif mutation == "unknown_role":
        p["stages"][0]["added_roles"] = ["different"]
    elif mutation == "missing_constraint":
        p["terminal_constraints"].pop("amine_head")
    elif mutation == "negative_inventory":
        p["stages"][1]["net_byproducts_per_event"]["Cl"] = -1
    else:
        p["stages"][0]["guessed_target_site"] = 0
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    with pytest.raises(LibraryAssemblyError):
        RegistryGroupedProgram.from_registry(
            path, program_id=p["program_id"], expected_sha256=str(sha256_file(path))
        )


def test_documented_stage_intermediate_is_verified_independently(contract):
    from forge.corpus.compose_lipid_grouped_replay import qualify_controls

    doc = json.loads(contract[1]["transform_controls"].read_text())
    doc["source_controls"][0]["expected_stage_products"][0] = "CCN"
    with pytest.raises(ComposeLipidError, match="intermediate"):
        qualify_controls(doc, contract[2])
