"""Two-source-thiol conservation and every-stage, unfiltered reconstruction gates."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import LibraryAssemblyError, constitutional_molecule
from forge.assembly.repeated_components import RepeatBounds
from forge.assembly.staged_program import RegistryStagedProgram
from forge.core.hashing import sha256_file
from forge.corpus.compose_lipid_staged_replay import load_contract, replay_record

ROOT = Path(__file__).resolve().parents[1]
FAMILY = "preassembled_thiol_yne_tail_amidation"
CONFIG = ROOT / "results/phase1/compose_lipid_thiol_yne_source_v1/replay-config.json"


@pytest.fixture(scope="module")
def contract():
    return load_contract(ROOT, CONFIG)


@pytest.fixture
def source(contract):
    return json.loads(contract[1]["transform_controls"].read_text())["source_controls"][0]


def test_source_full_structure_and_distinct_secondary_amine_library_control(contract):
    for label in ["Fig1B_A1C11", "Fig1A_library_B7C6"]:
        result = contract[-1][label]["replay"]
        assert result["computed_consistency_pass"]
        assert all(result["checks"].values())
        assert [len(layer) for layer in result["forward_layers"]] == [1, 1, 1]
        assert len(result["inverse"]["candidate_components"]) == 1
        assert not result["experimental_selectivity_qualified"]


def test_target_matching_cannot_select_between_head_sites(contract):
    r = contract[-1]["non_equivalent_head_amines"]["replay"]
    assert len(r["forward_layers"][-1]) == 2
    assert not r["computed_consistency_pass"]
    assert not r["checks"]["unique_each_forward_stage"]


def test_reordering_atoms_preserves_full_inverse_and_inventory(contract, source):
    def reorder(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(
            Chem.RenumberAtoms(m, list(reversed(range(m.GetNumAtoms())))), canonical=False
        )

    p = contract[2][FAMILY]["program"]
    assert p.replay(
        {r: reorder(s) for r, s in source["components"].items()},
        reorder(source["expected_product"]),
    )["computed_consistency_pass"]


def test_two_different_tails_fail_even_with_a_matching_net_graph(contract, source):
    p = contract[2][FAMILY]["program"]
    parts = source["components"]
    parts["thiol_second"] = "CCCCCCS"
    r = p.replay(parts, "CN(C)CCNC(=O)CCC(SCCCCCC)CSCCCCCCCCCCC")
    assert not r["checks"]["declared_component_equalities"]
    assert r["checks"]["unique_forward_exact"]
    assert not r["computed_consistency_pass"]


def test_single_source_tail_is_counted_twice_in_inventory(contract, source):
    result = contract[2][FAMILY]["program"].replay(source["components"], source["expected_product"])
    assert result["balance"]["reactants"]["S"] == 2
    assert (
        result["balance"]["reactants"]["H"] == result["balance"]["product_and_net_byproducts"]["H"]
    )


@pytest.mark.parametrize(
    "bound", [{"maximum_states": 2}, {"maximum_transitions": 1}, {"maximum_outcomes": 1}]
)
def test_exhaustion_never_passes(contract, source, bound):
    p = replace(contract[2][FAMILY]["program"], bounds=RepeatBounds(**bound))
    result = p.replay(source["components"], source["expected_product"])
    assert not result["computed_consistency_pass"]
    assert not result["checks"]["complete_search"]


def prepared(source):
    parts = source["components"]
    structures = {
        "linker": parts["alkynoic_linker"],
        "head": parts["amine_head"],
        "tail": parts["thiol_first"],
    }
    item = {
        "source": {"constitution": source["expected_product"]},
        "preparation": {
            "eligible_for_program_preparation": True,
            "constitution_id": hashlib.sha256(
                constitutional_molecule(source["expected_product"])[0].encode()
            ).hexdigest(),
            "component_instances": [
                ["alkynoic_linker", "linker", 1],
                ["amine_head", "head", 1],
                ["thiol_tail", "tail", 2],
            ],
        },
    }
    return item, structures


def test_global_component_quantity_expands_without_splitting_identity(contract, source):
    item, structures = prepared(source)
    before = copy.deepcopy(item)
    r = replay_record(item, structures, contract[2][FAMILY])
    assert r["computed_consistency_pass"]
    assert r["verified_target_constitution_id"] == item["preparation"]["constitution_id"]
    assert item == before


@pytest.mark.parametrize("quantity", [1, 3, 0, -1, True])
def test_wrong_tail_quantity_cannot_be_repaired_from_target(contract, source, quantity):
    item, structures = prepared(source)
    item["preparation"]["component_instances"][-1][-1] = quantity
    if type(quantity) is not int or quantity < 1:
        with pytest.raises(ComposeLipidError):
            replay_record(item, structures, contract[2][FAMILY])
    else:
        assert not replay_record(item, structures, contract[2][FAMILY])["computed_consistency_pass"]


def test_duplicate_source_role_cannot_supply_two_tail_identities(contract, source):
    item, structures = prepared(source)
    item["preparation"]["component_instances"][-1][-1] = 1
    structures["other"] = "CCCCCCS"
    item["preparation"]["component_instances"].append(["thiol_tail", "other", 1])
    assert (
        replay_record(item, structures, contract[2][FAMILY])["disposition"]
        == "unsupported_source_role_tuple"
    )


def test_protected_input_rejected_before_invalid_molecule_parsing(contract):
    with pytest.raises(ComposeLipidError, match="Protected"):
        replay_record(
            {
                "source": {"constitution": "invalid"},
                "preparation": {"eligible_for_program_preparation": False},
            },
            {},
            contract[2][FAMILY],
        )


def test_identity_hash_substitution_fails(contract, source):
    item, structures = prepared(source)
    item["preparation"]["constitution_id"] = "0" * 64
    with pytest.raises(ComposeLipidError, match="identity"):
        replay_record(item, structures, contract[2][FAMILY])


@pytest.mark.parametrize(
    "change",
    [
        "missing_tail",
        "duplicate_added_role",
        "wrong_adapter_roles",
        "unknown_equality_role",
        "overlap_equality",
        "negative_inventory",
        "zero_stages",
    ],
)
def test_registry_stage_schema_is_strict(contract, tmp_path, change):
    registry = json.loads(contract[1]["registry"].read_text())
    p = registry["programs"][0]
    if change == "missing_tail":
        p["terminal_constraints"].pop("thiol_second")
    elif change == "duplicate_added_role":
        p["stages"][0]["added_roles"] = ["thiol_first", "thiol_first"]
    elif change == "wrong_adapter_roles":
        p["stages"][0]["added_roles"] = ["thiol_first", "amine_head"]
    elif change == "unknown_equality_role":
        p["equal_component_groups"] = [["thiol_first", "missing"]]
    elif change == "overlap_equality":
        p["equal_component_groups"] *= 2
    elif change == "negative_inventory":
        p["stages"][0]["net_byproducts"] = {"H": -1}
    else:
        p["stages"] = []
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    with pytest.raises(LibraryAssemblyError):
        RegistryStagedProgram.from_registry(
            path, program_id=p["program_id"], expected_sha256=str(sha256_file(path))
        )


@pytest.mark.parametrize("change", ["quantity", "equality", "missing_negative", "formula"])
def test_contract_tampering_fails_before_corpus_access(contract, tmp_path, change):
    cfg = copy.deepcopy(contract[0])
    doc = json.loads(contract[1]["transform_controls"].read_text())
    if change == "quantity":
        cfg["families"][FAMILY]["source_role_occurrences"]["thiol_tail"].pop()
    elif change == "equality":
        cfg["families"][FAMILY]["source_role_occurrences"]["amine_head"] = ["thiol_first"]
    elif change == "missing_negative":
        doc["regression_controls"] = []
    else:
        doc["source_controls"][0]["expected_formula"] = "C"
    if change in ["quantity", "equality"]:
        doc["families"] = cfg["families"]
    d = tmp_path / "adjudication.json"
    d.write_text(json.dumps(doc))
    cfg["inputs"]["transform_controls"] = {"path": str(d), "sha256": str(sha256_file(d))}
    c = tmp_path / "config.json"
    c.write_text(json.dumps(cfg))
    with pytest.raises((ComposeLipidError, ValueError)):
        load_contract(ROOT, c)
