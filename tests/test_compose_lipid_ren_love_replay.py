"""New SI controls, protected records, occupancy and constitutional invariance."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from rdkit import Chem

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import constitutional_molecule
from forge.corpus.compose_lipid_ren_love_replay import load_contract, replay_record

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "results/phase1/compose_lipid_user_supplements_v1/replay-config.json"


@pytest.fixture(scope="module")
def contract():
    return load_contract(ROOT, CONFIG)


def source_item(contract, key):
    source = json.loads(contract[1]["source_control"].read_text())["controls"][key]
    structures = {f"global:{r}": s for r, s in source["components"].items()}
    item = {
        "source": {"constitution": source["expected_product"]},
        "preparation": {
            "family": source["family"],
            "eligible_for_program_preparation": True,
            "construction_basis": "v8_exact_family_task",
            "constitution_id": hashlib.sha256(
                constitutional_molecule(source["expected_product"])[0].encode()
            ).hexdigest(),
            "component_instances": [[r, f"global:{r}", n] for r, n in source["quantities"].items()],
        },
    }
    return item, structures


@pytest.mark.parametrize("key", ["ren", "love"])
def test_drawn_source_closes_forward_inverse_inventory_and_preserves_inputs(contract, key):
    item, structures = source_item(contract, key)
    original = copy.deepcopy(item)
    result = replay_record(item, structures, contract[2])
    assert result["computed_consistency_pass"]
    assert all(result["checks"].values())
    assert result["balance"]["reactants"] == result["balance"]["product_and_net_byproducts"]
    assert item == original
    assert not result["training_admitted"]
    assert not result["experimental_execution_admitted"]


def test_protected_input_is_rejected_before_molecular_decoding(contract):
    with pytest.raises(ComposeLipidError, match="Protected"):
        replay_record({"preparation": {"eligible_for_program_preparation": False}}, {}, contract[2])


@pytest.mark.parametrize("quantity", [True, 1.0, 0])
def test_bad_quantity_cannot_be_coerced(contract, quantity):
    item, structures = source_item(contract, "ren")
    item["preparation"]["component_instances"][0][2] = quantity
    with pytest.raises(ComposeLipidError, match="quantity"):
        replay_record(item, structures, contract[2])


def test_love_seven_feed_equivalents_cannot_replace_five_incorporated_arms(contract):
    item, structures = source_item(contract, "love")
    for row in item["preparation"]["component_instances"]:
        if row[0] == "epoxide_tail":
            row[2] = 7
    assert (
        replay_record(item, structures, contract[2])["disposition"]
        == "outside_qualified_program_multiplicity"
    )


def test_target_does_not_select_occupancy_or_regioisomer(contract):
    item, structures = source_item(contract, "love")
    for row in item["preparation"]["component_instances"]:
        if row[0] == "epoxide_tail":
            row[2] = 4
    result = replay_record(item, structures, contract[2])
    assert result["source_occupancy"] == 4
    assert not result["computed_consistency_pass"]
    assert not result["checks"]["full_element_hydrogen_charge_balance"]


def test_ren_tertiary_amine_is_not_quaternized(contract):
    item, structures = source_item(contract, "ren")
    structures["global:amine_head"] = "CN(C)CCCCCO"
    result = replay_record(item, structures, contract[2])
    assert not result["computed_consistency_pass"]
    assert not result["checks"]["every_declared_event_replayed"]


def test_ren_oxygen_is_not_used_as_substitute_for_amine(contract):
    item, structures = source_item(contract, "ren")
    structures["global:amine_head"] = "OCCCCCO"
    assert not replay_record(item, structures, contract[2])["computed_consistency_pass"]


def test_missing_original_occupancy_stays_pending(contract):
    item, structures = source_item(contract, "love")
    item["preparation"]["construction_basis"] = "inferred_from_target"
    assert (
        replay_record(item, structures, contract[2])["disposition"]
        == "missing_original_task_occupancy"
    )


@pytest.mark.parametrize("key", ["ren", "love"])
def test_atom_order_and_component_order_do_not_change_evidence(contract, key):
    item, structures = source_item(contract, key)
    before = replay_record(item, structures, contract[2])
    for role, smiles in structures.items():
        mol = Chem.MolFromSmiles(smiles)
        reordered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        structures[role] = Chem.MolToSmiles(reordered, canonical=False)
    item["preparation"]["component_instances"].reverse()
    assert replay_record(item, structures, contract[2]) == before
