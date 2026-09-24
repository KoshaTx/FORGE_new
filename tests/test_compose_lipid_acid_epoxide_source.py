"""Source-drawn regiochemistry and the explicitly retained Xu naming discrepancy."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.assembly.families import constitutional_molecule
from forge.assembly.repeated_components import RepeatBounds
from forge.corpus.compose_lipid_staged_replay import load_contract, replay_record

ROOT = Path(__file__).resolve().parents[1]
FAMILY = "acid_epoxide_diester_multistep"
CONFIG = ROOT / "results/phase1/compose_lipid_acid_epoxide_source_v2/replay-config.json"


@pytest.fixture(scope="module")
def contract():
    return load_contract(ROOT, CONFIG)


@pytest.fixture
def source(contract):
    return json.loads(contract[1]["transform_controls"].read_text())["source_controls"][0]


def test_independent_drawing_and_intermediate_both_reconstruct(contract, source):
    result = contract[-1][source["label"]]["replay"]
    assert all(result["checks"].values())
    assert result["forward_layers"][1:] == [
        [constitutional_molecule(s)[0]] for s in source["expected_stage_products"]
    ]
    assert result["balance"]["reactants"] == result["balance"]["product_and_net_byproducts"]
    assert not result["experimental_selectivity_qualified"]


def test_drawn_formula_agrees_with_reported_mass_but_printed_name_does_not(contract, source):
    ion = source["expected_product"].replace("CN(C)", "C[NH+](C)", 1)
    mol = Chem.MolFromSmiles(ion)
    assert rdMolDescriptors.CalcMolFormula(mol) == "C34H68NO4+"
    assert abs(rdMolDescriptors.CalcExactMolWt(mol) - 554.5143) < 0.0001
    negative = contract[-1]["printed_shorter_head_name"]["replay"]
    assert not negative["computed_consistency_pass"]
    assert not negative["checks"]["full_element_hydrogen_charge_balance"]


def test_same_formula_wrong_site_order_fails(contract):
    result = contract[-1]["head_and_hydrophobe_site_swap"]["replay"]
    assert result["checks"]["full_element_hydrogen_charge_balance"]
    assert not result["checks"]["unique_forward_exact"]
    assert not result["computed_consistency_pass"]


@pytest.mark.parametrize(
    "role,replacement",
    [
        ("epoxide", "CCCCCC1OC1CCCC"),
        ("epoxide", "OCCCCC1CO1"),
        ("hydrophobic_acid", "O=C(O)CCCCC(=O)O"),
        ("amine_acid_head", "CNCCCC(=O)O"),
        ("amine_acid_head", "CN(C)CC(O)CC(=O)O"),
    ],
)
def test_unqualified_competing_or_wrong_interfaces_fail(contract, source, role, replacement):
    parts = {**source["components"], role: replacement}
    result = contract[2][FAMILY]["program"].replay(parts, source["expected_product"])
    assert not result["computed_consistency_pass"]
    assert result["forward_layers"][-1] == []


def test_complete_global_recipe_is_preserved(contract, source):
    item = {
        "source": {"constitution": source["expected_product"]},
        "preparation": {
            "eligible_for_program_preparation": True,
            "constitution_id": hashlib.sha256(
                constitutional_molecule(source["expected_product"])[0].encode()
            ).hexdigest(),
            "component_instances": [[r, "global:" + r, 1] for r in source["components"]],
        },
    }
    structures = {"global:" + r: s for r, s in source["components"].items()}
    before = copy.deepcopy(item)
    assert replay_record(item, structures, contract[2][FAMILY])["computed_consistency_pass"]
    assert item == before
    item["preparation"]["component_instances"][0][2] = 2
    assert (
        replay_record(item, structures, contract[2][FAMILY])["disposition"]
        == "outside_qualified_program_multiplicity"
    )


def test_atom_order_does_not_change_source_admission(contract, source):
    def reorder(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(
            Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms())))), canonical=False
        )

    result = contract[2][FAMILY]["program"].replay(
        {r: reorder(s) for r, s in source["components"].items()},
        reorder(source["expected_product"]),
    )
    assert result["computed_consistency_pass"]


def test_incomplete_search_never_passes(contract, source):
    program = replace(contract[2][FAMILY]["program"], bounds=RepeatBounds(maximum_states=1))
    result = program.replay(source["components"], source["expected_product"])
    assert not result["computed_consistency_pass"]
    assert not result["checks"]["complete_search"]
