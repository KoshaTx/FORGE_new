"""Source drawings, ambiguity and protection constrain newly added family replay."""

import copy
import json
from pathlib import Path

import pytest
from rdkit import Chem

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.repeated_components import RepeatBounds, replay_repeated_components
from forge.corpus.compose_lipid_current_replay import load_contract, qualify_controls
from forge.corpus.compose_lipid_supplied_replay import replay_supplied

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/multireaction/compose_lipid_supplied_maleate_ester_v1.json"


@pytest.fixture(scope="module")
def contract():
    return load_contract(ROOT, CONFIG)


def run(contract, family, components, product, events, bounds=None):
    _, _, adapters, programs, default_bounds, _ = contract
    program = programs[family]
    return replay_repeated_components(
        adapters[family],
        components,
        product,
        accumulator_role=program["accumulator_role"],
        events=events,
        byproducts_per_event=program["net_byproducts_per_event"],
        bounds=bounds or default_bounds,
    )


@pytest.mark.parametrize("label", ["maleate_5D8", "esterification_AA3_DLin"])
def test_drawn_source_products_close_forward_inverse_and_atom_balance(contract, label):
    result = contract[-1][label]["replay"]
    assert result["computed_consistency_pass"]
    assert all(result["checks"].values())
    assert not result["experimental_selectivity_qualified"]


@pytest.mark.parametrize(
    "label",
    ["asymmetric_maleate_no_orientation_selection", "nonequivalent_alcohols_no_site_selection"],
)
def test_ambiguous_sites_remain_ambiguous_even_when_target_matches(contract, label):
    result = contract[-1][label]["replay"]
    assert not result["computed_consistency_pass"]
    assert len(result["forward_layers"][-1]) == 2
    assert result["reverse_layers"][0][0] in result["forward_layers"][-1]
    assert result["checks"]["full_element_hydrogen_charge_balance"]


def test_primary_amine_cannot_take_second_maleate_at_same_nitrogen(contract):
    parts = {"amine_head": "CN", "maleate": "COC(=O)C=CC(=O)OC"}
    mono = "CNC(CC(=O)OC)C(=O)OC"
    assert run(contract, "maleate_addition", parts, mono, 1)["computed_consistency_pass"]
    double = "CN(C(CC(=O)OC)C(=O)OC)C(CC(=O)OC)C(=O)OC"
    result = run(contract, "maleate_addition", parts, double, 2)
    assert not result["computed_consistency_pass"]
    assert result["forward_layers"][-1] == []
    assert result["checks"]["full_element_hydrogen_charge_balance"]


@pytest.mark.parametrize("head", ["CN(C)C", "CC(=O)NC", "CN(C)CCS"])
def test_tertiary_amide_and_thiol_are_not_maleate_amine_sites(contract, head):
    adapter = contract[2]["maleate_addition"]
    assert not adapter.forward_products(
        {"amine_head": head, "maleate": "COC(=O)C=CC(=O)OC"}
    ).products


@pytest.mark.parametrize("head", ["NCCO", "CNCCO", "OCc1ccccc1N"])
def test_competing_free_amine_is_outside_o_esterification_scope(contract, head):
    adapter = contract[2]["o_esterification"]
    assert not adapter.forward_products(
        {"aminoalcohol_head": head, "acid_tail": "CCCC(=O)O"}
    ).products


@pytest.mark.parametrize("acid", ["O=C(O)CCC(=O)O", "CC(=O)OC", "OCCCC(=O)O"])
def test_polyacid_ester_or_competing_alcohol_acid_is_not_one_acid_tail(contract, acid):
    adapter = contract[2]["o_esterification"]
    assert not adapter.forward_products(
        {"aminoalcohol_head": "CN(C)CCO", "acid_tail": acid}
    ).products


@pytest.mark.parametrize("index", [0, 1])
def test_atom_order_and_smiles_serialization_do_not_change_disposition(contract, index):
    doc = json.loads(contract[1]["transform_controls"].read_text())
    c = doc["source_controls"][index]
    alternate = {}
    for role, smiles in c["components"].items():
        molecule = Chem.MolFromSmiles(smiles)
        reordered = Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms()))))
        alternate[role] = Chem.MolToSmiles(reordered, canonical=False)
    assert run(contract, c["family"], alternate, c["expected_product"], c["events"])[
        "computed_consistency_pass"
    ]


def test_enumeration_bound_never_admits_partial_match(contract):
    c = json.loads(contract[1]["transform_controls"].read_text())["source_controls"][0]
    result = run(
        contract,
        c["family"],
        c["components"],
        c["expected_product"],
        c["events"],
        RepeatBounds(maximum_outcomes=1),
    )
    assert not result["computed_consistency_pass"]
    assert not result["checks"]["complete_search"]


@pytest.mark.parametrize("corruption", ["formula", "omit_ambiguity", "reaction", "expected_pass"])
def test_control_tampering_fails_before_corpus_open(contract, corruption):
    doc = json.loads(contract[1]["transform_controls"].read_text())
    if corruption == "formula":
        doc["source_controls"][0]["expected_formula"] = "C"
    if corruption == "omit_ambiguity":
        doc["ambiguity_controls"].pop()
    if corruption == "reaction":
        doc["source_controls"][0]["reaction_id"] = "other"
    if corruption == "expected_pass":
        doc["ambiguity_controls"][0]["expected_computed_consistency_pass"] = True
    with pytest.raises(ComposeLipidError):
        qualify_controls(doc, contract[2], contract[3], contract[4])


def test_protected_product_is_never_parsed(contract):
    family = "maleate_addition"
    with pytest.raises(ComposeLipidError, match="Protected target"):
        replay_supplied(
            contract[2][family],
            {
                "source": {"constitution": "invalid"},
                "preparation": {"eligible_for_program_preparation": False},
            },
            {},
            contract[0]["families"][family],
            contract[3][family],
            contract[4],
        )


def test_registry_pin_cannot_be_replaced(contract, tmp_path):
    doc = copy.deepcopy(contract[0])
    doc["inputs"]["registry"]["sha256"] = "0" * 64
    cfg = tmp_path / "bad.json"
    cfg.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="[Hh]ash|[Ss][Hh][Aa]"):
        load_contract(ROOT, cfg)


@pytest.fixture(scope="module")
def thiol_a3_contract():
    return load_contract(
        ROOT, ROOT / "configs/multireaction/compose_lipid_supplied_thiol_a3_v1.json"
    )


@pytest.mark.parametrize("label", ["maleate_library_49D8", "Han_11(aB)2", "Han_31hP-M"])
def test_thiol_and_a3_source_controls_close(thiol_a3_contract, label):
    assert thiol_a3_contract[-1][label]["replay"]["computed_consistency_pass"]


@pytest.mark.parametrize(
    "label", ["asymmetric_maleate_thiol_ambiguous_orientation", "a3_unresolved_amine_site"]
)
def test_separate_thiol_and_a3_ambiguity_controls_abstain(thiol_a3_contract, label):
    result = thiol_a3_contract[-1][label]["replay"]
    assert not result["computed_consistency_pass"]
    assert len(result["forward_layers"][-1]) == 2


@pytest.mark.parametrize("head", ["CNC", "NCCS", "SCCS"])
def test_thiol_program_rejects_amine_competitors_and_multiple_thiols(thiol_a3_contract, head):
    adapter = thiol_a3_contract[2]["maleate_addition"]
    assert not adapter.forward_products(
        {"thiol_head": head, "maleate": "COC(=O)C=CC(=O)OC"}
    ).products


@pytest.mark.parametrize("aldehyde", ["O=CO", "O=CN", "CC(=O)C", "O=CCC=O"])
def test_a3_requires_one_complete_aldehyde(thiol_a3_contract, aldehyde):
    adapter = thiol_a3_contract[2]["a3_amine_aldehyde_alkyne"]
    assert not adapter.forward_products(
        {"amine_head": "CNC", "aldehyde": aldehyde, "alkyne": "C#CC"}
    ).products


def test_a3_architecture_conflict_is_preserved(thiol_a3_contract):
    registry = json.loads(thiol_a3_contract[1]["registry"].read_text())
    a3 = next(
        r
        for r in registry["reactions"]
        if r["reaction_id"] == "source_fixed_components_repeated_a3"
    )
    assert not a3["architecture_qualified"]
    assert any(
        c["id"] == "han_asymmetric_event_count" and c["disposition"] == "reject_claim"
        for c in a3["architecture_conflicts"]
    )


def test_different_repeated_a3_quantities_cannot_be_target_selected(thiol_a3_contract):
    family = "a3_amine_aldehyde_alkyne"
    item = {
        "source": {"constitution": "invalid", "primary_metadata": {}},
        "preparation": {
            "eligible_for_program_preparation": True,
            "component_instances": [
                ["amine_head", "h", 1],
                ["aldehyde", "a", 2],
                ["alkyne", "t", 1],
            ],
        },
    }
    with pytest.raises(ComposeLipidError, match="inconsistent quantities"):
        replay_supplied(
            thiol_a3_contract[2][family],
            item,
            {"h": "CN", "a": "C=O", "t": "C#CC"},
            thiol_a3_contract[0]["families"][family],
            thiol_a3_contract[3][family],
            thiol_a3_contract[4],
        )
