"""Independent source structure, occupancy, scaffold identity, and protected-row guards."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import constitutional_molecule
from forge.assembly.repeated_components import RepeatBounds
from forge.corpus.compose_lipid_aema_replay import (
    FAMILY,
    component_domain,
    load_contract,
    replay_record,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "results/phase1/compose_lipid_aema_source_v1/replay-config.json"


@pytest.fixture(scope="module")
def contract():
    return load_contract(ROOT, CONFIG)


@pytest.fixture
def source(contract):
    return json.loads(contract[1]["source_control"].read_text())


def prepared(source):
    structures = {f"global:{r}": s for r, s in source["components"].items()}
    item = {
        "source": {"constitution": source["expected_product"]},
        "preparation": {
            "family": FAMILY,
            "eligible_for_program_preparation": True,
            "construction_basis": "v8_exact_family_task",
            "constitution_id": hashlib.sha256(
                constitutional_molecule(source["expected_product"])[0].encode()
            ).hexdigest(),
            "component_instances": [
                [r, f"global:{r}", n] for r, n in source["declared_quantities"].items()
            ],
        },
    }
    return item, structures


def test_source_drawing_closes_both_ordered_stages(contract, source):
    result = next(iter(contract[3].values()))
    assert result["computed_consistency_pass"]
    assert all(result["checks"].values())
    assert list(map(len, result["forward_layers"])) == [1, 1, 1]
    assert result["declared_quantities"] == source["declared_quantities"]
    assert result["balance"]["reactants"] == result["balance"]["product_and_net_byproducts"]
    assert result["balance"]["reactants"]["C"] == 67


def test_global_ids_quantities_and_inputs_are_unchanged(contract, source):
    item, structures = prepared(source)
    before = copy.deepcopy(item)
    result = replay_record(item, structures, contract[2])
    assert result["computed_consistency_pass"]
    assert result["verified_target_constitution_id"] == item["preparation"]["constitution_id"]
    assert result["source_occupancy"] == 4
    assert not result["training_admitted"]
    assert not result["experimental_execution_admitted"]
    assert item == before


def test_protected_row_rejected_before_decoding_any_structure(contract):
    with pytest.raises(ComposeLipidError, match="Protected"):
        replay_record({"preparation": {"eligible_for_program_preparation": False}}, {}, contract[2])


@pytest.mark.parametrize("quantity", [0, True, 1.0])
def test_invalid_quantities_are_not_coerced(contract, source, quantity):
    item, structures = prepared(source)
    item["preparation"]["component_instances"][0][2] = quantity
    with pytest.raises(ComposeLipidError, match="quantity"):
        replay_record(item, structures, contract[2])


def test_maximum_nh_or_target_cannot_replace_source_occupancy(contract, source):
    item, structures = prepared(source)
    for row in item["preparation"]["component_instances"]:
        if row[0] != "amine_core":
            row[2] = 3
    result = replay_record(item, structures, contract[2])
    assert result["source_occupancy"] == 3
    assert not result["computed_consistency_pass"]
    assert not result["checks"]["full_element_hydrogen_charge_balance"]


def test_source_basis_is_required(contract, source):
    item, structures = prepared(source)
    item["preparation"]["construction_basis"] = "inferred_from_target"
    assert (
        replay_record(item, structures, contract[2])["disposition"]
        == "missing_original_task_occupancy"
    )


def test_incorporated_aema_and_thiol_counts_must_match(contract, source):
    item, structures = prepared(source)
    for row in item["preparation"]["component_instances"]:
        if row[0] == "thiol_periphery":
            row[2] = 5
    assert (
        replay_record(item, structures, contract[2])["disposition"]
        == "outside_qualified_program_multiplicity"
    )


def test_formula_equivalent_wrong_fixed_scaffold_fails(contract, source):
    item, structures = prepared(source)
    structures["global:AEMA"] = "C=CC(=O)OCCCOC(=O)C=C"
    assert (
        replay_record(item, structures, contract[2])["disposition"]
        == "wrong_complete_AEMA_scaffold"
    )


def test_mixed_peripheral_identities_are_not_collapsed(contract, source):
    item, structures = prepared(source)
    structures["different"] = "CCCCCCS"
    item["preparation"]["component_instances"].append(["thiol_periphery", "different", 1])
    assert (
        replay_record(item, structures, contract[2])["disposition"]
        == "unsupported_source_role_tuple"
    )


@pytest.mark.parametrize(
    "core,thiol",
    [
        ("CN(CCCN)CCCN", "CCCCOC(=O)CCCCS"),
        ("CN(CCCN)CCCN", "NCCCCS"),
        ("NCCCn1ccnc1", "CCCCCCS"),
        ("CN(CCCN)CCCN", "C#CCCCCS"),
        ("CN(CCCN)CCCN", "SCCCCS"),
    ],
)
def test_unreviewed_components_stay_outside_transfer_domain(contract, core, thiol):
    assert not component_domain(core, thiol, contract[2][4])


def test_wrong_methacrylate_attachment_is_not_selected_by_target(contract, source):
    wrong = source["expected_product"].replace("C(C)CSCCCCC=C", "C(CC)SCCCCC=C")
    result = contract[2][4]["program"].replay(source["components"], wrong)
    assert result["checks"]["full_element_hydrogen_charge_balance"]
    assert not result["checks"]["unique_forward_exact"]
    assert not result["computed_consistency_pass"]


def test_stage_order_cannot_be_reversed(contract, source):
    program = contract[2][4]["program"]
    spec = copy.deepcopy(program.specification)
    spec["stages"].reverse()
    reversed_program = replace(
        program, specification=spec, adapters=tuple(reversed(program.adapters))
    )
    assert not reversed_program.replay(source["components"], source["expected_product"])[
        "computed_consistency_pass"
    ]


def test_atom_order_does_not_change_admission(contract, source):
    def reorder(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(
            Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms())))), canonical=False
        )

    source["components"] = {r: reorder(s) for r, s in source["components"].items()}
    source["expected_product"] = reorder(source["expected_product"])
    item, structures = prepared(source)
    assert replay_record(item, structures, contract[2])["computed_consistency_pass"]


def test_incomplete_search_cannot_pass(contract, source):
    program = replace(
        contract[2][4]["program"],
        bounds=RepeatBounds(maximum_events=12, maximum_states=2),
    )
    result = program.replay(source["components"], source["expected_product"])
    assert not result["computed_consistency_pass"]
    assert not result["checks"]["complete_search"]


def test_authenticated_target_identity_is_required(contract, source):
    item, structures = prepared(source)
    item["preparation"]["constitution_id"] = "0" * 64
    with pytest.raises(ComposeLipidError, match="identity"):
        replay_record(item, structures, contract[2])
