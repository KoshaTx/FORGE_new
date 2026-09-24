"""Source controls and adversarial repeated A3 chemistry checks."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem

from forge.assembly.families import LibraryAssemblyError, RegistryAssemblyAdapter
from forge.assembly.repeated_components import RepeatBounds, replay_repeated_components
from forge.core.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/vendor/qualified_a3_source_program_v1.json"
SOURCE = ROOT / "results/phase1/compose_lipid_v8_a3_source_v1/adjudication.json"


@pytest.fixture
def adapter():
    return RegistryAssemblyAdapter.from_registry(
        REGISTRY,
        reaction_id="a3_amine_aldehyde_terminal_alkyne",
        expected_sha256=str(sha256_file(REGISTRY)),
    )


def replay(adapter, components, target, events, **kwargs):
    return replay_repeated_components(
        adapter,
        components,
        target,
        accumulator_role="amine_head",
        events=events,
        byproducts_per_event=json.loads(REGISTRY.read_text())["curation"]["byproducts_per_event"],
        **kwargs,
    )


@pytest.mark.parametrize("index", [0, 1])
def test_independently_transcribed_source_controls(adapter, index):
    control = json.loads(SOURCE.read_text())["source_controls"][index]
    result = replay(adapter, control["components"], control["expected_product"], control["events"])
    assert result["computed_consistency_pass"]
    assert len(result["forward_layers"]) == control["events"] + 1
    assert len(result["reverse_layers"]) == control["events"] + 1
    assert not result["experimental_selectivity_qualified"]
    assert not result["physical_event_order_qualified"]


@pytest.mark.parametrize(
    "events,head,target",
    [
        (4, "NCCCN", "CC#CCN(CC#CC)CCCN(CC#CC)CC#CC"),
        (6, "NCCN(CCN)CCN", "CC#CCN(CC#CC)CCN(CCN(CC#CC)CC#CC)CCN(CC#CC)CC#CC"),
    ],
)
def test_full_repetition_can_pass_after_ambiguous_intermediates(adapter, events, head, target):
    result = replay(
        adapter, {"amine_head": head, "aldehyde": "C=O", "alkyne": "C#CC"}, target, events
    )
    assert result["computed_consistency_pass"]
    assert any(len(layer) > 1 for layer in result["forward_layers"])
    assert len(result["forward_layers"][-1]) == 1


def test_does_not_select_target_site_from_ambiguous_outcomes(adapter):
    result = replay(
        adapter,
        {"amine_head": "NCCCN", "aldehyde": "C=O", "alkyne": "C#CC"},
        "CC#CCN(CC#CC)CCCN",
        2,
    )
    assert result["checks"]["full_element_hydrogen_charge_balance"]
    assert len(result["forward_layers"][-1]) == 2
    assert not result["checks"]["unique_forward_exact"]
    assert not result["computed_consistency_pass"]


def test_declared_count_cannot_be_replaced_by_target_fit(adapter):
    control = json.loads(SOURCE.read_text())["source_controls"][0]
    result = replay(adapter, control["components"], control["expected_product"], 1)
    assert not result["computed_consistency_pass"]
    assert not result["checks"]["full_element_hydrogen_charge_balance"]


@pytest.mark.parametrize(
    "role,substitute",
    [
        ("amine_head", "CN(C)C"),
        ("amine_head", "CC(=O)N"),
        ("aldehyde", "CC(C)=O"),
        ("alkyne", "CC#CC"),
    ],
)
def test_unqualified_reactive_handles_do_not_produce_a_program(adapter, role, substitute):
    components = {"amine_head": "CN", "aldehyde": "C=O", "alkyne": "C#CC"}
    components[role] = substitute
    result = replay(adapter, components, "CC#CCNC", 1)
    assert not result["computed_consistency_pass"]
    assert not result["checks"]["declared_event_count_replayed"]
    assert result["forward_layers"][-1] == []


def test_aldehyde_role_does_not_admit_another_carbonyl_site(adapter):
    components = {"amine_head": "CNC", "aldehyde": "O=CCC(=O)C", "alkyne": "C#CC"}
    products = adapter.forward_products(components)
    assert len(products.products) == 1
    assert products.products == (Chem.MolToSmiles(Chem.MolFromSmiles("CC#CC(CC(=O)C)N(C)C")),)


@pytest.mark.parametrize(
    "bound",
    [
        {"maximum_outcomes": 1},
        {"maximum_states": 1},
        {"maximum_transitions": 1},
    ],
)
def test_saturation_never_admits_partial_replay(adapter, bound):
    control = json.loads(SOURCE.read_text())["source_controls"][0]
    result = replay(
        adapter,
        control["components"],
        control["expected_product"],
        2,
        bounds=replace(RepeatBounds(), **bound),
    )
    assert result["bound_reasons"]
    assert not result["checks"]["complete_search"]
    assert not result["computed_consistency_pass"]


def test_atom_order_and_smiles_order_do_not_change_replay(adapter):
    control = json.loads(SOURCE.read_text())["source_controls"][0]
    components = control["components"]
    expected = replay(adapter, components, control["expected_product"], 2)
    permuted = {}
    for role, value in components.items():
        mol = Chem.MolFromSmiles(value)
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        permuted[role] = Chem.MolToSmiles(mol, canonical=False)
    assert replay(adapter, permuted, control["expected_product"], 2) == expected


@pytest.mark.parametrize("events", [True, 0, -1, 1.5, 33])
def test_invalid_event_counts_fail_loudly(adapter, events):
    with pytest.raises(LibraryAssemblyError, match="event count"):
        replay(
            adapter, {"amine_head": "CN", "aldehyde": "C=O", "alkyne": "C#CC"}, "CC#CCNC", events
        )


def test_missing_role_is_not_a_partial_program(adapter):
    with pytest.raises(LibraryAssemblyError, match="complete registry roles"):
        replay(adapter, {"amine_head": "CN", "aldehyde": "C=O"}, "CC#CCNC", 1)
