"""Full repeated inverse checks use source controls and never infer missing occupancy."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.assembly.families import LibraryAssemblyError, RegistryAssemblyAdapter
from forge.assembly.repeated_components import RepeatBounds, replay_repeated_components
from forge.assembly.repeated_inverse import infer_repeated_components
from forge.core.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/vendor/qualified_michael_source_program_v1.json"
SOURCE = ROOT / "results/phase1/compose_lipid_v8_michael_source_v1/adjudication.json"


def adapter(family):
    return RegistryAssemblyAdapter.from_registry(
        REGISTRY, reaction_id="source_" + family, expected_sha256=str(sha256_file(REGISTRY))
    )


def inverse(adapt, target, events, **kwargs):
    return infer_repeated_components(
        adapt, target, accumulator_role="amine_head", events=events, **kwargs
    )


def replay(adapt, components, target, events):
    return replay_repeated_components(
        adapt,
        components,
        target,
        accumulator_role="amine_head",
        events=events,
        byproducts_per_event={"formal_charge": 0},
    )


@pytest.mark.parametrize("index", [0, 1, 2])
def test_published_structures_and_formulas_recover_complete_precursors(index):
    control = json.loads(SOURCE.read_text())["source_controls"][index]
    adapt = adapter(control["family"])
    found = inverse(adapt, control["expected_product"], control["events"])
    assert found["complete_search"]
    assert found["candidate_components"] == [control["components"]]
    assert replay(adapt, control["components"], control["expected_product"], control["events"])[
        "computed_consistency_pass"
    ]
    assert (
        rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(control["expected_product"]))
        == control["expected_formula"]
    )
    assert not found["physical_event_order_qualified"]
    assert not found["experimental_selectivity_qualified"]


def test_published_five_tail_isomer_requires_source_resolved_sites():
    control = json.loads(SOURCE.read_text())["ambiguity_controls"][0]
    adapt = adapter(control["family"])
    found = inverse(adapt, control["expected_product"], 5)
    assert found["candidate_components"] == [control["components"]]
    result = replay(adapt, control["components"], control["expected_product"], 5)
    assert len(result["forward_layers"][-1]) == control["expected_forward_product_count"]
    assert not result["computed_consistency_pass"]
    assert result["checks"]["full_element_hydrogen_charge_balance"]
    assert (
        rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(control["expected_product"]))
        == control["expected_formula"]
    )


@pytest.mark.parametrize(
    "family,tail,target",
    [
        ("aza_michael_acrylate", "C=CC(=O)OC", "COC(=O)CCN(CCC(=O)OC)CCN(CCC(=O)OC)CCC(=O)OC"),
        ("aza_michael_acrylamide", "C=CC(=O)NC", "CNC(=O)CCN(CCC(=O)NC)CCN(CCC(=O)NC)CCC(=O)NC"),
    ],
)
def test_complete_four_tail_program_recovers_head_without_counting_tail_amides(
    family, tail, target
):
    adapt = adapter(family)
    result = inverse(adapt, target, 4)
    assert result["candidate_components"] == [{"amine_head": "NCCN", "acceptor_tail": tail}]
    assert replay(adapt, result["candidate_components"][0], target, 4)["computed_consistency_pass"]


@pytest.mark.parametrize(
    "family,other",
    [("aza_michael_acrylate", "C=CC(=O)NCCCC"), ("aza_michael_acrylamide", "C=CC(=O)OCCCC")],
)
def test_ester_and_amide_interfaces_cannot_be_pooled(family, other):
    assert (
        not adapter(family).forward_products({"amine_head": "CN", "acceptor_tail": other}).products
    )


@pytest.mark.parametrize(
    "head,tail",
    [
        ("CC(=O)N", "C=CC(=O)OCCC"),
        ("CN(C)C", "C=CC(=O)OCCC"),
        ("CN", "C=CC(=O)OCCN"),
        ("NCCOC(=O)C=C", "C=CC(=O)OCCC"),
        ("CN", "C=CC(=O)OCCOC(=O)C=C"),
    ],
)
def test_unsupported_nucleophile_and_cross_reaction_sites_are_excluded(head, tail):
    assert (
        not adapter("aza_michael_acrylate")
        .forward_products({"amine_head": head, "acceptor_tail": tail})
        .products
    )


def test_two_different_tails_are_not_one_repeated_component():
    adapt = adapter("aza_michael_acrylate")
    target = "CN(CCC(=O)OC)CCC(=O)OCC"
    assert not inverse(adapt, target, 2)["candidate_components"]


def test_more_than_one_complete_tuple_is_retained():
    adapt = adapter("aza_michael_acrylate")
    target = "CN(CCC(=O)OC)CCC(=O)OCC"
    found = inverse(adapt, target, 1)
    assert len(found["candidate_components"]) == 2
    assert {c["acceptor_tail"] for c in found["candidate_components"]} == {
        "C=CC(=O)OC",
        "C=CC(=O)OCC",
    }


@pytest.mark.parametrize(
    "limit", [{"maximum_outcomes": 1}, {"maximum_states": 1}, {"maximum_transitions": 1}]
)
def test_truncation_discards_all_admissible_candidates(limit):
    control = json.loads(SOURCE.read_text())["source_controls"][0]
    found = inverse(
        adapter(control["family"]),
        control["expected_product"],
        2,
        bounds=replace(RepeatBounds(), **limit),
    )
    assert not found["complete_search"]
    assert not found["candidate_components"]
    assert found["bound_reasons"]


@pytest.mark.parametrize("events", [True, 0, -1, 1.5, 33])
def test_invalid_event_count_fails_loudly(events):
    with pytest.raises(LibraryAssemblyError, match="event count"):
        inverse(adapter("aza_michael_acrylate"), "COC(=O)CCNC", events)


def test_input_atom_order_does_not_change_inverse_witnesses():
    control = json.loads(SOURCE.read_text())["source_controls"][2]
    adapt = adapter(control["family"])
    mol = Chem.MolFromSmiles(control["expected_product"])
    permuted = Chem.MolToSmiles(
        Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms())))), canonical=False
    )
    assert inverse(adapt, permuted, 2) == inverse(adapt, control["expected_product"], 2)
