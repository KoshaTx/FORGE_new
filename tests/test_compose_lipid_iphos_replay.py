"""iPhos charge forms, per-nitrogen occupancy and ambiguity are independent gates."""

import json
from pathlib import Path

import pytest

from forge.assembly.repeated_components import RepeatBounds, replay_repeated_components
from forge.corpus.compose_lipid_current_replay import load_contract

ROOT = Path(__file__).resolve().parents[1]
FAMILY = "iphos_ring_opening"


@pytest.fixture(scope="module", params=["neutral_proton_transfer", "zwitterion"])
def contract(request):
    return request.param, load_contract(
        ROOT, ROOT / f"configs/multireaction/compose_lipid_supplied_iphos_{request.param}_v1.json"
    )


def run(contract, head, target, events=1, bounds=None):
    _, c = contract
    p = c[3][FAMILY]
    return replay_repeated_components(
        c[2][FAMILY],
        {"amine_head": head, "phosphate_tail": "COP1(=O)OCCO1"},
        target,
        accumulator_role=p["accumulator_role"],
        events=events,
        byproducts_per_event=p["net_byproducts_per_event"],
        bounds=bounds or c[4],
    )


def test_independent_source_graphs_and_all_atom_inventories_close(contract):
    form, c = contract
    for label in ("9A1P9", "10A1P10"):
        result = c[-1][label + "_" + form]["replay"]
        assert result["computed_consistency_pass"]
        assert result["checks"]["full_element_hydrogen_charge_balance"]
        assert not result["experimental_selectivity_qualified"]


def test_partial_occupancy_remains_ambiguous(contract):
    form, c = contract
    result = c[-1]["non_equivalent_nitrogens_" + form]["replay"]
    assert len(result["forward_layers"][-1]) == 2
    assert not result["computed_consistency_pass"]


def test_two_complete_primary_nitrogens_each_open_once(contract):
    form, _ = contract
    target = (
        "COP(=O)(O)OCCNCCNCCOP(=O)(O)OC"
        if form == "neutral_proton_transfer"
        else "COP(=O)([O-])OCC[NH2+]CC[NH2+]CCOP(=O)([O-])OC"
    )
    assert run(contract, "NCCN", target, 2)["computed_consistency_pass"]
    third = run(contract, "NCCN", target, 3)
    assert third["forward_layers"][-1] == []
    assert not third["computed_consistency_pass"]


def test_single_primary_nitrogen_cannot_take_two_units(contract):
    form, _ = contract
    target = (
        "CCCCNCCOP(=O)(O)OC" if form == "neutral_proton_transfer" else "CCCC[NH2+]CCOP(=O)([O-])OC"
    )
    assert run(contract, "CCCCN", target)["computed_consistency_pass"]
    result = run(contract, "CCCCN", target, 2)
    assert result["forward_layers"][-1] == []


def test_tertiary_amine_requires_permanent_zwitterion_form(contract):
    form, _ = contract
    result = run(contract, "CN(C)C", "COP(=O)([O-])OCC[N+](C)(C)C")
    assert result["computed_consistency_pass"] is (form == "zwitterion")


@pytest.mark.parametrize(
    "tail", ["CP1(=O)OCCO1", "CNP1(=O)OCCO1", "CC(=O)OP1(=O)OCCO1", "O=P1(OCCO1)OCCOP1(=O)OCCO1"]
)
def test_nonalkoxy_or_multiple_ring_phosphates_are_not_complete_pm(contract, tail):
    _, c = contract
    assert (
        not c[2][FAMILY].forward_products({"amine_head": "CCCCN", "phosphate_tail": tail}).products
    )


@pytest.mark.parametrize("head", ["CC(=O)NC", "CNC(=O)OC", "c1ccncc1", "C[N+](C)(C)C"])
def test_acyl_aromatic_and_quaternary_nitrogens_are_not_new_amine_ports(contract, head):
    _, c = contract
    assert (
        not c[2][FAMILY]
        .forward_products({"amine_head": head, "phosphate_tail": "COP1(=O)OCCO1"})
        .products
    )


def test_charge_is_not_normalized_to_manufacture_exact_match(contract):
    form, _ = contract
    wrong = (
        "CCCC[NH2+]CCOP(=O)([O-])OC" if form == "neutral_proton_transfer" else "CCCCNCCOP(=O)(O)OC"
    )
    result = run(contract, "CCCCN", wrong)
    assert not result["computed_consistency_pass"]
    assert result["checks"]["full_element_hydrogen_charge_balance"]


def test_saturated_search_is_never_admitted(contract):
    result = run(contract, "CCCCN", "CCCCNCCOP(=O)(O)OC", bounds=RepeatBounds(maximum_outcomes=1))
    assert not result["checks"]["complete_search"]
    assert not result["computed_consistency_pass"]


def test_printed_analytical_conflict_is_preserved(contract):
    _, c = contract
    doc = json.loads(c[1]["transform_controls"].read_text())
    assert any("563.6" in item for item in doc["limitations"])
    assert not doc["experimental_execution_admitted"]


@pytest.fixture(scope="module")
def anionic_contract():
    return load_contract(
        ROOT, ROOT / "configs/multireaction/compose_lipid_supplied_iphos_deprotonated_v1.json"
    )


def test_source_anionic_form_requires_explicit_released_proton(anionic_contract):
    c = anionic_contract
    parts = {"amine_head": "CCCCN", "phosphate_tail": "COP1(=O)OCCO1"}
    target = "CCCCNCCOP(=O)([O-])OC"
    correct = replay_repeated_components(
        c[2][FAMILY],
        parts,
        target,
        accumulator_role="amine_head",
        events=1,
        byproducts_per_event={"H": 1, "formal_charge": 1},
        bounds=c[4],
    )
    assert correct["computed_consistency_pass"]
    missing = replay_repeated_components(
        c[2][FAMILY],
        parts,
        target,
        accumulator_role="amine_head",
        events=1,
        byproducts_per_event={"formal_charge": 0},
        bounds=c[4],
    )
    assert not missing["checks"]["full_element_hydrogen_charge_balance"]
    assert not missing["computed_consistency_pass"]


def test_anionic_program_does_not_rescue_tertiary_amine(anionic_contract):
    c = anionic_contract
    assert (
        not c[2][FAMILY]
        .forward_products({"amine_head": "CN(C)C", "phosphate_tail": "COP1(=O)OCCO1"})
        .products
    )


def test_anionic_control_admits_no_positive_ion_analytical_label(anionic_contract):
    c = anionic_contract
    doc = json.loads(c[1]["transform_controls"].read_text())
    assert all(x["source_analytical_form_is_not_this_anion"] for x in doc["source_controls"])
    assert all(
        v["replay"]["computed_consistency_pass"]
        for v in c[-1].values()
        if v["kind"] == "source_controls"
    )
    assert not doc["experimental_execution_admitted"]


def test_anionic_charge_is_balanced_per_event_without_repeated_attack_on_same_n(anionic_contract):
    c = anionic_contract
    parts = {"amine_head": "NCCN", "phosphate_tail": "COP1(=O)OCCO1"}
    target = "COP(=O)([O-])OCCNCCNCCOP(=O)([O-])OC"
    result = replay_repeated_components(
        c[2][FAMILY],
        parts,
        target,
        accumulator_role="amine_head",
        events=2,
        byproducts_per_event=c[3][FAMILY]["net_byproducts_per_event"],
        bounds=c[4],
    )
    assert result["computed_consistency_pass"]
    assert result["balance"]["reactants"] == result["balance"]["product_and_byproducts"]
