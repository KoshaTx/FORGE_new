"""Source-scaffold checks preserve coupling, position, complete arms and ambiguity."""

import copy
import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.assembly.families import LibraryAssemblyError, RegistryAssemblyAdapter
from forge.assembly.precursor_scaffolds import assess_precursor_scaffold
from forge.assembly.repeated_components import replay_repeated_components
from forge.core.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/vendor/qualified_reductive_source_program_v1.json"
CONTROLS = ROOT / "results/phase1/compose_lipid_v8_reductive_source_v1/controls.json"


def reaction(family):
    return next(
        r
        for r in json.loads(REGISTRY.read_text())["reactions"]
        if r["reaction_id"] == "source_" + family
    )


def adapter(family):
    return RegistryAssemblyAdapter.from_registry(
        REGISTRY, reaction_id="source_" + family, expected_sha256=str(sha256_file(REGISTRY))
    )


@pytest.mark.parametrize("index", [0, 1, 2])
def test_independent_primary_products_and_formulas_reconstruct(index):
    c = json.loads(CONTROLS.read_text())["controls"][index]
    a = adapter(c["family"])
    candidates = a.decompose(c["expected_product"], maximum_outcomes=256)
    assert len(candidates) == 1
    result = replay_repeated_components(
        a,
        dict(candidates[0].components),
        c["expected_product"],
        accumulator_role="amine_head",
        events=1,
        byproducts_per_event=reaction(c["family"])["precursor_scaffolds"][
            "net_byproducts_per_event"
        ],
    )
    assert result["computed_consistency_pass"]
    assert (
        rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(c["expected_product"]))
        == c["expected_formula"]
    )
    assert assess_precursor_scaffold(c["components"]["coupled_aldehyde"], reaction(c["family"]))[
        "computed_scaffold_pass"
    ]


def test_a3_source_precursor_is_three_arms_at_246_positions():
    c = json.loads(CONTROLS.read_text())["precursor_controls"][0]
    result = assess_precursor_scaffold(c["smiles"], reaction(c["family"]))
    assert result["computed_scaffold_pass"]
    witness = result["witnesses"][0]
    assert witness["scaffold_id"] == c["expected_scaffold"]
    assert len(witness["detached_arm_fragments"]) == 3
    assert rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(c["smiles"])) == c["expected_formula"]


@pytest.mark.parametrize("index", [0, 2])
def test_unequal_arms_are_retained_as_a_failed_witness(index):
    c = json.loads(CONTROLS.read_text())["controls"][index]
    smiles = c["components"]["coupled_aldehyde"].replace("OC(=O)C", "OC(=O)CC", 1)
    result = assess_precursor_scaffold(smiles, reaction(c["family"]))
    assert result["complete_search"]
    assert len(result["witnesses"]) == 1
    assert not result["witnesses"][0]["checks"]["identical_arms"]
    assert not result["computed_scaffold_pass"]


@pytest.mark.parametrize(
    "smiles",
    [
        "O=Cc1ccc(OC(=O)CCCC)cc1OC(=O)CCCC",  # 2,4 does not become source 3,5.
        "O=Cc1cc(OC(=O)CCCC)c(OC(=O)CCCC)cc1OC(=O)CCCC",  # 2,4,5 is not 2,4,6.
        "O=CCCCCCC",  # Generic aldehyde is not a complete source precursor.
    ],
)
def test_wrong_positions_or_generic_aldehyde_do_not_qualify(smiles):
    assert not assess_precursor_scaffold(smiles, reaction("aryl_reductive_amination"))[
        "computed_scaffold_pass"
    ]


def test_aliphatic_and_aryl_scaffolds_are_not_pooled():
    controls = json.loads(CONTROLS.read_text())["controls"]
    for control, other in (
        (controls[0], "aryl_reductive_amination"),
        (controls[2], "reductive_amination"),
    ):
        assert not adapter(other).decompose(control["expected_product"])


def test_extra_aldehyde_does_not_hide_inside_an_allowed_core():
    smiles = "O=Cc1cc(OC(=O)CCCC=O)cc(OC(=O)CCCC=O)c1"
    result = assess_precursor_scaffold(smiles, reaction("aryl_reductive_amination"))
    assert len(result["witnesses"]) == 1
    assert not result["additional_handle_count_pass"]
    assert not result["computed_scaffold_pass"]


@pytest.mark.parametrize("arm", ["CCN(C)C", "c2ccccc2"])
def test_non_source_arm_elements_or_aromaticity_fail(arm):
    smiles = f"O=Cc1cc(OC(=O){arm})cc(OC(=O){arm})c1"
    result = assess_precursor_scaffold(smiles, reaction("aryl_reductive_amination"))
    assert not result["computed_scaffold_pass"]


def test_incomplete_symmetry_search_cannot_qualify():
    c = json.loads(CONTROLS.read_text())["precursor_controls"][0]
    result = assess_precursor_scaffold(c["smiles"], reaction(c["family"]), maximum_matches=2)
    assert not result["complete_search"]
    assert not result["computed_scaffold_pass"]


def test_atom_permutation_preserves_the_entire_witness():
    c = json.loads(CONTROLS.read_text())["precursor_controls"][0]
    mol = Chem.MolFromSmiles(c["smiles"])
    permuted = Chem.MolToSmiles(
        Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms())))), canonical=False
    )
    assert assess_precursor_scaffold(permuted, reaction(c["family"])) == assess_precursor_scaffold(
        c["smiles"], reaction(c["family"])
    )


def test_forward_amine_site_ambiguity_is_not_pruned_by_target():
    c = json.loads(CONTROLS.read_text())["controls"][2]
    a = adapter(c["family"])
    components = {**c["components"], "amine_head": "NCCCNCC"}
    products = a.forward_products(components, maximum_outcomes=256)
    assert len(products.products) == 2
    result = replay_repeated_components(
        a,
        components,
        products.products[0],
        accumulator_role="amine_head",
        events=1,
        byproducts_per_event={"O": 1},
    )
    assert not result["computed_consistency_pass"]
    assert not result["checks"]["unique_forward_exact"]


def test_oxygen_removal_is_required_for_net_reduction_balance():
    c = json.loads(CONTROLS.read_text())["controls"][2]
    result = replay_repeated_components(
        adapter(c["family"]),
        c["components"],
        c["expected_product"],
        accumulator_role="amine_head",
        events=1,
        byproducts_per_event={"formal_charge": 0},
    )
    assert not result["checks"]["full_element_hydrogen_charge_balance"]


def test_invalid_attachment_map_fails_loudly():
    spec = copy.deepcopy(reaction("reductive_amination"))
    spec["precursor_scaffolds"]["scaffolds"][0]["cut_bond_maps"] = [[11, 999]]
    with pytest.raises(LibraryAssemblyError, match="anchor or attachment"):
        assess_precursor_scaffold("CC=O", spec)
