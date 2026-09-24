"""Source ring controls, exact precursor scope and uncensored cyclic outcomes."""

import json
from pathlib import Path

import pytest
from rdkit import Chem

from forge.corpus.compose_lipid_fixed_replay import load_contract

ROOT = Path(__file__).resolve().parents[1]
FAMILY = "alpha_isocyanoester_dihydroimidazole"


@pytest.fixture(scope="module")
def contract():
    return load_contract(
        ROOT, ROOT / "results/phase1/compose_lipid_miao_cyclic_source_v2/replay-config.json"
    )


def test_two_independent_isolated_source_controls(contract):
    for label in ("A12_Iso5_2DC18", "A2_Iso5_2DC18"):
        r = contract[-1][label]["replay"]
        assert r["computed_consistency_pass"]
        assert r["inverse_candidate_count"] == 1
        assert all(r["checks"].values())
        assert not r["experimental_execution_admitted"]


@pytest.mark.parametrize("substituent", ["C", "CC", "C(C)(C)C"])
def test_three_drawn_spectator_ester_groups_are_retained(contract, substituent):
    run = contract[2][FAMILY]["run"]
    parts = {
        "amine_head": "CN(C)CCN",
        "coupled_ketone": "CCC(=O)CC",
        "isocyanide": "[C-]#[N+]CC(=O)O" + substituent,
    }
    target = "CN(C)CCN1C(CC)(CC)C=NC1C(=O)O" + substituent
    result = run(parts, target)
    assert result["computed_consistency_pass"]
    for key in parts:
        mol = Chem.MolFromSmiles(parts[key])
        parts[key] = Chem.MolToSmiles(
            Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms())))), canonical=False
        )
    assert run(parts, target)["computed_consistency_pass"]


@pytest.mark.parametrize(
    "isocyanide",
    [
        "[C-]#[N+]CC(=O)OCCC",
        "[C-]#[N+]CC(=O)Oc1ccccc1",
        "[C-]#[N+]C(C)C(=O)OC(C)(C)C",
        "[C-]#[N+]CCC(=O)OCC",
        "[C-]#[N+]CC(=O)O",
        "[C-]#[N+]CC",
    ],
)
def test_other_isocyanides_do_not_enter_source_scope(contract, isocyanide):
    run = contract[2][FAMILY]["run"]
    result = run(
        {"amine_head": "CN(C)CCN", "coupled_ketone": "CCC(=O)CC", "isocyanide": isocyanide},
        "CN(C)CCN1C(CC)(CC)C=NC1C(=O)OCC",
    )
    assert not result["computed_consistency_pass"]


def test_target_matching_does_not_remove_competing_head_sites(contract):
    r = contract[-1]["non_equivalent_primary_amines"]["replay"]
    assert len(r["forward_products"]) == 2
    assert not r["checks"]["unique_unfiltered_forward_exact"]
    assert not r["computed_consistency_pass"]


def test_scope_does_not_promote_source_library_to_execution(contract):
    document = json.loads(contract[1]["transform_controls"].read_text())
    assert not document["source_library_scope"]["individual_execution_admitted"]
    assert not document["training_admitted"]
    assert not document["experimental_execution_admitted"]
