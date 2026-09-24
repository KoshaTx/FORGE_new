import json
from pathlib import Path

import pytest
from rdkit import Chem

from forge.corpus.compose_lipid_miao_primary import component_domain, load_contract

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "results/phase1/compose_lipid_chemistry_resolution_v1"


@pytest.fixture(scope="module")
def contract():
    return load_contract(ROOT, HERE / "miao-config.json")


def test_published_acyclic_control_and_secondary_abstention(contract):
    controls = contract[3]
    assert controls["patent_method_3"]["replay"]["computed_consistency_pass"] is True
    assert controls["secondary_amine_not_supported"]["replay"]["computed_consistency_pass"] is False


@pytest.mark.parametrize(
    "role,smiles",
    [
        ("isocyanide", "[C-]#[N+]CC(=O)OC(C)(C)C"),
        ("coupled_ketone", "O=C1CCCCC1"),
        ("amine_head", "C[NH+](C)CCN"),
    ],
)
def test_unsupported_domains_abstain_independent_of_target(contract, role, smiles):
    source = json.loads((HERE / "miao-source-control.json").read_text())
    executor = contract[2]["ketone_isocyanide_amide"]
    parts = {**source["control"], role: smiles}
    assert not component_domain(parts, executor["domain"])
    molecule = Chem.MolFromSmiles(smiles)
    shuffled = Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms()))))
    parts[role] = Chem.MolToSmiles(shuffled, canonical=False)
    assert not component_domain(parts, executor["domain"])


def test_ambiguous_primary_sites_are_not_selected_by_target(contract):
    parts = {"amine_head": "NCCN(C)CCCN", "coupled_ketone": "CC(=O)CC", "isocyanide": "CC[N+]#[C-]"}
    expected = "CN(CCN)CCCNC(C)(CC)C(=O)NCC"
    result = contract[2]["ketone_isocyanide_amide"]["run"](parts, expected)
    assert len(result["forward_products"]) == 2
    assert not result["computed_consistency_pass"]
    assert not result["checks"]["unique_unfiltered_forward_exact"]
