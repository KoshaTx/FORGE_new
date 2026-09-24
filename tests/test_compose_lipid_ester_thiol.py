from pathlib import Path

import pytest
from rdkit import Chem

from forge.corpus.compose_lipid_ester_thiol import component_domain, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "results/phase1/compose_lipid_chemistry_resolution_v1/thiol-config.json"


@pytest.fixture(scope="module")
def contract():
    return load_contract(ROOT, CONFIG)


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("COC(=O)CS", True),
        ("CC(=O)OCCS", True),
        ("O=C(O)CS", False),
        ("OCCS", False),
        ("COC(=O)CC=CCS", False),
        ("COC(=O)C1CC(S)C1", False),
        ("COC(=O)CC(S)CS", False),
    ],
)
def test_bounded_domain_is_atom_order_invariant(contract, smiles, expected):
    domain = contract[2]
    molecule = Chem.MolFromSmiles(smiles)
    reordered = Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms()))))
    assert component_domain(smiles, domain) is expected
    assert component_domain(Chem.MolToSmiles(reordered, canonical=False), domain) is expected


def test_independent_source_formula_and_product(contract):
    assert contract[3]["inventory"]["C"] == 14
    assert contract[3]["inventory"]["H"] == 26
    assert len(contract[3]["products"]) == 1


def test_full_generated_sequence_retains_ambiguous_amines(contract):
    program = contract[1]
    parts = {
        "alkynoic_linker": "C#CCCC(=O)O",
        "amine_head": "CN(C)CCN",
        "thiol_first": "COC(=O)CS",
        "thiol_second": "COC(=O)CS",
    }
    product = "CN(C)CCNC(=O)CCC(SCC(=O)OC)CSCC(=O)OC"
    assert program.replay(parts, product)["computed_consistency_pass"] is True
    parts["amine_head"] = "NCCN(C)CCCN"
    ambiguous = "NCCN(C)CCCNC(=O)CCC(SCC(=O)OC)CSCC(=O)OC"
    result = program.replay(parts, ambiguous)
    assert result["computed_consistency_pass"] is False
    assert result["checks"]["unique_forward_exact"] is False
