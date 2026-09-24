"""Independent source product and adversarial boundaries for the AEMA extension."""

import copy
import gzip
import json
from pathlib import Path

import pytest
from rdkit import Chem

from forge.assembly.compose_lipid import ComposeLipidError
from forge.corpus.compose_lipid_aema_ester import FAMILY, load_contract, replay_record
from forge.corpus.compose_lipid_ester_thiol import component_domain

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "results/phase1/compose_lipid_fast_qualification_v1/aema-config.json"


@pytest.fixture(scope="module")
def contract():
    return load_contract(ROOT, CONFIG)


@pytest.fixture(scope="module")
def original():
    ledger = (
        ROOT
        / "results/phase1/compose_lipid_chemistry_resolution_v1/remaining-source-records.jsonl.gz"
    )
    with gzip.open(ledger, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["preparation"]["family"] == FAMILY:
                return (
                    {
                        "preparation": row["preparation"],
                        "source": {"constitution": row["product_smiles"]},
                    },
                    {p["component_id"]: p["smiles"] for p in row["components"]},
                )
    raise AssertionError("Missing independent source record")


def test_independent_published_methacrylate_control(contract):
    control = contract[3]["Bowman_ester_thiol_methacrylate"]
    assert control["source_control_pass"]
    assert control["products"] == ["CCCCOC(=O)CSCC(C)C(=O)OC"]
    assert control["inventory"] == {"C": 11, "H": 20, "O": 4, "S": 1, "formal_charge": 0}


def test_original_construction_is_exact_and_computed_only(contract, original):
    item, structures = copy.deepcopy(original)
    before = copy.deepcopy(item)
    result = replay_record(item, structures, contract[2])
    assert result["computed_consistency_pass"]
    assert result["checks"]["bounded_ester_thiol_domain"]
    assert result["evidence_basis"] == "computed_transform_consistency"
    assert not result["experimental_execution_admitted"]
    assert not result["training_admitted"]
    assert item == before


@pytest.mark.parametrize(
    "smiles", ["OCCCCS", "O=C(O)CCS", "COC(=O)CSS", "COC(=O)C=CS", "COC(=O)C1CC1S"]
)
def test_unreviewed_thiol_domains_remain_out(contract, smiles):
    assert not component_domain(smiles, contract[2][1]["ester_thiol_domain"])


def test_aromatic_amine_core_stays_out(contract, original):
    item, structures = copy.deepcopy(original)
    identity = next(
        i for r, i, _ in item["preparation"]["component_instances"] if r == "amine_core"
    )
    structures[identity] = "NCCCn1ccnc1"
    result = replay_record(item, structures, contract[2])
    assert result["disposition"] == "outside_qualified_source_component_domain"


def test_original_incorporated_counts_cannot_be_inferred_from_target(contract, original):
    item, structures = copy.deepcopy(original)
    for row in item["preparation"]["component_instances"]:
        if row[0] != "amine_core":
            row[2] = 1
    result = replay_record(item, structures, contract[2])
    assert not result["computed_consistency_pass"]
    assert not result["checks"]["full_element_hydrogen_charge_balance"]


def test_atom_order_invariance(contract, original):
    item, structures = copy.deepcopy(original)
    for identity, smiles in structures.items():
        mol = Chem.MolFromSmiles(smiles)
        structures[identity] = Chem.MolToSmiles(
            Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms())))), canonical=False
        )
    assert replay_record(item, structures, contract[2])["computed_consistency_pass"]


def test_protected_rows_rejected_before_structure_access(contract):
    with pytest.raises(ComposeLipidError, match="Protected"):
        replay_record({"preparation": {"eligible_for_program_preparation": False}}, {}, contract[2])


def test_ester_thioether_without_sh_handle_cannot_reconstruct(contract, original):
    item, structures = copy.deepcopy(original)
    identity = next(
        i for r, i, _ in item["preparation"]["component_instances"] if r == "thiol_periphery"
    )
    structures[identity] = "COC(=O)CCSC"
    result = replay_record(item, structures, contract[2])
    assert not result["computed_consistency_pass"]
    assert not result["checks"]["unique_forward_exact"]
