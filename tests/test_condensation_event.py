"""Ugi-4 water loss cannot weaken uniqueness, site, role, or charge gates."""

import copy
import json
import random
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.assembly.condensation_event import check_condensation_event
from forge.assembly.families import LibraryAssemblyError, RegistryAssemblyAdapter
from forge.assembly.source_event import check_source_event
from forge.core.hashing import resolve_pin

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def control():
    source = json.loads(
        (ROOT / "results/phase1/compose_lipid_v8_ugi4_source_v1/adjudication.json").read_text()
    )
    adapter = RegistryAssemblyAdapter.from_registry(
        resolve_pin(source["registry"], ROOT, label="registry"),
        reaction_id=source["reaction_id"],
        expected_sha256=source["registry"]["sha256"],
    )
    queries = {}
    contract = source["source_contract"]
    for role, ref in contract["query_references"].items():
        registry = json.loads(resolve_pin(source[ref["input"]], ROOT, label=role).read_text())
        reaction = next(r for r in registry["reactions"] if r["reaction_id"] == ref["reaction_id"])
        spec = next(r for r in reaction["reactant_roles"] if r["name"] == ref["role"])
        queries[role] = Chem.MolFromSmarts(spec["required_handle_smarts"])
    return (
        adapter,
        source["source_controls"][0],
        {
            "site_contract": contract["site_contract"],
            "role_queries": queries,
            "net_byproducts": contract["net_byproducts"],
        },
    )


def test_independently_transcribed_119_23_has_exact_roundtrip_and_source_formula(control):
    adapter, row, kwargs = control
    result = check_condensation_event(adapter, row["components"], row["expected_product"], **kwargs)
    assert result["computed_consistency_pass"]
    assert len(result["site_witnesses"]) == 1
    assert not result["experimental_execution_admitted"]
    assert (
        rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(row["expected_product"]))
        == row["expected_formula"]
    )
    original = check_source_event(
        adapter,
        row["components"],
        row["expected_product"],
        **{k: v for k, v in kwargs.items() if k != "net_byproducts"},
    )
    assert not original["computed_consistency_pass"]
    assert not original["checks"]["full_element_hydrogen_charge_balance"]


@pytest.mark.parametrize("seed", range(5))
def test_atom_order_does_not_change_constitution_or_source_witness(control, seed):
    adapter, row, kwargs = control
    rng = random.Random(seed)

    def permute(smiles):
        mol = Chem.MolFromSmiles(smiles)
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        return Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)

    result = check_condensation_event(
        adapter,
        {role: permute(smi) for role, smi in row["components"].items()},
        permute(row["expected_product"]),
        **kwargs,
    )
    assert result == check_condensation_event(
        adapter, row["components"], row["expected_product"], **kwargs
    )


@pytest.mark.parametrize(
    "inventory", [{"H": 1, "O": 1}, {"H": 2, "O": 2}, {"H": 2, "O": 1, "formal_charge": -1}]
)
def test_wrong_byproduct_atoms_or_negative_charge_never_balance(control, inventory):
    adapter, row, kwargs = control
    kwargs["net_byproducts"] = inventory
    result = check_condensation_event(adapter, row["components"], row["expected_product"], **kwargs)
    assert not result["checks"]["full_element_hydrogen_charge_balance"]
    assert not result["computed_consistency_pass"]


@pytest.mark.parametrize("inventory", [{}, {"H": -2}, {"H": True}, {"O": 1.0}, None])
def test_invalid_byproduct_contract_fails_loudly(control, inventory):
    adapter, row, kwargs = control
    kwargs["net_byproducts"] = inventory
    with pytest.raises(LibraryAssemblyError, match="byproduct inventory"):
        check_condensation_event(adapter, row["components"], row["expected_product"], **kwargs)


@pytest.mark.parametrize(
    "role,value",
    [
        ("amine_head", "CN(C)C"),
        ("amine_head", "CC(=O)N"),
        ("oxoester_aldehyde_body_tail", "CC(=O)CC"),
        ("oxoester_aldehyde_body_tail", "COC=O"),
        ("carboxylic_acid_tail", "CCOC(C)=O"),
        ("carboxylic_acid_tail", "O=C(O)CCC(=O)O"),
        ("isocyanide_tail", "CCCC#N"),
    ],
)
def test_other_reactive_groups_are_not_admitted(control, role, value):
    adapter, row, kwargs = control
    components = {**row["components"], role: value}
    result = check_condensation_event(adapter, components, row["expected_product"], **kwargs)
    assert not result["computed_consistency_pass"]


def test_source_primary_amine_requirement_cannot_be_ignored(control):
    adapter, row, kwargs = control
    kwargs = copy.deepcopy(kwargs)
    kwargs["site_contract"][0]["properties"]["total_hydrogens"] = 1
    result = check_condensation_event(adapter, row["components"], row["expected_product"], **kwargs)
    assert result["checks"]["full_element_hydrogen_charge_balance"]
    assert not result["checks"]["source_reactive_site_witness"]
    assert not result["computed_consistency_pass"]


def test_additional_primary_amine_products_are_retained(control):
    adapter, row, kwargs = control
    components = {**row["components"], "amine_head": "NCC(C)CCN"}
    products = adapter.forward_products(components).products
    assert len(products) == 2
    result = check_condensation_event(adapter, components, products[0], **kwargs)
    assert len(result["forward_products"]) == 2
    assert not result["checks"]["unique_unfiltered_forward_exact"]
    assert not result["computed_consistency_pass"]
