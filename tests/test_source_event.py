"""Source predicates restrict actual reaction sites without pruning competing products."""

import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.assembly.families import LibraryAssemblyError, RegistryAssemblyAdapter
from forge.assembly.source_event import check_source_event
from forge.core.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source_event():
    source = json.loads(
        (ROOT / "results/phase1/compose_lipid_v8_ugi3_source_v1/adjudication.json").read_text()
    )
    adapter = RegistryAssemblyAdapter.from_registry(
        ROOT / source["registry"]["path"],
        reaction_id=source["reaction_id"],
        expected_sha256=source["registry"]["sha256"],
    )
    queries = {}
    for role, ref in source["source_contract"]["query_references"].items():
        registry = json.loads((ROOT / source[ref["input"]]["path"]).read_text())
        reaction = next(r for r in registry["reactions"] if r["reaction_id"] == ref["reaction_id"])
        spec = next(r for r in reaction["reactant_roles"] if r["name"] == ref["role"])
        queries[role] = Chem.MolFromSmarts(spec["required_handle_smarts"])
    return (
        adapter,
        source,
        {
            "site_contract": source["source_contract"]["site_contract"],
            "role_queries": queries,
        },
    )


@pytest.mark.parametrize("index", [0, 1, 2])
def test_independent_published_controls(source_event, index):
    adapter, source, kwargs = source_event
    control = source["source_controls"][index]
    result = check_source_event(
        adapter, control["components"], control["expected_product"], **kwargs
    )
    assert result["computed_consistency_pass"]
    assert (
        rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(control["expected_product"]))
        == control["expected_formula"]
    )
    assert not result["experimental_execution_admitted"]
    assert str(sha256_file(adapter.registry_path)) == source["registry"]["sha256"]


@pytest.mark.parametrize("head,carbonyl", [("CNC", "CC=O"), ("CN", "O=COC"), ("CN", "O=CN")])
def test_broad_registry_match_is_not_source_qualification(source_event, head, carbonyl):
    adapter, _, kwargs = source_event
    components = dict(zip(adapter.roles, [head, carbonyl, "CCC[N+]#[C-]"], strict=True))
    (target,) = adapter.forward_products(components).products
    result = check_source_event(adapter, components, target, **kwargs)
    assert result["checks"]["unique_unfiltered_forward_exact"]
    assert not result["checks"]["source_reactive_site_witness"]
    assert not result["computed_consistency_pass"]


def test_primary_spectator_does_not_certify_secondary_attachment_or_hide_ambiguity(source_event):
    adapter, _, kwargs = source_event
    components = dict(zip(adapter.roles, ["NCCNC", "CC=O", "CCC[N+]#[C-]"], strict=True))
    products = adapter.forward_products(components).products
    assert len(products) == 2
    results = [check_source_event(adapter, components, p, **kwargs) for p in products]
    assert sorted(r["checks"]["source_reactive_site_witness"] for r in results) == [False, True]
    assert all(not r["checks"]["unique_unfiltered_forward_exact"] for r in results)
    assert all(not r["computed_consistency_pass"] for r in results)
    assert all(len(r["forward_products"]) == 2 for r in results)


def test_negative_charge_inventory_is_not_dropped(source_event):
    adapter, _, kwargs = source_event
    components = dict(zip(adapter.roles, ["CN", "O=CCC(=O)[O-]", "CCC[N+]#[C-]"], strict=True))
    (target,) = adapter.forward_products(components).products
    result = check_source_event(adapter, components, target, **kwargs)
    assert result["computed_consistency_pass"]
    assert Chem.GetFormalCharge(Chem.MolFromSmiles(target)) == -1


def test_input_atom_order_does_not_change_witness_indices(source_event):
    adapter, source, kwargs = source_event
    control = source["source_controls"][2]
    reversed_smiles = {
        role: Chem.MolToSmiles(
            Chem.MolFromSmiles(smi),
            canonical=False,
            rootedAtAtom=len(Chem.MolFromSmiles(smi).GetAtoms()) - 1,
        )
        for role, smi in control["components"].items()
    }
    expected = check_source_event(
        adapter, control["components"], control["expected_product"], **kwargs
    )
    assert (
        check_source_event(adapter, reversed_smiles, control["expected_product"], **kwargs)
        == expected
    )


def test_saturated_enumeration_is_not_qualification(source_event):
    adapter, source, kwargs = source_event
    control = source["source_controls"][0]
    with pytest.raises(LibraryAssemblyError, match="saturated"):
        check_source_event(
            adapter,
            control["components"],
            control["expected_product"],
            maximum_outcomes=1,
            **kwargs,
        )


def test_missing_role_or_wrong_role_map_is_rejected(source_event):
    adapter, source, kwargs = source_event
    control = source["source_controls"][0]
    kwargs["role_queries"].pop(adapter.roles[0])
    with pytest.raises(LibraryAssemblyError, match="explicit role/site"):
        check_source_event(adapter, control["components"], control["expected_product"], **kwargs)
    kwargs["role_queries"][adapter.roles[0]] = adapter.reaction.handles[0]
    kwargs["site_contract"][0]["atom_map"] = 4
    with pytest.raises(LibraryAssemblyError, match="declared reactant role"):
        check_source_event(adapter, control["components"], control["expected_product"], **kwargs)
