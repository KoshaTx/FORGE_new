"""Behavioral checks for diagnostic flags, without imposing a lipid whitelist."""

from types import SimpleNamespace

import numpy as np
from rdkit import Chem

from forge.model.compose_lipid_structural_audit import (
    RegistryQuery,
    audit_layout_rings,
    audit_smiles,
)


def audit(smiles):
    # This deliberately broad unit-test query is not a basicity prediction or
    # admitted chemistry rule. Production queries come from pinned registries.
    definition = RegistryQuery("test nitrogen", "[#7]", "unit-test-only")
    query = definition.compile()
    return audit_smiles(
        smiles,
        registry_queries=[(definition, query)],
        basic_query=(definition, query),
        policy={
            "maximum_heavy_atoms": 254,
            "maximum_independent_cycles": 12,
            "small_endocyclic_allene_maximum_ring_size": 7,
            "small_endocyclic_alkyne_maximum_ring_size": 7,
        },
        allowed_elements={"C", "N", "O", "S", "P", "Br"},
        allowed_atom_charges={
            ("C", 0),
            ("N", 0),
            ("N", 1),
            ("O", 0),
            ("O", -1),
            ("S", 0),
            ("P", 0),
            ("Br", 0),
        },
        allowed_kekule_bonds={"SINGLE", "DOUBLE", "TRIPLE"},
    )


def codes(result, tier=None):
    return {v["code"] for v in result["flags"] if tier is None or v["tier"] == tier}


def test_invalid_and_disconnected_outputs_are_retained_as_issues():
    assert not audit("bad_smiles")["valid"]
    separated = audit("CCCC.N")
    assert separated["valid"] and not separated["connected"]
    assert "disconnected_product" in codes(separated)


def test_small_endocyclic_allene_is_distinct_from_valid_acyclic_allene():
    cyclic = audit("C1=C=CCCC1")
    flag = next(v for v in cyclic["flags"] if v["code"] == "small_endocyclic_allene")
    mol = Chem.MolFromSmiles(cyclic["canonical_smiles"])
    assert len(flag["atoms"]) == flag["minimum_containing_ring_size"]
    assert (
        sum(
            b.GetBondType() == Chem.BondType.DOUBLE
            for b in mol.GetAtomWithIdx(flag["central_atom"]).GetBonds()
        )
        == 2
    )
    acyclic = audit("CCC=C=CCCC")
    assert "other_allene" in codes(acyclic)
    assert not codes(acyclic, "justified_chemical_alert")


def test_cyclic_alkyne_size_boundary_and_ordinary_benzene():
    assert "small_endocyclic_alkyne" in codes(audit("C1#CCCCC1"))
    assert "small_endocyclic_alkyne" not in codes(audit("C1#CCCCCCC1"))
    assert "cyclobutadiene_like_ring" in codes(audit("C1=CC=C1"))
    assert not codes(audit("c1ccccc1"), "justified_chemical_alert")
    assert not codes(audit("c1ccccc1"), "demonstrated_representation_issue")


def test_orthoester_is_context_only_and_atom_order_is_canonical():
    first = audit("COC(OC)(OC)CCCCN")
    second = audit(Chem.MolToSmiles(Chem.MolFromSmiles("COC(OC)(OC)CCCCN"), rootedAtAtom=8))
    assert first == second
    assert "saturated_carbon_at_least_three_oxygens" in codes(first, "context_required")
    assert not codes(first, "justified_chemical_alert")


def test_carbon_domains_do_not_cross_heteroatom_linkers():
    result = audit("NCCCCOCCCCCC")
    assert sorted(v["carbon_atoms"] for v in result["organization"]["carbon_domains"]) == [4, 6]
    assert len(result["organization"]["basic_candidate_atoms"]) == 1
    assert all(
        v["minimum_distance_to_basic_candidate"] is not None
        for v in result["organization"]["carbon_domains"]
    )


def test_query_bounds_are_censored_instead_of_silently_complete():
    definition = RegistryQuery("test carbon", "[#6]", "unit-test-only")
    result = audit_smiles(
        "CCCCN",
        registry_queries=[(definition, definition.compile())],
        basic_query=(definition, definition.compile()),
        policy={
            "maximum_heavy_atoms": 254,
            "maximum_independent_cycles": 12,
            "small_endocyclic_allene_maximum_ring_size": 7,
            "small_endocyclic_alkyne_maximum_ring_size": 7,
        },
        allowed_elements={"C", "N"},
        maximum_query_matches=2,
    )
    assert result["registry_handles"][0]["censored"]
    assert len(result["registry_handles"][0]["matches"]) == 2


def test_role_cycle_count_and_tree_size_are_different_conditions():
    layout = SimpleNamespace(
        record=SimpleNamespace(
            node_count=6,
            role_states=np.array([1, 1, 1, 1, 2, 2]),
            core_position_states=np.array([2, 1, 1, 1, 2, 1]),
            component_blocks=[
                SimpleNamespace(role_state=1, role="body"),
                SimpleNamespace(role_state=2, role="head"),
            ],
        ),
        variable_closures_by_role={1: 1, 2: 0},
        ring_sizes_by_role={1: (6,), 2: ()},
    )
    edges = np.zeros((6, 6), dtype=int)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5)]:
        edges[a, b] = edges[b, a] = 1
    state = {"parents": [0, 0, 1, 2, 0, 4], "closure_left": [0], "closure_right": [3]}
    result = audit_layout_rings(edges, layout, tree_state=state, tree_basis="observed raw")
    assert not result["cycle_allocation_mismatch"]
    assert result["fundamental_size_mismatch"]
    assert result["roles"][1]["observed_fundamental_rings"][0]["size"] == 4
    unknown = audit_layout_rings(edges, layout, tree_state=None, tree_basis="unavailable")
    assert unknown["fundamental_size_mismatch"] is None
    assert not unknown["cycle_allocation_mismatch"]
