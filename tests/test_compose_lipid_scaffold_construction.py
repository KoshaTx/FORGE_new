"""Source chemistry and conservation checks for constructive terminal scaffolds."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from forge.assembly.families import RegistryAssemblyAdapter
from forge.assembly.program import repair_template_hydrogens
from forge.assembly.repeated_components import RepeatBounds
from forge.core.hashing import sha256_file
from forge.corpus.compose_lipid_family_replay import _scaffold_replay
from forge.corpus.qualified_program_cache import QualifiedProgramExample
from forge.model.compose_lipid_component_diversity import admit_component_diverse_proposals
from forge.model.compose_lipid_layout import build_layout, summarize_layout
from forge.model.compose_lipid_scaffold_construction import construct_scaffold_proposals
from tests.test_compose_lipid_generation import gold_predictions
from tests.test_source_instance_coordinates import record


def fixture(index):
    path = Path("data/vendor/qualified_reductive_source_program_v1.json")
    reaction = json.loads(path.read_text())["reactions"][min(index, 1)]
    adapter = RegistryAssemblyAdapter.from_registry(
        path, reaction_id=reaction["reaction_id"], expected_sha256=str(sha256_file(path))
    )
    product = reaction["known_positive_examples"][0]["expected"]
    if index == 2:
        controls = json.loads(
            Path("results/phase1/compose_lipid_v8_reductive_source_v1/controls.json").read_text()
        )
        precursor = controls["precursor_controls"][0]["smiles"]
        parts = dict(
            zip(adapter.roles, reaction["known_positive_examples"][0]["reactants"], strict=True)
        )
        parts[reaction["precursor_scaffolds"]["precursor_role"]] = precursor
        products = adapter.forward_products(parts)
        assert len(products.products) == 1 and not products.saturated
        product = products.products[0]
    molecule = Chem.MolFromSmiles(product)
    for outcome in adapter.reaction.reverse.RunReactants((molecule,)):
        repaired = [repair_template_hydrogens(m) for m in outcome]
        if any(x is None for x in repaired):
            continue
        parts = {role: value[0] for role, value in zip(adapter.roles, repaired, strict=True)}
        if all(x.qualified for x in adapter.assess_roles(parts)):
            break
    else:
        raise AssertionError("Source fixture has no qualified inverse")
    roles, labels = [None] * molecule.GetNumAtoms(), ["exterior"] * molecule.GetNumAtoms()
    for role, fragment in zip(adapter.roles, outcome, strict=True):
        for atom in fragment.GetAtoms():
            if atom.HasProp("react_atom_idx"):
                i = atom.GetIntProp("react_atom_idx")
                roles[i] = role
                if atom.HasProp("old_mapno"):
                    labels[i] = f"map_{atom.GetIntProp('old_mapno')}"
    r, _, atoms = record(product, roles, labels)
    example = QualifiedProgramExample(r, "test", tuple((x, x, 1) for x in adapter.roles), ())
    bundle, choices, rings = summarize_layout(example)
    layout = build_layout(bundle, choices, rings, identity="construction-test")
    pred = {k: v[0].numpy() for k, v in gold_predictions(r, atoms).items()}
    block = next(b for b in r.component_blocks if b.role == "coupled_aldehyde")
    # Deliberately remove all learned scaffold chemistry. Only the registry may
    # restore it; variable arms prefer simple generated carbon chains.
    for key in ("nodes", "parents", "parent_bonds"):
        pred[key][block.start : block.stop] = 0
    carbon = next(i for i, a in enumerate(atoms) if a.symbol == "C" and not a.aromatic)
    pred["nodes"][block.start : block.stop, carbon] = 100
    pred["parent_bonds"][block.start : block.stop, 0] = 100
    for child in range(block.start + 1, block.stop):
        pred["parents"][child, child - 1] = 100
    return layout, pred, atoms, reaction, adapter


@pytest.mark.needs_vendor
@pytest.mark.parametrize("index", [0, 1, 2])
def test_missing_scaffold_is_constructed_and_passes_independent_complete_source_check(index):
    layout, pred, atoms, reaction, adapter = fixture(index)
    before = copy.deepcopy(layout)
    proposed = construct_scaffold_proposals(layout, pred, atoms, reaction)
    valid = [p for p in proposed["proposals"] if p["smiles"]]
    assert valid, proposed
    assert proposed == construct_scaffold_proposals(layout, pred, atoms, reaction)
    assert np.array_equal(layout.record.graph.node_states, before.record.graph.node_states)
    bounds = RepeatBounds(maximum_events=1, maximum_outcomes=256)
    for proposal in valid:
        molecule = Chem.MolFromSmiles(proposal["smiles"])
        assert molecule.GetNumAtoms() == layout.record.node_count
        assert (
            molecule.GetNumBonds() - molecule.GetNumAtoms() + 1 == layout.record.graph.closure_count
        )
        assert any(
            _scaffold_replay(adapter, reaction, bounds, dict(c.components), proposal["smiles"])[
                "computed_consistency_pass"
            ]
            for c in adapter.decompose(proposal["smiles"], maximum_outcomes=256)
        )
    # Source examples are test fixtures only; production construction must not
    # consult them or introduce sample identity into its scaffold definition.
    stripped = dict(reaction, known_positive_examples=[], known_negative_examples=[])
    assert construct_scaffold_proposals(layout, pred, atoms, stripped) == proposed


@pytest.mark.needs_vendor
def test_unsupported_cycles_and_nonfinite_predictions_fail_without_resizing():
    layout, pred, atoms, reaction, _ = fixture(1)
    block = next(b for b in layout.record.component_blocks if b.role == "coupled_aldehyde")
    budgets = dict(layout.variable_closures_by_role)
    budgets[block.role_state] += 1
    proposed = construct_scaffold_proposals(
        replace(layout, variable_closures_by_role=budgets), pred, atoms, reaction
    )
    assert all(not p["smiles"] for p in proposed["proposals"])
    pred["nodes"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite predictions"):
        construct_scaffold_proposals(layout, pred, atoms, reaction)


def exact(head, tail):
    return dict(
        status="evaluated",
        exact=True,
        checks=[dict(accepted_components=[dict(head=head, tail=tail)])],
    )


def test_component_guard_rejects_concentration_even_for_distinct_novel_products():
    no = dict(status="evaluated", exact=False)
    rows = [
        dict(family="f", baseline_smiles="A", baseline_check=exact("H1", "T1")),
        dict(family="f", baseline_smiles="B", baseline_check=exact("H2", "T2")),
        dict(
            family="f",
            baseline_smiles=None,
            baseline_check=no,
            proposals=[dict(smiles="C", check=exact("H1", "T3"))],
        ),
        dict(
            family="f",
            baseline_smiles=None,
            baseline_check=no,
            proposals=[dict(smiles="D", check=exact("H3", "T4"))],
        ),
    ]
    selected = admit_component_diverse_proposals(rows, is_train_product=lambda _: False)
    assert [r["selected_smiles"] for r in selected] == ["A", "B", None, "D"]
    assert (
        selected[2]["proposal_decisions"][0]["disposition"]
        == "would_reduce_component_effective_count"
    )
    bad = copy.deepcopy(rows)
    bad[2]["proposals"][0]["check"]["checks"][0]["accepted_components"].append(
        dict(head="X", tail="Y")
    )
    with pytest.raises(ValueError, match="one accepted precursor tuple"):
        admit_component_diverse_proposals(bad, is_train_product=lambda _: False)


def test_component_selection_keeps_attempts_novelty_and_product_uniqueness():
    no = dict(status="evaluated", exact=False)
    rows = [
        dict(
            family="f",
            baseline_smiles="novel",
            baseline_check=no,
            proposals=[dict(smiles="TRAIN", check=exact("H1", "T1"))],
        ),
        dict(
            family="f",
            baseline_smiles=None,
            baseline_check=no,
            proposals=[dict(smiles="new", check=exact("H2", "T2"))],
        ),
        dict(
            family="f",
            baseline_smiles=None,
            baseline_check=no,
            proposals=[dict(smiles="new", check=exact("H2", "T2"))],
        ),
    ]
    result = admit_component_diverse_proposals(rows, is_train_product=lambda s: s == "TRAIN")
    assert [r["selected_smiles"] for r in result] == ["novel", "new", None]
    assert result[0]["proposal_decisions"][0]["disposition"] == "would_reduce_train_novelty"
    assert result[2]["proposal_decisions"][0]["disposition"] == "would_concentrate_products"
