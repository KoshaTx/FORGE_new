"""Attachment reservation and registered internal-arm correspondence invariants."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from forge.assembly.families import RegistryAssemblyAdapter
from forge.core.hashing import sha256_file
from forge.corpus.qualified_program_cache import QualifiedProgramExample
from forge.model.compose_lipid_family_rules import (
    admit_family_proposals,
    propose_scaffold_completion,
    reserve_core_attachments,
    reserve_ordered_core_attachments,
)
from forge.model.compose_lipid_generation import constrained_readout
from forge.model.compose_lipid_layout import (
    build_layout,
    collate_generated_layouts,
    summarize_layout,
)
from forge.model.synthesis_program_sampling import _terminal_smiles
from tests.test_compose_lipid_generation import gold_predictions
from tests.test_source_instance_coordinates import record


def test_core_reservation_prevents_greedy_underattachment_without_changing_layout():
    r, _, atoms = record("C(C)(C)C", ["one"] * 4, ["core", "exterior", "exterior", "exterior"])
    e = QualifiedProgramExample(r, "f", (("one", "anonymous", 1),), ())
    bundle, choices, rings = summarize_layout(e)
    layout = build_layout(bundle, choices, rings, identity="test")
    pred = gold_predictions(r, atoms)
    pred["parents"][0, 2, 1] = 1000
    pred["parents"][0, 3, 2] = 1000
    batch = collate_generated_layouts([layout], maximum_closures=12)
    _, reasons = constrained_readout(pred, batch, [layout], atoms)
    assert reasons == ("reaction_core_saturation_unmet",)
    before = copy.deepcopy(layout)
    changed, reason = reserve_core_attachments(layout, pred["parents"][0].numpy())
    assert reason is None
    assert np.array_equal(
        layout.record.fixed_parent_bond_mask, before.record.fixed_parent_bond_mask
    )
    state, reasons = constrained_readout(pred, batch, [changed], atoms)
    assert reasons == (None,)
    assert _terminal_smiles(state, 0, r.node_count, 0, atoms) == Chem.MolToSmiles(
        Chem.MolFromSmiles(r.graph.canonical_smiles)
    )


def test_core_reservation_abstains_on_insufficient_support_and_rejects_nonfinite_scores():
    r, _, atoms = record("CC", ["one"] * 2, ["core", "exterior"])
    e = QualifiedProgramExample(r, "f", (("one", "anonymous", 1),), ())
    bundle, choices, rings = summarize_layout(e)
    layout = build_layout(bundle, choices, rings, identity="test")
    units = layout.core_units.copy()
    units[0] = 8
    changed, reason = reserve_core_attachments(replace(layout, core_units=units), np.zeros((2, 2)))
    assert changed is None and reason == "insufficient_ordered_core_attachment_slots"
    with pytest.raises(ValueError, match="finite square"):
        reserve_core_attachments(layout, np.full((2, 2), np.nan))


def test_core_reservation_keeps_first_exterior_attachment_before_high_scoring_late_edges():
    r, _, atoms = record("C(CC)C", ["one"] * 4, ["core", "exterior", "exterior", "exterior"])
    e = QualifiedProgramExample(r, "f", (("one", "anonymous", 1),), ())
    bundle, choices, rings = summarize_layout(e)
    layout = build_layout(bundle, choices, rings, identity="test")
    pred = gold_predictions(r, atoms)
    pred["parents"][0, 1, 0] = -1000
    pred["parents"][0, 2, 0] = 1000
    pred["parents"][0, 3, 0] = 900
    changed, reason = reserve_core_attachments(layout, pred["parents"][0].numpy())
    assert reason is None and changed.record.fixed_parent_bond_mask[1]
    batch = collate_generated_layouts([layout], maximum_closures=12)
    _, reasons = constrained_readout(pred, batch, [changed], atoms)
    assert reasons == (None,)


def test_ordered_reservation_connects_late_core_islands_without_inventing_children():
    r, _, atoms = record("OCCN", ["one"] * 4, ["core_left", "exterior", "exterior", "core_right"])
    e = QualifiedProgramExample(r, "f", (("one", "anonymous", 1),), ())
    bundle, choices, rings = summarize_layout(e)
    layout = build_layout(bundle, choices, rings, identity="islands")
    pred = gold_predictions(r, atoms)
    changed, reason = reserve_core_attachments(layout, pred["parents"][0].numpy())
    assert changed is None and reason == "insufficient_ordered_core_attachment_slots"
    changed, reason = reserve_ordered_core_attachments(layout, pred["parents"][0].numpy())
    assert reason is None
    assert np.array_equal(changed.core_units, layout.core_units)
    assert changed.record.node_count == r.node_count
    assert not layout.record.fixed_parent_bond_mask[3]
    state, reasons = constrained_readout(
        pred, collate_generated_layouts([layout], maximum_closures=12), [changed], atoms
    )
    assert reasons == (None,)
    assert _terminal_smiles(state, 0, r.node_count, 0, atoms) == "NCCO"


@pytest.mark.needs_vendor
def test_scaffold_completion_copies_generated_arms_and_preserves_source_core():
    # The source control supplies a test fixture; the proposal only receives its
    # generated-state representation and the registered query, never this SMILES.
    path = Path("data/vendor/qualified_reductive_source_program_v1.json")
    registry = json.loads(path.read_text())
    reaction = registry["reactions"][1]
    adapter = RegistryAssemblyAdapter.from_registry(
        path, reaction_id=reaction["reaction_id"], expected_sha256=str(sha256_file(path))
    )
    product = reaction["known_positive_examples"][0]["expected"]
    mol = Chem.MolFromSmiles(product)
    from forge.assembly.program import repair_template_hydrogens

    outcome = None
    for values in adapter.reaction.reverse.RunReactants((mol,)):
        repaired = [repair_template_hydrogens(v) for v in values]
        if any(x is None for x in repaired):
            continue
        parts = {role: value[0] for role, value in zip(adapter.roles, repaired, strict=True)}
        if all(x.qualified for x in adapter.assess_roles(parts)):
            outcome = values
            break
    assert outcome is not None
    role_nodes = {
        a.GetIntProp("react_atom_idx")
        for a in outcome[adapter.roles.index("coupled_aldehyde")].GetAtoms()
        if a.HasProp("react_atom_idx")
    }
    roles = [
        "coupled_aldehyde" if i in role_nodes else "amine_head" for i in range(mol.GetNumAtoms())
    ]
    # Exact reaction-core maps are obtained from the reverse transformation.
    labels = ["exterior"] * mol.GetNumAtoms()
    for fragment in outcome:
        for atom in fragment.GetAtoms():
            if atom.HasProp("old_mapno") and atom.HasProp("react_atom_idx"):
                labels[atom.GetIntProp("react_atom_idx")] = f"map_{atom.GetIntProp('old_mapno')}"
    r, _, atoms = record(product, roles, labels)
    e = QualifiedProgramExample(r, "f", tuple((role, role, 1) for role in adapter.roles), ())
    bundle, choices, rings = summarize_layout(e)
    layout = build_layout(bundle, choices, rings, identity="test")
    state = {
        k: getattr(r.graph, {"nodes": "node_states"}.get(k, k)).tolist()
        for k in (
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
        )
    }
    proposed = propose_scaffold_completion(layout, state, atoms, adapter, reaction)
    assert len(proposed["proposals"]) == 1, proposed
    assert proposed["proposals"][0]["smiles"] == Chem.MolToSmiles(
        Chem.MolFromSmiles(r.graph.canonical_smiles)
    )
    assert proposed["proposals"][0]["atom_edits"] == 0
    ambiguous = copy.deepcopy(reaction)
    duplicate = copy.deepcopy(ambiguous["precursor_scaffolds"]["scaffolds"][0])
    duplicate["id"] += "_conflicting_source"
    ambiguous["precursor_scaffolds"]["scaffolds"].append(duplicate)
    assert not propose_scaffold_completion(layout, state, atoms, adapter, ambiguous)["proposals"]
    with pytest.raises(ValueError, match="search bound"):
        propose_scaffold_completion(layout, state, atoms, adapter, reaction, maximum_matches=1)

    # An unequal generated arm passes simple inverse/forward matching but must
    # fail the complete source scaffold contract until equality is restored.
    from forge.assembly.repeated_components import RepeatBounds
    from forge.corpus.compose_lipid_family_replay import _scaffold_replay
    from forge.model.precursor_reuse_projection import graph_smiles, state_graph

    nodes, edges = state_graph(state)
    block = next(b for b in layout.record.component_blocks if b.role == "coupled_aldehyde")
    leaf = next(
        i
        for i in range(block.start, block.stop)
        if atoms[int(nodes[i])].symbol == "C"
        and np.count_nonzero(edges[i]) == 1
        and not layout.record.fixed_atom_mask[i]
    )
    unequal = copy.deepcopy(state)
    unequal["nodes"][leaf] = next(i for i, a in enumerate(atoms) if a.symbol == "O")
    smiles = graph_smiles(*state_graph(unequal), atoms)
    candidates = adapter.decompose(smiles, maximum_outcomes=256)
    assert candidates
    bounds = RepeatBounds(maximum_events=1, maximum_outcomes=256)
    assert not any(
        _scaffold_replay(adapter, reaction, bounds, dict(c.components), smiles)[
            "computed_consistency_pass"
        ]
        for c in candidates
    )
    completed = propose_scaffold_completion(layout, unequal, atoms, adapter, reaction)
    result = completed["proposals"][0]["smiles"]
    assert result != smiles
    assert any(
        _scaffold_replay(adapter, reaction, bounds, dict(c.components), result)[
            "computed_consistency_pass"
        ]
        for c in adapter.decompose(result, maximum_outcomes=256)
    )


def test_admission_recovers_abstention_and_preserves_exact_novel_diverse_population():
    yes = dict(status="evaluated", exact=True)
    no = dict(status="evaluated", exact=False)
    rows = [
        dict(
            family="f",
            baseline_smiles="A",
            baseline_check=yes,
            proposal_smiles="B",
            proposal_check=yes,
        ),
        dict(
            family="f",
            baseline_smiles=None,
            baseline_check=no,
            proposal_smiles="B",
            proposal_check=yes,
        ),
        dict(
            family="f",
            baseline_smiles=None,
            baseline_check=no,
            proposal_smiles="B",
            proposal_check=yes,
        ),
        dict(
            family="f",
            baseline_smiles="C",
            baseline_check=no,
            proposal_smiles="TRAIN",
            proposal_check=yes,
        ),
        dict(
            family="f",
            baseline_smiles="D",
            baseline_check=no,
            proposal_smiles="E",
            proposal_check=no,
        ),
    ]
    result = admit_family_proposals(rows, is_train_product=lambda s: s == "TRAIN")
    assert [r["selected_smiles"] for r in result] == ["A", "B", None, "C", "D"]
    assert result[2]["disposition"] == "would_concentrate_products"
    assert result[3]["disposition"] == "would_reduce_train_novelty"
    bad = dict(rows[1], proposal_check=dict(status="abstained", exact=True))
    with pytest.raises(ValueError, match="evaluated full-source"):
        admit_family_proposals([bad], is_train_product=lambda _: False)
