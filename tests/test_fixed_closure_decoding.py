from dataclasses import replace

import numpy as np
import torch

from forge.model.defog_feasibility import AtomState
from forge.model.fixed_closure_decoding import decode_fixed_closure_reserved_argmax
from forge.model.reaction_program_flow import collate_synthesis_program_layouts
from forge.model.sparse_topology_feasibility import SparseGraphRecord
from forge.model.synthesis_program_graph import (
    SynthesisProgramComponentBlock,
    SynthesisProgramGraphRecord,
)
from forge.model.synthesis_program_sampling import (
    _fixed_state_exact_records,
    _terminal_smiles,
    decode_synthesis_program_strict_argmax,
)


def ring_case():
    # Fixed C-O-C triangle: O already has its two bonds, one represented as a closure.
    atoms = (AtomState("C", 0, False, 0), AtomState("O", 0, False, 0))
    graph = SparseGraphRecord(
        structure_id="ring-capacity-regression",
        canonical_smiles="",
        node_states=np.array([0, 1, 0, 0]),
        parents=np.array([0, 0, 0, 0]),
        parent_bonds=np.zeros(4, dtype=np.int64),
        closure_left=np.array([1]),
        closure_right=np.array([2]),
        closure_bonds=np.array([0]),
        edges=np.empty((0, 0), dtype=np.int8),
    )
    record = SynthesisProgramGraphRecord(
        graph=graph,
        canonical_atom_order=np.arange(4),
        program_id="ring-regression",
        program_state=1,
        program_depth=1,
        role_states=np.ones(4, dtype=np.int64),
        core_position_states=np.array([2, 3, 4, 1]),
        component_blocks=(SynthesisProgramComponentBlock("role", 1, 0, 4),),
        fixed_atom_mask=np.array([True, True, True, False]),
        fixed_parent_bond_mask=np.array([False, True, True, False]),
        fixed_closure_bond_mask=np.array([True]),
    )
    layout = collate_synthesis_program_layouts([record], maximum_closures=1)
    logits = {
        "nodes": torch.tensor([[[8.0, 0.0]] * 4]),
        "parents": torch.zeros((1, 4, 4)),
        "parent_bonds": torch.tensor([[[8.0, 0.0, 0.0, 0.0]] * 4]),
        "closure_left": torch.zeros((1, 1, 4)),
        "closure_right": torch.zeros((1, 1, 4)),
        "closure_bonds": torch.tensor([[[8.0, 0.0, 0.0, 0.0]]]),
    }
    logits["parents"][0, 3] = torch.tensor([5.0, 10.0, 0.0, -10.0])
    return record, layout, logits, atoms


def test_fixed_ring_capacity_precedes_generated_parent_choices():
    record, layout, logits, atoms = ring_case()
    old, old_reason = decode_synthesis_program_strict_argmax(logits, layout, [record], atoms)
    assert old_reason == ("fixed_closure_valence_exceeds_support",)
    before = {k: v.clone() for k, v in logits.items()}
    new, reason = decode_fixed_closure_reserved_argmax(logits, layout, [record], atoms)
    assert reason == (None,)
    assert int(new["parents"][0, 3]) == 0
    assert _fixed_state_exact_records(new, [record])
    assert _terminal_smiles(new, 0, 4, 1, atoms) is not None
    assert all(torch.equal(logits[k], value) for k, value in before.items())


def test_no_fixed_closure_remains_exactly_the_frozen_decoder():
    record, _, logits, atoms = ring_case()
    record = replace(record, fixed_closure_bond_mask=np.array([False]))
    layout = collate_synthesis_program_layouts([record], maximum_closures=1)
    old, a = decode_synthesis_program_strict_argmax(logits, layout, [record], atoms)
    new, b = decode_fixed_closure_reserved_argmax(logits, layout, [record], atoms)
    assert a == b
    assert all(torch.equal(old[k], new[k]) for k in old)


def test_impossible_immutable_valence_still_abstains():
    record, _, logits, atoms = ring_case()
    record = replace(record, graph=replace(record.graph, closure_bonds=np.array([1])))
    layout = collate_synthesis_program_layouts([record], maximum_closures=1)
    _, reason = decode_fixed_closure_reserved_argmax(logits, layout, [record], atoms)
    assert reason == ("immutable_edge_valence_exceeds_support",)
