from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.precursor_reuse import PrecursorReuseError, PrecursorReusePlan
from forge.model.precursor_reuse_projection import (
    TerminalTrace,
    complete_reuse,
    copy_generated_component,
    fixed_graph_preserved,
    graph_smiles,
    state_graph,
)
from forge.model.synthesis_program_graph import SynthesisProgramComponentBlock
from forge.model.synthesis_program_sampling import (
    load_synthesis_program_checkpoint,
    sample_synthesis_program_products,
)

REPO = Path(__file__).resolve().parents[1]


def example_graph():
    record = SimpleNamespace(
        graph=SimpleNamespace(
            structure_id="fixture",
            node_states=np.asarray([0, 1, 2, 3]),
            parents=np.asarray([0, 0, 1, 2]),
            parent_bonds=np.asarray([0, 1, 0, 0]),
            closure_left=np.asarray([], dtype=int),
            closure_right=np.asarray([], dtype=int),
            closure_bonds=np.asarray([], dtype=int),
        ),
        component_blocks=(
            SynthesisProgramComponentBlock("role", 1, 0, 2),
            SynthesisProgramComponentBlock("role", 1, 2, 4),
        ),
        fixed_atom_mask=np.zeros(4, dtype=bool),
        fixed_parent_bond_mask=np.zeros(4, dtype=bool),
        fixed_closure_bond_mask=np.zeros(0, dtype=bool),
    )
    state = {
        "nodes": [0, 1, 2, 3],
        "parents": [0, 0, 1, 2],
        "parent_bonds": [0, 1, 0, 0],
        "closure_left": [],
        "closure_right": [],
        "closure_bonds": [],
    }
    return record, PrecursorReusePlan("fixture", (1, 2, 1, 2), "qualified", 1), state


def test_graph_copy_preserves_cross_edges_and_copies_generated_internal_states():
    record, plan, state = example_graph()
    nodes, edges = state_graph(state)
    new_nodes, new_edges = copy_generated_component(nodes, edges, record, plan, 1)
    assert new_nodes.tolist() == [2, 3, 2, 3]
    assert new_edges[0, 1] == new_edges[2, 3] == 1
    assert new_edges[1, 2] == edges[1, 2]
    assert nodes.tolist() == state["nodes"] and edges[0, 1] == 2
    assert np.array_equal(new_edges, new_edges.T)


def test_copy_does_not_read_variable_source_targets():
    record, plan, state = example_graph()
    nodes, edges = state_graph(state)
    before = copy_generated_component(nodes, edges, record, plan, 0)
    record.graph.node_states[:] = 99
    record.graph.parent_bonds[:] = 99
    after = copy_generated_component(nodes, edges, record, plan, 0)
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))


def test_fixed_atom_and_bond_changes_are_detected():
    record, plan, state = example_graph()
    nodes, edges = state_graph(state)
    assert fixed_graph_preserved(nodes, edges, record)
    record.fixed_atom_mask[0] = True
    changed = copy_generated_component(nodes, edges, record, plan, 1)
    assert not fixed_graph_preserved(*changed, record)
    record.fixed_atom_mask[0] = False
    record.fixed_parent_bond_mask[1] = True
    assert not fixed_graph_preserved(*changed, record)


@pytest.mark.parametrize("donor", [-1, 2, True])
def test_invalid_donor_fails(donor):
    record, plan, state = example_graph()
    with pytest.raises(PrecursorReuseError):
        copy_generated_component(*state_graph(state), record, plan, donor)


def test_invalid_parent_or_duplicate_closure_fails():
    _, _, state = example_graph()
    state["parents"][2] = 3
    with pytest.raises(PrecursorReuseError, match="parent"):
        state_graph(state)
    state["parents"][2] = 1
    state.update(closure_left=[0], closure_right=[1], closure_bonds=[0])
    with pytest.raises(PrecursorReuseError, match="closure"):
        state_graph(state)


def test_invalid_original_remains_an_explicit_failed_attempt():
    record, plan, _ = example_graph()
    result = complete_reuse(
        record=record,
        plan=plan,
        terminal={"state": None},
        original={"canonical_smiles": None, "valid_connected": False},
        atoms=(),
        adapter=None,
        policy={},
        limits={},
    )
    assert result["selected_smiles"] is None and result["disposition"] == "original_invalid"
    assert result["proposals"] == [] and result["additional_assembly_checks"] == 0


def test_trace_reproduces_unchanged_sampler_and_canonical_graphs():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, vocabulary, atoms, nodes, bonds, _ = load_synthesis_program_checkpoint(
            REPO / "results/phase1/combinatorial_checkpoint_v1/checkpoint.json", device="cpu"
        )
        with SynthesisProgramProductionCache(
            REPO / "results/phase1/combinatorial_program_cache_v2/cache.npz"
        ) as cache:
            records = cache.records(
                [
                    int(cache.indices(program_id=f, fold="train")[0])
                    for f in vocabulary.program_states[1:]
                ]
            )
        trace = TerminalTrace(model, records, atoms)
        results = [
            sample_synthesis_program_products(
                network,
                records,
                atoms,
                nodes,
                bonds,
                samples_per_program=1,
                sample_steps=2,
                batch_size=4,
                seed=101,
                device="cpu",
                terminal_decode_policy="strict_valence_topology_argmax",
            )
            for network in (model, trace)
        ]
        assert results[0] == results[1]
        assert len(trace.terminals) == len(records)
        for row, terminal in zip(results[0][0], trace.terminals, strict=True):
            if terminal["state"] is not None:
                assert (
                    graph_smiles(*state_graph(terminal["state"]), atoms) == row["canonical_smiles"]
                )
    finally:
        torch.set_num_threads(previous)
