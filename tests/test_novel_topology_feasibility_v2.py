"""Protect correspondence scope and actual saved input schema without proposals."""

import networkx as nx
import pytest
import torch

from experiments.phase1.multireaction.novel_topology_feasibility_v2 import (
    ROOT,
    as_state,
    correspondence,
    existing_role_graph,
)
from forge.model.precursor_reuse_projection import state_graph


def path_graph():
    graph = nx.path_graph(5)
    for node in graph:
        graph.nodes[node].update(
            block=0,
            core=0 if node == 0 else -1,
            ports=(),
            fixed_atom=1 if node == 0 else -1,
            atom=1,
        )
    nx.set_edge_attributes(graph, 1, "bond")
    return graph


def test_exterior_order_is_not_chemistry():
    graph = path_graph()
    permuted = nx.relabel_nodes(graph, {1: 4, 2: 3, 3: 2, 4: 1})
    assert correspondence(graph, permuted, "atom_bond_labelled")["status"] == "match"


def test_levels_are_separate():
    left = path_graph()
    right = left.copy()
    right.nodes[3]["atom"] = 2
    assert correspondence(left, right, "topology")["status"] == "match"
    assert correspondence(left, right, "bond_labelled")["status"] == "match"
    assert correspondence(left, right, "atom_bond_labelled")["status"] == "no_match"
    right.edges[2, 3]["bond"] = 2
    assert correspondence(left, right, "topology")["status"] == "match"
    assert correspondence(left, right, "bond_labelled")["status"] == "no_match"


@pytest.mark.parametrize(
    "field,value", [("core", 7), ("block", 2), ("ports", ((4, 1, 1),)), ("fixed_atom", 2)]
)
def test_core_origin_and_ports_protected(field, value):
    left = path_graph()
    right = left.copy()
    right.nodes[0][field] = value
    assert correspondence(left, right, "topology")["status"] == "no_match"


def test_search_limit_is_unknown():
    assert (
        correspondence(path_graph(), path_graph(), "topology", visits=0)["status"]
        == "unknown_search_limit"
    )


def test_actual_schema_source_positive():
    layouts = torch.load(
        ROOT / "results/phase1/compose_lipid_quality_confirmation_v1/input.pt",
        map_location="cpu",
        weights_only=False,
    )["layouts"]
    from experiments.phase1.multireaction.component_feasibility_census_v2 import pin, read
    from forge.corpus.mapped_program_cache import MappedProgramCache

    prior = (
        ROOT
        / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1/b_quality/component_feasibility_v1/run_v2"
    )
    profiles = read(prior / "TRAIN_profiles.json")["profiles"]
    requests = read(prior / "result.json")["requests"]
    with MappedProgramCache(
        ROOT,
        manifest=pin(
            ROOT / "results/phase1/compose_lipid_mapped_preparation_v3/cache/manifest.json"
        ),
    ) as cache:
        for family in ("a3_amine_aldehyde_alkyne", "ketone_ugi4"):
            req = next(r for r in requests if r["family"] == family)
            role = next(k for k, v in req["roles"].items() if "profile_witnesses" in v)
            p = profiles[next(iter(req["roles"][role]["profile_witnesses"].values()))[0]]
            rec = cache.record(p["witness_index"]).record
            nodes, edges = state_graph(as_state(rec))
            graph = existing_role_graph(rec, nodes, edges, role)
            assert correspondence(graph, graph, "atom_bond_labelled")["status"] == "match"
            assert (
                len(layouts[req["request"]].record.core_position_states)
                == layouts[req["request"]].record.node_count
            )
            with pytest.raises(ValueError, match="dimensions"):
                existing_role_graph(rec, nodes[:-1], edges, role)
            broken = edges.copy()
            blocks = rec.component_blocks
            a = next(
                i
                for b in blocks
                if b.role == role
                for i in range(b.start, b.stop)
                if rec.core_position_states[i] == 1
            )
            b = next(
                i
                for b in blocks
                if b.role != role
                for i in range(b.start, b.stop)
                if rec.core_position_states[i] == 1
            )
            broken[a, b] = broken[b, a] = 1
            with pytest.raises(ValueError, match="cross-origin"):
                existing_role_graph(rec, nodes, broken, role)


def test_empty_closure_arrays_use_integer_indices():
    from types import SimpleNamespace

    import numpy as np

    from experiments.phase1.multireaction.novel_topology_feasibility_v2 import saved_morphology

    state = dict(
        nodes=[1, 1],
        parents=[0, 0],
        parent_bonds=[0, 0],
        closure_left=[],
        closure_right=[],
        closure_bonds=[],
    )
    record = SimpleNamespace(
        node_count=2, role_states=np.array([1, 1]), core_position_states=np.array([2, 1])
    )
    actual = saved_morphology(record, state)
    assert actual.tolist() == [[2, 1, 1, 2], [2, 1, 1, 2]]
