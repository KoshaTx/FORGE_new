from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from forge.assembly.families import load_assembly_libraries
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.precursor_occurrence_reuse import OccurrenceReusePlan
from forge.model.precursor_reuse import PrecursorReuseError
from forge.model.precursor_reuse_projection import state_graph
from forge.model.program_connection_reuse import (
    ProgramConnectionPlan,
    connect_generated_occurrences,
    occurrence_labels,
    propose_program_connections,
    qualify_program_connections,
)
from forge.model.synthesis_program_sampling import load_synthesis_program_checkpoint

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source():
    config = json.loads(
        (REPO / "configs/multireaction/library_program_dataset_v1.json").read_text()
    )
    libraries = load_assembly_libraries(
        [
            (REPO / config["inputs"][k]["path"], config["inputs"][k]["sha256"])
            for k in config["registries"]
        ],
        expected_families=config["programs"],
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _, vocabulary, atoms, *_ = load_synthesis_program_checkpoint(
            REPO / "results/phase1/combinatorial_checkpoint_v1/checkpoint.json", device="cpu"
        )
    finally:
        torch.set_num_threads(previous)
    layouts = json.loads(
        (REPO / "results/phase1/combinatorial_graph_reuse_discovery_v1/layouts.json").read_text()
    )
    with SynthesisProgramProductionCache(
        REPO / "results/phase1/combinatorial_program_cache_v2/cache.npz"
    ) as cache:
        records = {}
        for row in layouts:
            if row["depth"] > 1 and (row["family"], row["depth"]) not in records:
                records[row["family"], row["depth"]] = cache.record(row["index"])
    return config, libraries, vocabulary, atoms, records


def test_connection_projection_keeps_generated_interiors_and_replaces_only_cross_edges():
    occurrence = OccurrenceReusePlan("example", ((0, 2), (1, 3)), "qualified", 1)
    plan = ProgramConnectionPlan("example", "qualified", occurrence, ((0, 4, 1), (1, 4, 1)))
    nodes = np.array([0, 1, 2, 3, 4])
    edges = np.zeros((5, 5), dtype=int)
    edges[0, 2] = edges[2, 0] = 2
    edges[1, 3] = edges[3, 1] = 1
    edges[0, 1] = edges[1, 0] = 1
    edges[3, 4] = edges[4, 3] = 1
    generated, projected = connect_generated_occurrences(nodes, edges, plan, 1)
    assert generated.tolist() == [1, 1, 3, 3, 4]
    assert projected[0, 2] == projected[1, 3] == 1
    assert projected[0, 1] == projected[3, 4] == 0
    assert projected[0, 4] == projected[1, 4] == 1
    assert edges[0, 1] == 1 and nodes.tolist() == [0, 1, 2, 3, 4]


def test_invalid_or_duplicate_connection_plan_fails():
    occurrence = OccurrenceReusePlan("example", ((0, 2), (1, 3)), "qualified", 1)
    with pytest.raises(PrecursorReuseError, match="identity"):
        ProgramConnectionPlan("other", "qualified", occurrence, ((0, 4, 1), (1, 4, 1)))
    with pytest.raises(PrecursorReuseError, match="connection program"):
        ProgramConnectionPlan("example", "qualified", occurrence, ((0, 4, 1), (0, 4, 1)))


def test_repeated_source_connections_preserve_graphs_or_abstain_on_ambiguity(source):
    config, libraries, vocabulary, _, records = source
    covered = set()
    for (family, depth), record in records.items():
        plan = qualify_program_connections(
            record, libraries[family], config["programs"][family], config["limits"], vocabulary
        )
        if family == "iphos_amine_dioxaphospholane":
            assert plan.status == "trace_not_qualified"
            assert plan.reason == "ambiguous occurrence correspondence"
            assert not plan.connections
            covered.add(family)
            continue
        assert plan.status == "qualified", (family, depth, plan)
        state = {
            name: getattr(record.graph, name).tolist()
            for name in (
                "node_states",
                "parents",
                "parent_bonds",
                "closure_left",
                "closure_right",
                "closure_bonds",
            )
        }
        state["nodes"] = state.pop("node_states")
        nodes, edges = state_graph(state)
        labels = occurrence_labels(record.node_count, plan.occurrence_plan)
        assert len(plan.connections) == depth
        for a, b, _ in plan.connections:
            assert labels[a] != labels[b]
            assert record.core_position_states[a] > 1 and record.core_position_states[b] > 1
        for donor in range(depth):
            changed = connect_generated_occurrences(nodes, edges, plan, donor)
            assert np.array_equal(changed[0], nodes) and np.array_equal(changed[1], edges)
        covered.add(family)
    assert covered == {
        f for f, p in config["programs"].items() if p["accumulator_role"] is not None
    }


def test_registry_core_position_mismatch_abstains(source):
    config, libraries, vocabulary, _, records = source
    record = records["urea_amine_isocyanate", 4]
    changed = replace(record, core_position_states=np.ones(record.node_count, dtype=np.int64))
    plan = qualify_program_connections(
        changed,
        libraries[record.program_id],
        config["programs"][record.program_id],
        config["limits"],
        vocabulary,
    )
    assert plan.status == "connection_not_in_product_template"


def test_exact_original_retained_without_terminal_reconstruction(source):
    config, libraries, vocabulary, atoms, records = source
    r = records["urea_amine_isocyanate", 4]
    plan = qualify_program_connections(
        r, libraries[r.program_id], config["programs"][r.program_id], config["limits"], vocabulary
    )
    result = propose_program_connections(
        record=r,
        plan=plan,
        terminal={"state": None},
        original={"valid_connected": True, "assembly": {"status": "exact_computed_program"}},
        atoms=atoms,
        adapter=libraries[r.program_id],
        policy=config["programs"][r.program_id],
        limits=config["limits"],
    )
    assert result["proposals"] == [] and result["disposition"] == "original_exact"
