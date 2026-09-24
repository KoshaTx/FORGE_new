from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from forge.assembly.families import load_assembly_libraries
from forge.assembly.library_generation import check_generated_program
from forge.assembly.library_occurrences import (
    OccurrenceTraceError,
    normalize_occurrences,
    trace_precursor_occurrences,
)
from forge.assembly.library_programs import LibraryProgramLimits
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.precursor_occurrence_reuse import (
    OccurrenceReusePlan,
    copy_occurrence_graph,
    qualify_occurrence_reuse,
)
from forge.model.precursor_reuse import PrecursorReuseError, qualify_reuse

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def examples():
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
    layouts = json.loads(
        (REPO / "results/phase1/combinatorial_graph_reuse_discovery_v1/layouts.json").read_text()
    )
    with SynthesisProgramProductionCache(
        REPO / "results/phase1/combinatorial_program_cache_v2/cache.npz"
    ) as cache:
        records = {
            depth: cache.record(
                next(
                    r["index"]
                    for r in layouts
                    if r["family"] == "urea_amine_isocyanate" and r["depth"] == depth
                )
            )
            for depth in (2, 3, 4)
        }
    return config, libraries, records


def test_normalization_preserves_relation_but_discards_occurrence_and_atom_names():
    value = normalize_occurrences(((0, 2, 4), (1, 3, 5)))
    assert value == normalize_occurrences(((5, 1, 3), (4, 0, 2)))
    assert value != normalize_occurrences(((0, 2, 4), (3, 1, 5)))


@pytest.mark.parametrize("value", [(), ((0,), ()), ((0, 1), (1, 2)), ((0,), (True,))])
def test_invalid_correspondences_fail(value):
    with pytest.raises(OccurrenceTraceError):
        normalize_occurrences(value)


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_exact_replay_recovers_each_precursor_occurrence(examples, depth):
    config, libraries, records = examples
    record = records[depth]
    plan = qualify_occurrence_reuse(
        record,
        libraries[record.program_id],
        config["programs"][record.program_id],
        config["limits"],
    )
    assert plan.status == "qualified", plan
    assert len(plan.occurrences) == depth
    assert len(set(i for o in plan.occurrences for i in o)) == sum(map(len, plan.occurrences))
    if depth > 2:
        old = qualify_reuse(
            record,
            libraries[record.program_id],
            config["programs"][record.program_id],
            config["limits"],
        )
        assert old.status == "occurrence_partition_mismatch"


def test_occurrence_copy_handles_noncontiguous_atoms_and_preserves_cross_edges():
    plan = OccurrenceReusePlan("example", ((0, 2), (1, 3)), "qualified", 1)
    nodes = np.asarray([0, 1, 2, 3])
    edges = np.zeros((4, 4), dtype=int)
    edges[0, 2] = edges[2, 0] = 2
    edges[1, 3] = edges[3, 1] = 1
    edges[0, 1] = edges[1, 0] = 1
    before = (nodes.copy(), edges.copy())
    changed, adjacency = copy_occurrence_graph(nodes, edges, plan, 1)
    assert changed.tolist() == [1, 1, 3, 3]
    assert adjacency[0, 2] == adjacency[1, 3] == adjacency[0, 1] == 1
    assert np.array_equal(nodes, before[0]) and np.array_equal(edges, before[1])


def test_trace_abstains_on_saturated_enumeration_and_wrong_intermediate(examples):
    config, libraries, records = examples
    r = records[4]
    p = config["programs"][r.program_id]
    limits = LibraryProgramLimits(p["maximum_steps"], **config["limits"])
    check = check_generated_program(
        libraries[r.program_id],
        r.graph.canonical_smiles,
        depth=4,
        accumulator_role=p["accumulator_role"],
        limits=limits,
    )
    w = check.programs[0]
    with pytest.raises(OccurrenceTraceError, match="outcome limit"):
        trace_precursor_occurrences(
            libraries[r.program_id],
            w["components"],
            w["intermediate_products"],
            accumulator_role=p["accumulator_role"],
            limits=replace(limits, maximum_outcomes=1),
        )
    wrong = copy.deepcopy(w["intermediate_products"])
    wrong[-1] = w["components"][p["accumulator_role"]]
    with pytest.raises(OccurrenceTraceError, match="exact step"):
        trace_precursor_occurrences(
            libraries[r.program_id],
            w["components"],
            wrong,
            accumulator_role=p["accumulator_role"],
            limits=limits,
        )


def test_existing_role_partition_conflict_cannot_be_silently_relabelled(examples):
    config, libraries, records = examples
    r = records[4]
    blocks = tuple(replace(b, role="unexpected") for b in r.component_blocks)
    changed = replace(r, component_blocks=blocks)
    plan = qualify_occurrence_reuse(
        changed, libraries[r.program_id], config["programs"][r.program_id], config["limits"]
    )
    assert plan.status == "source_role_partition_conflict"


def test_identical_precursors_do_not_authorize_unequal_product_atom_states(examples):
    config, libraries, records = examples
    r = records[4]
    plan = qualify_occurrence_reuse(
        r, libraries[r.program_id], config["programs"][r.program_id], config["limits"]
    )
    states = r.graph.node_states.copy()
    states[plan.occurrences[1][0]] += 1
    changed = replace(r, graph=replace(r.graph, node_states=states))
    checked = qualify_occurrence_reuse(
        changed, libraries[r.program_id], config["programs"][r.program_id], config["limits"]
    )
    assert checked.status == "product_occurrence_states_differ"


def test_internal_atom_symmetry_still_abstains(examples):
    config, libraries, _ = examples
    base = REPO / "results/phase1/combinatorial_graph_reuse_discovery_v1"
    layouts = json.loads((base / "layouts.json").read_text())
    plans = json.loads((base / "reuse_plans.json").read_text())
    index = next(
        row["index"]
        for row, plan in zip(layouts, plans, strict=True)
        if plan["status"] == "ambiguous_atom_correspondence"
    )
    with SynthesisProgramProductionCache(
        REPO / "results/phase1/combinatorial_program_cache_v2/cache.npz"
    ) as cache:
        record = cache.record(index)
    plan = qualify_occurrence_reuse(
        record,
        libraries[record.program_id],
        config["programs"][record.program_id],
        config["limits"],
    )
    assert plan.status == "trace_not_qualified" and "ambiguous" in plan.reason


def test_occurrence_verifier_rejects_changed_policy_before_scoring(tmp_path):
    from experiments.phase1.multireaction.combinatorial_occurrence_reuse import verify

    result = json.loads(
        (
            REPO / "results/phase1/combinatorial_occurrence_reuse_discovery_v1/result.json"
        ).read_text()
    )
    result["policy"]["gate_changes"] = True
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    with pytest.raises(PrecursorReuseError, match="contract"):
        verify(REPO, path)


def test_occurrence_verifier_rejects_recomputed_summary_substitution(tmp_path, monkeypatch):
    from experiments.phase1.multireaction import combinatorial_occurrence_reuse as experiment

    result = json.loads(
        (
            REPO / "results/phase1/combinatorial_occurrence_reuse_discovery_v1/result.json"
        ).read_text()
    )
    # Isolate the comparison boundary; the full saved calculation is also exercised by the CLI.
    expected = {"total_exact_gain": result["total_exact_gain"]}
    monkeypatch.setattr(experiment, "calculate", lambda *_: (expected, {}))
    result["total_exact_gain"] += 1
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    with pytest.raises(PrecursorReuseError, match="summary/decision"):
        experiment.verify(REPO, path)
