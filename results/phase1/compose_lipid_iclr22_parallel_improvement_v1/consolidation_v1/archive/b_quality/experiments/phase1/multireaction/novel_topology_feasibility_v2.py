"""Compare existing saved graphs only; never construct a replacement molecule."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import torch

from experiments.phase1.multireaction import component_feasibility_census_v2 as census
from forge.corpus.mapped_program_cache import MappedProgramCache
from forge.model.precursor_reuse_projection import state_graph
from forge.model.reaction_program_flow import derive_role_morphology_states
from forge.model.sparse_topology_feasibility import INDEX_TO_DENSE_BOND
from results.phase1.compose_lipid_structure_repair_v1.evaluation.replay import load_candidate_graphs

ROOT, WORKTREE, N = census.ROOT, census.WORKTREE, census.N
OUT = N / "b_quality/novel_topology_feasibility_v1/run_v2"
PRIOR = N / "b_quality/component_feasibility_v1/run_v2"
read, write, pin = census.read, census.write, census.pin
LEVELS = ("topology", "bond_labelled", "atom_bond_labelled")


class SearchLimitError(RuntimeError):
    pass


class BoundedMatcher(nx.algorithms.isomorphism.GraphMatcher):
    def __init__(self, *args, cpu_limit=0.15, visits=20000, **kwargs):
        self.started, self.cpu_limit, self.limit, self.visits = (
            time.process_time(),
            cpu_limit,
            visits,
            0,
        )
        super().__init__(*args, **kwargs)

    def semantic_feasibility(self, a, b):
        self.visits += 1
        if self.visits > self.limit or time.process_time() - self.started > self.cpu_limit:
            raise SearchLimitError("bounded_existing_graph_correspondence")
        return super().semantic_feasibility(a, b)


def correspondence(left, right, level, *, visits=20000, cpu_limit=0.15):
    """Existing graph isomorphism; exterior node indices never constrain matching."""
    if level not in LEVELS:
        raise ValueError("Unknown correspondence level")

    def node_match(a, b):
        if any(a[k] != b[k] for k in ("block", "core", "ports", "fixed_atom")):
            return False
        return level != "atom_bond_labelled" or a["atom"] == b["atom"]

    edge_match = None if level == "topology" else lambda a, b: a["bond"] == b["bond"]
    matcher = BoundedMatcher(
        left,
        right,
        node_match=node_match,
        edge_match=edge_match,
        visits=visits,
        cpu_limit=cpu_limit,
    )
    try:
        matched = matcher.is_isomorphic()
        return {
            "status": "match" if matched else "no_match",
            "mapping": (
                sorted([int(a), int(b)] for a, b in matcher.mapping.items()) if matched else []
            ),
            "search_visits": matcher.visits,
        }
    except SearchLimitError:
        return {"status": "unknown_search_limit", "mapping": [], "search_visits": matcher.visits}


def existing_role_graph(record, nodes, edges, role):
    """Represent a saved role subgraph; anchored ports retain the rest of its source core."""
    n = record.node_count
    if nodes.shape != (n,) or edges.shape != (n, n):
        raise ValueError("Saved graph dimensions do not match the request")
    if not np.array_equal(edges, edges.T) or np.any(np.diag(edges)):
        raise ValueError("Malformed saved adjacency")
    blocks = [b for b in record.component_blocks if b.role == role]
    if not blocks:
        raise ValueError("Absent role")
    selected = [i for b in blocks for i in range(b.start, b.stop)]
    selected_set = set(selected)
    core_nodes = np.flatnonzero(record.core_position_states > 1)
    core_rank = {int(node): i for i, node in enumerate(core_nodes)}
    graph = nx.Graph()
    for occurrence, block in enumerate(blocks):
        for i in range(block.start, block.stop):
            ports = []
            for j in np.flatnonzero(edges[i]):
                if int(j) in selected_set:
                    continue
                if i not in core_rank or int(j) not in core_rank:
                    raise ValueError("Unqualified cross-origin edge")
                ports.append((core_rank[int(j)], int(nodes[j]), int(edges[i, j])))
            graph.add_node(
                i,
                atom=int(nodes[i]),
                block=occurrence,
                core=core_rank.get(i, -1),
                ports=tuple(sorted(ports)),
                fixed_atom=int(nodes[i]) if i in core_rank else -1,
            )
    for a in selected:
        for b in selected:
            if b > a and edges[a, b]:
                graph.add_edge(a, b, bond=int(edges[a, b]))
    return graph


def graph_inventory(graph):
    return {
        "nodes": len(graph),
        "edges": graph.number_of_edges(),
        "atom_states": dict(sorted(Counter(d["atom"] for _, d in graph.nodes(data=True)).items())),
        "bond_states": dict(
            sorted(Counter(d["bond"] for _, _, d in graph.edges(data=True)).items())
        ),
    }


def saved_morphology(record, state):
    """Empty saved closures are integer index arrays, including the zero-cycle case."""
    return derive_role_morphology_states(
        SimpleNamespace(
            node_count=record.node_count,
            role_states=record.role_states,
            core_position_states=record.core_position_states,
            graph=SimpleNamespace(**{k: np.asarray(v, dtype=np.int64) for k, v in state.items()}),
        )
    )


def schema_preflight(layouts, requests, generated, files):
    """Parse every actual selected/witness shape without graph matching."""
    start = time.process_time()
    profiles = read(files["profiles"])["profiles"]
    profile_ids = sorted(
        {
            j
            for r in requests
            for v in r["roles"].values()
            for js in v.get("profile_witnesses", {}).values()
            for j in js
        }
    )
    counts = Counter()
    for request in requests:
        i = request["request"]
        rec, saved = layouts[i].record, generated[i]
        state = saved["tree_state"]
        if state is None:
            raise ValueError("Selected graph lacks an auditable serialization")
        nodes, edges = state_graph(state)
        if not np.array_equal(nodes, saved["nodes"]) or not np.array_equal(edges, saved["edges"]):
            raise ValueError("Selected saved serialization mismatch")
        if nodes.shape != (rec.node_count,):
            raise ValueError("Selected count mismatch")
        morphology = saved_morphology(rec, state)
        assert morphology.shape == (rec.node_count, 4)
        counts["selected_records"] += 1
        counts["selected_zero_closures"] += not state["closure_left"]
        counts["selected_nonexact"] += not next(
            v["selected"]["joint"]["source_exact"]
            for v in request["roles"].values()
            if "selected" in v
        )
        counts["selected_repeated_role_blocks"] += len(rec.component_blocks) > len(
            {b.role for b in rec.component_blocks}
        )
    with MappedProgramCache(ROOT, manifest=pin(files["manifest"])) as cache:
        for j in profile_ids:
            profile = profiles[j]
            example = cache.record(profile["witness_index"])
            census.require_source_correspondence(
                profile["witness_target"], profile["family"], example
            )
            state = as_state(example.record)
            nodes, edges = state_graph(state)
            existing_role_graph(example.record, nodes, edges, profile["role"])
            saved_morphology(example.record, state)
            counts["witness_profiles"] += 1
            counts["witness_zero_closures"] += not state["closure_left"]
    return {
        "passed": True,
        "counts": dict(counts),
        "matching_calls": 0,
        "CPU_seconds": time.process_time() - start,
    }


def as_state(record):
    g = record.graph
    return {
        "nodes": g.node_states.tolist(),
        **{
            k: getattr(g, k).tolist()
            for k in ("parents", "parent_bonds", "closure_left", "closure_right", "closure_bonds")
        },
    }


def paths():
    return {
        "producer": Path(__file__),
        "tests": WORKTREE / "tests/test_novel_topology_feasibility_v2.py",
        "test_receipt": OUT / "tests.xml",
        "census": PRIOR / "result.json",
        "profiles": PRIOR / "TRAIN_profiles.json",
        "census_verification": PRIOR / "verification.json",
        "census_protocol": PRIOR / "protocol.json",
        "joint": census.paths()["joint"],
        "layouts": census.paths()["layouts"],
        "manifest": census.paths()["manifest"],
        "compact": ROOT
        / "results/phase1/compose_lipid_quality_confirmation_v1/compact_attempts.json",
        "B_result": N / "b_quality/closure_v1/result.json",
        "B_handoff": N / "b_quality/closure_v1/handoff.json",
        "preserved_v1_failure": OUT.parent / "run_failure.json",
        "preserved_v1_protocol": OUT.parent / "protocol.json",
    }


def collect_existing(files):
    """Recover exact chosen saved graphs; reconstruction is serialization, not new chemistry."""
    layouts = torch.load(files["layouts"], weights_only=False, map_location="cpu")["layouts"]
    requests = read(files["census"])["requests"]
    wanted = {r["request"] for r in requests}
    joint = read(files["joint"])
    chosen = {r["request"]: r for r in joint["selection"] if r["request"] in wanted}
    evidence = {r["request"]: r for r in joint["selected_evidence"] if r["request"] in wanted}
    generated, sources = {}, {}
    b_wanted = {i for i in wanted if evidence[i]["source_arm"] == "B"}
    for receipt in read(files["B_result"])["parent_receipts"]:
        path = ROOT / receipt["path"]
        if census.pin(path)["sha256"] != receipt["sha256"]:
            raise ValueError("Changed B parent")
        parent = read(path)
        i = parent["request"]
        if i not in b_wanted:
            continue
        sources[str(path)] = pin(path)
        options = [p for p in parent["assessed_proposals"] if p["candidate"] == chosen[i]]
        if len(options) != 1:
            raise ValueError("Ambiguous selected B identity")
        g = options[0]["graph"]
        generated[i] = {k: g[k] for k in ("nodes", "edges", "tree_state", "tree_basis", "smiles")}
    compact = read(files["compact"])
    ordinal_set = {(i, chosen[i]["ordinal"]) for i in wanted - b_wanted}
    for i, ordinal, nodes, edges, state, basis, source in load_candidate_graphs(
        compact, "d1", ordinal_set, layouts
    ):
        row = next(r for r in compact if r["index"] == i)["branches"]["d1"]
        candidate = [row["raw"], *row["proposals"]][ordinal]
        if candidate["smiles"] != chosen[i]["smiles"]:
            raise ValueError("Saved graph/selected identity mismatch")
        if source:
            path = ROOT / source["path"]
            if pin(path)["sha256"] != source["sha256"]:
                raise ValueError("Changed selected graph source")
            sources[str(path)] = pin(path)
        generated[i] = dict(
            nodes=nodes.tolist(),
            edges=edges.tolist(),
            tree_state=state,
            tree_basis=basis,
            smiles=chosen[i]["smiles"],
        )
    if set(generated) != wanted or len(wanted) != 128:
        raise ValueError("Incomplete 128-request recovery")
    return layouts, requests, chosen, evidence, generated, sources


def prepare():
    started = time.process_time()
    files = paths()
    layouts, requests, _, _, generated, sources = collect_existing(files)
    write(OUT / "schema_preflight.json", schema_preflight(layouts, requests, generated, files))
    files["schema_preflight"] = OUT / "schema_preflight.json"
    write(OUT / "generated_graphs.json", {"graphs": generated, "sources": sources})
    files["generated_graphs"] = OUT / "generated_graphs.json"
    modules = {
        name: pin(Path(module.__file__))
        for name, module in sys.modules.items()
        if name.startswith(
            (
                "forge.",
                "experiments.phase1.multireaction.component_feasibility",
                "results.phase1.compose_lipid_structure_repair_v1.evaluation.replay",
                "results.phase1.compose_lipid_quality_decode_v2.run",
            )
        )
        and getattr(module, "__file__", None)
    }
    assert all(
        Path(p["path"]).is_relative_to(WORKTREE)
        for k, p in modules.items()
        if k.startswith("forge.")
    )
    write(
        OUT / "protocol.json",
        {
            "schema": "forge.existing_novel_topology_feasibility.v1",
            "inputs": {k: pin(p) for k, p in files.items()},
            "source_modules": modules,
            "graph_sources": sources,
            "requests": [r["request"] for r in requests],
            "CPU_cap_seconds": 240,
            "aggregate_authorized_CPU_cap_seconds": 300,
            "threads": 1,
            "seed": None,
            "deterministic": True,
            "levels": LEVELS,
            "per_pair_level_CPU_cap_seconds": 0.15,
            "per_pair_level_visits": 20000,
            "mapping": "Preserve matched source origin-block occurrence and global core-coordinate anchors/ports; arbitrary exterior-node permutation is allowed. No assumption that serialized exterior order is chemistry.",
            "criterion": "Existing generated role graph has an isomorphism to a matched TRAIN witness topology. Bond-labelled and full atom/bond-labelled matches reported separately. A novel identity already occupying a supported topology is evidence only about that existing graph.",
            "boundary": "Changed-topology transport with retained generated chemistry is not tested. Atom/bond inventory agreement is necessary only; no new adjacency, molecule, source replay or selection is produced.",
            "unknown": "Search limits are unknown; nonmatches do not establish chemical impossibility or exhaustive source inadequacy.",
            "population": "Current selected A3/ketoneUgi4, 64 each, TRAIN-derived request layouts; no source-disjoint or independent-realism claim.",
            "prepare_CPU_seconds": time.process_time() - started,
        },
    )
    print(
        {
            "protocol": pin(OUT / "protocol.json"),
            "prepare_CPU_seconds": time.process_time() - started,
        },
        flush=True,
    )


def run():
    started = time.process_time()
    torch.set_num_threads(1)
    protocol = read(OUT / "protocol.json")
    for item in [
        *protocol["inputs"].values(),
        *protocol["source_modules"].values(),
        *protocol["graph_sources"].values(),
    ]:
        if pin(Path(item["path"])) != item:
            raise ValueError("Changed input " + item["path"])

    def timeout(*_):
        raise TimeoutError("CPU audit cap reached; incomplete evidence is not admitted")

    signal.signal(signal.SIGPROF, timeout)
    signal.setitimer(signal.ITIMER_PROF, 235)
    files = {k: Path(v["path"]) for k, v in protocol["inputs"].items()}
    layouts = torch.load(files["layouts"], weights_only=False, map_location="cpu")["layouts"]
    requests = read(files["census"])["requests"]
    profiles = read(files["profiles"])["profiles"]
    generated = read(files["generated_graphs"])["graphs"]
    joint = read(files["joint"])
    evidence = {r["request"]: r for r in joint["selected_evidence"]}
    source_graphs, results = {}, []
    with MappedProgramCache(ROOT, manifest=pin(files["manifest"])) as cache:
        for request in requests:
            i, layout = request["request"], layouts[request["request"]]
            saved = generated[str(i)]
            nodes, edges = np.asarray(saved["nodes"]), np.asarray(saved["edges"])
            state = saved["tree_state"]
            if state is None:
                raise ValueError("No auditable saved tree")
            sn, se = state_graph(state)
            if not np.array_equal(sn, nodes) or not np.array_equal(se, edges):
                raise ValueError("Saved tree/dense graph mismatch")
            rec = layout.record
            core = np.flatnonzero(rec.core_position_states > 1)
            core_bundle = __import__("json").loads(request["template"])
            expected_core = np.zeros((len(core), len(core)), dtype=np.int64)
            for a, b, bond in core_bundle["core_tree"] + core_bundle["core_closures"]:
                expected_core[a, b] = expected_core[b, a] = INDEX_TO_DENSE_BOND[bond]
            fixed_core = np.array_equal(nodes[core], core_bundle["core_nodes"]) and np.array_equal(
                edges[np.ix_(core, core)], expected_core
            )
            morphology = saved_morphology(rec, state)
            roles = {}
            for role, requirement in request["roles"].items():
                if "profile_witnesses" not in requirement:
                    continue
                try:
                    graph = existing_role_graph(rec, nodes, edges, role)
                except ValueError as exc:
                    roles[role] = {
                        "status": "abstain_saved_graph",
                        "reason": str(exc),
                        "levels": {},
                    }
                    continue
                witness_rows = []
                for j in sorted(
                    {j for js in requirement["profile_witnesses"].values() for j in js}
                ):
                    profile = profiles[j]
                    if j not in source_graphs:
                        example = cache.record(profile["witness_index"])
                        census.require_source_correspondence(
                            profile["witness_target"], request["family"], example
                        )
                        ns, es = state_graph(as_state(example.record))
                        source_graphs[j] = existing_role_graph(example.record, ns, es, role)
                    source = source_graphs[j]
                    comparisons = {level: correspondence(graph, source, level) for level in LEVELS}
                    witness_rows.append(
                        {
                            "profile": j,
                            "identity": profile["identity"],
                            "witness_index": profile["witness_index"],
                            "levels": comparisons,
                            "atom_and_bond_inventory_equal": graph_inventory(graph)
                            == graph_inventory(source),
                        }
                    )
                states = {}
                for level in LEVELS:
                    statuses = [w["levels"][level]["status"] for w in witness_rows]
                    states[level] = (
                        "match"
                        if "match" in statuses
                        else (
                            "unknown_search_limit"
                            if "unknown_search_limit" in statuses
                            else "no_match"
                        )
                    )
                role_state = requirement["requirement"]["role_state"]
                pos = int(np.flatnonzero(rec.role_states == role_state)[0])
                roles[role] = {
                    "status": "assessed_existing_graph",
                    "inventory": graph_inventory(graph),
                    "levels": states,
                    "quantity": requirement["requirement"]["quantity"],
                    "morphology_matches_requested": morphology[pos].tolist()
                    == requirement["requirement"]["morphology"],
                    "selected": requirement["selected"]["joint"],
                    "witnesses": witness_rows,
                }
            results.append(
                {
                    "request": i,
                    "family": request["family"],
                    "smiles": saved["smiles"],
                    "source_exact": evidence[i]["design_assessment"]["source_exact"],
                    "design_pass": request["joint_design"],
                    "fixed_core_preserved": fixed_core,
                    "tree_basis": saved["tree_basis"],
                    "roles": roles,
                    "all_roles_match": {
                        level: all(v["levels"].get(level) == "match" for v in roles.values())
                        for level in LEVELS
                    },
                }
            )
    summary = {}
    for family in census.FAMILIES:
        rows = [r for r in results if r["family"] == family]
        summary[family] = {
            "requests": len(rows),
            "source_exact": sum(r["source_exact"] for r in rows),
            "design_failures": sum(not r["design_pass"] for r in rows),
            "fixed_core_preserved": sum(r["fixed_core_preserved"] for r in rows),
            "all_roles_match": {
                level: sum(r["all_roles_match"][level] for r in rows) for level in LEVELS
            },
            "roles": {
                role: {
                    "requests": len(rows),
                    "statuses": {
                        level: dict(
                            Counter(r["roles"][role]["levels"].get(level, "abstain") for r in rows)
                        )
                        for level in LEVELS
                    },
                    "morphology_matches": sum(
                        r["roles"][role].get("morphology_matches_requested", False) for r in rows
                    ),
                    "novel_topology_matches": sum(
                        r["roles"][role].get("selected", {}).get("TRAIN_novel_by_saved_metric")
                        is True
                        and r["roles"][role]["levels"].get("topology") == "match"
                        for r in rows
                    ),
                    "design_failed_topology_matches": sum(
                        not r["design_pass"]
                        and r["roles"][role]["levels"].get("topology") == "match"
                        for r in rows
                    ),
                }
                for role in rows[0]["roles"]
            },
        }
    write(
        OUT / "result.json",
        {
            "complete": True,
            "protocol": pin(OUT / "protocol.json"),
            "requests": results,
            "summary": summary,
            "distinct_witness_profiles_compared": len(source_graphs),
            "CPU_seconds": time.process_time() - started,
            "new_graph_proposals_molecules_source_executors_models_TEST_network_GPU_calls": 0,
            "topology_changing_repair_feasibility": "unassessed_requires_new_adjacency_chemistry_and_source_checks",
            "production_or_cohort_changed": False,
        },
    )
    print(
        {"result": pin(OUT / "result.json"), "CPU_seconds": time.process_time() - started},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    try:
        (prepare if args.stage == "prepare" else run)()
    except BaseException as exc:
        failure = OUT / (args.stage + "_failure.json")
        if not failure.exists():
            write(failure, {"complete": False, "error": type(exc).__name__, "message": str(exc)})
        raise
