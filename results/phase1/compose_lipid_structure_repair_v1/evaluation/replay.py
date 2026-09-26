"""Bounded replay of original and added candidate graphs with explicit provenance."""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from forge.model.precursor_reuse_projection import state_graph
from results.phase1.compose_lipid_quality_decode_v2.run import serial_state
from results.phase1.compose_lipid_quality_selection_v2.run import pin
from results.phase1.compose_lipid_structural_audit_v1.layout_audit import location

ROOT = Path(__file__).resolve().parents[4]


def load_candidate_graphs(rows, branch, requested_ordinals, layouts):
    """Yield request/ordinal, graph, exact tree basis, and source pin without inference.

    Candidate ordinal0 is branch.raw; ordinal i+1 is branch.proposals[i]. Original
    branch provenance may name d1 even when a caller uses another branch label.
    New compact candidates may supply inline graphs/state or a pinned shard locator.
    Each full graph shard is loaded once, then released.
    """
    grouped = defaultdict(list)
    observed = set()

    def materialize(index, ordinal, proposal, source):
        state = proposal.get("tree_state", proposal.get("state"))
        if proposal.get("nodes") is not None and proposal.get("edges") is not None:
            nodes, edges = np.asarray(proposal["nodes"]), np.asarray(proposal["edges"])
            if state is not None:
                tree_nodes, tree_edges = state_graph(state)
                if not np.array_equal(tree_nodes, nodes) or not np.array_equal(tree_edges, edges):
                    raise ValueError("Supplied candidate tree does not reconstruct dense graph")
        elif state is not None:
            nodes, edges = state_graph(state)
        else:
            raise ValueError("Candidate has neither dense graph nor reconstructing tree")
        if state is None:
            state = serial_state(nodes, edges, layouts[index])
            basis = (
                "reconstructed_ordered_lowest_parent" if state else "no_ordered_tree_reconstruction"
            )
        else:
            basis = proposal.get("tree_basis", "observed_saved_raw_tree")
        if (index, ordinal) in observed:
            raise ValueError("Candidate graph was replayed twice")
        observed.add((index, ordinal))
        return index, ordinal, nodes, edges, state, basis, source

    for row in rows:
        index = row["index"]
        branch_row = row["branches"][branch]
        proposals = [dict(branch_row["raw"], kind="raw"), *branch_row.get("proposals", [])]
        for ordinal, proposal in enumerate(proposals):
            if (index, ordinal) not in requested_ordinals:
                continue
            if (
                proposal.get("tree_state") is not None
                or proposal.get("state") is not None
                or (proposal.get("nodes") is not None and proposal.get("edges") is not None)
            ):
                yield materialize(index, ordinal, proposal, None)
                continue
            provenance = proposal.get("graph_provenance")
            kind = proposal.get("kind")
            if provenance is None:
                entry, kind = location(branch_row, kind)
                if kind == "raw" and entry.get("raw", {}).get("state") is not None:
                    yield materialize(index, ordinal, entry["raw"], None)
                    continue
                provenance = entry["construction_provenance"]
            source = provenance["source"]
            grouped[source["path"]].append((index, ordinal, proposal, provenance, source, kind))
    for path, queries in sorted(grouped.items()):
        path = ROOT / path
        expected = queries[0][4]
        if pin(path) != expected or any(q[4] != expected for q in queries):
            raise ValueError("Candidate graph provenance pin mismatch")
        payload = path.read_bytes()
        shard = json.loads(gzip.decompress(payload) if path.suffix == ".gz" else payload)
        if isinstance(shard, dict) and "attempts" in shard:
            shard = shard["attempts"]
        shard_by_index = {r["index"]: r for r in shard}
        for index, ordinal, proposal, provenance, source, kind in queries:
            full = shard_by_index[provenance.get("index", index)]["branches"][
                provenance.get("branch", branch)
            ]
            if "proposal_ordinal" in provenance:
                resolved = full["proposals"][provenance["proposal_ordinal"]]
                if resolved["smiles"] != proposal["smiles"] or resolved["kind"] != proposal["kind"]:
                    raise ValueError("Explicit proposal locator does not match compact candidate")
            else:
                matches = [
                    p
                    for p in full["proposals"]
                    if p["kind"] == kind and p["smiles"] == proposal["smiles"]
                ]
                if len(matches) != 1:
                    raise ValueError("Compact candidate does not identify one full graph")
                resolved = matches[0]
                if kind == "raw":
                    resolved = {**resolved, "state": full["raw"]["state"]}
            yield materialize(index, ordinal, resolved, source)
    if observed != set(requested_ordinals):
        raise ValueError("Requested graph coverage is incomplete")
