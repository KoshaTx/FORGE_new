"""Recover selected graphs from pinned ledgers and assess sampled ring conditions."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from forge.model.compose_lipid_structural_audit import audit_layout_rings
from forge.model.defog_feasibility import AtomState, graph_to_molecule
from forge.model.precursor_reuse_projection import graph_smiles, state_graph
from results.phase1.compose_lipid_quality_decode_v2.run import serial_state
from results.phase1.compose_lipid_quality_selection_v2.run import pin, read, write

ROOT = Path(__file__).resolve().parents[3]


def location(branch, kind):
    if kind.startswith("retained_domain:"):
        _, draw, kind = kind.split(":", 2)
        entry = next(r for r in branch["domain_followups"] if r["draw"] == int(draw[4:]))
    elif kind.startswith("draw"):
        draw, kind = kind.split(":", 1)
        entry = next(r for r in branch["additional_draws"] if r["draw"] == int(draw[4:]))
    else:
        entry = branch
    return entry, kind


def recover_graphs(selected, ledger):
    """Load each needed full shard once; do not infer molecular identities."""
    by_index = {r["index"]: r for r in ledger}
    pending = defaultdict(list)
    pins = {}
    for row in selected:
        branch = by_index[row["index"]]["branches"][row["source_branch"]]
        entry, kind = location(branch, row["selected_kind"])
        if kind == "raw" and entry.get("raw", {}).get("state") is not None:
            state = entry["raw"]["state"]
            nodes, edges = state_graph(state)
            yield row, nodes, edges, state, "observed_saved_raw_tree", None
        else:
            source = entry["construction_provenance"]["source"]
            pending[source["path"]].append((row, kind, source))
    for path, requests in sorted(pending.items()):
        path = ROOT / path
        expected = requests[0][2]
        if pin(path) != expected or any(r[2] != expected for r in requests):
            raise ValueError("Selected graph shard pin mismatch")
        pins[expected["path"]] = expected
        raw = path.read_bytes()
        shard = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
        shard = {r["index"]: r for r in shard}
        for row, kind, _ in requests:
            full = shard[row["index"]]["branches"][row["source_branch"]]
            matches = [
                p
                for p in full["proposals"]
                if p["kind"] == kind and p["smiles"] == row["selected_smiles"]
            ]
            if len(matches) != 1:
                raise ValueError("Selected candidate does not identify exactly one graph")
            p = matches[0]
            state = full["raw"]["state"] if kind == "raw" else None
            yield (
                row,
                np.asarray(p["nodes"]),
                np.asarray(p["edges"]),
                state,
                "observed_saved_raw_tree" if state else "reconstructed_ordered_lowest_parent",
                expected,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--layout-payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(1)
    vocab = ROOT / "results/phase1/compose_lipid_v8_representation_v1/atom_vocabulary.json"
    paths = {
        "ledger": args.ledger,
        "selected": args.selected,
        "layout": args.layout_payload,
        "vocabulary": vocab,
        "runner": Path(__file__).resolve(),
        "module": ROOT / "forge/model/compose_lipid_structural_audit.py",
        "tree_reconstruction": ROOT / "results/phase1/compose_lipid_quality_decode_v2/run.py",
        "rubric": Path(__file__).with_name("rubric.json"),
    }
    pins = {k: pin(v.resolve()) for k, v in paths.items()}
    atoms = tuple(
        AtomState(**{k: v for k, v in a.items() if k not in {"index", "train_atoms"}})
        for a in read(vocab)["atom_vocabulary"]
    )
    layouts = torch.load(args.layout_payload, weights_only=False, map_location="cpu")["layouts"]
    selected = read(args.selected)["attempts"]
    records, source_pins = [], {}
    for row, nodes, edges, state, basis, source in recover_graphs(selected, read(args.ledger)):
        layout = layouts[row["index"]]
        if (
            layout.family != row["family"]
            or graph_smiles(nodes, edges, atoms) != row["selected_smiles"]
        ):
            raise ValueError("Recovered graph does not reproduce selected identity")
        if source is not None:
            source_pins[source["path"]] = source
        if state is None:
            state = serial_state(nodes, edges, layout)
            if state is None:
                basis = "no_ordered_tree_reconstruction"
        mol = graph_to_molecule(nodes, edges, atoms)
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        if smiles != row["selected_smiles"]:
            raise ValueError("Canonical atom mapping changed selected identity")
        order = json.loads(mol.GetProp("_smilesAtomOutputOrder"))
        audited = audit_layout_rings(edges, layout, tree_state=state, tree_basis=basis)
        audited["canonical_atom_to_generated_node"] = order
        records.append(
            {
                "index": row["index"],
                "family": row["family"],
                "selected_kind": row["selected_kind"],
                "canonical_smiles": smiles,
                "exact_l1": row["selected_check"]["exact"],
                "audit": audited,
            }
        )
    if len(records) != len(selected) or len({r["index"] for r in records}) != len(selected):
        raise ValueError("Every selected request must retain its graph audit")
    groups = defaultdict(list)
    for row in records:
        groups[row["family"]].append(row)
    summary = {}
    for family, rows in sorted(groups.items()):
        summary[family] = {
            "requests": len(rows),
            "cycle_allocation_mismatch": sum(r["audit"]["cycle_allocation_mismatch"] for r in rows),
            "fundamental_size_mismatch": sum(
                r["audit"]["fundamental_size_mismatch"] is True for r in rows
            ),
            "fundamental_size_unassessed": sum(
                r["audit"]["fundamental_size_mismatch"] is None for r in rows
            ),
            "unexpected_cross_origin_edges": sum(
                bool(r["audit"]["unexpected_cross_origin_edges"]) for r in rows
            ),
            "tree_basis": dict(Counter(r["audit"]["tree_basis"] for r in rows)),
        }
    write(
        args.output,
        {
            "schema_version": "forge.compose_lipid_selected_layout_ring_audit.v1",
            "inputs": pins,
            "full_graph_shards": source_pins,
            "by_family": summary,
            "attempts": sorted(records, key=lambda r: r["index"]),
            "runtime_seconds": time.monotonic() - started,
            "limitations": [
                "Graph conditioning mismatch is separate from chemical impossibility.",
                "Fundamental ring sizes depend on the explicitly reported tree basis; no SSSR substitution.",
                "Positive TRAIN controls have no paired sampled layout; this condition axis is unassessed for controls.",
            ],
        },
    )
    print(json.dumps({"requests": len(records), "runtime_seconds": time.monotonic() - started}))


if __name__ == "__main__":
    main()
