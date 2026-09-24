"""Explain edge-count rejection at attempt level without changing any generation gate."""

import json
from pathlib import Path

import numpy as np

from forge.core.hashing import resolve_pin, sha256_file
from forge.model.precursor_reuse_projection import state_graph

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": str(sha256_file(path))}


def main():
    inputs = [pin(Path(__file__)), pin(ROOT / "forge/model/precursor_reuse_projection.py")]
    populations = {}

    def read_pin(value, lines=False):
        inputs.append(value)
        text = resolve_pin(value, ROOT, label="connection obstruction").read_text()
        return [json.loads(s) for s in text.splitlines()] if lines else json.loads(text)

    for label in ("discovery", "replication_1", "replication_2"):
        path = ROOT / f"results/phase1/combinatorial_occurrence_reuse_{label}_v1/result.json"
        result = read_pin(pin(path))
        baseline = read_pin(result["baseline_result"])
        parent = read_pin(baseline["parent_result"])
        terminals = read_pin(parent["artifacts"]["terminals.jsonl"], True)
        layouts = read_pin(parent["artifacts"]["layouts.json"])
        plans = read_pin(result["artifacts"]["plans.json"])
        proposals = read_pin(result["artifacts"]["proposals.jsonl"], True)
        rows = []
        for index, (terminal, layout, plan, entry) in enumerate(
            zip(terminals, layouts, plans, proposals, strict=True)
        ):
            if not entry["proposals"]:
                continue
            _, edges = state_graph(terminal["state"])
            internal = [
                int(np.count_nonzero(np.triu(edges[np.ix_(o, o)], k=1)))
                for o in plan["occurrences"]
            ]
            deltas = [len(internal) * count - sum(internal) for count in internal]
            if [p["status"] == "changed_cycle_count" for p in entry["proposals"]] != [
                delta != 0 for delta in deltas
            ]:
                raise ValueError(
                    "saved edge-count rejection disagrees with independent count calculation"
                )
            rows.append(
                {
                    "sample_index": index,
                    "family": layout["family"],
                    "depth": layout["depth"],
                    "internal_edge_counts": internal,
                    "edge_count_deltas_by_donor": deltas,
                    "no_existing_donor_can_preserve_edge_count": 0 not in deltas,
                }
            )
        populations[label] = {
            "attempts_with_proposals": len(rows),
            "attempts_blocked_for_every_donor_by_edge_count": sum(
                r["no_existing_donor_can_preserve_edge_count"] for r in rows
            ),
            "rows": rows,
        }
    result = {
        "schema_version": "forge.occurrence_connection_obstruction.v1",
        "inputs": inputs,
        "populations": populations,
        "random_sampling_used": False,
        "invariant": "With m occurrences, internal edge counts k_i and fixed external edges, copying donor d changes the total edge count by m*k_d - sum(k_i).",
        "interpretation": "When no donor delta is zero, whole-occurrence copying with fixed external edges cannot pass the current edge-count gate. This does not justify raising the cycle budget; interiors and connections require a joint proposal.",
    }
    (OUT / "connection_obstruction.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        {
            k: {name: value for name, value in v.items() if name != "rows"}
            for k, v in populations.items()
        }
    )


if __name__ == "__main__":
    main()
