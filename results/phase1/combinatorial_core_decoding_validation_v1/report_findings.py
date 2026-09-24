"""Authenticate paired core experiments and independently check core/order preservation."""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem

from forge.core.hashing import resolve_pin, sha256_file

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def core_graph_signature(core):
    labels = [(role, c) for role, states in core["semantic_key"][2] for c in states]
    graph = Chem.RWMol()
    for (role, coordinate), state in zip(labels, core["node_states"], strict=True):
        atom = Chem.Atom(6)
        atom.SetIsotope((role * 64 + coordinate) * 32 + state)
        graph.AddAtom(atom)
    for a, b, bond in core["edges"]:
        graph.AddBond(
            a,
            b,
            (
                Chem.BondType.SINGLE,
                Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE,
                Chem.BondType.AROMATIC,
            )[bond],
        )
    return Chem.MolToSmiles(graph, canonical=True)


def main():
    inputs = {}

    def pin(path):
        result = {"path": str(path.relative_to(ROOT)), "sha256": str(sha256_file(path))}
        inputs[result["path"]] = result
        return result

    def load(path):
        pin(path)
        return json.loads(path.read_text())

    def result(name):
        r = load(ROOT / f"results/phase1/{name}/result.json")
        config = load(resolve_pin(r["config"], ROOT, label="core findings config"))
        assert r["sources"] == config["source_pins"] and r["inputs"] == config["inputs"]
        for value in [*r["sources"], *r["inputs"].values(), *r["artifacts"].values()]:
            pin(resolve_pin(value, ROOT, label="core findings input"))
        return r

    runs, layouts = {}, {}
    for label, name in (
        ("initial", "combinatorial_core_decoding"),
        ("ordered", "combinatorial_ordered_core_decoding"),
    ):
        r, replay = result(name + "_v1"), result(name + "_replay_v1")
        ignore = {"created_at_utc", "duration_seconds", "artifacts"}
        assert {k: v for k, v in r.items() if k not in ignore} == {
            k: v for k, v in replay.items() if k not in ignore
        }
        assert {k: p["sha256"] for k, p in r["artifacts"].items()} == {
            k: p["sha256"] for k, p in replay["artifacts"].items()
        }
        layouts[label] = load(ROOT / r["artifacts"]["layouts.json"]["path"])
        rows = [
            json.loads(line)
            for line in (ROOT / r["artifacts"]["attempts.jsonl"]["path"]).read_text().splitlines()
        ]
        assert len(rows) == 1536 and r["neural_attempts"] == 768
        assert all(
            p["identical_trajectory_and_terminal_scores"]
            and p["identity_control_exact"]
            and p["fixed_states_exact"]
            for p in r["pairing"]
        )
        metrics = (
            "attempts",
            "valid_connected",
            "exact_program",
            "unique_valid_products",
            "novel_valid_attempts_vs_all_cache_train",
            "unique_novel_exact_products",
        )
        totals = {
            arm: {m: sum(v[m] for v in values.values()) for m in metrics}
            for arm, values in r["comparison"]["per_arm"].items()
        }
        runs[label] = {
            "totals": totals,
            "per_family": r["comparison"]["per_arm"],
            "qualification_by_family": r["qualification_by_family"],
            "core_support": r["core_support"],
            "constraint_abstentions": {
                arm: dict(
                    Counter(
                        x["constraint_abstention_reason"]
                        for x in rows
                        if x["arm"] == arm and x["constraint_abstention_reason"]
                    )
                )
                for arm in ("baseline", "graph_reuse")
            },
            "paired_preservation_pass": r["comparison"]["all_family_preservation_screen_passed"],
            "existing_pipeline_preservation_pass": r["comparison_vs_existing_pipeline"][
                "all_family_preservation_screen_passed"
            ],
            "replay_scientific_fields_and_artifact_hashes_identical": True,
        }
    preserved = 0
    for a, b in zip(layouts["initial"], layouts["ordered"], strict=True):
        assert all(
            a[k] == b[k]
            for k in (
                "sample_index",
                "program_id",
                "layout_record_id",
                "depth",
                "nodes",
                "closures",
            )
        )
        assert core_graph_signature(a["core"]) == core_graph_signature(b["core"])
        preserved += 1
    pin(Path(__file__))
    payload = {
        "schema_version": "forge.combinatorial_core_decoding_findings.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "runs": runs,
        "count_and_typed_core_graphs_preserved_by_ordering": preserved,
        "promoted": False,
        "goal_complete": False,
        "decision": "Retain supported core/order compilation and both failed preservation screens. Keep the existing completed pipeline.",
        "next_mechanism": "The frozen strict decoder reserves fixed tree edges before variable parents but adds fixed closures after them. Reserve every immutable edge before variable choices, keep the same valence bounds, and test a fixed-ring regression before any new pilot.",
        "limits": [
            "Both arms within each run share one neural trajectory and exactly the same terminal logits. The two runs themselves differ in layout ordering and are not a paired causal comparison.",
            "Ordering retains the identical typed core graph in all 768 requests; the isotope-labelled graphs used to check isomorphism are test encodings, not chemical product proposals.",
            "These are bounded TRAIN-supported diagnostics, not learned-model improvement, heldout evidence, chemical realism, or complete L2/L3 route closure.",
        ],
    }
    for value in inputs.values():
        resolve_pin(value, ROOT, label="final core findings input")
    path = OUT / "findings.json"
    if path.exists():
        raise ValueError("findings already exist")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "counts_and_core_graphs_preserved": preserved,
                "totals": {k: v["totals"] for k, v in runs.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
