"""Audit selected completions and report all three frozen saved-run comparisons."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rdkit import rdBase

from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.precursor_occurrence_reuse import OccurrenceReusePlan, copy_occurrence_graph
from forge.model.precursor_reuse_projection import fixed_graph_preserved, graph_smiles, state_graph
from forge.model.program_connection_reuse import (
    ProgramConnectionPlan,
    connect_generated_occurrences,
    occurrence_labels,
)
from forge.model.synthesis_program_sampling import load_synthesis_program_checkpoint

ROOT = Path(__file__).resolve().parents[3]


def main():
    inputs = {}

    def pin(path):
        inputs[str(path.relative_to(ROOT))] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": str(sha256_file(path)),
        }

    def read(path):
        pin(path)
        text = path.read_text()
        return (
            [json.loads(line) for line in text.splitlines()]
            if path.suffix == ".jsonl"
            else json.loads(text)
        )

    pin(Path(__file__))
    report = {}
    for name in ("discovery", "replication_1", "replication_2"):
        base = ROOT / f"results/phase1/combinatorial_connection_reuse_{name}_v1"
        result = read(base / "result.json")
        previous = read(resolve_pin(result["baseline_result"], ROOT, label="connection evidence"))
        admitted = read(resolve_pin(previous["baseline_result"], ROOT, label="connection evidence"))
        raw = read(resolve_pin(admitted["parent_result"], ROOT, label="connection evidence"))
        for source in result["sources"]:
            pin(resolve_pin(source, ROOT, label="connection evidence"))
        for label in ("cache", "checkpoint"):
            pin(resolve_pin(raw["inputs"][label], ROOT, label="connection evidence"))
        _, _, atoms, *_ = load_synthesis_program_checkpoint(
            ROOT / raw["inputs"]["checkpoint"]["path"], device="cpu"
        )

        def artifact(r, filename):
            return read(resolve_pin(r["artifacts"][filename], ROOT, label="connection evidence"))

        layouts = artifact(raw, "layouts.json")
        terminals = artifact(raw, "terminals.jsonl")
        rows = artifact(result, "attempts.jsonl")
        originals = [r for r in rows if r["arm"] == "baseline"]
        completed = [r for r in rows if r["arm"] == "graph_reuse"]
        plans = artifact(result, "plans.json")
        decisions = artifact(result, "admissions.jsonl")
        proposals = artifact(result, "proposals.jsonl")
        selected, counts, qualification = [], Counter(), Counter()
        with SynthesisProgramProductionCache(ROOT / raw["inputs"]["cache"]["path"]) as cache:
            for layout, terminal, original, final, p, decision, proposal in zip(
                layouts, terminals, originals, completed, plans, decisions, proposals, strict=True
            ):
                qualification[layout["family"], p["status"]] += 1
                assert original["sample_index"] == final["sample_index"]
                assert original["layout_record_id"] == final["layout_record_id"]
                assert original["valid_connected"] == final["valid_connected"]
                assert original["requested_depth"] == final["requested_depth"]
                if decision["selected_donor"] is None:
                    assert final["canonical_smiles"] == original["canonical_smiles"]
                    continue
                assert original["valid_connected"] and final["valid_connected"]
                assert cache.fold(layout["index"]) == "train"
                record = cache.record(layout["index"])
                occurrence = p["occurrence_plan"]
                occurrence["occurrences"] = tuple(tuple(row) for row in occurrence["occurrences"])
                plan = ProgramConnectionPlan(
                    p["record_id"],
                    p["status"],
                    OccurrenceReusePlan(**occurrence),
                    tuple(tuple(edge) for edge in p["connections"]),
                    p["reason"],
                )
                donor = decision["selected_donor"]
                nodes, edges = state_graph(terminal["state"])
                output_nodes, output_edges = connect_generated_occurrences(
                    nodes, edges, plan, donor
                )
                labels = occurrence_labels(len(nodes), plan.occurrence_plan)
                accumulator = np.flatnonzero(labels == 0)
                source = list(plan.occurrence_plan.occurrences[donor])
                assert len(nodes) == len(output_nodes)
                assert np.count_nonzero(edges) == np.count_nonzero(output_edges)
                assert fixed_graph_preserved(output_nodes, output_edges, record)
                assert np.array_equal(output_nodes[accumulator], nodes[accumulator])
                assert np.array_equal(
                    output_edges[np.ix_(accumulator, accumulator)],
                    edges[np.ix_(accumulator, accumulator)],
                )
                for occurrence in plan.occurrence_plan.occurrences:
                    target = list(occurrence)
                    assert np.array_equal(output_nodes[target], nodes[source])
                    assert np.array_equal(
                        output_edges[np.ix_(target, target)], edges[np.ix_(source, source)]
                    )
                with rdBase.BlockLogs():
                    assert (
                        graph_smiles(output_nodes, output_edges, atoms) == final["canonical_smiles"]
                    )
                assert final["assembly"]["status"] == "exact_computed_program"
                assert original["assembly"]["status"] != "exact_computed_program"
                copy_nodes, copy_edges = copy_occurrence_graph(
                    nodes, edges, plan.occurrence_plan, donor
                )
                cross_edits = int(np.count_nonzero(np.triu(output_edges != copy_edges, k=1)))
                edge_delta = int((np.count_nonzero(copy_edges) - np.count_nonzero(edges)) // 2)
                counts["accepted"] += 1
                counts["accepted_with_connection_change"] += int(cross_edits > 0)
                counts["same_donor_copy_only_violates_edge_count"] += int(edge_delta != 0)
                counts["novel_product_vs_all_train"] += int(final["novel_vs_train"])
                counts["every_witness_has_novel_component"] += int(
                    final["all_witnesses_have_novel_component"]
                )
                selected.append(
                    {
                        "sample_index": original["sample_index"],
                        "family": original["program_id"],
                        "depth": original["requested_depth"],
                        "donor": donor,
                        "connection_edge_edits": cross_edits,
                        "copy_only_edge_count_difference": edge_delta,
                        "canonical_smiles": final["canonical_smiles"],
                        "proposal": next(r for r in proposal["proposals"] if r["donor"] == donor),
                    }
                )
        assert counts["accepted"] == result["total_exact_gain"]
        report[name] = {
            "counts": dict(counts),
            "all_selected_atom_count_edge_count_fixed_graph_and_generated_interior_checks_pass": True,
            "all_original_invalid_and_exact_outputs_preserved": True,
            "preservation_passes_for_every_family": result["all_family_preservation_screen_passed"],
            "qualification_by_family": [
                {"family": f, "status": s, "count": n}
                for (f, s), n in sorted(qualification.items())
            ],
            "totals": {
                arm: {
                    metric: sum(row[metric] for row in families.values())
                    for metric in (
                        "attempts",
                        "valid_connected",
                        "exact_program",
                        "unique_valid_products",
                        "novel_valid_attempts_vs_all_cache_train",
                        "unique_novel_exact_products",
                    )
                }
                for arm, families in result["per_arm"].items()
            },
            "exact_gain_by_family": {
                f: d["exact_program"] for f, d in result["per_family_deltas"].items()
            },
            "urea_by_depth": [
                {k: row[k] for k in ("arm", "depth", "attempts", "exact_program")}
                for row in result["by_depth"]
                if row["family"] == "urea_amine_isocyanate"
            ],
            "selected": selected,
        }
    payload = {
        "schema_version": "forge.combinatorial_connection_reuse_evidence.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "verified_selected_invariants",
        "inputs": inputs,
        "runs": report,
        "new_random_sampling": False,
        "interpretation": [
            "The same-donor comparison is a diagnostic of admitted outputs, not a new randomized causal experiment.",
            "The additional connection context is from exact source programs. Generated interiors and accumulator graphs are preserved as specified; new-layout autonomy is unmeasured.",
            "Unique product totals sum per-family unique counts, rather than deduplicating across families.",
        ],
    }
    for value in inputs.values():
        resolve_pin(value, ROOT, label="connection evidence")
    path = Path(__file__).with_name("evidence.json")
    if path.exists():
        raise ValueError("evidence output already exists")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: r["counts"] for name, r in report.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
