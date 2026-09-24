"""Preserve source-stage evidence and unresolved complete-component scope counterexamples."""

import json
from collections import Counter
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    inputs = {
        k: ROOT / "results/phase1" / v
        for k, v in {
            "previous_scope_audit": "compose_lipid_training_readiness_v2/source-scope-audit.json",
            "han_db": "compose_lipid_supplied_han_db_v1/result.json",
            "b5_source_controls": "compose_lipid_b5_source_v1/control-transcriptions.json",
            "b5_draft_replay": "compose_lipid_b5_source_v1/draft-control-replay.json",
            "b5_adversarial": "compose_lipid_b5_source_v1/draft-adversarial-audit.json",
            "b5_bindings": "compose_lipid_b5_source_v1/eligible-bindings.json",
        }.items()
    }
    docs = {k: json.loads(p.read_text()) for k, p in inputs.items()}
    h = docs["han_db"]
    counts = Counter()
    widths = Counter()
    for row in rows(resolve_pin(h["artifact"], ROOT, label="Han ledger")):
        r = row["replay"]
        counts["rows"] += 1
        counts["exact"] += r["computed_consistency_pass"]
        counts["with_converging_intermediate_branches"] += any(
            any(len(layer) > 1 for layer in s["layers"][1:-1]) for s in r["event_searches"]
        )
        widths[str([list(map(len, s["layers"])) for s in r["event_searches"]])] += 1
        if r["declared_quantities"] != {"amine_head": 1, "epoxide_tail": 2, "acyl_tail": 2}:
            raise ValueError("Han source quantities changed")
    series = Counter()
    for group in docs["b5_bindings"]["groups"]:
        series[json.loads(group["profile"])["metadata"]["series"]] += group["rows"]
    draft = docs["b5_draft_replay"]["controls"]
    adverse = docs["b5_adversarial"]
    dump(
        HERE / "source-scope-audit.json",
        {
            "schema_version": "forge.compose_lipid_source_scope_audit.v2",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {k: pin(ROOT, v) for k, v in inputs.items()},
            "han_db": {
                **dict(counts),
                "event_layer_widths": dict(widths),
                "scope": "All source-stage endpoints unique; all within-stage branches retained. Two acyl chloride copies; net two HCl. No isolated mono-adduct or generated-product experimental selectivity claim.",
            },
            "vitamin_b5": {
                "eligible_by_series": dict(series),
                "source_controls": len(draft),
                "exact_source_control_replays": sum(
                    c["computed_consistency_pass"] and c["independent_stage_endpoints_match"]
                    for c in draft.values()
                ),
                "mechanical_invariant_checks": len(adverse["mechanical_invariant_checks"]),
                "unresolved_scope_counterexamples": list(
                    adverse["remaining_complete_terminal_scope_counterexamples"]
                ),
                "corpus_replay_qualified": False,
                "scope": "Draft net programs reproduce independent source structures and isolated endpoints. Source-site selectivity follows source drawings, not target filtering. Complete precursor grammar and metadata binding are still required.",
            },
            "training_admitted": False,
            "experimental_execution_admitted": False,
        },
    )


if __name__ == "__main__":
    main()
