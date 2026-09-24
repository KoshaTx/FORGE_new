"""Summarize actual failed checks from completed replays without changing their decisions."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    paths = [
        ROOT / "results/phase1/compose_lipid_full_replay_v1/michael/result.json",
        ROOT / "results/phase1/compose_lipid_full_program_replay_v1/programs/result.json",
        ROOT / "results/phase1/compose_lipid_full_iphos_replay_v1/anionic/result.json",
    ]
    counts = defaultdict(Counter)
    failures = defaultdict(Counter)
    bounds = defaultdict(Counter)
    signatures = defaultdict(Counter)
    examples = defaultdict(dict)
    for path in paths:
        result = json.loads(path.read_text())
        if not result["complete"]:
            raise ValueError("Cannot summarize an incomplete replay")
        for shard_pin in result["shards"]:
            shard = json.loads(resolve_pin(shard_pin, ROOT, label="shard").read_text())
            for row in rows(resolve_pin(shard["artifact"], ROOT, label="replay ledger")):
                family, replay = row["family"], row["replay"]
                counts[family][replay["disposition"]] += 1
                if replay["computed_consistency_pass"]:
                    continue
                failed = sorted(k for k, v in replay.get("checks", {}).items() if v is not True)
                failures[family].update(failed)
                bounds[family].update(replay.get("bound_reasons", []))
                signature = json.dumps([replay["disposition"], failed], separators=(",", ":"))
                signatures[family][signature] += 1
                examples[family].setdefault(signature, row["target_id"])
        for family, expected in result["summary"].items():
            if dict(counts[family]) != {k: v for k, v in expected.items() if k != "rows"}:
                raise ValueError("Residual audit does not reproduce replay counts")
    worklist_path = HERE / "all-family-worklist.json"
    worklist = json.loads(worklist_path.read_text())
    for family, values in counts.items():
        pending = sum(v for k, v in values.items() if k != "exact_computed_reconstruction")
        if pending != worklist["families"][family]["pending_eligible_reconstructions"]:
            raise ValueError("Remaining failures disagree with cumulative evidence")
    dump(
        HERE / "residual-check-audit.json",
        {
            "schema_version": "forge.compose_lipid_residual_check_audit.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": [pin(ROOT, p) for p in [*paths, worklist_path]],
            "by_family": {
                family: {
                    "counts": counts[family],
                    "failed_check_counts_nonexclusive": failures[family],
                    "bound_reason_counts_nonexclusive": bounds[family],
                    "failure_signatures": signatures[family],
                    "example_target_per_signature": examples[family],
                }
                for family in sorted(counts)
            },
            "eligible_families_without_qualified_replay": {
                family: row["eligible_preparation_rows"]
                for family, row in worklist["families"].items()
                if row["eligible_preparation_rows"] > 0 and family not in counts
            },
            "training_admitted": False,
            "training_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
