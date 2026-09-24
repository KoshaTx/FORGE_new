"""Require the routing increment to preserve every unrelated B5 replay verbatim."""

import json
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    paths = {
        "previous": ROOT / "results/phase1/compose_lipid_supplied_b5_v1/result.json",
        "current": ROOT / "results/phase1/compose_lipid_supplied_b5_v2/result.json",
    }
    reports = {k: json.loads(p.read_text()) for k, p in paths.items()}
    records = {
        k: {r["target_id"]: r for r in rows(resolve_pin(v["artifact"], ROOT, label=k))}
        for k, v in reports.items()
    }
    if records["previous"].keys() != records["current"].keys():
        raise ValueError("B5 routing changed the eligible population")
    changed, same, old_exact, new_exact = [], 0, 0, 0
    for target, current in sorted(records["current"].items()):
        previous = records["previous"][target]
        old_exact += previous["result"]["computed_consistency_pass"]
        new_exact += current["result"]["computed_consistency_pass"]
        if current == previous:
            same += 1
            continue
        if (
            {k: v for k, v in current.items() if k != "result"}
            != {k: v for k, v in previous.items() if k != "result"}
            or previous["result"]["computed_consistency_pass"] is not False
            or current["result"]["computed_consistency_pass"] is not True
            or current["result"]["binding"]["source_pair_routing"]["baseline_binding"]
            != previous["result"]["binding"]
        ):
            raise ValueError("B5 routing changed unrelated evidence or source identity")
        changed.append(
            {
                "target_id": target,
                "original_task_reference": current["original_task_reference"],
                "component_instances": current["component_instances"],
                "program_id": current["result"]["binding"]["program_id"],
                "route_id": current["result"]["binding"]["source_pair_routing"]["route_id"],
            }
        )
    dump(
        HERE / "replay-comparison.json",
        {
            "schema_version": "forge.b5_source_pair_increment_audit.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {k: pin(ROOT, p) for k, p in paths.items()},
            "rows": len(records["current"]),
            "unchanged_replay_rows": same,
            "previous_exact": old_exact,
            "current_exact": new_exact,
            "new_exact": len(changed),
            "changed_rows": changed,
            "all_previous_evidence_preserved": new_exact == old_exact + len(changed),
            "training_admitted": False,
        },
    )
    print(json.dumps({"unchanged": same, "new_exact": len(changed)}))


if __name__ == "__main__":
    main()
