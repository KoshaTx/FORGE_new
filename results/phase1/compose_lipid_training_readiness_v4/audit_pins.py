"""Verify every directly referenced file pin in this checkpoint's new JSON receipts."""

import json
from collections import defaultdict
from pathlib import Path

from forge.core.hashing import sha256_file
from forge.corpus.compose_lipid_original_binding import selected_lines
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    extraction_path = ROOT / "results/phase1/compose_lipid_b5_original_tasks_v1/result.json"
    original_sources = json.loads(extraction_path.read_text())["task_shards"]
    original_references = defaultdict(dict)
    documents = []
    for name in [
        "compose_lipid_b5_source_v2",
        "compose_lipid_b5_original_tasks_v1",
        "compose_lipid_supplied_b5_v1",
        "compose_lipid_readiness_b5_v1",
        "compose_lipid_training_goal_validation_v4",
        "compose_lipid_training_readiness_v4",
    ]:
        documents.extend(
            p for p in (ROOT / "results/phase1" / name).glob("*.json") if p.name != "pin-audit.json"
        )
    checked = {}

    def visit(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value and isinstance(value["path"], str):
                key = (value["path"], value["sha256"])
                if "line" in value and "payload_sha256" in value:
                    local = original_sources.get(value["path"])
                    if local is None or local["sha256"] != value["sha256"]:
                        raise ValueError("Unknown or changed original task locator")
                    number, digest = value["line"], value["payload_sha256"]
                    previous = original_references[value["path"]].get(number)
                    if previous is not None and previous != digest:
                        raise ValueError("Conflicting original task payload hashes")
                    original_references[value["path"]][number] = digest
                    key = (local["path"], local["sha256"])
                if key not in checked:
                    checked[key] = str(sha256_file(ROOT / key[0])) == key[1]
            for v in value.values():
                visit(v)
        elif isinstance(value, list):
            for v in value:
                visit(v)

    for p in documents:
        visit(json.loads(p.read_text()))
    original_payloads_checked = 0
    for source, expected in sorted(original_references.items()):
        local = ROOT / original_sources[source]["path"]
        for number, _, digest in selected_lines(local, set(expected)):
            if digest != expected[number]:
                raise ValueError("Original task payload hash changed")
            original_payloads_checked += 1
    dump(
        HERE / "pin-audit.json",
        {
            "schema_version": "forge.checkpoint_pin_audit.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {str(p.relative_to(ROOT)): pin(ROOT, p) for p in sorted(documents)},
            "pins_checked": len(checked),
            "original_task_payloads_checked": original_payloads_checked,
            "mismatches": [
                {"path": p, "expected_sha256": h}
                for (p, h), ok in sorted(checked.items())
                if not ok
            ],
            "pass": all(checked.values()),
            "training_admitted": False,
        },
    )
    if not all(checked.values()):
        raise ValueError("Checkpoint file pins changed")


if __name__ == "__main__":
    main()
