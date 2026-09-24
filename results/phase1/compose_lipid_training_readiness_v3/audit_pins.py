"""Verify every directly referenced file pin in this checkpoint's new JSON receipts."""

import json
from pathlib import Path

from forge.core.hashing import sha256_file
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    documents = []
    for name in [
        "compose_lipid_han_db_source_v1",
        "compose_lipid_supplied_han_db_v1",
        "compose_lipid_readiness_han_db_v1",
        "compose_lipid_b5_source_v1",
        "compose_lipid_training_goal_validation_v3",
        "compose_lipid_training_readiness_v3",
    ]:
        documents.extend(
            p for p in (ROOT / "results/phase1" / name).glob("*.json") if p.name != "pin-audit.json"
        )
    checked = {}

    def visit(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value and isinstance(value["path"], str):
                key = (value["path"], value["sha256"])
                if key not in checked:
                    checked[key] = str(sha256_file(ROOT / value["path"])) == value["sha256"]
            for v in value.values():
                visit(v)
        elif isinstance(value, list):
            for v in value:
                visit(v)

    for p in documents:
        visit(json.loads(p.read_text()))
    dump(
        HERE / "pin-audit.json",
        {
            "schema_version": "forge.checkpoint_pin_audit.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {str(p.relative_to(ROOT)): pin(ROOT, p) for p in sorted(documents)},
            "pins_checked": len(checked),
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
