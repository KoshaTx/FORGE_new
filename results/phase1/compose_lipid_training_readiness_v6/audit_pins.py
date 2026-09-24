"""Authenticate every directly referenced local pin in the new partition checkpoint."""

import json
from pathlib import Path

from forge.core.hashing import sha256_file
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    documents = []
    for name in (
        "compose_lipid_split_source_recovery_v2",
        "compose_lipid_full_split_projection_v1",
        "compose_lipid_training_goal_validation_v6",
        "compose_lipid_training_readiness_v6",
    ):
        documents.extend(
            p
            for p in (ROOT / "results/phase1" / name).rglob("*.json")
            if p.name != "pin-audit.json"
            and "upstream" not in p.parts
            and not p.name.endswith("-blob.json")
        )
    documents.extend(
        ROOT / "data/vendor" / name
        for name in (
            "compose_lipid_frozen_split_policy_v1.json",
            "compose_lipid_historical_morphology_policy_v1.json",
        )
    )
    checked = {}

    def visit(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value and isinstance(value["path"], str):
                key = (value["path"], value["sha256"])
                if key not in checked:
                    checked[key] = sha256_file(ROOT / key[0]) == key[1]
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
            "pass": all(checked.values()),
            "mismatches": [
                {"path": p, "expected_sha256": h}
                for (p, h), ok in sorted(checked.items())
                if not ok
            ],
            "training_admitted": False,
        },
    )
    if not all(checked.values()):
        raise ValueError("Checkpoint pins changed")


if __name__ == "__main__":
    main()
