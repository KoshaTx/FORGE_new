"""Verify unchanged production inputs before reusing the recorded full test result."""

import json
from pathlib import Path

from forge.corpus.compose_lipid_source_view import dump, pin
from results.phase1.compose_lipid_training_goal_validation_v7.validate import snapshot

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    previous = ROOT / "results/phase1/compose_lipid_training_goal_validation_v7"
    before = json.loads((previous / "source-snapshot.json").read_text())
    current = snapshot()
    changed = sorted(k for k in before.keys() | current.keys() if before.get(k) != current.get(k))
    dump(
        HERE / "validation-reuse.json",
        {
            "schema_version": "forge.validation_snapshot_reuse.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {
                name: pin(ROOT, previous / name)
                for name in ("source-snapshot.json", "validation_report.json", "validate.py")
            },
            "checked_paths": len(current),
            "changed_paths": changed,
            "source_config_test_snapshot_unchanged": not changed,
            "full_suite_rerun": False,
            "prior_full_suite_pass": False,
            "reused_validation_qualified": not changed,
            "training_admitted": False,
        },
    )
    if changed:
        raise ValueError("Production inputs changed; prior test receipt cannot be reused")


if __name__ == "__main__":
    main()
