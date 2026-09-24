"""Record the fresh narrow checks and the explicitly reused full-suite evidence."""

import json
from pathlib import Path

from forge.corpus.compose_lipid_source_view import dump, pin
from results.phase1.compose_lipid_training_goal_validation_v7.validate import junit

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    tests, issues = junit(HERE / "maleate-tests.xml")
    reuse = json.loads((HERE / "validation-reuse.json").read_text())
    vendor = (HERE / "vendor-verify.log").read_text()
    if issues["failures"] or issues["errors"] or tests["passed"] == 0:
        raise ValueError("Fresh program tests do not pass")
    if (
        not reuse["reused_validation_qualified"]
        or "all 30 present vendored assets verified" not in vendor
    ):
        raise ValueError("Source snapshot reuse or vendor verification failed")
    dump(
        HERE / "validation-report.json",
        {
            "schema_version": "forge.incremental_preparation_validation.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {
                name: pin(ROOT, HERE / name)
                for name in (
                    "maleate-tests.xml",
                    "maleate-tests.log",
                    "vendor-verify.log",
                    "validation-reuse.json",
                )
            },
            "fresh_tests": tests,
            "fresh_test_scope": "tests/test_compose_lipid_current_replay.py",
            "vendor_verify_observed_exit_code": 0,
            "source_config_test_snapshot_unchanged": True,
            "full_suite_rerun": False,
            "full_suite_gate_pass": False,
            "training_admitted": False,
        },
    )


if __name__ == "__main__":
    main()
