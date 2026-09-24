"""Assemble the preparation closeout from authenticated completed receipts."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def pin(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def read(name):
    return json.loads((OUT / name).read_text())


def main():
    reconstruction = read("reconstruction_report.json")
    validation = read("validation_report.json")
    reproduction = read("reproduction.json")
    for receipt in (reconstruction, validation, reproduction):
        for value in receipt.get("inputs", {}).values():
            path = ROOT / value["path"]
            if pin(path) != value:
                raise ValueError(f"Receipt input changed: {path}")
        implementation = receipt["implementation"]
        if pin(ROOT / implementation["path"]) != implementation:
            raise ValueError("Receipt implementation changed")
    if not validation["source_snapshot_unchanged"]:
        raise ValueError("Validation source snapshot changed")
    if validation["new_failures"] or validation["new_errors"]:
        raise ValueError("New repository failures or errors require investigation")
    if validation["focused"]["failures"] or validation["focused"]["errors"]:
        raise ValueError("Focused preparation tests failed")
    if validation["vendor_verify_exit_code"] != 0 or not reproduction["byte_identical"]:
        raise ValueError("Vendor verification or reproduction failed")
    for name, expected in reproduction["after_sha256"].items():
        if pin(OUT / name)["sha256"] != expected:
            raise ValueError(f"Reproduced artifact changed: {name}")
    result = {
        "schema_version": "forge.compose_lipid_full_preparation_closeout.v1",
        "status": "preparation_delivered_training_unqualified",
        "seed": 0,
        "implementation": pin(Path(__file__)),
        "inputs": {
            name: pin(OUT / name)
            for name in (
                "reconstruction_report.json",
                "validation_report.json",
                "reproduction.json",
            )
        },
        "totals": reconstruction["totals"],
        "scientific_checks": reconstruction["checks"],
        "focused_tests": validation["focused"],
        "full_tests": validation["full"],
        "new_test_failures_or_errors": 0,
        "vendor_verification_passed": True,
        "byte_identical_report_and_ledger_reproduction": True,
        "remaining_gates": reconstruction["remaining_gates"],
        "phase1_definition_of_done_met": validation["phase1_definition_of_done_met"],
        "training_admitted": False,
        "training_calls": 0,
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
