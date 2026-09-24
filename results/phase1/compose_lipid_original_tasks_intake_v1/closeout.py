"""Collect the authenticated transfer, metadata and completed validation receipts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from verify_restore import CACHE, OUT, ROOT, pin


def read_receipt(name):
    result = json.loads((OUT / name).read_text())
    for value in result.get("inputs", {}).values():
        if pin(ROOT / value["path"]) != value:
            raise ValueError(f"Changed receipt input in {name}: {value['path']}")
    implementation = result["implementation"]
    if pin(ROOT / implementation["path"]) != implementation:
        raise ValueError(f"Changed receipt implementation: {name}")
    return result


def main():
    package = read_receipt("package_check.json")
    audit = read_receipt("record_audit.json")
    validation = read_receipt("validation_report.json")
    if package["remaining_missing_manifest_files"]:
        raise ValueError("Source package remains incomplete")
    manifest = json.loads((CACHE / "MANIFEST.json").read_text())
    for item in manifest["files"]:
        if pin(CACHE / item["path"])["sha256"] != item["sha256"]:
            raise ValueError(f"Verified package file changed: {item['path']}")
    for item in package["restored_files"]:
        if pin(ROOT / item["path"])["sha256"] != item["sha256"]:
            raise ValueError(f"Restored task changed: {item['path']}")
    for field in ("new_bundle", "completed_previous_package"):
        check = package[field]
        if check["exit_code"] != 0 or pin(ROOT / check["log"]["path"]) != check["log"]:
            raise ValueError(f"Checksum command failed or log changed: {field}")
    if not validation["source_snapshot_unchanged"]:
        raise ValueError("Validation source snapshot changed")
    if validation["new_failures"] or validation["new_errors"]:
        raise ValueError("New test failures or errors require investigation")
    if validation["focused"]["failures"] or validation["focused"]["errors"]:
        raise ValueError("Focused validation failed")
    if validation["vendor_verify_exit_code"] != 0:
        raise ValueError("Vendor verification failed")
    result = {
        "schema_version": "forge.compose_lipid_original_tasks_intake.v1",
        "status": "source_transfer_complete_training_unqualified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": 0,
        "implementation": pin(Path(__file__)),
        "inputs": {
            name: pin(OUT / name)
            for name in ("package_check.json", "record_audit.json", "validation_report.json")
        },
        "previously_missing_files_resolved": package["previously_missing_files_resolved"],
        "new_bundle_verified_files": package["new_bundle"]["verified_files"],
        "completed_package_verified_files": package["completed_previous_package"]["verified_files"],
        "remaining_missing_manifest_files": [],
        "family_rows": audit["family_rows"],
        "ugi4_all_referenced_ids_resolve": audit["ugi4_catalog"]["all_referenced_ids_resolve"],
        "focused_tests": validation["focused"],
        "full_tests": validation["full"],
        "new_test_failures_or_errors": 0,
        "phase1_definition_of_done_met": validation["phase1_definition_of_done_met"],
        "molecular_graphs_parsed": 0,
        "reactions_executed": 0,
        "training_admitted": False,
        "training_calls": 0,
        "next_work": [
            "Use restored source records for their dependent construction/provenance checks.",
            "Preserve current eligibility and qualify unresolved family programs independently.",
            "Retain full-universe partition, representation and balanced-weight gates.",
        ],
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
