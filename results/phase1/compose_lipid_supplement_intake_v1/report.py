"""Consolidate intake and validation evidence without promoting training readiness."""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import xml.etree.ElementTree as ET
from pathlib import Path

from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def pin(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 << 20), b""):
            h.update(data)
    return {"path": str(path.relative_to(ROOT)), "sha256": h.hexdigest()}


def junit(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        cases = list(ET.parse(stream).iter("testcase"))
    issues = {"failures": [], "errors": []}
    skipped = 0
    for case in cases:
        identifier = case.get("classname", "") + "::" + case.get("name", "")
        if case.find("failure") is not None:
            issues["failures"].append(identifier)
        elif case.find("error") is not None:
            issues["errors"].append(identifier)
        elif case.find("skipped") is not None:
            skipped += 1
    return {
        "total": len(cases),
        "passed": len(cases) - skipped - sum(map(len, issues.values())),
        "skipped_or_xfailed": skipped,
        **{key: len(value) for key, value in issues.items()},
    }, {key: sorted(value) for key, value in issues.items()}


def main():
    names = [
        "result.json",
        "package_check.json",
        "checks.json",
        "intake_checks.json",
        "source-snapshot.json",
        "snapshot-comparison.json",
        "prior_precursor_concordance.json",
        "precursor_split_check.json",
        "missing_source_files.json",
        "shasum.log",
    ]
    inputs = {name: pin(OUT / name) for name in names}
    intake = json.loads((OUT / "result.json").read_text())
    package = json.loads((OUT / "package_check.json").read_text())
    concordance = json.loads((OUT / "prior_precursor_concordance.json").read_text())
    protection_path = OUT / "component_exclusions.json"
    protection = json.loads(protection_path.read_text())
    inputs["component_exclusions"] = pin(protection_path)
    checks = json.loads((OUT / "checks.json").read_text())
    snapshot = json.loads((OUT / "snapshot-comparison.json").read_text())
    full, issues = junit(OUT / "full-tests.xml")
    focused, _ = junit(OUT / "focused-tests.xml")
    old = ROOT / "results/phase1/compose_lipid_v8_universe_validation_v1/full-tests.xml.gz"
    _, old_issues = junit(old)
    inputs["previous_full_tests"] = pin(old)
    inputs["full_tests"] = pin(OUT / "full-tests.xml")
    inputs["focused_tests"] = pin(OUT / "focused-tests.xml")
    result = {
        "schema_version": "forge.compose_lipid_supplement_intake_report.v1",
        "status": (
            "core_data_verified_supporting_package_incomplete"
            if package["missing_paths"]
            else "core_intake_verified_training_unqualified"
        ),
        "inputs": inputs,
        "implementation": pin(Path(__file__)),
        "environment": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "source_package": {
            key: package[key]
            for key in (
                "status",
                "exit_code",
                "manifest_files",
                "verified_files",
                "missing_paths",
                "mismatched_files",
            )
        },
        "precursors": intake["precursors"],
        "corrected_split": intake["corrected_split"],
        "constructions": intake["constructions"],
        "prior_exact_replay_precursor_agreement": concordance["totals"],
        "known_component_exclusions": protection["totals"],
        "validation": {
            "focused": focused,
            "full": full,
            "prior_failure_and_error_ids_unchanged": issues == old_issues,
            "new_failures": sorted(set(issues["failures"]) - set(old_issues["failures"])),
            "new_errors": sorted(set(issues["errors"]) - set(old_issues["errors"])),
            "source_snapshot_unchanged": all(snapshot.values()),
            "vendor_verify_exit_code": checks["vendor-verify"]["exit_code"],
        },
        "training_admitted": False,
        "training_calls": 0,
        "seed": 0,
        "remaining_gates": [
            "complete source-package checksum verification",
            "global precursor/product/study/combination protection with prior frozen exclusions",
            "all-family reaction-program qualification",
            "full-size molecular representation qualification",
            "canonical deduplication and balanced verified training measure",
            "repository-wide test failures and setup errors",
        ],
    }
    (OUT / "summary_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name in ("full-tests.log", "full-tests.xml"):
        path = OUT / name
        with (
            path.open("rb") as source,
            (OUT / (name + ".gz")).open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as destination,
        ):
            while data := source.read(1 << 20):
                destination.write(data)
    print(json.dumps(result["validation"], indent=2))


if __name__ == "__main__":
    main()
