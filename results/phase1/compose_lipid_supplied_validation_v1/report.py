"""Audit supplied replay coverage and retain the completed test comparison."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from forge.core.hashing import resolve_pin  # noqa: E402
from forge.corpus.compose_lipid_source_view import dump, pin  # noqa: E402
from forge.corpus.compose_lipid_supplement import rows  # noqa: E402


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
    source_view = ROOT / "results/phase1/compose_lipid_source_view_v1/result.json"
    replay_path = ROOT / "results/phase1/compose_lipid_supplied_michael_v1/result.json"
    audit_path = ROOT / "results/phase1/compose_lipid_v8_precursor_audit_v2/result.json"
    concordance_path = (
        ROOT / "results/phase1/compose_lipid_supplement_intake_v1/prior_precursor_concordance.json"
    )
    view, replay, audit = [
        json.loads(path.read_text()) for path in (source_view, replay_path, audit_path)
    ]
    config = json.loads(resolve_pin(replay["config"], ROOT, label="replay config").read_text())
    for receipt in (view, replay):
        for name, value in receipt["implementation"].items():
            resolve_pin(value, ROOT, label=name)
    ledger = resolve_pin(view["artifact"], ROOT, label="preparation")
    current = {
        row["target_id"]: row for row in rows(ledger) if row["eligible_for_program_preparation"]
    }
    by_family = defaultdict(Counter)
    for row in current.values():
        by_family[row["family"]]["protected_preparation_rows"] += 1
    prior_ledger = resolve_pin(
        audit["artifacts"]["rows.jsonl.gz"], ROOT, label="prior precursor audit"
    )
    previous = set()
    for row in rows(prior_ledger):
        if row["target_id"] in current and row["exact_program_evidence"]:
            if row["constitution_id"] != current[row["target_id"]]["constitution_id"]:
                raise ValueError("Prior exact evidence has a different target identity")
            previous.add(row["target_id"])
            by_family[row["family"]]["prior_exact_program_rows"] += 1
    seen, exact = set(), set()
    failures = Counter()
    proof = resolve_pin(replay["artifact"], ROOT, label="supplied proof ledger")
    for row in rows(proof):
        target = row["target_id"]
        if target not in current or target in seen:
            raise ValueError("Replay leaked a protected/unassigned target or duplicated a target")
        seen.add(target)
        preparation = current[target]
        if any(
            row[name] != preparation[name]
            for name in ("family", "constitution_id", "component_instances", "construction_basis")
        ):
            raise ValueError("Replay changed source component identities, roles or quantities")
        if row["training_admitted"] or row["experimental_execution_admitted"]:
            raise ValueError("Replay promoted an unsupported evidence claim")
        value = row["replay"]
        by_family[row["family"]]["supplied_replay_rows"] += 1
        if value["computed_consistency_pass"]:
            if not all(value["checks"].values()) or len(value["forward_layers"][-1]) != 1:
                raise ValueError("Incomplete or ambiguous reconstruction promoted to exact")
            product = value["forward_layers"][-1][0]
            if hashlib.sha256(product.encode()).hexdigest() != row["constitution_id"]:
                raise ValueError("Forward replay differs from authenticated target graph identity")
            exact.add(target)
            by_family[row["family"]]["supplied_exact_reconstruction_rows"] += 1
        else:
            failures[value["disposition"]] += 1
            by_family[row["family"]][value["disposition"]] += 1
    expected = {target for target, row in current.items() if row["family"] in config["families"]}
    if seen != expected:
        raise ValueError("Source replay lost eligible targets")
    for target in previous | exact:
        by_family[current[target]["family"]]["exact_program_evidence_from_either_receipt"] += 1
    for family, counts in by_family.items():
        for field in (
            "prior_exact_program_rows",
            "supplied_replay_rows",
            "supplied_exact_reconstruction_rows",
            "exact_program_evidence_from_either_receipt",
        ):
            counts.setdefault(field, 0)
        counts["without_exact_program_evidence"] = (
            counts["protected_preparation_rows"]
            - counts["exact_program_evidence_from_either_receipt"]
        )
    reconstruction = {
        "schema_version": "forge.compose_lipid_supplied_reconstruction_report.v1",
        "implementation": pin(ROOT, Path(__file__)),
        "inputs": {
            "view": pin(ROOT, source_view),
            "replay": pin(ROOT, replay_path),
            "prior_audit": pin(ROOT, audit_path),
            "prior_precursor_set_concordance": pin(ROOT, concordance_path),
        },
        "by_family": {family: dict(counts) for family, counts in sorted(by_family.items())},
        "totals": dict(sum(by_family.values(), Counter())),
        "independent_checks": {
            "target_coverage_exact": True,
            "protected_or_unassigned_replay_targets": 0,
            "replay_duplicates": 0,
            "component_role_or_quantity_changes": 0,
            "forward_product_identity_mismatches": 0,
        },
        "known_limits": [
            "Prior exact programs have precursor-structure-set concordance, not newly established source-role/multiplicity agreement.",
            "These are computed reconstructions, not experimental selectivity or historical-route evidence.",
            "Current preparation is the protected intersection of the initial 200k selection. Full 3,182,837-record partition and representation remain unfinished.",
            "All-family training remains unqualified. Missing supporting task files are required only for checks that depend on them; they are not blanket replay prerequisites.",
        ],
        "seed": 0,
        "training_admitted": False,
        "training_calls": 0,
    }
    dump(OUT / "reconstruction_report.json", reconstruction)
    if not (OUT / "snapshot-comparison.json").exists():
        print("Reconstruction report complete; full validation still running.")
        return
    checks = json.loads((OUT / "checks.json").read_text())
    snapshot = json.loads((OUT / "snapshot-comparison.json").read_text())
    full, issues = junit(OUT / "full-tests.xml")
    focused, _ = junit(OUT / "focused-tests.xml")
    previous_tests = ROOT / "results/phase1/compose_lipid_supplement_intake_v1/full-tests.xml.gz"
    _, previous_issues = junit(previous_tests)
    result = {
        "schema_version": "forge.compose_lipid_supplied_validation.v1",
        "implementation": pin(ROOT, Path(__file__)),
        "inputs": {
            **{
                name: pin(ROOT, OUT / name)
                for name in (
                    "checks.json",
                    "source-snapshot.json",
                    "snapshot-comparison.json",
                    "focused-tests.xml",
                    "full-tests.xml",
                    "reconstruction_report.json",
                )
            },
            "previous_full_tests": pin(ROOT, previous_tests),
        },
        "focused": focused,
        "full": full,
        "new_failures": sorted(set(issues["failures"]) - set(previous_issues["failures"])),
        "new_errors": sorted(set(issues["errors"]) - set(previous_issues["errors"])),
        "prior_failure_and_error_ids_unchanged": issues == previous_issues,
        "source_snapshot_unchanged": all(snapshot.values()),
        "vendor_verify_exit_code": checks["vendor-verify"]["exit_code"],
        "training_admitted": False,
        "seed": 0,
    }
    dump(OUT / "validation_report.json", result)
    for name in ("full-tests.log", "full-tests.xml"):
        path = OUT / name
        with (
            path.open("rb") as source,
            (OUT / (name + ".gz")).open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as destination,
        ):
            shutil.copyfileobj(source, destination)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
