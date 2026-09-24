"""Summarize preserved test reports and authenticate the bounded reuse experiment."""

import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from forge.core.hashing import sha256_file

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def pin(path):
    return {"path": str(path.relative_to(REPO)), "sha256": str(sha256_file(path))}


def report(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failures = []
    for case in cases:
        for kind in ("failure", "error"):
            if case.find(kind) is not None:
                failures.append(
                    {
                        "test": f"{case.get('classname')}::{case.get('name')}",
                        "kind": kind,
                        "message": case.find(kind).get("message"),
                    }
                )
    counts = {
        "tests": len(cases),
        "failures": sum(row["kind"] == "failure" for row in failures),
        "errors": sum(row["kind"] == "error" for row in failures),
        "skipped": sum(case.find("skipped") is not None for case in cases),
    }
    counts["passed"] = counts["tests"] - sum(counts[k] for k in ("failures", "errors", "skipped"))
    new_cases = [case for case in cases if case.get("classname") == "tests.test_precursor_reuse"]
    return {
        "xml": pin(path),
        "counts": counts,
        "failures": failures,
        "new_test_cases": len(new_cases),
        "new_test_skips": sum(case.find("skipped") is not None for case in new_cases),
    }


def main():
    reports = {
        "focused": report(OUT / "focused-tests.xml"),
        "full": report(OUT / "full-tests.xml"),
        "baseline": report(
            REPO / "results/phase1/combinatorial_repeat_audit_validation_v1/full-tests.xml"
        ),
    }
    ids = {k: {r["test"] for r in v["failures"]} for k, v in reports.items()}
    replay = json.loads((OUT / "replay-verification.json").read_text())
    checks = {
        "all_30_vendor_assets_verified": "all 30 present vendored assets verified"
        in (OUT / "verify.log").read_text(),
        "focused_pass": not reports["focused"]["failures"]
        and reports["focused"]["counts"]["tests"] == 104,
        "new_tests_pass_in_full": reports["full"]["new_test_cases"] == 18
        and reports["full"]["new_test_skips"] == 0
        and not any("test_precursor_reuse" in name for name in ids["full"]),
        "all_attempts_and_reuse_relations_recomputed": replay["status"] == "verified"
        and replay["attempts_recomputed"] == 3072
        and replay["reuse_plans_recomputed"] == 2698,
        "scientific_payloads_checkpoints_and_ledgers_replay_identical": replay["replay_identical"],
        "black_pass": "left unchanged" in (OUT / "black.log").read_text(),
        "ruff_pass": "All checks passed!" in (OUT / "ruff.log").read_text(),
        "whitespace_pass": all(
            subprocess.run(command, cwd=REPO, capture_output=True).returncode == 0
            for command in (["git", "diff", "--check"], ["git", "diff", "--cached", "--check"])
        ),
    }
    inputs = [p for p in OUT.iterdir() if p.name != "result.json"]
    inputs += [
        REPO / name
        for name in (
            "configs/multireaction/combinatorial_reuse_pilot_v1.json",
            "results/phase1/combinatorial_reuse_pilot_v1/result.json",
            "results/phase1/combinatorial_reuse_pilot_replay_v1/result.json",
            "forge/model/precursor_reuse.py",
            "experiments/phase1/multireaction/combinatorial_reuse_pilot.py",
            "experiments/phase1/multireaction/combinatorial_reuse_verify.py",
            "tests/test_precursor_reuse.py",
            "Makefile",
            "pyproject.toml",
            "uv.lock",
        )
    ]
    result = {
        "schema_version": "forge.combinatorial_reuse_validation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "scoped_checks_pass_global_suite_failed"
            if all(checks.values()) and ids["full"]
            else "inspect_checks"
        ),
        "phase1_definition_of_done_met": all(checks.values()) and not ids["full"],
        "checks": checks,
        "inputs": [pin(p) for p in sorted(inputs)],
        "test_reports": reports,
        "failure_comparison": {
            "same_failure_and_error_ids": ids["full"] == ids["baseline"],
            "new": sorted(ids["full"] - ids["baseline"]),
            "resolved": sorted(ids["baseline"] - ids["full"]),
        },
        "commands": {
            "vendor": "UV_CACHE_DIR=/private/tmp/forge-uv-cache make verify",
            "full": "UV_CACHE_DIR=/private/tmp/forge-uv-cache PYTEST_ADDOPTS='--tb=short --junitxml=results/phase1/combinatorial_reuse_validation_v1/full-tests.xml' make test",
            "focused": "uv run pytest -q tests/test_precursor_reuse.py tests/test_combinatorial_repeat_audit.py tests/test_combinatorial_generation.py tests/test_library_programs.py tests/test_assembly_families.py tests/test_pinned_sources_are_tracked.py",
            "read_only_verifier": "uv run python -m experiments.phase1.multireaction.combinatorial_reuse_verify --result results/phase1/combinatorial_reuse_pilot_v1/result.json --replay results/phase1/combinatorial_reuse_pilot_replay_v1/result.json",
        },
        "nonclaims": [
            "The model intervention failed its scientific preservation rule.",
            "No global test pass or model promotion is claimed.",
            "Matching failure IDs does not fully diagnose all legacy failures.",
        ],
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "checks": checks,
                "failure_comparison": result["failure_comparison"],
                "counts": {k: v["counts"] for k, v in reports.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
