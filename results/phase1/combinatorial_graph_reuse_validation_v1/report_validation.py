"""Summarize preserved graph-reuse test outcomes without concealing baseline failures."""

import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from forge.core.hashing import resolve_pin, sha256_file

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
NEW_TESTS = {
    "tests.test_precursor_reuse_projection",
    "tests.test_precursor_reuse_admission",
    "tests.test_combinatorial_graph_reuse",
}


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": str(sha256_file(path))}


def report(path):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    failures = []
    for case in cases:
        for kind in ("failure", "error"):
            entry = case.find(kind)
            if entry is not None:
                failures.append(
                    {
                        "test": f"{case.get('classname')}::{case.get('name')}",
                        "kind": kind,
                        "message": entry.get("message"),
                    }
                )
    counts = {
        "tests": len(cases),
        "failures": sum(r["kind"] == "failure" for r in failures),
        "errors": sum(r["kind"] == "error" for r in failures),
        "skipped": sum(c.find("skipped") is not None for c in cases),
    }
    counts["passed"] = counts["tests"] - sum(counts[k] for k in ("failures", "errors", "skipped"))
    new = [c for c in cases if c.get("classname") in NEW_TESTS]
    return {
        "xml": pin(path),
        "counts": counts,
        "failures": failures,
        "new_tests": len(new),
        "new_test_skips": sum(c.find("skipped") is not None for c in new),
    }


def main():
    reports = {
        "focused": report(OUT / "focused-tests.xml"),
        "full": report(OUT / "full-tests.xml"),
        "baseline": report(
            ROOT / "results/phase1/combinatorial_reuse_validation_v1/full-tests.xml"
        ),
    }
    findings = json.loads((OUT / "findings.json").read_text())
    for value in findings["inputs"]:
        resolve_pin(value, ROOT, label="verified findings input")
    ids = {key: {r["test"] for r in value["failures"]} for key, value in reports.items()}
    checks = {
        "vendor_assets_verified": "all 30 present vendored assets verified"
        in (OUT / "verify.log").read_text(),
        "focused_pass": reports["focused"]["counts"]["passed"] == 178 and not ids["focused"],
        "new_tests_pass_in_full": reports["full"]["new_tests"] == 26
        and reports["full"]["new_test_skips"] == 0
        and not any(name.split("::")[0] in NEW_TESTS for name in ids["full"]),
        "three_populations_verified": len(findings["populations"]) == 3
        and all(
            p["admission_verification"]["status"] == "verified"
            for p in findings["populations"].values()
        ),
        "all_family_declared_metrics_preserved": all(
            p["all_family_count_preservation"] for p in findings["populations"].values()
        ),
        "positive_exact_gain_in_all_populations": all(
            p["exact_gain"] > 0 for p in findings["populations"].values()
        ),
        "discovery_full_replay_identical": findings["discovery_replay"]["replay_identical"],
        "black_pass": "left unchanged" in (OUT / "black.log").read_text(),
        "ruff_pass": "All checks passed!" in (OUT / "ruff.log").read_text(),
        "whitespace_pass": all(
            subprocess.run(cmd, cwd=ROOT, capture_output=True).returncode == 0
            for cmd in (["git", "diff", "--check"], ["git", "diff", "--cached", "--check"])
        ),
    }
    files = [p for p in OUT.iterdir() if p.is_file() and p.name != "result.json"]
    files.extend(
        ROOT / name
        for name in (
            "forge/model/precursor_reuse_projection.py",
            "forge/model/precursor_reuse_admission.py",
            "experiments/phase1/multireaction/combinatorial_graph_reuse.py",
            "experiments/phase1/multireaction/combinatorial_graph_reuse_verify.py",
            "experiments/phase1/multireaction/combinatorial_reuse_admission.py",
            "tests/test_precursor_reuse_projection.py",
            "tests/test_precursor_reuse_admission.py",
            "tests/test_combinatorial_graph_reuse.py",
            "Makefile",
            "pyproject.toml",
            "uv.lock",
        )
    )
    result = {
        "schema_version": "forge.combinatorial_graph_reuse_validation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "scoped_checks_pass_global_suite_failed"
            if all(checks.values()) and ids["full"]
            else "inspect_checks"
        ),
        "phase1_definition_of_done_met": all(checks.values()) and not ids["full"],
        "checks": checks,
        "inputs": [pin(p) for p in sorted(files)],
        "test_reports": reports,
        "failure_comparison": {
            "same_failure_and_error_ids": ids["full"] == ids["baseline"],
            "new": sorted(ids["full"] - ids["baseline"]),
            "resolved": sorted(ids["baseline"] - ids["full"]),
        },
        "commands": {
            "vendor": "UV_CACHE_DIR=/private/tmp/forge-uv-cache make verify",
            "focused": "uv run pytest -q tests/test_combinatorial*.py tests/test_library*.py tests/test_assembly_families.py tests/test_precursor_reuse*.py tests/test_pinned_sources_are_tracked.py",
            "full": "UV_CACHE_DIR=/private/tmp/forge-uv-cache PYTEST_ADDOPTS='--tb=short --junitxml=results/phase1/combinatorial_graph_reuse_validation_v1/full-tests.xml' make test",
            "findings": "uv run python results/phase1/combinatorial_graph_reuse_validation_v1/report_findings.py",
        },
        "limits": [
            "The broad goal remains active; this is a bounded composite-pipeline result on TRAIN layouts.",
            "No repository-wide test pass, learned-model improvement, complete-IL qualification of all twelve scopes or chemical realism claim is made.",
            "Matching baseline failure IDs does not fully diagnose every failure; no failing check was weakened or skipped for this work.",
        ],
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "checks": checks,
                "failure_comparison": result["failure_comparison"],
                "counts": {k: r["counts"] for k, r in reports.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
