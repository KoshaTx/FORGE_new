"""Preserve engineering checks separately from the failed count-prior experiment."""

import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from forge.core.hashing import resolve_pin, sha256_file

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
NEW_TESTS = {"tests.test_generated_program_layout", "tests.test_combinatorial_layout_prior"}
FILES = (
    "forge/model/generated_program_layout.py",
    "forge/model/generated_program_expansion.py",
    "forge/model/combinatorial_layout_prior.py",
    "experiments/phase1/multireaction/combinatorial_generated_layouts.py",
    "experiments/phase1/multireaction/combinatorial_context_expansion.py",
    "experiments/phase1/multireaction/combinatorial_count_layout_generation.py",
    "tests/test_generated_program_layout.py",
    "tests/test_combinatorial_layout_prior.py",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
)


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
            ROOT / "results/phase1/combinatorial_connection_reuse_validation_v1/full-tests.xml"
        ),
    }
    findings = json.loads((OUT / "findings.json").read_text())
    for value in findings["inputs"].values():
        resolve_pin(value, ROOT, label="generated-layout findings")
    ids = {
        key: {(r["test"], r["kind"]) for r in value["failures"]} for key, value in reports.items()
    }
    checks = {
        "vendor_assets_verified": "all 30 present vendored assets verified"
        in (OUT / "verify.log").read_text(),
        "focused_pass": reports["focused"]["counts"]["passed"] == 72 and not ids["focused"],
        "new_tests_pass_in_full": reports["full"]["new_tests"] == 13
        and reports["full"]["new_test_skips"] == 0
        and not any(name.split("::")[0] in NEW_TESTS for name, _ in ids["full"]),
        "expansion_and_nested_layout_audit_verified": json.loads(
            (OUT / "expansion-verification.log").read_text()
        )["status"]
        == "verified",
        "count_generation_verified": json.loads((OUT / "count-verification.log").read_text())[
            "status"
        ]
        == "verified",
        "count_replay_identical": findings[
            "count_replay_all_scientific_fields_and_artifact_hashes_identical"
        ],
        "negative_result_preserved_without_promotion": findings["promoted"] is False
        and not findings["count_prior_preserves_source_layout_baseline"]
        and not findings["count_prior_preserves_completed_baseline"],
        "black_pass": "left unchanged" in (OUT / "black.log").read_text(),
        "ruff_pass": "All checks passed!" in (OUT / "ruff.log").read_text(),
        "whitespace_pass": all(
            subprocess.run(cmd, cwd=ROOT, capture_output=True).returncode == 0
            for cmd in (["git", "diff", "--check"], ["git", "diff", "--cached", "--check"])
        ),
    }
    files = [p for p in OUT.iterdir() if p.is_file() and p.name != "result.json"]
    files.extend(ROOT / name for name in FILES)
    scoped_pass = all(checks.values())
    result = {
        "schema_version": "forge.combinatorial_generated_layouts_validation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            ("scoped_checks_pass_global_suite_failed" if ids["full"] else "checks_pass")
            if scoped_pass
            else "inspect_checks"
        ),
        "phase1_definition_of_done_met": scoped_pass and not ids["full"],
        "scientific_preservation_screen_passed": False,
        "goal_complete": False,
        "promoted": False,
        "checks": checks,
        "inputs": [pin(p) for p in sorted(files)],
        "test_reports": reports,
        "failure_comparison": {
            "same_failure_and_error_ids_and_kinds": ids["full"] == ids["baseline"],
            "new": sorted(ids["full"] - ids["baseline"]),
            "resolved": sorted(ids["baseline"] - ids["full"]),
        },
        "commands": {
            "vendor": "UV_CACHE_DIR=/private/tmp/forge-uv-cache make verify",
            "focused": "uv run pytest -q tests/test_generated_program_layout.py tests/test_combinatorial_layout_prior.py tests/test_program_connection_reuse.py tests/test_combinatorial_connection_reuse.py tests/test_precursor_occurrences.py tests/test_library_semantics.py tests/test_library_programs.py tests/test_pinned_sources_are_tracked.py",
            "full": "UV_CACHE_DIR=/private/tmp/forge-uv-cache PYTEST_ADDOPTS='--tb=short --junitxml=results/phase1/combinatorial_generated_layouts_validation_v1/full-tests.xml' make test",
            "findings": "uv run python results/phase1/combinatorial_generated_layouts_validation_v1/report_findings.py",
            "expansion_verify": "uv run python -m experiments.phase1.multireaction.combinatorial_context_expansion --verify results/phase1/combinatorial_context_expansion_v1/result.json",
            "count_verify": "uv run python -m experiments.phase1.multireaction.combinatorial_count_layout_generation --verify results/phase1/combinatorial_count_layout_generation_v1/result.json",
        },
        "limits": [
            "Engineering checks and exact replay do not turn the failed scientific preservation screen into an improvement.",
            "The count-only prior is not promoted; the existing completion pipeline remains the reference.",
            "Matching failure IDs and kinds does not fully diagnose all repository failures. No failing check was weakened or skipped.",
            "No chemical-realism improvement, all-library ionizable-lipid qualification or L2/L3 closure is established.",
        ],
    }
    path = OUT / "result.json"
    if path.exists():
        raise ValueError("validation output already exists")
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "checks": checks,
                "counts": {k: r["counts"] for k, r in reports.items()},
                "failure_comparison": result["failure_comparison"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
