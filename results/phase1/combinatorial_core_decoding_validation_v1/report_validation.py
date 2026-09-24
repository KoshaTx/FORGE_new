"""Record validation without promoting either failed scientific preservation screen."""

import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from forge.core.hashing import resolve_pin, sha256_file

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
NEW_TESTS = {"tests.test_combinatorial_core_scaffold", "tests.test_combinatorial_core_order"}


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
        "failures": sum(f["kind"] == "failure" for f in failures),
        "errors": sum(f["kind"] == "error" for f in failures),
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
        "focused": report(OUT / "final-focused-tests.xml"),
        "full": report(OUT / "final-full-tests.xml"),
        "initial_full": report(OUT / "full-tests.xml"),
        "baseline": report(
            ROOT / "results/phase1/combinatorial_generated_layouts_validation_v1/full-tests.xml"
        ),
    }
    findings = json.loads((OUT / "findings.json").read_text())
    for p in findings["inputs"].values():
        resolve_pin(p, ROOT, label="core findings")
    ids = {name: {(f["test"], f["kind"]) for f in r["failures"]} for name, r in reports.items()}
    checks = {
        "vendor_assets_verified": "all 30 present vendored assets verified"
        in (OUT / "final-verify.log").read_text(),
        "focused_pass": reports["focused"]["counts"]["passed"] == 53 and not ids["focused"],
        "new_tests_pass_in_full": reports["full"]["new_tests"] == 6
        and reports["full"]["new_test_skips"] == 0
        and not any(name.split("::")[0] in NEW_TESTS for name, _ in ids["full"]),
        "both_experiments_semantically_verified": all(
            json.loads((OUT / name).read_text())["status"] == "verified"
            for name in ("verification.log", "ordered-verification.log")
        ),
        "both_fresh_replays_identical": all(
            r["replay_scientific_fields_and_artifact_hashes_identical"]
            for r in findings["runs"].values()
        ),
        "all_768_counts_and_typed_core_graphs_preserved_by_ordering": findings[
            "count_and_typed_core_graphs_preserved_by_ordering"
        ]
        == 768,
        "negative_results_preserved_without_promotion": findings["promoted"] is False
        and all(
            not r["paired_preservation_pass"] and not r["existing_pipeline_preservation_pass"]
            for r in findings["runs"].values()
        ),
        "black_pass": "left unchanged" in (OUT / "black.log").read_text(),
        "ruff_pass": "All checks passed!" in (OUT / "ruff.log").read_text(),
        "whitespace_pass": all(
            subprocess.run(cmd, cwd=ROOT, capture_output=True).returncode == 0
            for cmd in (["git", "diff", "--check"], ["git", "diff", "--cached", "--check"])
        ),
    }
    files = [p for p in OUT.iterdir() if p.is_file() and p.name != "result.json"]
    files.extend(
        ROOT / p
        for p in (
            "forge/model/combinatorial_core_scaffold.py",
            "forge/model/combinatorial_core_order.py",
            "forge/model/paired_core_decoding.py",
            "experiments/phase1/multireaction/combinatorial_core_decoding.py",
            "experiments/phase1/multireaction/combinatorial_ordered_core_decoding.py",
            "tests/test_combinatorial_core_scaffold.py",
            "tests/test_combinatorial_core_order.py",
            "Makefile",
            "pyproject.toml",
            "uv.lock",
        )
    )
    scoped = all(checks.values())
    result = {
        "schema_version": "forge.combinatorial_core_decoding_validation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            ("scoped_checks_pass_global_suite_failed" if ids["full"] else "checks_pass")
            if scoped
            else "inspect_checks"
        ),
        "phase1_definition_of_done_met": scoped and not ids["full"],
        "scientific_preservation_screen_passed": False,
        "promoted": False,
        "goal_complete": False,
        "checks": checks,
        "inputs": [pin(p) for p in sorted(files)],
        "test_reports": reports,
        "failure_comparison": {
            "same_failure_and_error_ids_and_kinds": ids["full"] == ids["baseline"],
            "new": sorted(ids["full"] - ids["baseline"]),
            "resolved": sorted(ids["baseline"] - ids["full"]),
            "initial_new_failures": sorted(ids["initial_full"] - ids["baseline"]),
        },
        "commands": {
            "vendor": "UV_CACHE_DIR=/private/tmp/forge-uv-cache make verify",
            "focused": "uv run pytest -q tests/test_combinatorial_core_scaffold.py tests/test_combinatorial_core_order.py tests/test_combinatorial_layout_prior.py tests/test_generated_program_layout.py tests/test_library_programs.py tests/test_library_semantics.py tests/test_pinned_sources_are_tracked.py",
            "full": "UV_CACHE_DIR=/private/tmp/forge-uv-cache PYTEST_ADDOPTS='--tb=short --junitxml=results/phase1/combinatorial_core_decoding_validation_v1/final-full-tests.xml' make test",
            "findings": "uv run python results/phase1/combinatorial_core_decoding_validation_v1/report_findings.py",
        },
        "limits": [
            "Both terminal core experiments fail scientific preservation; passing implementation checks does not establish improved generation under the user goal.",
            "The initial full run began before the ordering follow-up was staged and its new test was collected. Its two new provenance failures are retained; the final run follows staging of all pinned files.",
            "Matching failure IDs and kinds is not a full diagnosis of every repository failure. No failing gate or test was weakened or skipped.",
        ],
    }
    path = OUT / "result.json"
    if path.exists():
        raise ValueError("validation result exists")
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
