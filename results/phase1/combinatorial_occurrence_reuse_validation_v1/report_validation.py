"""Record scoped validation and compare repository failures with the preceding baseline."""

import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from forge.core.hashing import resolve_pin, sha256_file

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


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
    new = [c for c in cases if c.get("classname") == "tests.test_precursor_occurrences"]
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
            ROOT / "results/phase1/combinatorial_graph_reuse_validation_v1/full-tests.xml"
        ),
    }
    evidence = json.loads((OUT / "evidence.json").read_text())
    for value in evidence["inputs"]:
        resolve_pin(value, ROOT, label="verified occurrence evidence")
    ids = {key: {r["test"] for r in value["failures"]} for key, value in reports.items()}
    checks = {
        "vendor_assets_verified": "all 30 present vendored assets verified"
        in (OUT / "verify.log").read_text(),
        "focused_pass": reports["focused"]["counts"]["passed"] == 66 and not ids["focused"],
        "new_tests_pass_in_full": reports["full"]["new_tests"] == 15
        and reports["full"]["new_test_skips"] == 0
        and not any("tests.test_precursor_occurrences::" in name for name in ids["full"]),
        "all_three_populations_verified": len(evidence["populations"]) == 3
        and all(
            p["verification"]["status"] == "verified" for p in evidence["populations"].values()
        ),
        "all_predeclared_population_rules_pass": evidence["all_populations_pass"],
        "discovery_replay_identical": evidence["scientific_payload_replay_identical"],
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
            "forge/assembly/library_occurrences.py",
            "forge/model/precursor_occurrence_reuse.py",
            "experiments/phase1/multireaction/combinatorial_occurrence_reuse.py",
            "tests/test_precursor_occurrences.py",
            "Makefile",
            "pyproject.toml",
            "uv.lock",
        )
    )
    scoped_pass = all(checks.values())
    result = {
        "schema_version": "forge.combinatorial_occurrence_reuse_validation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            ("scoped_checks_pass_global_suite_failed" if ids["full"] else "checks_pass")
            if scoped_pass
            else "inspect_checks"
        ),
        "phase1_definition_of_done_met": scoped_pass and not ids["full"],
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
            "focused": "uv run pytest -q tests/test_precursor_occurrences.py tests/test_precursor_reuse_projection.py tests/test_precursor_reuse_admission.py tests/test_library_semantics.py tests/test_library_programs.py tests/test_pinned_sources_are_tracked.py",
            "full": "UV_CACHE_DIR=/private/tmp/forge-uv-cache PYTEST_ADDOPTS='--tb=short --junitxml=results/phase1/combinatorial_occurrence_reuse_validation_v1/full-tests.xml' make test",
            "evidence": "uv run python results/phase1/combinatorial_occurrence_reuse_validation_v1/report_evidence.py",
        },
        "limits": [
            "The active broad goal is not marked complete by this bounded improvement.",
            "No repository-wide pass, new-layout autonomy, learned-model improvement, all-library IL qualification or chemical realism is claimed.",
            "Matching failure IDs does not fully diagnose every existing failure. No failing check was weakened or skipped.",
        ],
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
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
