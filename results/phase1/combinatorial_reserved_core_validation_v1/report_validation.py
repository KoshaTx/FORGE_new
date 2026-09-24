"""Record scoped validation and preserve the independent repository failure baseline."""

import gzip
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from forge.core.hashing import resolve_pin, sha256_file

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
NEW_TESTS = {"tests.test_fixed_closure_decoding", "tests.test_source_core_completion"}
PYTHON = (
    "forge/model/fixed_closure_decoding.py",
    "forge/model/paired_reserved_core_decoding.py",
    "forge/model/source_core_completion.py",
    "experiments/phase1/multireaction/combinatorial_reserved_core_decoding.py",
    "experiments/phase1/multireaction/combinatorial_source_core_completion.py",
    "tests/test_fixed_closure_decoding.py",
    "tests/test_source_core_completion.py",
)


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": str(sha256_file(path))}


def report(path):
    content = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    cases = list(ET.fromstring(content).iter("testcase"))
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
        "initial_focused": report(OUT / "focused-tests.xml.gz"),
        "focused": report(OUT / "final-focused-tests.xml"),
        "full": report(OUT / "full-tests.xml"),
        "baseline": report(
            ROOT / "results/phase1/combinatorial_core_decoding_validation_v1/final-full-tests.xml"
        ),
    }
    ids = {name: {(f["test"], f["kind"]) for f in r["failures"]} for name, r in reports.items()}
    findings = json.loads((OUT / "findings.json").read_text())
    for value in findings["inputs"].values():
        resolve_pin(value, ROOT, label="reserved-core validation")
    failed_runs = []
    for path in sorted(OUT.glob("failed_source_attempt_*/failure.json")):
        value = json.loads(path.read_text())
        assert value["status"] == "failed_no_result_admitted"
        for p in value["inputs"]:
            assert str(sha256_file(ROOT / p["snapshot_path"])) == p["sha256"]
        failed_runs.append(pin(path))
    checks = {
        "vendor_assets_verified": "all 30 present vendored assets verified"
        in (OUT / "verify.log").read_text(),
        "focused_pass": reports["focused"]["counts"]["passed"] == 62 and not ids["focused"],
        "all_five_new_tests_pass_in_full": reports["full"]["new_tests"] == 5
        and reports["full"]["new_test_skips"] == 0
        and not any(name.split("::")[0] in NEW_TESTS for name, _ in ids["full"]),
        "source_completion_semantically_verified_in_three_populations": all(
            json.loads((OUT / f"source-{label}-verification.log").read_text())["status"]
            == "verified"
            for label in ("discovery", "replication_1", "replication_2")
        ),
        "reserved_decoder_semantically_verified": json.loads(
            (OUT / "reserved-verification.log").read_text()
        )["status"]
        == "verified",
        "both_fresh_replays_identical": findings["closure_correction"]["fresh_replay_identical"]
        and findings["source_core_completion"]["discovery"]["fresh_replay_identical"],
        "admitted_source_core_stage_passes_three_preservation_screens": all(
            r["exact_gain"] > 0 and r["all_eight_metrics_preserved_in_every_family"]
            for r in findings["source_core_completion"].values()
        ),
        "negative_raw_core_screen_preserved": not findings["closure_correction"][
            "paired_preservation_pass"
        ],
        "failed_execution_snapshots_preserved": len(failed_runs) == 2,
        "black_pass": "left unchanged" in (OUT / "all-black.log").read_text(),
        "ruff_pass": "All checks passed!" in (OUT / "all-ruff.log").read_text(),
        "whitespace_pass": all(
            subprocess.run(cmd, cwd=ROOT, capture_output=True).returncode == 0
            for cmd in (["git", "diff", "--check"], ["git", "diff", "--cached", "--check"])
        ),
    }
    paths = [p for p in OUT.rglob("*") if p.is_file() and p.name != "result.json"]
    paths.extend(ROOT / p for p in (*PYTHON, "Makefile", "pyproject.toml", "uv.lock"))
    scoped = all(checks.values())
    result = {
        "schema_version": "forge.combinatorial_reserved_core_validation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            ("scoped_checks_pass_global_suite_failed" if ids["full"] else "checks_pass")
            if scoped
            else "inspect_checks"
        ),
        "phase1_definition_of_done_met": scoped and not ids["full"],
        "goal_complete": False,
        "promoted": False,
        "checks": checks,
        "inputs": [pin(p) for p in sorted(paths)],
        "test_reports": reports,
        "failed_execution_records": failed_runs,
        "failure_comparison": {
            "same_failure_and_error_ids_and_kinds": ids["full"] == ids["baseline"],
            "new": sorted(ids["full"] - ids["baseline"]),
            "resolved": sorted(ids["baseline"] - ids["full"]),
        },
        "commands": {
            "vendor": "UV_CACHE_DIR=/private/tmp/forge-uv-cache make verify",
            "focused": "uv run pytest -q tests/test_fixed_closure_decoding.py tests/test_source_core_completion.py tests/test_combinatorial_core_scaffold.py tests/test_combinatorial_core_order.py tests/test_precursor_reuse_admission.py tests/test_precursor_reuse_projection.py tests/test_library_programs.py tests/test_library_semantics.py tests/test_pinned_sources_are_tracked.py",
            "full": "UV_CACHE_DIR=/private/tmp/forge-uv-cache PYTEST_ADDOPTS='--tb=short --junitxml=results/phase1/combinatorial_reserved_core_validation_v1/full-tests.xml' make test",
            "findings": "uv run python results/phase1/combinatorial_reserved_core_validation_v1/report_findings.py",
        },
        "limits": [
            "The retained pipeline gain uses explicit source-derived core context. It does not establish learned improvement, chemical realism, heldout performance or complete L2/L3 route closure.",
            "The initial focused failure expected unchanged morphology after exterior masking. Only computed morphology differs; the corrected test verifies every other context field plus sparse-flow context identity. Original neural trajectory identity is independently tested and checked on every saved attempt.",
            "Both implementation exceptions preceded any admitted result. Source snapshots, config revisions and failure logs are retained. Seeds, budgets, chemistry gates and acceptance criteria were not changed.",
            "Matching historical failure IDs and kinds does not diagnose every repository failure. No failing scientific gate or repository test was weakened or skipped.",
            "Original failed-test log/XML bytes are preserved in gzip files because pytest's failure text contains trailing spaces. The first validation report and its original producer and raw logs remain in initial-validation.tar.gz, authenticated by its receipt; no whitespace rule was changed.",
        ],
    }
    target = OUT / "result.json"
    if target.exists():
        raise ValueError("validation result already exists")
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": checks,
                "counts": {k: r["counts"] for k, r in reports.items()},
                "failure_comparison": result["failure_comparison"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
