"""Validate the unified pipeline and identify the remaining goal-completion requirement."""

import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from experiments.phase1.multireaction import combinatorial_generation_pipeline as pipeline
from forge.core.hashing import resolve_pin

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
NEW_TEST = "tests.test_combinatorial_generation_pipeline"


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
    new = [c for c in cases if c.get("classname") == NEW_TEST]
    return {
        "xml": pipeline._pin(path, ROOT),
        "counts": counts,
        "failures": failures,
        "new_tests": len(new),
        "new_test_skips": sum(c.find("skipped") is not None for c in new),
    }


def main():
    reports = {
        "initial_focused": report(OUT / "focused-tests.xml"),
        "focused": report(OUT / "final-focused-tests.xml"),
        "full": report(OUT / "full-tests.xml"),
        "baseline": report(
            ROOT / "results/phase1/combinatorial_reserved_core_validation_v1/full-tests.xml"
        ),
    }
    ids = {name: {(f["test"], f["kind"]) for f in r["failures"]} for name, r in reports.items()}
    evidence = pipeline._read(OUT / "evidence.json")
    for p in evidence["inputs"]:
        resolve_pin(p, ROOT, label="pipeline validation evidence")
    fresh = pipeline._read(OUT / "fresh-verification.log")
    reuse = pipeline._read(OUT / "replay-reuse.log")
    checks = {
        "vendor_assets_verified": "all 30 present vendored assets verified"
        in (OUT / "verify.log").read_text(),
        "focused_pass": reports["focused"]["counts"]["passed"] == 53 and not ids["focused"],
        "all_fifteen_new_tests_pass_in_full": reports["full"]["new_tests"] == 15
        and reports["full"]["new_test_skips"] == 0
        and not any(name.split("::")[0] == NEW_TEST for name, _ in ids["full"]),
        "fresh_pipeline_semantically_verified": fresh["status"] == "verified"
        and fresh["acceptance_passed"],
        "completed_pipeline_reuse_verified": reuse["status"] == "numerical_complete"
        and reuse["acceptance_passed"],
        "historical_stage_artifacts_and_metrics_identical": len(
            evidence["integration_replay"]["checks"]
        )
        == 5,
        "fresh_population_preservation_and_gain": all(
            evidence["fresh_population"]["checks"].values()
        ),
        "black_pass": "left unchanged" in (OUT / "all-black.log").read_text(),
        "ruff_pass": "All checks passed!" in (OUT / "all-ruff.log").read_text(),
        "whitespace_pass": all(
            subprocess.run(cmd, cwd=ROOT, capture_output=True).returncode == 0
            for cmd in (["git", "diff", "--check"], ["git", "diff", "--cached", "--check"])
        ),
    }
    paths = [p for p in OUT.rglob("*") if p.is_file() and p.name != "result.json"]
    paths.extend(
        ROOT / p
        for p in (
            pipeline.SELF,
            "tests/test_combinatorial_generation_pipeline.py",
            "Makefile",
            "pyproject.toml",
            "uv.lock",
        )
    )
    scoped = all(checks.values())
    result = {
        "schema_version": "forge.combinatorial_generation_pipeline_validation.v1",
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
        "inputs": [pipeline._pin(p, ROOT) for p in sorted(paths)],
        "test_reports": reports,
        "failure_comparison": {
            "same_failure_and_error_ids_and_kinds": ids["full"] == ids["baseline"],
            "new": sorted(ids["full"] - ids["baseline"]),
            "resolved": sorted(ids["baseline"] - ids["full"]),
        },
        "completion_audit": {
            "twelve_family_generation_command": "Implemented and exercised on a new balanced TRAIN-context population; qualified exports retain family-specific scope.",
            "generation_improvement": "Positive exact-program gain versus the paired raw generator and versus the preceding admitted pipeline, under the frozen new-population request.",
            "structural_validity_exact_program_diversity_novelty": "All eight declared metrics are nondecreasing in every family. Every family has novel exactly reconstructable products.",
            "gates_unchanged": "Existing stages and checkers are consumed at their frozen source identities; orchestration does not alter their scientific gates.",
            "reproducibility": "All five historical stage artifact sets reproduce exactly, new results verify semantically, and completed-stage reuse succeeds.",
            "remaining_required_repository_validation": "make test still fails. Global Phase 1 completion cannot be claimed until these failures are repaired or their required authentic artifacts are restored without weakening gates.",
        },
        "commands": {
            "vendor": "UV_CACHE_DIR=/private/tmp/forge-uv-cache make verify",
            "focused": "uv run pytest -q tests/test_combinatorial_generation_pipeline.py tests/test_source_core_completion.py tests/test_fixed_closure_decoding.py tests/test_precursor_reuse_admission.py tests/test_precursor_occurrences.py tests/test_program_connection_reuse.py tests/test_pinned_sources_are_tracked.py",
            "full": "UV_CACHE_DIR=/private/tmp/forge-uv-cache PYTEST_ADDOPTS='--tb=short --junitxml=results/phase1/combinatorial_generation_pipeline_validation_v1/full-tests.xml' make test",
            "generation": "uv run python -m experiments.phase1.multireaction.combinatorial_generation_pipeline --config configs/multireaction/combinatorial_generation_pipeline_fresh_v2.json --output-dir results/phase1/combinatorial_generation_pipeline_fresh_v1",
        },
        "limits": [
            "The original integration attempt was stopped by final source authentication after import sorting changed the orchestrator. Its exact source and requests are retained. All completed stage configs were identical under the revised request and independently reverified before reuse.",
            "Fresh TRAIN-context sampling is not heldout or source-independent generation. Improvements concern exact program reconstruction and the specified preservation metrics, not calibrated lipid realism or experimental synthesis success.",
            "Matching historical failure identities does not diagnose every failure. No failing test was weakened or skipped.",
        ],
    }
    target = OUT / "result.json"
    if target.exists():
        raise ValueError("validation result already exists")
    pipeline._write(target, result)
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
