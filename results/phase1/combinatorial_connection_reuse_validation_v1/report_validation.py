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
    new = [
        c
        for c in cases
        if c.get("classname")
        in {"tests.test_program_connection_reuse", "tests.test_combinatorial_connection_reuse"}
    ]
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
            ROOT / "results/phase1/combinatorial_occurrence_reuse_validation_v1/full-tests.xml"
        ),
    }
    evidence = json.loads((OUT / "evidence.json").read_text())
    replay = json.loads((OUT / "replay_equivalence.json").read_text())
    for value in [*evidence["inputs"].values(), *replay["inputs"].values()]:
        resolve_pin(value, ROOT, label="verified connection evidence")
    ids = {key: {r["test"] for r in value["failures"]} for key, value in reports.items()}
    checks = {
        "vendor_assets_verified": "all 30 present vendored assets verified"
        in (OUT / "verify.log").read_text(),
        "focused_pass": reports["focused"]["counts"]["passed"] == 117 and not ids["focused"],
        "new_tests_pass_in_full": reports["full"]["new_tests"] == 10
        and reports["full"]["new_test_skips"] == 0
        and not any(
            name.startswith(
                (
                    "tests.test_program_connection_reuse::",
                    "tests.test_combinatorial_connection_reuse::",
                )
            )
            for name in ids["full"]
        ),
        "all_three_populations_verified": all(
            json.loads((OUT / p).read_text())["status"] == "verified"
            for p in (
                "discovery-verification.log",
                "replication-2-verification.log",
                "replication_1_verification.json",
            )
        ),
        "all_predeclared_population_rules_pass": len(evidence["runs"]) == 3
        and all(
            r["preservation_passes_for_every_family"] and r["counts"]["accepted"] > 0
            for r in evidence["runs"].values()
        ),
        "selected_graph_invariants_pass": all(
            r["all_selected_atom_count_edge_count_fixed_graph_and_generated_interior_checks_pass"]
            and r["all_original_invalid_and_exact_outputs_preserved"]
            for r in evidence["runs"].values()
        ),
        "discovery_replay_identical": replay["status"]
        == "verified_scientific_payload_and_artifact_equivalence",
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
            "forge/model/program_connection_reuse.py",
            "experiments/phase1/multireaction/combinatorial_connection_reuse.py",
            "tests/test_program_connection_reuse.py",
            "tests/test_combinatorial_connection_reuse.py",
            "Makefile",
            "pyproject.toml",
            "uv.lock",
        )
    )
    scoped_pass = all(checks.values())
    result = {
        "schema_version": "forge.combinatorial_connection_reuse_validation.v1",
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
            "focused": "uv run pytest -q tests/test_program_connection_reuse.py tests/test_combinatorial_connection_reuse.py tests/test_precursor_occurrences.py tests/test_precursor_reuse.py tests/test_precursor_reuse_projection.py tests/test_precursor_reuse_admission.py tests/test_combinatorial_graph_reuse.py tests/test_library_programs.py tests/test_assembly_families.py tests/test_pinned_sources_are_tracked.py",
            "full": "UV_CACHE_DIR=/private/tmp/forge-uv-cache PYTEST_ADDOPTS='--tb=short --junitxml=results/phase1/combinatorial_connection_reuse_validation_v1/full-tests.xml' make test",
            "evidence": "uv run python results/phase1/combinatorial_connection_reuse_validation_v1/report_evidence.py",
        },
        "limits": [
            "The active broad goal is not marked complete by this bounded improvement.",
            "No repository-wide pass, new-layout autonomy, learned-model improvement, all-library IL qualification or chemical realism is claimed.",
            "Matching failure IDs does not fully diagnose every existing failure. No failing check was weakened or skipped.",
        ],
    }
    if (OUT / "result.json").exists():
        raise ValueError("validation output already exists")
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
