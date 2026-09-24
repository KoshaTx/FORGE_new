"""Recount source-scaffold qualification and its final-snapshot test evidence."""

import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
PREFIX = "results/phase1/compose_lipid_v8_"


def read(path):
    return json.loads((ROOT / path).read_text())


def pin(path):
    path = (ROOT / path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest.hexdigest()}


def test_report(path):
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    tree = ET.fromstring(data)
    cases = list(tree.iter("testcase"))
    counts = {
        "tests": len(cases),
        "failures": sum(c.find("failure") is not None for c in cases),
        "errors": sum(c.find("error") is not None for c in cases),
        "skipped_or_xfail": sum(c.find("skipped") is not None for c in cases),
    }
    counts["passed"] = (
        counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped_or_xfail"]
    )
    return {
        "counts": counts,
        "duration_seconds": sum(float(t.get("time", 0)) for t in tree.iter("testsuite")),
        **{
            label: {
                f"{c.get('classname')}::{c.get('name')}": c.find(kind).get("message")
                for c in cases
                if c.find(kind) is not None
            }
            for label, kind in (("failures", "failure"), ("errors", "error"))
        },
    }, cases


def main():
    previous_path = PREFIX + "michael_validation_v1/result.json"
    baseline_path = PREFIX + "a3_validation_v1/result.json"
    program_path = PREFIX + "reductive_program_v1/result.json"
    previous, baseline, program = map(read, (previous_path, baseline_path, program_path))
    source = read(program["inputs"]["adjudication"]["path"])
    with gzip.open(ROOT / program["artifacts"]["programs.jsonl.gz"]["path"], "rt") as stream:
        rows = [json.loads(line) for line in stream]
    families = previous["family_scope"]
    for family, summary in program["summary"]["by_family"].items():
        selected = [r for r in rows if r["family"] == family]
        assert len(selected) == summary["rows"]
        assert dict(Counter(r["status"] for r in selected)) == summary["by_status"]
        assert (
            sum(r["computed_consistency_pass"] for r in selected)
            == summary["computed_consistency_pass"]
        )
        assert (
            sum(r["eligible_after_known_exclusions"] for r in selected)
            == summary["eligible_after_known_exclusions"]
        )
        assert (
            sum(
                r["computed_consistency_pass"] and r["historical_protected_precursor"]
                for r in selected
            )
            == summary["consistent_with_known_protected_precursor"]
        )
        assert not any(r["training_admitted"] for r in selected)
        families[family]["additional_source_scaffold_program_evidence"] = {
            "receipt": pin(program_path),
            "summary": summary,
            "source_bank_membership_qualified": False,
        }
    focused, cases = test_report(OUTPUT / "focused-tests.xml")
    full, full_cases = test_report(OUTPUT / "full-tests.xml.gz")
    baseline_bad = set(baseline["full"]["failures"]) | set(baseline["full"]["errors"])
    current_bad = set(full["failures"]) | set(full["errors"])
    verification = read(OUTPUT / "verification.json")
    execution = read(OUTPUT / "execution.json")
    full_execution = read(OUTPUT / "full-test-execution.json")
    snapshot = read(OUTPUT / "full-test-source-snapshot.json")
    snapshot_unchanged = {path: pin(path)["sha256"] == sha for path, sha in snapshot.items()}
    assert all(snapshot_unchanged.values())
    assert verification["exit_code"] == 0 and verification["stdout"] == "verified\n"
    assert execution["exit_code"] == 0
    assert full_execution["exit_code"] != 0
    probe = read(OUTPUT / "multiplicity-probe.json")
    assert probe["training_rows_admitted"] == 0
    inputs = {
        previous_path,
        baseline_path,
        program_path,
        "Makefile",
        "pyproject.toml",
        "uv.lock",
        Path(__file__).relative_to(ROOT).as_posix(),
        "experiments/phase1/multireaction/compose_lipid_scaffold_event.py",
    }
    inputs.update(p["path"] for p in program["inputs"].values())
    inputs.update(p["path"] for p in program["implementation"].values())
    inputs.update(p["path"] for p in program["artifacts"].values())
    inputs.add(program["config"]["path"])
    inputs.update(p["path"] for p in source["assets"].values())
    inputs.update(p["path"] for p in probe["inputs"].values())
    test_modules = sorted({c.get("classname").removeprefix("tests.") for c in cases})
    inputs.update(f"tests/{name}.py" for name in test_modules)
    new_modules = {"tests.test_precursor_scaffolds", "tests.test_compose_lipid_scaffold_event"}
    new_failures = sorted(name for name in current_bad if name.partition("::")[0] in new_modules)
    artifacts = {
        p.name: pin(p)
        for p in sorted(OUTPUT.iterdir())
        if p.name not in {"result.json", "summarize.py", "full-tests.xml", "full-tests.log"}
        and p.is_file()
    }
    result = {
        "schema_version": "forge.compose_lipid_reductive_validation.v1",
        "status": "source_scaffold_programs_replayed_training_unqualified",
        "phase1_definition_of_done_met": False,
        "training_ready": False,
        "training_calls": 0,
        "random_sampling_used": False,
        "training_rows_admitted": 0,
        "inputs": {path: pin(path) for path in sorted(inputs)},
        "artifacts": artifacts,
        "focused": focused,
        "full": full,
        "full_snapshot_unchanged": snapshot_unchanged,
        "full_run_scope": "Completed after all production and test changes in this increment. The later multiplicity probe and this recount are separate diagnostics.",
        "failure_comparison": {
            "baseline": pin(baseline_path),
            "new_failure_or_error_ids": sorted(current_bad - baseline_bad),
            "resolved_failure_or_error_ids": sorted(baseline_bad - current_bad),
            "new_scaffold_audit_failures": new_failures,
        },
        "new_test_count": sum(c.get("classname") in new_modules for c in cases),
        "new_test_count_in_full_run": sum(c.get("classname") in new_modules for c in full_cases),
        "vendor_assets_verified": sum(
            line.strip().startswith("ok ")
            for line in (OUTPUT / "vendor-verify.log").read_text().splitlines()
        ),
        "black_pass": "would be left unchanged" in (OUTPUT / "black.log").read_text(),
        "ruff_pass": "All checks passed!" in (OUTPUT / "ruff.log").read_text(),
        "program_summary": program["summary"],
        "multiplicity_probe": {
            "receipt": pin(OUTPUT / "multiplicity-probe.json"),
            "rows": probe["rows"],
            "rows_by_source_head_code": probe["rows_by_source_head_code"],
            "inferred_heads_by_source_code": probe["inferred_heads_by_source_code"],
            "candidate_replay_counts": probe["candidate_replay_counts"],
        },
        "family_count": len(families),
        "family_scope": families,
        "latest_full_suite_reference": {
            "receipt": "this_receipt.full",
            "xml": pin(OUTPUT / "full-tests.xml.gz"),
            "log": pin(OUTPUT / "full-tests.log.gz"),
            "snapshot": pin(OUTPUT / "full-test-source-snapshot.json"),
            "counts": full["counts"],
        },
        "commands": {
            "recount": "uv run python results/phase1/compose_lipid_v8_reductive_validation_v1/summarize.py",
            "execution": execution["command"],
            "verification": verification["command"],
            "vendor": "make verify",
            "focused": "uv run pytest -q " + " ".join(f"tests/{name}.py" for name in test_modules),
            "full": full_execution,
            "multiplicity_probe": "uv run python results/phase1/compose_lipid_v8_reductive_validation_v1/multiplicity_probe.py",
        },
        "nonclaims": [
            "Source-scaffold consistency is assessed on the enforced preparation population, not all provider TRAIN rows.",
            "Complete coupled aldehydes retain the source core and identical arms; scaffold checks are not experimental bank membership.",
            "Primary controls and adversarial tests do not calibrate corpus-wide chemical precision or experimental selectivity.",
            "The two-event diagnostic is outside the one-event key and preserves every competing forward product.",
            "No L2 route, global precursor holdout or complete 23-family training-readiness claim is made.",
            "The repository-wide suite remains failing; no historical checks were skipped or relaxed.",
        ],
    }
    assert focused["counts"]["failures"] == focused["counts"]["errors"] == 0
    assert not new_failures
    assert result["vendor_assets_verified"] == 30
    assert result["black_pass"] and result["ruff_pass"]
    (OUTPUT / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(result["status"], focused["counts"], full["counts"], pin(OUTPUT / "result.json"))


if __name__ == "__main__":
    main()
