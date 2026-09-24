"""Recount the complete STAAR stage replay and its attributable validation."""

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
    raw = path.read_bytes()
    tree = ET.fromstring(gzip.decompress(raw) if path.suffix == ".gz" else raw)
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
    previous_path = PREFIX + "reductive_validation_v1/result.json"
    baseline_path = PREFIX + "manifest_recovery_v1/result.json"
    program_path = PREFIX + "staar_program_v1/result.json"
    previous, baseline, program = map(read, (previous_path, baseline_path, program_path))
    source = read(program["inputs"]["adjudication"]["path"])
    with gzip.open(ROOT / program["artifacts"]["programs.jsonl.gz"]["path"], "rt") as stream:
        rows = [json.loads(line) for line in stream]
    families = previous["family_scope"]
    for family, summary in program["summary"]["by_family"].items():
        selected = [r for r in rows if r["family"] == family]
        assert len(selected) == summary["rows"]
        assert dict(Counter(r["status"] for r in selected)) == summary["by_status"]
        for field in ("computed_consistency_pass", "eligible_after_known_exclusions"):
            assert sum(r[field] for r in selected) == summary[field]
        assert (
            sum(
                r["computed_consistency_pass"] and r["historical_protected_precursor"]
                for r in selected
            )
            == summary["consistent_with_known_protected_precursor"]
        )
        assert (
            sum(r["component_label_conflict"] for r in selected)
            == summary["component_label_conflicts"]
        )
        assert not any(r["training_admitted"] for r in selected)
        families[family]["additional_sequential_program_evidence"] = {
            "receipt": pin(program_path),
            "summary": summary,
            "source_bank_membership_qualified": False,
        }
    focused, cases = test_report(OUTPUT / "focused-tests.xml")
    full, _ = test_report(OUTPUT / "full-tests.xml.gz")
    baseline_bad = set(baseline["full"]["failures"]) | set(baseline["full"]["errors"])
    current_bad = set(full["failures"]) | set(full["errors"])
    verification = read(OUTPUT / "verification.json")
    execution = read(OUTPUT / "execution.json")
    checks = read(OUTPUT / "checks.json")
    snapshot = read(OUTPUT / "source-snapshot.json")
    snapshot_unchanged = {path: pin(path)["sha256"] == sha for path, sha in snapshot.items()}
    assert all(snapshot_unchanged.values())
    assert verification["exit_code"] == 0 and verification["stdout"] == "verified\n"
    assert execution["exit_code"] == 0
    assert "full-tests" in checks
    assert all(item["exit_code"] == 0 for key, item in checks.items() if key != "full-tests")
    inputs = {
        previous_path,
        baseline_path,
        program_path,
        Path(__file__).relative_to(ROOT).as_posix(),
    }
    inputs.update(snapshot)
    for section in ("inputs", "implementation", "artifacts"):
        inputs.update(p["path"] for p in program[section].values())
    inputs.add(program["config"]["path"])
    inputs.update(p["path"] for p in source["assets"].values())
    test_modules = sorted({c.get("classname").removeprefix("tests.") for c in cases})
    inputs.update(f"tests/{name}.py" for name in test_modules)
    new_modules = {"tests.test_sequential_program", "tests.test_compose_lipid_sequential"}
    new_failures = sorted(name for name in current_bad if name.partition("::")[0] in new_modules)
    result = {
        "schema_version": "forge.compose_lipid_staar_validation.v1",
        "status": "complete_sequential_programs_replayed_training_unqualified",
        "phase1_definition_of_done_met": False,
        "training_ready": False,
        "training_calls": 0,
        "random_sampling_used": False,
        "training_rows_admitted": 0,
        "inputs": {path: pin(path) for path in sorted(inputs)},
        "artifacts": {
            p.name: pin(p)
            for p in sorted(OUTPUT.iterdir())
            if p.is_file()
            and p.name not in {"result.json", "summarize.py", "full-tests.xml", "full-tests.log"}
        },
        "focused": focused,
        "full": full,
        "full_snapshot_unchanged": snapshot_unchanged,
        "full_run_scope": "Completed after production, source-contract and test changes. This subsequent recount does not change their bytes.",
        "failure_comparison": {
            "baseline": pin(baseline_path),
            "new_failure_or_error_ids": sorted(current_bad - baseline_bad),
            "resolved_failure_or_error_ids": sorted(baseline_bad - current_bad),
            "new_sequential_audit_failures": new_failures,
        },
        "new_test_count": sum(c.get("classname") in new_modules for c in cases),
        "vendor_assets_verified": sum(
            line.strip().startswith("ok ")
            for line in (OUTPUT / "vendor-verify.log").read_text().splitlines()
        ),
        "black_pass": checks["black"]["exit_code"] == 0,
        "ruff_pass": checks["ruff"]["exit_code"] == 0,
        "program_summary": program["summary"],
        "source_control_formulas": {
            label: control["neutral_formula"]
            for label, control in program["source_controls"].items()
        },
        "family_count": len(families),
        "family_scope": families,
        "latest_full_suite_reference": {
            "receipt": "this_receipt.full",
            "xml": pin(OUTPUT / "full-tests.xml.gz"),
            "log": pin(OUTPUT / "full-tests.log.gz"),
            "snapshot": pin(OUTPUT / "source-snapshot.json"),
            "counts": full["counts"],
        },
        "commands": {
            "recount": "uv run python results/phase1/compose_lipid_v8_staar_validation_v1/summarize.py",
            "execution": execution["command"],
            "verification": verification["command"],
            "checks": checks,
        },
        "nonclaims": [
            "Complete aminolysis followed by thiol-acrylate addition is assessed on every enforced preparation row for this family, after prior-product exclusion.",
            "Reagent addition order and covalent stage order are recorded separately; neither implies experimental selectivity for new products.",
            "Independent source drawings and adversarial tests check transform consistency, not corpus-wide chemical precision or exact source-bank membership.",
            "Constitutional identity is stereo-free; the basic-site query is not a measured ionization or pKa claim.",
            "Known protected-precursor overlaps remain excluded; global precursor holdouts and the final balanced TRAIN dataset are still unqualified.",
            "No L2 route, experimental success, biological supervision or complete 23-family training-readiness claim is made.",
            "The repository-wide suite remains failing; no historical checks were skipped or relaxed.",
        ],
    }
    assert focused["counts"]["failures"] == focused["counts"]["errors"] == 0
    assert not new_failures
    assert result["vendor_assets_verified"] == 30
    (OUTPUT / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(result["status"], focused["counts"], full["counts"], pin(OUTPUT / "result.json"))


if __name__ == "__main__":
    main()
