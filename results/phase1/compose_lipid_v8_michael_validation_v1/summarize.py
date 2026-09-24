"""Recount the repeated Michael increment without promoting training readiness."""

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


def main():
    previous_path = PREFIX + "source_event_validation_v1/result.json"
    first_path = PREFIX + "michael_program_v1/result.json"
    program_path = PREFIX + "michael_program_v2/result.json"
    previous, first, program = map(read, (previous_path, first_path, program_path))
    source = read(program["inputs"]["adjudication"]["path"])
    with gzip.open(ROOT / program["artifacts"]["programs.jsonl.gz"]["path"], "rt") as stream:
        rows = [json.loads(line) for line in stream]
    comparison = {
        name: first[name] == program[name]
        for name in ("config", "inputs", "policy", "source_controls", "summary")
    }
    comparison["complete_ledger_bytes"] = (
        pin(first["artifacts"]["programs.jsonl.gz"]["path"])["sha256"]
        == pin(program["artifacts"]["programs.jsonl.gz"]["path"])["sha256"]
    )
    assert all(comparison.values())
    families = previous["family_scope"]
    for family, summary in program["summary"]["by_family"].items():
        selected = [row for row in rows if row["family"] == family]
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
        assert not any(r["training_admitted"] for r in selected)
        families[family]["additional_repeated_program_evidence"] = {
            "receipt": pin(program_path),
            "summary": summary,
            "source_architecture_assignment_complete": False,
        }
    cases = list(ET.parse(OUTPUT / "focused-tests.xml").iter("testcase"))
    counts = {
        "tests": len(cases),
        "failures": sum(c.find("failure") is not None for c in cases),
        "errors": sum(c.find("error") is not None for c in cases),
        "skipped_or_xfail": sum(c.find("skipped") is not None for c in cases),
    }
    counts["passed"] = (
        counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped_or_xfail"]
    )
    verification = read((OUTPUT / "verification.json").relative_to(ROOT))
    execution = read((OUTPUT / "execution.json").relative_to(ROOT))
    assert verification["exit_code"] == 0 and verification["stdout"] == "verified\n"
    assert execution["exit_code"] == 0
    inputs = {
        previous_path,
        first_path,
        program_path,
        PREFIX + "michael_program_v1/implementation_snapshot.json",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
        Path(__file__).relative_to(ROOT).as_posix(),
        "experiments/phase1/multireaction/compose_lipid_repeated_inverse.py",
    }
    inputs.update(p["path"] for p in program["inputs"].values())
    inputs.update(p["path"] for p in program["implementation"].values())
    inputs.update(p["path"] for p in program["artifacts"].values())
    inputs.add(program["config"]["path"])
    inputs.update(p["path"] for p in source["assets"].values())
    test_modules = sorted({c.get("classname").removeprefix("tests.") for c in cases})
    inputs.update(f"tests/{name}.py" for name in test_modules)
    artifacts = {
        p.name: pin(p)
        for p in sorted(OUTPUT.iterdir())
        if p.suffix in (".xml", ".log") or p.name in ("execution.json", "verification.json")
    }
    result = {
        "schema_version": "forge.compose_lipid_michael_validation.v1",
        "status": "repeated_michael_programs_replayed_training_unqualified",
        "phase1_definition_of_done_met": False,
        "training_ready": False,
        "training_calls": 0,
        "random_sampling_used": False,
        "training_rows_admitted": 0,
        "inputs": {path: pin(path) for path in sorted(inputs)},
        "artifacts": artifacts,
        "focused": counts,
        "new_test_count": sum(
            c.get("classname")
            in {"tests.test_repeated_inverse", "tests.test_compose_lipid_repeated_inverse"}
            for c in cases
        ),
        "vendor_assets_verified": sum(
            line.strip().startswith("ok ")
            for line in (OUTPUT / "vendor-verify.log").read_text().splitlines()
        ),
        "black_pass": "would be left unchanged" in (OUTPUT / "black.log").read_text(),
        "ruff_pass": "All checks passed!" in (OUTPUT / "ruff.log").read_text(),
        "program_summary": program["summary"],
        "replay_equivalence": comparison,
        "replay_scope": "Source-asset authentication was tightened between the two runs. All compared scientific fields and the entire ledger reproduce identically.",
        "family_count": len(families),
        "family_scope": families,
        "latest_full_suite_reference": previous["latest_full_suite_reference"],
        "commands": {
            "recount": "uv run python results/phase1/compose_lipid_v8_michael_validation_v1/summarize.py",
            "execution": execution["command"],
            "verification": verification["command"],
            "vendor": "make verify",
            "focused": "uv run pytest -q " + " ".join(f"tests/{name}.py" for name in test_modules),
        },
        "nonclaims": [
            "The shared repeated-event program is checked; source architecture and precursor-bank assignments are not inferred.",
            "Reagent feed ratios do not prove maximum occupancy. Missing occupancy is excluded without graph-based imputation.",
            "The five-tail source isomer is an ambiguity control of attribution, not an experimental failure.",
            "Primary controls and adversarial checks do not establish corpus-wide chemical precision or experimental selectivity.",
            "No global precursor holdout or complete 23-family training-readiness claim is made.",
            "The full suite remains failing at its last retained run and was not repeated for this increment.",
        ],
    }
    assert counts["failures"] == counts["errors"] == 0
    assert result["vendor_assets_verified"] == 30
    assert result["black_pass"] and result["ruff_pass"]
    (OUTPUT / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(result["status"], counts, pin(OUTPUT / "result.json"))


if __name__ == "__main__":
    main()
