"""Recount the source-event increment; do not substitute focused checks for full readiness."""

import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
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
    previous_path = PREFIX + "a3_validation_v1/result.json"
    protection_path = PREFIX + "product_protection_v1/result.json"
    program_path = PREFIX + "ugi3_program_v1/result.json"
    previous, protection, program = map(read, (previous_path, protection_path, program_path))
    with gzip.open(ROOT / program["artifacts"]["programs.jsonl.gz"]["path"], "rt") as stream:
        rows = [json.loads(line) for line in stream]
    by_lane = defaultdict(Counter)
    for row in rows:
        lane = row["source_metadata"].get("design_lane", "unspecified")
        by_lane[lane]["rows"] += 1
        by_lane[lane][row["status"]] += 1
        by_lane[lane]["eligible_after_known_exclusions"] += row["eligible_after_known_exclusions"]
    assert len(rows) == program["summary"]["rows"]
    assert (
        sum(r["computed_consistency_pass"] for r in rows)
        == program["summary"]["computed_consistency_pass"]
    )
    assert (
        sum(r["eligible_after_known_exclusions"] for r in rows)
        == program["summary"]["eligible_after_known_exclusions"]
    )
    families = {}
    for family, earlier in previous["family_scope"].items():
        families[family] = {
            "complete_source_program_qualified_on_inspected_train": earlier[
                "complete_source_program_qualified_on_inspected_train"
            ],
            "architecture_subfamilies": earlier["architecture_subfamilies"],
            "current_product_exclusions": protection["summary"]["by_family"][family],
            "previous_evidence": pin(previous_path),
            "training_admitted": 0,
        }
    families["aldehyde_ugi3"]["additional_single_event_evidence"] = {
        "receipt": pin(program_path),
        "summary": program["summary"],
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
    assert verification["exit_code"] == 0 and verification["stdout"] == "verified\n"
    inputs = {
        previous_path,
        protection_path,
        program_path,
        "results/phase1/compose_lipid_v8_ugi3_source_v1/adjudication.json",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
        Path(__file__).relative_to(ROOT).as_posix(),
    }
    for result in (protection, program):
        inputs.add(result["config"]["path"])
        inputs.update(p["path"] for p in result["implementation"].values())
        inputs.update(p["path"] for p in result["artifacts"].values())
    for name in (
        "source_event",
        "compose_lipid_source_event",
        "compose_lipid_protection",
        "a3_source_boundary",
        "repeated_components",
        "compose_lipid_repeated_source",
        "compose_lipid_source_program",
        "compose_lipid_components",
        "compose_lipid_pretraining",
        "compose_lipid_representation",
        "compose_lipid",
        "pinned_sources_are_tracked",
    ):
        inputs.add(f"tests/test_{name}.py")
    for name in ("compose_lipid_source_event", "compose_lipid_protection"):
        inputs.add(f"experiments/phase1/multireaction/{name}.py")
    artifacts = {
        p.name: pin(p)
        for p in sorted(OUTPUT.iterdir())
        if p.suffix in (".xml", ".log") or p.name == "verification.json"
    }
    result = {
        "schema_version": "forge.compose_lipid_source_event_validation.v1",
        "status": "product_exclusions_enforced_single_event_subset_qualified_training_unqualified",
        "phase1_definition_of_done_met": False,
        "training_ready": False,
        "training_calls": 0,
        "training_rows_admitted": 0,
        "inputs": {path: pin(path) for path in sorted(inputs)},
        "artifacts": artifacts,
        "focused": counts,
        "new_test_count": sum(
            c.get("classname")
            in {
                "tests.test_source_event",
                "tests.test_compose_lipid_source_event",
                "tests.test_compose_lipid_protection",
            }
            for c in cases
        ),
        "vendor_assets_verified": sum(
            line.strip().startswith("ok ")
            for line in (OUTPUT / "vendor-verify.log").read_text().splitlines()
        ),
        "black_pass": "would be left unchanged" in (OUTPUT / "black.log").read_text(),
        "ruff_pass": "All checks passed!" in (OUTPUT / "ruff.log").read_text(),
        "product_protection_summary": protection["summary"],
        "ugi3_summary": program["summary"],
        "ugi3_consistent_with_protected_precursor": sum(
            r["computed_consistency_pass"] and r["historical_protected_precursor"] for r in rows
        ),
        "ugi3_by_preserved_provider_lane": {k: dict(v) for k, v in sorted(by_lane.items())},
        "family_count": len(families),
        "families_with_complete_program_on_inspected_train": sum(
            f["complete_source_program_qualified_on_inspected_train"] for f in families.values()
        ),
        "family_scope": families,
        "latest_full_suite_reference": {
            "receipt": pin(previous_path),
            "counts": previous["full"]["counts"],
            "scope": previous["full_run_scope"],
            "repeated_this_increment": False,
        },
        "commands": {
            "recount": "uv run python results/phase1/compose_lipid_v8_source_event_validation_v1/summarize.py",
            "source_verification": verification["command"],
            "product_verification": "uv run python -m experiments.phase1.multireaction.compose_lipid_protection --verify results/phase1/compose_lipid_v8_product_protection_v1/result.json",
            "vendor": "make verify",
            "focused": "uv run pytest -q "
            + " ".join(sorted(p for p in inputs if p.startswith("tests/"))),
        },
        "nonclaims": [
            "Known prior validation overlaps are excluded; nonmatching strings do not prove global holdout disjointness.",
            "Single-event computed Ugi-3 support does not establish source-bank assignment, experimental selectivity or execution.",
            "Complete-family qualification remains 1/23; all original families remain in scope.",
            "The full suite remains unsatisfied. The older full run is evidence of unresolved failures, not validation of this source snapshot.",
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
