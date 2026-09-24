"""Recount source qualification and completed checks without admitting training."""

import gzip
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from forge.core.hashing import sha256_file  # noqa: E402


def pin(path):
    path = (ROOT / path).resolve()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": str(sha256_file(path))}


def read(path):
    return json.loads((ROOT / path).read_text())


def tests(path):
    raw = path.read_bytes()
    cases = list(
        ET.fromstring(gzip.decompress(raw) if path.suffix == ".gz" else raw).iter("testcase")
    )
    bad = {}
    counts = {"tests": len(cases), "passed": 0, "failures": 0, "errors": 0, "skipped_or_xfail": 0}
    for case in cases:
        name = case.get("classname") + "::" + case.get("name")
        for kind, label in (
            ("failure", "failures"),
            ("error", "errors"),
            ("skipped", "skipped_or_xfail"),
        ):
            value = case.find(kind)
            if value is not None:
                counts[label] += 1
                if kind != "skipped":
                    bad[name] = {"kind": kind, "message": value.get("message")}
                break
        else:
            counts["passed"] += 1
    return {"counts": counts, "failure_or_error_details": bad}


def main():
    previous_path = Path("results/phase1/compose_lipid_v8_precursor_validation_v1/result.json")
    audit_path = Path("results/phase1/compose_lipid_v8_precursor_audit_v2/result.json")
    program_path = Path("results/phase1/compose_lipid_v8_ugi4_program_v1/result.json")
    source_dir = Path("results/phase1/compose_lipid_v8_ugi4_source_v1")
    exclusion_path = source_dir / "exclusion_diagnostic.json"
    dictionary_path = source_dir / "prior_dictionary_concordance.json"
    abstention_path = Path("results/phase1/compose_lipid_v8_epoxide_source_v1/adjudication.json")
    previous, audit, program = read(previous_path), read(audit_path), read(program_path)
    exclusion, dictionary = read(exclusion_path), read(dictionary_path)
    holdout_request = read(OUTPUT / "holdout-request.json")
    assert holdout_request["inputs"]["audit"] == pin(audit_path)
    assert holdout_request["status"] == "prepared_not_sent"
    checks = read(OUTPUT / "checks.json")
    assert "full-tests" in checks
    assert all(c["exit_code"] == 0 for name, c in checks.items() if name != "full-tests")
    snapshot = read(OUTPUT / "source-snapshot.json")
    comparison = {
        path: str(sha256_file(ROOT / path)) == digest for path, digest in snapshot.items()
    }
    assert all(comparison.values()), "source changed after validation started"
    assert all(read(OUTPUT / "snapshot-comparison.json").values())
    for name in ("full-tests.xml", "full-tests.log"):
        path = OUTPUT / name
        if path.exists():
            (OUTPUT / (name + ".gz")).write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    focused, full = tests(OUTPUT / "focused-tests.xml"), tests(OUTPUT / "full-tests.xml.gz")
    assert not focused["failure_or_error_details"]
    old_bad = set(previous["full"]["failure_or_error_details"])
    new_bad = set(full["failure_or_error_details"])
    family_scope = previous["family_scope"]
    summary = audit["summary"]
    assert set(summary["by_family"]) == set(family_scope)
    totals = summary["totals"]
    assert totals["preparation_rows"] == (
        totals["program_rows_evaluated"] + totals["preparation_rows_without_program_evidence"]
    )
    assert totals["program_rows_evaluated"] == (
        totals["exact_program_rows"] + totals["program_rows_not_exact"]
    )
    assert totals["exact_program_rows"] == (
        totals["exact_rows_clear_of_known_exclusions"]
        + totals["exact_rows_with_known_protected_precursor"]
    )
    with gzip.open(ROOT / audit["artifacts"]["rows.jsonl.gz"]["path"], "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == totals["inspection_rows"]
    assert sum(row["exact_program_evidence"] for row in rows) == totals["exact_program_rows"]
    assert sum(row["clear_of_known_exclusions"] for row in rows) == (
        totals["exact_rows_clear_of_known_exclusions"]
    )
    assert not any(row["training_admitted"] for row in rows)
    with gzip.open(ROOT / program["artifacts"]["programs.jsonl.gz"]["path"], "rt") as stream:
        programs = [json.loads(line) for line in stream]
    assert len(programs) == program["summary"]["rows"]
    assert sum(row["computed_consistency_pass"] for row in programs) == (
        program["summary"]["computed_consistency_pass"]
    )
    assert not any(row["training_admitted"] for row in programs)
    assert exclusion["inputs"]["program_result"] == pin(program_path)
    assert exclusion["rows"] == program["summary"]["by_status"]["excluded_no_exact_inverse"]
    assert exclusion["reasons_per_target"] == {"role_rejected:amine_head:count=3": 287}
    for family, counts in summary["by_family"].items():
        family_scope[family]["cross_family_precursor_identity_audit"] = {
            "receipt": pin(audit_path),
            "counts": counts,
            "complete_holdout_qualification": False,
        }
    family_scope["aldehyde_ugi4"]["additional_condensation_program_evidence"] = {
        "receipt": pin(program_path),
        "summary": program["summary"],
        "source_bank_membership_qualified": False,
        "excluded_multiplicity_diagnostic": pin(exclusion_path),
    }
    input_paths = {
        previous_path,
        audit_path,
        program_path,
        exclusion_path,
        dictionary_path,
        abstention_path,
        Path(__file__).relative_to(ROOT),
        Path("docs/COMPOSE_LIPID_V8_PRETRAINING.md"),
        Path("docs/COMPOSE_LIPID_V8_SOURCE_CONTRACT_REQUEST.md"),
        Path("results/phase1/compose_lipid_v8_precursor_validation_v1/missing-inputs.json"),
    }
    input_paths.update(Path(p) for p in snapshot)
    input_paths.update(p for p in source_dir.iterdir() if p.is_file())
    input_paths.update(Path(p["path"]) for p in read(abstention_path)["assets"].values())
    for result in (audit, program):
        input_paths.add(Path(result["config"]["path"]))
        input_paths.update(Path(p["path"]) for p in result["artifacts"].values())
    config = read(audit["config"]["path"])
    input_paths.add(Path(config["protection_result"]["path"]))
    input_paths.update(Path(p["path"]) for p in config["program_results"])
    result = {
        "schema_version": "forge.compose_lipid_ugi4_validation.v1",
        "status": (
            "blocked_source_contract_holdout_and_repository_inputs"
            if not new_bad - old_bad
            else "blocked_with_new_test_regressions"
        ),
        "phase1_definition_of_done_met": False,
        "training_ready": False,
        "training_calls": 0,
        "training_rows_admitted": 0,
        "seed": 0,
        "random_sampling_used": False,
        "inputs": {str(p): pin(p) for p in sorted(input_paths)},
        "artifacts": {
            p.name: pin(p)
            for p in sorted(OUTPUT.iterdir())
            if p.is_file() and p.name not in {"result.json", "full-tests.xml", "full-tests.log"}
        },
        "family_scope": family_scope,
        "ugi4_summary": program["summary"],
        "precursor_summary": summary,
        "excluded_ugi4_multiplicity": exclusion["reasons_per_target"],
        "older_dictionary_decision": dictionary["decision"],
        "epoxide_evidence_disposition": read(abstention_path)["disposition"],
        "holdout_identity_request": {
            "receipt": pin(OUTPUT / "holdout-request.json"),
            "rows": holdout_request["rows"],
            "distinct_unresolved_family_role_labels": holdout_request[
                "distinct_unresolved_family_role_labels"
            ],
            "malformed_metadata_rows": holdout_request["malformed_metadata_rows"],
            "sent": False,
        },
        "focused": focused,
        "full": full,
        "failure_comparison": {
            "new_failure_or_error_ids": sorted(new_bad - old_bad),
            "resolved_failure_or_error_ids": sorted(old_bad - new_bad),
        },
        "vendor_assets_verified": sum(
            line.strip().startswith("ok ")
            for line in (OUTPUT / "vendor-verify.log").read_text().splitlines()
        ),
        "implementation_snapshot_unchanged": all(comparison.values()),
        "checks": checks,
        "remaining_holds": [
            "complete_unambiguous_program_evidence_missing_for_remaining_family_populations",
            "unseen_precursor_chemical_identities_and_source_split_audit_unresolved",
            "source_architecture_assignments_unresolved_where_recorded",
            "final_balanced_train_dataset_unbuilt",
            "repository_tests_fail_missing_frozen_inputs_and_source_authentication_preconditions",
        ],
        "nonclaims": [
            "Exact replay is computed consistency, not experimental success or route certification.",
            "The three-handle amine exclusions do not establish chemical invalidity or ambiguous attachment.",
            "The earlier conditional A3 diagnostic is preserved without promotion to complete program evidence.",
            "No corpus-wide chemical precision estimate follows from these source controls and tests.",
            "All unseen-precursor tuples remain unresolved; zero label contradictions do not establish disjointness.",
            "No evaluation graphs, training weights, new model training or paid remote job were used.",
        ],
        "commands": {
            "program_verification": "uv run python -m experiments.phase1.multireaction.compose_lipid_condensation_event verify --result results/phase1/compose_lipid_v8_ugi4_program_v1/result.json",
            "audit_verification": "uv run python -m experiments.phase1.multireaction.compose_lipid_precursor_audit --verify results/phase1/compose_lipid_v8_precursor_audit_v2/result.json",
            "validation": "uv run python results/phase1/compose_lipid_v8_ugi4_validation_v1/run_checks.py",
            "recount": "uv run python results/phase1/compose_lipid_v8_ugi4_validation_v1/report.py",
        },
    }
    assert result["vendor_assets_verified"] == 30
    (OUTPUT / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(result["status"], focused["counts"], full["counts"], pin(OUTPUT / "result.json"))


if __name__ == "__main__":
    main()
