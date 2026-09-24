"""Recount pinned A3 replay, source ambiguity and repository validation evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def pin(path: Path) -> dict:
    return {
        "path": path.relative_to(REPO).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def read(path: str) -> dict:
    return json.loads((REPO / path).read_text())


def junit(path: Path) -> dict:
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    root = ET.fromstring(data)
    cases = list(root.iter("testcase"))
    failures, errors, skipped = {}, {}, []
    for case in cases:
        name = case.attrib["classname"] + "::" + case.attrib["name"]
        failure, error = case.find("failure"), case.find("error")
        if failure is not None:
            failures[name] = failure.attrib.get("message", "")
        elif error is not None:
            errors[name] = error.attrib.get("message", "")
        elif case.find("skipped") is not None:
            skipped.append(name)
    return {
        "report": pin(path),
        "counts": {
            "tests": len(cases),
            "passed": len(cases) - len(failures) - len(errors) - len(skipped),
            "failures": len(failures),
            "errors": len(errors),
            "skipped_or_xfail": len(skipped),
        },
        "duration_seconds": sum(float(s.attrib["time"]) for s in root.iter("testsuite")),
        "failures": failures,
        "errors": errors,
    }


def main() -> None:
    program_path = "results/phase1/compose_lipid_v8_a3_program_v3/result.json"
    prior_path = "results/phase1/compose_lipid_v8_passerini_validation_v1/result.json"
    source_path = "results/phase1/compose_lipid_v8_a3_source_v1/adjudication-v2.json"
    program, previous, source = read(program_path), read(prior_path), read(source_path)
    prior_identity = read("results/phase1/compose_lipid_v8_a3_validation_v1/prior_identity.json")
    ledger_path = REPO / program["artifacts"]["train_replay.jsonl.gz"]["path"]
    if pin(ledger_path) != program["artifacts"]["train_replay.jsonl.gz"]:
        raise ValueError("replay ledger does not match the recorded program")
    widths, missing, membership = Counter(), Counter(), Counter()
    with gzip.open(ledger_path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    for row in rows:
        if row["status"] == "reconstruction_or_site_hold":
            target = row["reverse_layers"][0][0]
            products = row["forward_layers"][-1]
            membership["target_present"] += target in products
            membership["target_absent"] += target not in products
            widths[str(len(products))] += 1
        missing.update(row["missing_roles"])
        if row["training_admitted"] or row["architecture_qualified"]:
            raise ValueError("diagnostic row was incorrectly admitted")
    family_scope = previous["family_scope"]
    for family, values in prior_identity["by_family"].items():
        family_scope[family]["new_v5_prior_product_identity_audit"] = values
    family_scope["a3_amine_aldehyde_alkyne"]["additional_repeated_program_diagnostic"] = {
        "receipt": pin(REPO / program_path),
        "summary": program["summary"],
        "source_conflicts": source["conflicts"],
    }
    full = junit(OUT / "full-tests.xml.gz")
    focused = junit(OUT / "focused-tests.xml")
    baseline = junit(
        REPO / "results/phase1/compose_lipid_v8_passerini_validation_v1/full-tests.xml.gz"
    )
    before = set(baseline["failures"]) | set(baseline["errors"])
    after = set(full["failures"]) | set(full["errors"])
    input_paths = [
        program_path,
        prior_path,
        source_path,
        "configs/multireaction/compose_lipid_v8_a3_program_v3.json",
        "data/vendor/qualified_a3_source_program_v2.json",
        "forge/assembly/repeated_components.py",
        "forge/corpus/compose_lipid_repeated_source.py",
        "experiments/phase1/multireaction/compose_lipid_repeated_source.py",
        "tests/test_repeated_components.py",
        "tests/test_compose_lipid_repeated_source.py",
        "tests/test_a3_source_boundary.py",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
    ]
    result = {
        "schema_version": "forge.compose_lipid_v8_a3_validation.v1",
        "status": "a3_partial_computed_reconstruction_training_unqualified",
        "training_ready": False,
        "phase1_definition_of_done_met": False,
        "training_rows_admitted": 0,
        "family_count": len(family_scope),
        "families_with_complete_program_on_inspected_train": sum(
            f["complete_source_program_qualified_on_inspected_train"] for f in family_scope.values()
        ),
        "family_scope": family_scope,
        "program_summary": program["summary"],
        "additional_diagnostics": {
            "ambiguous_target_membership": dict(membership),
            "competing_product_counts": dict(widths),
            "missing_role_counts": dict(missing),
        },
        "inputs": {name: pin(REPO / name) for name in input_paths},
        "artifacts": {
            p.relative_to(OUT).as_posix(): pin(p)
            for p in sorted(OUT.rglob("*"))
            if p.is_file() and p.name != "result.json" and "__pycache__" not in p.parts
        },
        "focused": focused,
        "full": full,
        "full_run_scope": "Development run started before the final aldehyde-boundary and cross-version identity corrections. Final focused tests exercise those corrections; no passing full-suite or final full-source-snapshot claim is made.",
        "prior_product_identity_audit": prior_identity["by_family"],
        "baseline": baseline,
        "failure_comparison": {
            "new_failure_or_error_identities": sorted(after - before),
            "removed_failure_or_error_identities": sorted(before - after),
            "scope": "test identity comparison; remaining failures are not all diagnosed",
        },
        "commands": {
            "run": "uv run python -m experiments.phase1.multireaction.compose_lipid_repeated_source --config configs/multireaction/compose_lipid_v8_a3_program_v3.json --output-dir results/phase1/compose_lipid_v8_a3_program_v3",
            "verification": "uv run python -m experiments.phase1.multireaction.compose_lipid_repeated_source --verify results/phase1/compose_lipid_v8_a3_program_v3/result.json",
            "full": "UV_CACHE_DIR=/private/tmp/forge-uv-cache make verify && UV_CACHE_DIR=/private/tmp/forge-uv-cache PYTEST_ADDOPTS='--tb=short --junitxml=/private/tmp/forge-v8-a3-full-tests.xml' make test",
            "focused": "uv run pytest -q tests/test_a3_source_boundary.py tests/test_repeated_components.py tests/test_compose_lipid_repeated_source.py tests/test_compose_lipid_source_program.py tests/test_compose_lipid_components.py tests/test_compose_lipid_pretraining.py tests/test_compose_lipid_representation.py tests/test_compose_lipid.py tests/test_pinned_sources_are_tracked.py",
            "cross_version_identity": "uv run python results/phase1/compose_lipid_v8_a3_validation_v1/audit_prior_identity.py",
            "recount": "uv run python results/phase1/compose_lipid_v8_a3_validation_v1/summarize.py",
        },
        "remaining_holds": program["remaining_holds"],
        "nonclaims": [
            "No whole-family A3 qualification or chemical-precision estimate.",
            "Prior v5 metadata supplies candidates, not experimental or v8 identity labels.",
            "No gates, historical evidence, or provider splits were weakened.",
            "No training, generation, model promotion, or heldout v8 graph inspection.",
        ],
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
