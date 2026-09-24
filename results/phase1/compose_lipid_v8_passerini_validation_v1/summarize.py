"""Recount preserved validation evidence and the explicitly qualified family scope."""

from __future__ import annotations

import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def pin(path: Path) -> dict:
    return {
        "path": path.relative_to(REPO).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def read(relative: str) -> dict:
    return json.loads((REPO / relative).read_text())


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
            "duration_seconds": sum(float(s.attrib["time"]) for s in root.iter("testsuite")),
        },
        "failures": failures,
        "errors": errors,
    }


def main() -> None:
    program_path = "results/phase1/compose_lipid_v8_passerini_program_v1/result.json"
    replay_path = "results/phase1/compose_lipid_v8_passerini_replay_v1/result.json"
    key_path = "data/source_cache/compose_lipid_drive_2026-09-18/family_decomposition_key.json"
    previous_path = "results/phase1/compose_lipid_v8_pretraining_v1/result.json"
    representation_path = "results/phase1/compose_lipid_v8_representation_v1/result.json"
    components_path = "results/phase1/compose_lipid_v8_components_v1/result.json"
    source_path = "results/phase1/compose_lipid_v8_passerini_source_v1/adjudication.json"
    program, replay = read(program_path), read(replay_path)
    previous, representation, components = (
        read(previous_path),
        read(representation_path),
        read(components_path),
    )
    key = read(key_path)
    family_scope = {}
    for family, contract in key.items():
        qualified = family == program["family"] and program["source_event_program_qualified"]
        family_scope[family] = {
            "rows_inspected": previous["by_family"][family]["rows"],
            "representation": representation["by_family"][family],
            "prior_component_audit": components["by_family"][family],
            "complete_source_program_qualified_on_inspected_train": qualified,
            "qualification_receipt": pin(REPO / program_path) if qualified else None,
            "architecture_subfamilies": [r["id"] for r in contract["architecture_subfamilies"]],
            "declared_variable_metadata": contract["variable"],
            "source_program_invariant": contract["reaction_program_invariant"],
            "source_never_collapse": contract["never_collapse"],
            "training_admitted_rows": 0,
        }
    compared = (
        "summary",
        "source_controls",
        "policy",
        "implementation",
        "evidence_basis",
        "scope",
        "remaining_holds",
        "rdkit_version",
    )
    equality = {field: program[field] == replay[field] for field in compared}
    equality["artifact_hashes"] = {
        name: p["sha256"] for name, p in program["artifacts"].items()
    } == {name: p["sha256"] for name, p in replay["artifacts"].items()}
    if not all(equality.values()):
        raise ValueError("fresh program replay differs from its first execution")
    baseline = junit(
        REPO / "results/phase1/compose_lipid_v8_readiness_portability_v1/full-tests.xml.gz"
    )
    focused = junit(OUT / "focused-tests.xml")
    full = junit(OUT / "full-tests.xml.gz")
    before = set(baseline["failures"]) | set(baseline["errors"])
    after = set(full["failures"]) | set(full["errors"])
    input_paths = [
        program_path,
        replay_path,
        key_path,
        previous_path,
        representation_path,
        components_path,
        source_path,
        "configs/multireaction/compose_lipid_v8_passerini_program_v1.json",
        "forge/assembly/atom_map_variant.py",
        "forge/corpus/compose_lipid_source_program.py",
        "experiments/phase1/multireaction/compose_lipid_source_program.py",
        "tests/test_compose_lipid_source_program.py",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
    ]
    result = {
        "schema_version": "forge.compose_lipid_v8_source_program_validation.v1",
        "status": "passerini_program_qualified_training_unqualified",
        "training_ready": False,
        "training_rows_admitted": 0,
        "phase1_definition_of_done_met": False,
        "family_count": len(key),
        "families_with_complete_program_on_inspected_train": sum(
            f["complete_source_program_qualified_on_inspected_train"] for f in family_scope.values()
        ),
        "family_scope": family_scope,
        "inputs": {path: pin(REPO / path) for path in input_paths},
        "artifacts": {
            p.name: pin(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "result.json"
        },
        "source_program": program["summary"],
        "fresh_replay_equality": equality,
        "replay_comparison_excludes": ["execution timestamp", "output directory paths"],
        "baseline": baseline,
        "focused": focused,
        "full": full,
        "failure_comparison": {
            "removed_failure_or_error_identities": sorted(before - after),
            "new_failure_or_error_identities": sorted(after - before),
            "claim": "Identity comparison only; remaining failures are not all diagnosed.",
        },
        "checks": {
            "vendored_assets_verified": sum(
                line.startswith("  ok ") for line in (OUT / "verify.log").read_text().splitlines()
            ),
            "vendor_verification_pass": "all 30 present vendored assets verified"
            in (OUT / "verify.log").read_text(),
            "black_check_pass": "4 files would be left unchanged"
            in (OUT / "black.log").read_text(),
            "ruff_check_pass": "All checks passed!" in (OUT / "ruff.log").read_text(),
            "all_focused_checks_pass": not focused["failures"] and not focused["errors"],
            "all_v8_tests_pass": not any(
                name.startswith("tests.test_compose_lipid") for name in after
            ),
            "full_repository_tests_pass": not after,
        },
        "commands": read(str((OUT / "commands.json").relative_to(REPO))),
        "remaining_holds": program["remaining_holds"],
        "seed": 0,
        "random_sampling_used": False,
        "training_calls": 0,
        "generation_calls": 0,
        "heldout_graphs_parsed": False,
        "call_accounting_scope": "No training or generation experiment launched; excludes test fixtures.",
        "nonclaims": [
            "Source program qualification is computed transform consistency, not experimental execution.",
            "Two primary controls do not estimate chemical decomposition precision over the corpus.",
            "Source mechanism atom origins are not physical isotope-tracing measurements.",
            "No complete provider component-ID-to-structure mapping or chemical holdout is qualified.",
            "No L2/procurement dossier, biological claim or model improvement is established.",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
