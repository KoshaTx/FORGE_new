"""Summarize preserved JUnit evidence without rerunning tests or changing source pins."""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASELINE = REPO / "results/phase1/compose_lipid_v8_components_validation_v1/full-tests.xml.gz"
MODULES = (
    "ugi3_targeted_aldehyde_evidence",
    "ugi3_targeted_exact_overlay",
    "ugi3_targeted_role_gap_evidence",
    "ugi3_high_leverage_head_terminals",
    "ugi3_second_wave_head_terminals",
    "ugi3_third_wave_head_terminals",
)
SOURCES = [
    "forge/core/input_locations.py",
    "forge/synthesis/evidence/ugi3_targeted_exact_overlay.py",
    *(
        f"forge/synthesis/terminals/ugi3_{wave}_head_terminals.py"
        for wave in ("high_leverage", "second_wave", "third_wave")
    ),
    "tests/test_core_input_locations.py",
    *(f"tests/test_{name}.py" for name in MODULES),
    "experiments/_runtime/historical.py",
    "forge/core/hashing.py",
    "docs/artifact_path_moves.json",
    "provenance/frozen-code/manifest.json",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
]


def pin(path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(REPO).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


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
    suites = list(root.iter("testsuite"))
    return {
        "report": pin(path),
        "counts": {
            "tests": len(cases),
            "passed": len(cases) - len(failures) - len(errors) - len(skipped),
            "failures": len(failures),
            "errors": len(errors),
            "skipped_or_xfail": len(skipped),
            "duration_seconds": sum(float(s.attrib["time"]) for s in suites),
        },
        "failures": failures,
        "errors": errors,
    }


def main() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    test_bodies = {}
    for module in MODULES:
        path = f"tests/test_{module}.py"
        old = subprocess.check_output(["git", "show", f"{head}:{path}"], cwd=REPO)
        current = (REPO / path).read_bytes()

        def functions(payload: bytes) -> list[str]:
            return [
                ast.dump(node, include_attributes=False)
                for node in ast.walk(ast.parse(payload))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]

        unchanged = functions(old) == functions(current)
        assert unchanged, f"scientific test/helper functions changed: {path}"
        test_bodies[path] = {
            "base_git_object": f"{head}:{path}",
            "base_sha256": hashlib.sha256(old).hexdigest(),
            "current": pin(REPO / path),
            "all_test_and_helper_function_asts_unchanged": unchanged,
        }
    baseline = junit(BASELINE)
    focused = junit(OUT / "focused-tests.xml")
    full = junit(OUT / "full-tests.xml.gz")
    before = set(baseline["failures"]) | set(baseline["errors"])
    after = set(full["failures"]) | set(full["errors"])
    # Archives preserve the exact four pre-edit implementations; never rewrite old results.
    archive = json.loads((REPO / "provenance/frozen-code/manifest.json").read_text())
    snapshots = [
        e
        for e in archive["entries"]
        if e["original_path"] in SOURCES
        and e["source"].get("reason")
        == "Input location authentication and replay portability; preserve prior source bytes"
    ]
    assert len(snapshots) == 4, "all four pre-edit implementations must remain recoverable"
    for snapshot in snapshots:
        assert pin(REPO / snapshot["blob_path"])["sha256"] == snapshot["sha256"]
    inputs = [REPO / name for name in SOURCES]
    inputs.extend(REPO / e["blob_path"] for e in snapshots)
    inputs.extend(
        REPO / f"results/phase1/compose_lipid_v8_{name}_v1/result.json"
        for name in ("import", "representation", "pretraining", "components")
    )
    inputs.extend(sorted((REPO / "configs/route").glob("phase1_ugi3_*head_terminal_audit_v1.json")))
    inputs.extend(
        REPO / f"configs/route/phase1_ugi3_{name}_audit_v1.json"
        for name in ("targeted_aldehyde_evidence", "targeted_role_gap_evidence")
    )
    report = {
        "schema_version": "forge.compose_lipid_v8_readiness_portability.v1",
        "status": "historical_replay_portability_improved_training_unqualified",
        "training_ready": False,
        "phase1_definition_of_done_met": False,
        "random_sampling_used": False,
        "git_head": head,
        "training_calls": 0,
        "generation_calls": 0,
        "call_accounting_scope": (
            "No training or generation experiment was launched. Test-internal fixtures "
            "are outside these experiment counters."
        ),
        "checks": {
            "vendored_assets_verified": sum(
                line.startswith("  ok ") for line in (OUT / "verify.log").read_text().splitlines()
            ),
            "vendor_verification_pass": "all 30 present vendored assets verified"
            in (OUT / "verify.log").read_text(),
            "ruff_check_pass": "All checks passed!" in (OUT / "ruff.log").read_text(),
            "black_check_pass": "12 files would be left unchanged"
            in (OUT / "black.log").read_text(),
            "all_new_location_tests_pass": not any(
                name.startswith("tests.test_core_input_locations::") for name in after
            ),
            "all_v8_tests_pass": not any(
                name.startswith("tests.test_compose_lipid") for name in after
            ),
        },
        "inputs": {p.relative_to(REPO).as_posix(): pin(p) for p in inputs},
        "artifacts": {
            p.name: pin(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "result.json"
        },
        "baseline": baseline,
        "focused": focused,
        "full": full,
        "failure_comparison": {
            "removed_failure_or_error_identities": sorted(before - after),
            "new_failure_or_error_identities": sorted(after - before),
            "claim": "Test identity comparison; remaining causes are not all diagnosed.",
        },
        "prechange_source_archive_entries": snapshots,
        "test_function_preservation": test_bodies,
        "scope": [
            "Authenticate relocated source by its declared path and exact SHA-256.",
            "Normalize only authenticated inputs.*.path in historical result comparisons.",
            "Keep all chemistry assertions, source/data digests and ledger checks intact.",
            "Read source as evidence bytes; do not execute the archived modules.",
            "No supplier retrieval, new procurement qualification or training occurred.",
            "Existing v8 source-program and precursor qualification holds remain in force.",
        ],
        "commands": {
            "vendor": "UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run make verify",
            "full": (
                "UV_CACHE_DIR=/private/tmp/forge-uv-cache "
                "PYTEST_ADDOPTS='--tb=short "
                "--junitxml=/private/tmp/forge-v8-portability-full-tests.xml' uv run make test"
            ),
            "focused": (
                "UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run pytest -q "
                "tests/test_core_input_locations.py tests/test_core_provenance_archive.py "
                "tests/test_pinned_sources_are_tracked.py "
                + " ".join(f"tests/test_{name}.py" for name in MODULES)
                + " --tb=short"
            ),
            "reproduce_summary": (
                "python3 results/phase1/compose_lipid_v8_readiness_portability_v1/summarize.py"
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
