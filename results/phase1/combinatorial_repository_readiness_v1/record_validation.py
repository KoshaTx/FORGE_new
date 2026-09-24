"""Record repository-readiness repairs and their exact validation evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent


def pin(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def xml_path(name: str) -> Path:
    compressed = OUTPUT / f"{name}.xml.gz"
    return compressed if compressed.is_file() else OUTPUT / f"{name}.xml"


def xml_root(path: Path) -> ET.Element:
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return ET.fromstring(data)


def test_report(path: Path) -> dict:
    cases = list(xml_root(path).iter("testcase"))
    failures = []
    skipped = []
    for case in cases:
        test = f"{case.get('classname')}::{case.get('name')}"
        for kind in ("failure", "error"):
            node = case.find(kind)
            if node is not None:
                failures.append({"test": test, "kind": kind, "message": node.get("message", "")})
        if case.find("skipped") is not None:
            skipped.append(test)
    return {
        "xml": pin(path),
        "counts": {
            "tests": len(cases),
            "failures": sum(row["kind"] == "failure" for row in failures),
            "errors": sum(row["kind"] == "error" for row in failures),
            "skipped_or_xfail": len(skipped),
            "passed": len(cases) - len(failures) - len(skipped),
        },
        "failures": failures,
    }


def main() -> None:
    baseline_path = (
        ROOT / "results/phase1/combinatorial_generation_pipeline_validation_v1/result.json"
    )
    baseline = read(baseline_path)
    reports = {
        name: test_report(xml_path(name))
        for name in ("engineering-tests", "graph-cache-tests", "full-tests")
    }
    if xml_path("l1-annotation-tests").is_file():
        reports["l1-annotation-tests"] = test_report(xml_path("l1-annotation-tests"))
    previous = {(row["test"], row["kind"]) for row in baseline["test_reports"]["full"]["failures"]}
    current = {(row["test"], row["kind"]) for row in reports["full-tests"]["failures"]}
    archive_before = read(OUTPUT / "archive_manifest_before.json")
    archive_after = read(ROOT / "provenance/frozen-code/manifest.json")
    before_by_key = {
        (row["original_path"], row["sha256"]): row for row in archive_before["entries"]
    }
    after_by_key = {(row["original_path"], row["sha256"]): row for row in archive_after["entries"]}
    smoke_before = read(
        ROOT
        / "provenance/frozen-code/sha256/a9/a94b3e407399c7ceb6337bfbbc5e3c554a6b0b8c807a0623a5db953b8ecb98fd"
    )
    smoke_after = read(ROOT / "experiments/installation_smoke/experiment.json")
    smoke_before["stages"][0]["inputs"]["project"]["sha256"] = pin(ROOT / "pyproject.toml")[
        "sha256"
    ]
    graph_restoration = read(OUTPUT / "graph_cache_restoration.json")
    cache_authority = read(ROOT / "results/m0_07/oracle_graph_tensor_cache_result.json")
    cache_exact = all(
        pin(ROOT / "results/m0_07" / name)["sha256"] == expected["sha256"]
        for name, expected in cache_authority["artifacts"].items()
    )
    input_paths = [
        baseline_path,
        Path(__file__).resolve(),
        ROOT / "experiments/installation_smoke/experiment.json",
        ROOT / "pyproject.toml",
        ROOT / "provenance/frozen-code/manifest.json",
        ROOT / "results/m0_07/oracle_graph_tensor_cache_result.json",
        ROOT / "results/phase1/combinatorial_generation_pipeline_fresh_v1/result.json",
        ROOT / "results/phase1/combinatorial_generation_pipeline_validation_v1/evidence.json",
        *sorted(path for path in OUTPUT.glob("*.xml") if not Path(str(path) + ".gz").is_file()),
        *sorted(OUTPUT.glob("*.xml.gz")),
        *sorted(path for path in OUTPUT.glob("*.log") if not Path(str(path) + ".gz").is_file()),
        *sorted(OUTPUT.glob("*.log.gz")),
        *[
            OUTPUT / name
            for name in (
                "archive_update.json",
                "installation-smoke-plan.json",
                "dependency_inventory.json",
                "graph_cache_recovery.json",
                "graph_cache_restoration.json",
                "compressed_logs.json",
            )
        ],
    ]
    annotation_receipt = OUTPUT / "l1_annotations_restoration.json"
    annotation_exact = None
    if annotation_receipt.is_file():
        input_paths.append(annotation_receipt)
        annotation_exact = all(
            pin(ROOT / value["path"]) == value for value in read(annotation_receipt)["restored"]
        )
    full_cases = {
        f"{case.get('classname')}::{case.get('name')}": case
        for case in xml_root(xml_path("full-tests")).iter("testcase")
    }
    annotation_resolved = []
    annotation_xml = xml_path("l1-annotation-tests")
    if annotation_xml.is_file():
        for case in xml_root(annotation_xml).iter("testcase"):
            key = f"{case.get('classname')}::{case.get('name')}"
            passed = all(case.find(kind) is None for kind in ("failure", "error", "skipped"))
            prior_failed = any(
                full_cases[key].find(kind) is not None for kind in ("failure", "error")
            )
            if passed and prior_failed:
                annotation_resolved.append(key)
    verified_assets = re.search(
        r"all (\d+) present vendored assets verified", (OUTPUT / "verify.log").read_text()
    )
    result = {
        "schema_version": "forge.combinatorial_repository_readiness.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "authentic_artifact_recovery_global_suite_incomplete",
        "goal_complete": False,
        "phase1_definition_of_done_met": False,
        "inputs": [pin(path) for path in sorted(set(input_paths))],
        "test_reports": reports,
        "commands": {
            "vendor_verification": "UV_CACHE_DIR=/private/tmp/forge-uv-cache make verify",
            "full_tests": "UV_CACHE_DIR=/private/tmp/forge-uv-cache PYTEST_ADDOPTS='--tb=short --junitxml=results/phase1/combinatorial_repository_readiness_v1/full-tests.xml' make test",
            "annotation_tests": "UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run pytest -q tests/test_ugi_chemistry_interface.py tests/test_ugi_morphology_program.py tests/test_ugi_morphology_corpus.py --junitxml=results/phase1/combinatorial_repository_readiness_v1/l1-annotation-tests.xml",
            "pipeline_verification": "UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run python -m experiments.phase1.multireaction.combinatorial_generation_pipeline --verify results/phase1/combinatorial_generation_pipeline_fresh_v1/result.json",
        },
        "later_annotation_test_resolutions": sorted(annotation_resolved),
        "full_suite_comparison": {
            "resolved": [{"test": test, "kind": kind} for test, kind in sorted(previous - current)],
            "new_failures": [
                {"test": test, "kind": kind} for test, kind in sorted(current - previous)
            ],
            "remaining": len(current),
            "scope": "Full suite after smoke/cache repair, before any L1 annotation restoration.",
        },
        "checks": {
            "vendored_assets_verified": int(verified_assets.group(1)) if verified_assets else None,
            "generation_pipeline_verified": read(OUTPUT / "pipeline-verification.log"),
            "prior_archive_entries_preserved": all(
                after_by_key.get(key) == value for key, value in before_by_key.items()
            ),
            "smoke_change_only_current_project_hash": smoke_before == smoke_after,
            "graph_cache_matches_all_original_artifact_hashes": cache_exact,
            "original_cache_result_unchanged": graph_restoration["original_result_unchanged"],
            "restored_annotation_tables_exact_if_present": annotation_exact,
        },
        "limits": [
            "The full suite remains failing; no tests, thresholds or scientific input pins were relaxed.",
            "Annotation tests, if present, postdate the full-suite run; no updated full-suite count is inferred.",
            "The inventory identifies first reported dependencies, not every transitive cause.",
            "Archived source availability does not authenticate current execution against old source pins.",
            "These repairs add no training, generation, model-quality or chemical-realism finding.",
        ],
    }
    with (OUTPUT / "result.json").open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: value["counts"] for name, value in reports.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
