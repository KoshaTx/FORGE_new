"""Verify recovered upstream bytes and recount the completed source-recovery validation."""

import gzip
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from forge.core.hashing import sha256_file  # noqa: E402


def read(path):
    return json.loads((ROOT / path).read_text())


def pin(path):
    path = (ROOT / path).resolve()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": str(sha256_file(path))}


def tests(path):
    raw = path.read_bytes()
    cases = ET.fromstring(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    counts = dict(tests=0, passed=0, failures=0, errors=0, skipped_or_xfail=0)
    details = {}
    for case in cases.iter("testcase"):
        counts["tests"] += 1
        for tag, label in (
            ("failure", "failures"),
            ("error", "errors"),
            ("skipped", "skipped_or_xfail"),
        ):
            child = case.find(tag)
            if child is not None:
                counts[label] += 1
                if tag != "skipped":
                    details[case.get("classname") + "::" + case.get("name")] = {
                        "kind": tag,
                        "message": child.get("message"),
                    }
                break
        else:
            counts["passed"] += 1
    return {"counts": counts, "failure_or_error_details": details}


def main():
    manifest = read(OUTPUT / "asset-manifest.json")
    tree = read(OUTPUT / "commit-tree.json")
    commit = read(OUTPUT / "commit.json")
    assert commit["sha"] == manifest["commit"]
    assert tree["sha"] == commit["commit"]["tree"]["sha"]
    assert tree["truncated"] is False
    blobs = {entry["path"]: entry["sha"] for entry in tree["tree"] if entry["type"] == "blob"}
    for asset in manifest["assets"]:
        data = (ROOT / asset["path"]).read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert digest == asset["git_blob_sha1"] == blobs[asset["upstream_path"]]
        assert hashlib.sha256(data).hexdigest() == asset["sha256"]
    before = read(OUTPUT / "before-pins.json")
    for path, digest in before["files"].items():
        assert str(sha256_file(OUTPUT / "before" / path)) == digest
    archive = read("provenance/frozen-code/manifest.json")
    old_digest = before["files"]["forge/synthesis/engine/single_step_proposal_benchmark.py"]
    original_entries = [
        entry
        for entry in archive["entries"]
        if not (
            entry["original_path"] == "forge/synthesis/engine/single_step_proposal_benchmark.py"
            and entry["sha256"] == old_digest
        )
    ]
    assert len(original_entries) == len(archive["entries"]) - 1
    # The edit retained all entries and sorted them. The recorded pre-edit digest alone does
    # not reconstruct the old array/key order, so do not claim a byte-for-byte replay of it.
    for entry in archive["entries"]:
        assert str(sha256_file(ROOT / entry["blob_path"])) == entry["sha256"]
    checks = read(OUTPUT / "checks.json")
    assert "full-tests" in checks
    assert all(
        value["exit_code"] == 0
        for name, value in checks.items()
        if name not in {"full-tests", "historical-benchmark-tests"}
    )
    snapshot = read(OUTPUT / "source-snapshot.json")
    assert all(str(sha256_file(ROOT / path)) == digest for path, digest in snapshot.items())
    assert all(read(OUTPUT / "snapshot-comparison.json").values())
    for name in ("full-tests.log", "full-tests.xml"):
        (OUTPUT / (name + ".gz")).write_bytes(gzip.compress((OUTPUT / name).read_bytes(), mtime=0))
    focused = tests(OUTPUT / "focused-tests.xml")
    full = tests(OUTPUT / "full-tests.xml.gz")
    assert focused["counts"]["passed"] == focused["counts"]["tests"]
    baseline_path = Path("results/phase1/compose_lipid_v8_ugi4_validation_v1/result.json")
    old_bad = set(read(baseline_path)["full"]["failure_or_error_details"])
    new_bad = set(full["failure_or_error_details"])
    upstream_audit = read(
        OUTPUT / "upstream/audits/post_instruction_generator_splits_v8_1_independent.json"
    )
    split_config = read(
        OUTPUT / "upstream/configs/corpus/post_instruction_generator_splits_v8_1.json"
    )
    needed = [
        {
            "directory": str(Path(split_config["component_manifest_receipt"]).parent),
            "receipt_sha256": split_config["component_manifest_receipt_sha256"],
            "files": ["receipt.json", "components.jsonl.gz", "summary.json"],
        },
        {
            "directory": split_config["output_dir"],
            "receipt_sha256": upstream_audit["audited_receipt_sha256"],
            "files": [
                "receipt.json",
                "assignments.jsonl.gz",
                "split_groups.jsonl.gz",
                "selected_groups.json",
                "summary.json",
            ],
        },
    ]
    assert all(
        not any(path.startswith(item["directory"] + "/") for path in blobs) for item in needed
    )
    request = {
        "schema_version": "forge.compose_lipid_full_universe_source_request.v1",
        "status": "prepared_not_sent",
        "upstream_commit": manifest["commit"],
        "upstream_evidence": pin(OUTPUT / "asset-manifest.json"),
        "missing_selection_artifacts": needed,
        "selection_scope_rows": upstream_audit["targets"],
        "full_universe_mapping_also_required": True,
        "full_universe_required_rows": 3182837,
        "full_universe_import": pin("results/phase1/compose_lipid_v8_import_v1/result.json"),
        "required_full_universe_fields": [
            "target_id",
            "canonical_complete_precursors",
            "roles",
            "multiplicities",
            "ordered_program_and_sites",
            "source_studies",
            "registry_and_construction_witness_pins",
        ],
        "unknown_full_universe_receipt_hash": None,
        "training_admitted": False,
    }
    (OUTPUT / "full-universe-source-request.json").write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n"
    )
    result = {
        "schema_version": "forge.compose_lipid_source_recovery.v1",
        "status": "upstream_reports_recovered_raw_component_and_split_artifacts_missing",
        "seed": 0,
        "training_calls": 0,
        "training_ready": False,
        "provider_splits_changed": False,
        "upstream_code_executed": False,
        "source_assets_verified": len(manifest["assets"]),
        "upstream_commit": manifest["commit"],
        "upstream_audit_claims_independently_replayed": False,
        "upstream_reported_old_panel_defect": {
            key: upstream_audit[key]
            for key in [
                "old_v8_unseen_precursor_panel_targets",
                "old_v8_panel_targets_without_structurally_unseen_component",
            ]
        },
        "benchmark_source_relocation": "exact_archived_source_evidence_only_data_pins_unchanged",
        "archive_blob_hashes_verified": True,
        "archive_before_bytes_independently_replayed": False,
        "benchmark_next_missing_input": "results/phase1/ugi3_high_leverage_route_evidence_worklist_v1/manifest.json",
        "focused": focused,
        "full": full,
        "new_failure_or_error_ids": sorted(new_bad - old_bad),
        "resolved_failure_or_error_ids": sorted(old_bad - new_bad),
        "inputs": {
            "baseline": pin(baseline_path),
            "source_archive": pin("provenance/frozen-code/manifest.json"),
        },
        "artifacts": {
            str(path.relative_to(OUTPUT)): pin(path)
            for path in sorted(OUTPUT.rglob("*"))
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.name not in {"result.json", "full-tests.log", "full-tests.xml"}
        },
        "validation_scope": "source_snapshot_only_universe_module_validated_separately",
        "phase1_definition_of_done_met": False,
    }
    (OUTPUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "focused": focused["counts"],
                "full": full["counts"],
                "new_failures": len(new_bad - old_bad),
            }
        )
    )


if __name__ == "__main__":
    main()
