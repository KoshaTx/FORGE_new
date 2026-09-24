"""Historical source can move; its bytes, identity and data boundaries cannot."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from forge.synthesis.engine.single_step_benchmark_manifest import (
    SingleStepBenchmarkManifestError,
    _implementation_evidence,
    build_manifest_payloads,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/single_step_proposal_lane_qualification_manifest_v1.json"


@pytest.fixture
def source_archive(tmp_path):
    # A valid pin must not cause the archived module to be imported or executed.
    blob = tmp_path / "provenance/frozen-code/sha256/blob"
    blob.parent.mkdir(parents=True)
    blob.write_text("raise RuntimeError('archived source must never execute')\n")
    spec = {
        "path": "scripts/original_builder.py",
        "sha256": hashlib.sha256(blob.read_bytes()).hexdigest(),
    }
    document = {
        "schema_version": "forge.historical_pin_archive.v1",
        "entries": [
            {
                "original_path": spec["path"],
                "sha256": spec["sha256"],
                "blob_path": blob.relative_to(tmp_path).as_posix(),
                "source": {"kind": "fixture"},
            }
        ],
    }
    manifest = tmp_path / "provenance/frozen-code/manifest.json"
    manifest.write_text(json.dumps(document))
    return tmp_path, blob, spec, manifest, document


def test_exact_archive_bytes_are_read_as_evidence_without_execution(source_archive):
    repo, blob, spec, _, _ = source_archive
    assert _implementation_evidence(repo, spec, label="builder") == blob.resolve()


@pytest.mark.parametrize(
    "mutation", ["bytes", "logical_path", "digest", "duplicate", "extra_field", "data", "symlink"]
)
def test_source_substitution_and_archive_tampering_fail_closed(source_archive, mutation):
    repo, blob, spec, manifest, document = source_archive
    if mutation == "bytes":
        blob.write_text("changed")
    elif mutation == "logical_path":
        document["entries"][0]["original_path"] = "scripts/another.py"
    elif mutation == "digest":
        spec["sha256"] = "0" * 64
    elif mutation == "duplicate":
        document["entries"].append(dict(document["entries"][0]))
    elif mutation == "extra_field":
        document["entries"][0]["allow_changed_hash"] = True
    elif mutation == "data":
        spec["path"] = "data/input.json"
        document["entries"][0]["original_path"] = spec["path"]
    else:
        actual = blob.with_name("actual")
        blob.rename(actual)
        blob.symlink_to(actual)
    manifest.write_text(json.dumps(document))
    with pytest.raises(SingleStepBenchmarkManifestError):
        _implementation_evidence(repo, spec, label="builder")


def test_reviewed_move_still_requires_original_digest(tmp_path):
    moved = tmp_path / "forge/new.py"
    moved.parent.mkdir()
    moved.write_text("source bytes")
    spec = {"path": "src/forge/old.py", "sha256": hashlib.sha256(moved.read_bytes()).hexdigest()}
    moves = tmp_path / "docs/artifact_path_moves.json"
    moves.parent.mkdir()
    moves.write_text(json.dumps({"moves": {spec["path"]: "forge/new.py"}}))
    assert _implementation_evidence(tmp_path, spec, label="builder") == moved.resolve()
    moved.write_text("changed")
    with pytest.raises(SingleStepBenchmarkManifestError):
        _implementation_evidence(tmp_path, spec, label="builder")


def test_rebuilt_scientific_payloads_equal_all_frozen_payloads():
    outputs = build_manifest_payloads(REPO, CONFIG)
    config = json.loads(CONFIG.read_text())
    frozen = REPO / config["output_directory"]
    # Equality covers every target, scoring truth and visibility mask, not counts alone.
    for name in ("lane_targets", "scoring_truth", "visibility_masks"):
        assert outputs[name] == (frozen / f"{name}.json.gz").read_bytes()
    result = json.loads(outputs["result"])
    assert result["implementation"] == config["implementation"]
    assert result["execution_provenance"]["archived_code_executed"] is False
    for label, binding in result["implementation_source_bindings"].items():
        assert (
            hashlib.sha256((REPO / binding["path"]).read_bytes()).hexdigest() == binding["sha256"]
        )
        assert binding["sha256"] == config["implementation"][label]["sha256"]
    current = result["execution_provenance"]["current_modules"]
    for item in current.values():
        assert hashlib.sha256((REPO / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert (
        json.loads(gzip.decompress(outputs["lane_targets"]))["known_routes_exposed_to_lanes"]
        is False
    )
