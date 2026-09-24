"""Benchmark preflight may relocate exact source evidence, never data or execution."""

import hashlib
import json

import pytest

from forge.synthesis.engine.single_step_proposal_benchmark import (
    BenchmarkContractError,
    _artifact_evidence,
)
from tests.test_single_step_manifest_source_relocation import source_archive  # noqa: F401


def test_benchmark_reads_exact_archived_source_without_executing_it(source_archive):  # noqa: F811
    repo, blob, spec, _, _ = source_archive
    assert _artifact_evidence(repo, spec, label="runner_source") == blob.resolve()


@pytest.mark.parametrize(
    "mutation", ["bytes", "logical_path", "digest", "duplicate", "extra_field", "data", "symlink"]
)
def test_benchmark_archive_changes_cannot_authenticate(source_archive, mutation):  # noqa: F811
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
    with pytest.raises(BenchmarkContractError):
        _artifact_evidence(repo, spec, label="runner_source")


def test_reviewed_source_move_requires_unchanged_frozen_bytes(tmp_path):
    moved = tmp_path / "forge/new.py"
    moved.parent.mkdir()
    moved.write_text("raise RuntimeError('source is evidence only')\n")
    spec = {"path": "src/forge/old.py", "sha256": hashlib.sha256(moved.read_bytes()).hexdigest()}
    moves = tmp_path / "docs/artifact_path_moves.json"
    moves.parent.mkdir()
    moves.write_text(json.dumps({"moves": {spec["path"]: "forge/new.py"}}))
    assert _artifact_evidence(tmp_path, spec, label="runner_source") == moved.resolve()
    moved.write_text("changed")
    with pytest.raises(BenchmarkContractError):
        _artifact_evidence(tmp_path, spec, label="runner_source")


@pytest.mark.parametrize("suffix", ["json", "csv.gz"])
def test_data_cannot_use_the_source_move_ledger(tmp_path, suffix):
    moved = tmp_path / f"new.{suffix}"
    moved.write_bytes(b"exact data bytes")
    spec = {"path": f"old.{suffix}", "sha256": hashlib.sha256(moved.read_bytes()).hexdigest()}
    moves = tmp_path / "docs/artifact_path_moves.json"
    moves.parent.mkdir()
    moves.write_text(json.dumps({"moves": {spec["path"]: moved.name}}))
    with pytest.raises(BenchmarkContractError, match="missing"):
        _artifact_evidence(tmp_path, spec, label="data")


@pytest.mark.parametrize("path", ["../outside.py", "forge/../old.py", "", None])
def test_source_paths_cannot_escape_or_change_logical_identity(tmp_path, path):
    with pytest.raises(BenchmarkContractError, match="path"):
        _artifact_evidence(tmp_path, {"path": path, "sha256": "0" * 64}, label="runner")


def test_data_at_its_declared_location_still_requires_exact_digest(tmp_path):
    data = tmp_path / "data.json"
    data.write_bytes(b"original")
    spec = {"path": data.name, "sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
    assert _artifact_evidence(tmp_path, spec, label="data") == data.resolve()
    data.write_bytes(b"changed")
    with pytest.raises(BenchmarkContractError, match="SHA-256 mismatch"):
        _artifact_evidence(tmp_path, spec, label="data")
