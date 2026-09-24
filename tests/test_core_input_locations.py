from __future__ import annotations

import json
from copy import deepcopy

import pytest

from forge.core.hashing import sha256_file
from forge.core.input_locations import input_location_matches, results_match_after_input_relocation


@pytest.fixture
def archived(tmp_path):
    asset = "src/forge/original.py"
    blob = tmp_path / "provenance/frozen-code/sha256/blob"
    blob.parent.mkdir(parents=True)
    blob.write_text("exact original source\n")
    digest = str(sha256_file(blob))
    manifest = tmp_path / "provenance/frozen-code/manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "forge.historical_pin_archive.v1",
                "entries": [
                    {
                        "original_path": asset,
                        "sha256": digest,
                        "blob_path": blob.relative_to(tmp_path).as_posix(),
                        "source": {"kind": "fixture"},
                    }
                ],
            }
        )
    )
    return tmp_path, asset, digest, blob


def test_archive_requires_exact_original_path_and_digest(archived):
    repo, asset, digest, path = archived
    assert input_location_matches(repo, path, asset, digest)
    assert not input_location_matches(repo, path, "src/forge/different.py", digest)
    assert not input_location_matches(repo, path, asset, "0" * 64)
    path.write_text("changed")
    assert not input_location_matches(repo, path, asset, digest)


def test_duplicate_archive_identity_fails_closed(archived):
    repo, asset, digest, path = archived
    manifest = repo / "provenance/frozen-code/manifest.json"
    value = json.loads(manifest.read_text())
    value["entries"].append(value["entries"][0])
    manifest.write_text(json.dumps(value))
    assert not input_location_matches(repo, path, asset, digest)


@pytest.mark.parametrize("mutation", ["absolute", "traversal", "source", "extra", "entries"])
def test_malformed_archive_binding_fails_closed(archived, mutation):
    repo, asset, digest, path = archived
    manifest = repo / "provenance/frozen-code/manifest.json"
    value = json.loads(manifest.read_text())
    entry = value["entries"][0]
    if mutation == "absolute":
        entry["blob_path"] = str(path)
    elif mutation == "traversal":
        entry["blob_path"] = "provenance/../" + entry["blob_path"]
    elif mutation == "source":
        entry["source"] = "not an attribution record"
    elif mutation == "extra":
        entry["unknown"] = True
    else:
        value["entries"] = {"entry": entry}
    manifest.write_text(json.dumps(value))
    assert not input_location_matches(repo, path, asset, digest)


def test_move_requires_reviewed_source_path_and_exact_bytes(tmp_path):
    path = tmp_path / "forge/new.py"
    path.parent.mkdir()
    path.write_text("source")
    digest = str(sha256_file(path))
    assert not input_location_matches(tmp_path, path, "src/forge/old.py", digest)
    moves = tmp_path / "docs/artifact_path_moves.json"
    moves.parent.mkdir()
    moves.write_text(json.dumps({"moves": {"src/forge/old.py": "forge/new.py"}}))
    assert input_location_matches(tmp_path, path, "src/forge/old.py", digest)
    path.write_text("changed")
    assert not input_location_matches(tmp_path, path, "src/forge/old.py", digest)


def test_data_cannot_use_source_archive_or_moves(archived):
    repo, _, digest, path = archived
    assert not input_location_matches(repo, path, "data/supplier.html", digest)


def test_symlink_and_traversal_cannot_supply_source_identity(archived):
    repo, asset, digest, path = archived
    link = repo / "alias.py"
    link.symlink_to(path)
    assert not input_location_matches(repo, link, asset, digest)
    assert not input_location_matches(repo, path, "src/../original.py", digest)


def _records(repo, asset, digest, path):
    stored = {
        "inputs": {"source": {"path": "/old/workstation/" + asset, "sha256": digest}},
        "summary": {"closed": 2},
        "artifacts": {"ledger": {"sha256": "a" * 64}},
    }
    fresh = deepcopy(stored)
    fresh["inputs"]["source"]["path"] = path.relative_to(repo).as_posix()
    kwargs = {
        "repo": repo,
        "specifications": {"source": {"asset": asset, "expected_sha256": digest}},
        "input_paths": {"source": path},
    }
    return stored, fresh, kwargs


def test_result_relocation_keeps_every_other_field_and_hash(archived):
    stored, fresh, kwargs = _records(*archived)
    untouched = deepcopy(stored)
    assert results_match_after_input_relocation(stored, fresh, **kwargs)
    assert stored == untouched
    changed = deepcopy(fresh)
    changed["summary"]["closed"] = 3
    assert not results_match_after_input_relocation(stored, changed, **kwargs)
    changed = deepcopy(fresh)
    changed["artifacts"]["ledger"]["sha256"] = "b" * 64
    assert not results_match_after_input_relocation(stored, changed, **kwargs)
    changed = deepcopy(fresh)
    changed["unreviewed_field"] = True
    assert not results_match_after_input_relocation(stored, changed, **kwargs)


def test_source_locators_are_not_silently_normalized(archived):
    stored, fresh, kwargs = _records(*archived)
    stored["source_locator"] = "/old/source#record"
    fresh["source_locator"] = "/new/source#record"
    assert not results_match_after_input_relocation(stored, fresh, **kwargs)


def test_wrong_logical_asset_or_current_binding_is_rejected(archived):
    stored, fresh, kwargs = _records(*archived)
    stored["inputs"]["source"]["path"] = "src/other.py"
    assert not results_match_after_input_relocation(stored, fresh, **kwargs)
    stored, fresh, kwargs = _records(*archived)
    fresh["inputs"]["source"]["path"] = "unbound.py"
    assert not results_match_after_input_relocation(stored, fresh, **kwargs)


def test_recorded_and_actual_input_hashes_are_both_checked(archived):
    stored, fresh, kwargs = _records(*archived)
    stored["inputs"]["source"]["sha256"] = "f" * 64
    assert not results_match_after_input_relocation(stored, fresh, **kwargs)
    stored, fresh, kwargs = _records(*archived)
    archived[-1].write_text("tampered actual input")
    assert not results_match_after_input_relocation(stored, fresh, **kwargs)


def test_explicit_absolute_fixture_is_verified_at_its_declared_location(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "evidence.json"
    data.write_text("{}")
    digest = str(sha256_file(data))
    assert input_location_matches(repo, data, str(data), digest)
    assert not input_location_matches(repo, data, "data/evidence.json", digest)
