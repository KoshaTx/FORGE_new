"""Existing gaps must never hide a new loss, even if the total gap count improves."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from forge_provenance.regression import compare, main, snapshot
from forge_provenance.resolver import HistoricalPinArchiveError


def write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def checkout(root: Path, *, missing: bool = False) -> Path:
    (root / "configs").mkdir(parents=True)
    expected = hashlib.sha256(b"original").hexdigest()
    write(root / "results/run.json", json.dumps({"path": "source.py", "sha256": expected}).encode())
    if not missing:
        write(root / "source.py", b"original")
    return root


def archive_source(root: Path) -> Path:
    digest = hashlib.sha256(b"original").hexdigest()
    blob = f"provenance/frozen-code/sha256/{digest[:2]}/{digest}"
    write(root / blob, b"original")
    write(
        root / "provenance/frozen-code/manifest.json",
        json.dumps(
            {
                "schema_version": "forge.historical_pin_archive.v1",
                "entries": [
                    {
                        "original_path": "source.py",
                        "sha256": digest,
                        "blob_path": blob,
                        "source": {"kind": "test"},
                    }
                ],
            }
        ).encode(),
    )
    return root / blob


def test_existing_gap_is_reported_as_incomplete_not_verified(tmp_path: Path) -> None:
    base = checkout(tmp_path / "base", missing=True)
    current = checkout(tmp_path / "current", missing=True)
    output = tmp_path / "report.json"
    assert (
        main(["--base-root", str(base), "--current-root", str(current), "--output", str(output)])
        == 0
    )
    report = json.loads(output.read_text())
    assert report["historical_availability"] == "incomplete"
    assert report["counts"]["current_available_identities"] == 0
    assert len(report["unresolved_declarations"]) == 1


@pytest.mark.parametrize("change", ["edit", "delete", "remove_declaration", "rewrite_digest"])
def test_new_evidence_loss_fails(tmp_path: Path, change: str) -> None:
    base = checkout(tmp_path / "base")
    current = checkout(tmp_path / "current")
    if change == "edit":
        (current / "source.py").write_bytes(b"changed")
    elif change == "delete":
        (current / "source.py").unlink()
    elif change == "remove_declaration":
        (current / "results/run.json").unlink()
    else:
        (current / "source.py").write_bytes(b"changed")
        (current / "results/run.json").write_text(
            json.dumps(
                {
                    "path": "source.py",
                    "sha256": hashlib.sha256(b"changed").hexdigest(),
                }
            )
        )
    assert compare(snapshot(base), snapshot(current))["regression_status"] == "fail"


def test_refactor_passes_when_exact_old_bytes_are_archived(tmp_path: Path) -> None:
    base = checkout(tmp_path / "base")
    current = checkout(tmp_path / "current")
    archive_source(current)
    (current / "source.py").write_bytes(b"refactored")
    report = compare(snapshot(base), snapshot(current))
    assert report["regression_status"] == "pass"
    assert report["historical_availability"] == "available"


def test_recovery_does_not_offset_a_different_loss(tmp_path: Path) -> None:
    base = checkout(tmp_path / "base", missing=True)
    current = checkout(tmp_path / "current")
    for root in (base, current):
        write(
            root / "results/other.json",
            json.dumps(
                {
                    "path": "other.py",
                    "sha256": hashlib.sha256(b"other").hexdigest(),
                }
            ).encode(),
        )
    write(base / "other.py", b"other")
    report = compare(snapshot(base), snapshot(current))
    assert report["counts"]["base_unresolved_declarations"] == 1
    assert report["counts"]["current_unresolved_declarations"] == 1
    assert report["regression_status"] == "fail"


def test_new_declaration_cannot_inherit_an_old_missing_identity(tmp_path: Path) -> None:
    base = checkout(tmp_path / "base", missing=True)
    current = checkout(tmp_path / "current", missing=True)
    write(current / "results/new.json", (current / "results/run.json").read_bytes())
    assert compare(snapshot(base), snapshot(current))["regression_status"] == "fail"


def test_gap_ledger_cannot_hide_a_loss(tmp_path: Path) -> None:
    base = checkout(tmp_path / "base")
    current = checkout(tmp_path / "current", missing=True)
    write(current / "docs/known_artifact_drift.json", b'{"known_drift": []}')
    assert compare(snapshot(base), snapshot(current))["regression_status"] == "fail"


def test_corrupt_archive_is_fatal_even_when_active_source_matches(tmp_path: Path) -> None:
    current = checkout(tmp_path / "current")
    archive_source(current).write_bytes(b"corrupt")
    with pytest.raises(HistoricalPinArchiveError, match="changed"):
        snapshot(current)


def test_archive_removal_fails_even_when_active_source_matches(tmp_path: Path) -> None:
    base = checkout(tmp_path / "base")
    archive_source(base)
    current = checkout(tmp_path / "current")
    assert compare(snapshot(base), snapshot(current))["regression_status"] == "fail"


def test_invalid_new_json_cannot_be_silently_skipped(tmp_path: Path) -> None:
    current = checkout(tmp_path / "current")
    write(current / "results/broken.json", b"{broken")
    with pytest.raises(json.JSONDecodeError):
        snapshot(current)


def test_wrong_base_path_is_not_an_empty_baseline(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="FORGE checkout"):
        snapshot(tmp_path)
