from __future__ import annotations

from pathlib import Path

import pytest

from forge.core.hashing import sha256_file
from forge.core.io import atomic_write, write_json
from forge.core.provenance_archive import (
    HistoricalPinArchive,
    HistoricalPinArchiveError,
)


def build_archive(tmp_path: Path) -> tuple[Path, Path, str]:
    blob = tmp_path / "provenance" / "frozen-code" / "sha256" / "aa" / ("a" * 64)
    atomic_write(blob, b"historical source\n")
    digest = str(sha256_file(blob))
    actual = blob.with_name(digest)
    blob.rename(actual)
    manifest = tmp_path / "provenance" / "frozen-code" / "manifest.json"
    write_json(
        manifest,
        {
            "schema_version": "forge.historical_pin_archive.v1",
            "entries": [
                {
                    "original_path": "src/forge/old.py",
                    "sha256": digest,
                    "blob_path": str(actual.relative_to(tmp_path)),
                    "source": {"kind": "git", "commit": "abc"},
                }
            ],
        },
    )
    return manifest, actual, digest


def test_resolves_exact_historical_bytes(tmp_path: Path) -> None:
    manifest, blob, digest = build_archive(tmp_path)
    archive = HistoricalPinArchive.load(manifest, tmp_path)
    assert archive.resolve("src/forge/old.py", digest) == blob.resolve()
    assert archive.resolve("src/forge/old.py", "b" * 64) is None


def test_tampered_historical_blob_fails_closed(tmp_path: Path) -> None:
    manifest, blob, digest = build_archive(tmp_path)
    archive = HistoricalPinArchive.load(manifest, tmp_path)
    atomic_write(blob, b"changed\n")
    with pytest.raises(HistoricalPinArchiveError, match="changed"):
        archive.resolve("src/forge/old.py", digest)


def test_duplicate_path_and_digest_is_rejected(tmp_path: Path) -> None:
    manifest, _, _ = build_archive(tmp_path)
    document = __import__("json").loads(manifest.read_text())
    document["entries"].append(dict(document["entries"][0]))
    write_json(manifest, document)
    with pytest.raises(HistoricalPinArchiveError, match="duplicate"):
        HistoricalPinArchive.load(manifest, tmp_path)


def test_extra_manifest_fields_are_rejected(tmp_path: Path) -> None:
    manifest, _, _ = build_archive(tmp_path)
    document = __import__("json").loads(manifest.read_text())
    document["diagnostics"] = {}
    write_json(manifest, document)
    with pytest.raises(HistoricalPinArchiveError, match="exactly"):
        HistoricalPinArchive.load(manifest, tmp_path)
