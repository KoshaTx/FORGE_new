"""Content-addressed storage for source/config bytes pinned by historical results.

Active source is allowed to improve.  A result's evidence chain is not.  The archive keeps exact
historical bytes outside the importable package and keys them by both their original path and their
digest, so one path may legitimately have several result-pinned revisions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.core.hashing import is_sha256, sha256_file

HISTORICAL_PIN_ARCHIVE_SCHEMA = "forge.historical_pin_archive.v1"


class HistoricalPinArchiveError(ValueError):
    """The archive manifest or a content-addressed blob is malformed."""


@dataclass(frozen=True)
class HistoricalPinEntry:
    original_path: str
    sha256: str
    blob_path: str
    source: Mapping[str, Any]


@dataclass(frozen=True)
class HistoricalPinArchive:
    repo: Path
    entries: Mapping[tuple[str, str], HistoricalPinEntry]

    @classmethod
    def load(cls, manifest_path: Path, repo: Path) -> HistoricalPinArchive:
        if not manifest_path.is_file():
            return cls(repo=repo.resolve(), entries={})
        try:
            document = json.loads(manifest_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HistoricalPinArchiveError(
                f"historical pin archive could not be read: {manifest_path}: {error}"
            ) from error
        if not isinstance(document, dict):
            raise HistoricalPinArchiveError("historical pin archive must contain a JSON object")
        if set(document) != {"schema_version", "entries"}:
            raise HistoricalPinArchiveError(
                "historical pin archive must define exactly schema_version and entries"
            )
        if document.get("schema_version") != HISTORICAL_PIN_ARCHIVE_SCHEMA:
            raise HistoricalPinArchiveError(
                f"unsupported historical pin archive schema: {document.get('schema_version')!r}"
            )
        raw_entries = document.get("entries")
        if not isinstance(raw_entries, list):
            raise HistoricalPinArchiveError("historical pin archive entries must be a JSON array")
        entries: dict[tuple[str, str], HistoricalPinEntry] = {}
        for index, raw in enumerate(raw_entries):
            if not isinstance(raw, dict) or set(raw) != {
                "original_path",
                "sha256",
                "blob_path",
                "source",
            }:
                raise HistoricalPinArchiveError(f"malformed historical pin entry at index {index}")
            original = raw["original_path"]
            digest = raw["sha256"]
            blob = raw["blob_path"]
            source = raw["source"]
            if not isinstance(original, str) or not original or original.startswith(("/", "..")):
                raise HistoricalPinArchiveError(f"invalid original path at index {index}")
            if not is_sha256(digest):
                raise HistoricalPinArchiveError(f"invalid digest at index {index}")
            if not isinstance(blob, str) or not blob or blob.startswith(("/", "..")):
                raise HistoricalPinArchiveError(f"invalid blob path at index {index}")
            if not isinstance(source, dict):
                raise HistoricalPinArchiveError(f"invalid source record at index {index}")
            key = (original, digest)
            if key in entries:
                raise HistoricalPinArchiveError(
                    f"duplicate historical pin entry for {original} at {digest}"
                )
            entries[key] = HistoricalPinEntry(
                original_path=original,
                sha256=digest,
                blob_path=blob,
                source=source,
            )
        return cls(repo=repo.resolve(), entries=entries)

    def resolve(self, original_path: str, sha256: str) -> Path | None:
        entry = self.entries.get((original_path, sha256))
        if entry is None:
            return None
        candidate = self.repo / entry.blob_path
        if candidate.is_symlink():
            raise HistoricalPinArchiveError(f"historical pin blob is a symlink: {candidate}")
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.repo)
        except ValueError as error:
            raise HistoricalPinArchiveError(
                f"historical pin blob leaves the repository: {resolved}"
            ) from error
        if not resolved.is_file():
            raise HistoricalPinArchiveError(f"historical pin blob is missing: {resolved}")
        observed = str(sha256_file(resolved))
        if observed != entry.sha256:
            raise HistoricalPinArchiveError(
                f"historical pin blob changed: expected {entry.sha256}, found {observed}: {resolved}"
            )
        return resolved


__all__ = [
    "HISTORICAL_PIN_ARCHIVE_SCHEMA",
    "HistoricalPinArchive",
    "HistoricalPinArchiveError",
    "HistoricalPinEntry",
]
