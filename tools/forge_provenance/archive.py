#!/usr/bin/env python3
"""Archive exact code/config bytes referenced by surviving result artifacts.

The operation is mechanical and idempotent.  Bytes are recovered from Git objects when possible,
falling back to the current working file only when it already matches the declared digest.  A blob
is admitted solely by SHA-256 equality; a path or commit name is never trusted on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from forge.core.io import atomic_write, write_json
from forge_provenance.pins import (
    FOREIGN_PREFIXES,
    Pin,
    collect_pins,
    load_baseline,
    load_moves,
)
from forge_provenance.resolver import HISTORICAL_PIN_ARCHIVE_SCHEMA

REPO = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = REPO / "provenance" / "frozen-code"

# Frozen configuration lives both in `configs/` and beside the active experiment applications.
# Scan both alongside result artifacts so moving a specification never drops its source pins from
# the archive boundary.
DEFAULT_ROOTS = (
    REPO / "results",
    REPO / "docs" / "provenance",
    REPO / "configs",
    REPO / "experiments",
)
# Historical prefixes remain eligible because frozen artifacts still name them. Current prefixes
# allow new records to preserve source identity without recreating the old layout.
ELIGIBLE_PREFIXES = (
    "cli/",
    "configs/",
    "experiments/",
    "forge/",
    "paper/",
    "scripts/",
    "src/",
    "tests/",
    "tools/",
)
ELIGIBLE_ROOT_FILES = frozenset({"Makefile", "pyproject.toml"})


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _eligible(path: str) -> bool:
    return path in ELIGIBLE_ROOT_FILES or path.startswith(ELIGIBLE_PREFIXES)


def _git(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(REPO), *arguments],
        capture_output=True,
        check=False,
    )


def _commits() -> tuple[str, ...]:
    completed = _git("rev-list", "--all")
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode(errors="replace"))
    return tuple(line for line in completed.stdout.decode().splitlines() if line)


def _git_blob(
    commits: tuple[str, ...], path: str, digest: str
) -> tuple[bytes, dict[str, Any]] | None:
    for commit in commits:
        completed = _git("show", f"{commit}:{path}")
        if completed.returncode == 0 and _sha256(completed.stdout) == digest:
            return completed.stdout, {"commit": commit, "kind": "git", "path": path}
    return None


def _working_blob(path: str, moved: str, digest: str) -> tuple[bytes, dict[str, Any]] | None:
    candidate = REPO / moved
    if not candidate.is_file() or candidate.is_symlink():
        return None
    payload = candidate.read_bytes()
    if _sha256(payload) != digest:
        return None
    return payload, {"kind": "working_tree", "path": moved, "original_path": path}


def _unique_code_pins(roots: tuple[Path, ...]) -> tuple[Pin, ...]:
    unique: dict[tuple[str, str], Pin] = {}
    for pin in collect_pins(roots):
        if pin.path.startswith(FOREIGN_PREFIXES) or not _eligible(pin.path):
            continue
        unique.setdefault((pin.path, pin.sha256), pin)
    return tuple(unique[key] for key in sorted(unique))


def archive(
    roots: tuple[Path, ...],
    *,
    archive_root: Path = ARCHIVE_ROOT,
) -> dict[str, Any]:
    commits = _commits()
    moves = load_moves(REPO / "docs" / "artifact_path_moves.json")
    baseline = load_baseline(REPO / "docs" / "known_artifact_drift.json")
    entries: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    known_unresolved: list[dict[str, Any]] = []
    for pin in _unique_code_pins(roots):
        recovered = _git_blob(commits, pin.path, pin.sha256)
        if recovered is None:
            recovered = _working_blob(pin.path, moves.get(pin.path, pin.path), pin.sha256)
        if recovered is None:
            record = {
                "declared_by": pin.declared_by,
                "original_path": pin.path,
                "sha256": pin.sha256,
            }
            if (pin.path, pin.sha256) in baseline:
                known_unresolved.append(record)
            else:
                unresolved.append(record)
            continue
        payload, source = recovered
        blob = archive_root / "sha256" / pin.sha256[:2] / pin.sha256
        if blob.is_file() and _sha256(blob.read_bytes()) != pin.sha256:
            raise RuntimeError(f"existing historical blob is corrupted: {blob}")
        if not blob.exists():
            atomic_write(blob, payload)
        entries.append(
            {
                "blob_path": str(blob.relative_to(REPO)),
                "original_path": pin.path,
                "sha256": pin.sha256,
                "source": source,
            }
        )

    document = {
        "entries": sorted(entries, key=lambda row: (row["original_path"], row["sha256"])),
        "known_unresolved": sorted(
            known_unresolved, key=lambda row: (row["original_path"], row["sha256"])
        ),
        "schema_version": HISTORICAL_PIN_ARCHIVE_SCHEMA,
        "summary": {
            "archived": len(entries),
            "known_unresolved": len(known_unresolved),
            "unresolved": len(unresolved),
        },
        "unresolved": sorted(unresolved, key=lambda row: (row["original_path"], row["sha256"])),
    }
    # The resolver intentionally accepts only schema_version + entries.  Keep diagnostic fields in
    # a separate report so the load-bearing manifest remains a minimal strict contract.
    write_json(
        archive_root / "manifest.json",
        {"schema_version": document["schema_version"], "entries": document["entries"]},
    )
    write_json(archive_root / "archive_report.json", document)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", default=None)
    args = parser.parse_args(argv)
    roots = tuple(args.root) if args.root else DEFAULT_ROOTS
    document = archive(roots)
    print(json.dumps(document["summary"], indent=2, sort_keys=True))
    if document["unresolved"]:
        print("unresolved non-baseline historical pins:", file=sys.stderr)
        for row in document["unresolved"]:
            print(
                f"  {row['original_path']} {row['sha256']} declared by {row['declared_by']}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
