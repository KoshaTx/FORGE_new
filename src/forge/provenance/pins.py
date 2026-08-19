#!/usr/bin/env python3
"""Re-hash every pinned input declared by a result artifact and report drift.

Artifacts under `results/` record the inputs they were computed from as `{"path", "sha256"}`
pairs. Together those pins are the paper's evidence chain: they are what lets a number in the
manuscript be traced to the bytes it came from. This script walks all of them and checks that
the bytes on disk still hash to what the artifact says they did.

The distinction that matters for refactoring:

  drift    a pinned file is present but hashes differently -- always fatal, something changed
           that was supposed to be frozen
  absent   a pinned file is not on this machine -- reported, not fatal by default, because many
           inputs legitimately live only on the workstation that produced them

So a refactor is safe when `verified` is unchanged and `drift` is zero. A dropping `verified`
count means a file went missing; any drift at all means a supposedly-frozen byte moved.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Set
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge.core.hashing import is_sha256
from forge.core.provenance_archive import HistoricalPinArchive

REPO = Path(__file__).resolve().parents[3]

# Pins recorded against a container-side path (a Modal mount, a remote checkout) describe the
# same bytes we may hold locally under a different root. They are not local files and their
# absence is never a finding.
FOREIGN_PREFIXES = ("/", "..")


@dataclass(frozen=True)
class Pin:
    """One declared (path, sha256) input, and who declared it."""

    path: str
    sha256: str
    declared_by: str


@dataclass
class Report:
    verified: list[Pin] = field(default_factory=list)
    archived: list[Pin] = field(default_factory=list)
    drift: list[tuple[Pin, str]] = field(default_factory=list)
    known_drift: list[tuple[Pin, str | None]] = field(default_factory=list)
    absent: list[Pin] = field(default_factory=list)
    foreign: list[Pin] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.drift


def load_moves(path: Path) -> dict[str, str]:
    """Old path -> new path, for files a restructure relocated.

    A pin binds a path and a digest. Moving a file changes the path but not the bytes, so without
    this the verifier reports the input as absent and the pin count silently drops -- which is what
    made restructuring look impossible. Resolving through the map keeps the content check exactly as
    strict: a moved file whose bytes changed still fails.
    """
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text()).get("moves", {}))


def load_baseline(path: Path) -> frozenset[tuple[str, str]]:
    """Return the exact unrecoverable ``(path, expected_sha256)`` exceptions.

    These are historical bytes that were already unavailable when the repository was consolidated.
    The exception binds the missing identity, never the mutable hash of today's file. Recoverable
    historical bytes belong in the content-addressed archive instead.
    """
    if not path.is_file():
        return frozenset()
    document = json.loads(path.read_text())
    expected_fields = {"schema_version", "recorded_utc", "why", "how_to_add", "known_drift"}
    if not isinstance(document, dict) or set(document) != expected_fields:
        raise ValueError("known artifact drift ledger has invalid top-level fields")
    if document["schema_version"] != "forge.known_artifact_drift.v2":
        raise ValueError("known artifact drift ledger has an unsupported schema")
    entries = document["known_drift"]
    if not isinstance(entries, list):
        raise ValueError("known artifact drift entries must be a list")
    identities: set[tuple[str, str]] = set()
    entry_fields = {"path", "expected", "actual", "declared_by", "reason"}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != entry_fields:
            raise ValueError(f"known artifact drift entry {index} is malformed")
        identity = (entry["path"], entry["expected"])
        if (
            not isinstance(identity[0], str)
            or not identity[0]
            or identity[0].startswith(("/", ".."))
            or not is_sha256(identity[1])
            or not is_sha256(entry["actual"])
        ):
            raise ValueError(f"known artifact drift entry {index} has an invalid identity")
        if identity in identities:
            raise ValueError(f"duplicate known artifact drift identity: {identity}")
        identities.add(identity)
    return frozenset(identities)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Any | None:
    try:
        if path.name.endswith(".json.gz"):
            with gzip.open(path, "rt") as handle:
                return json.load(handle)
        return json.loads(path.read_text())
    except (OSError, ValueError, EOFError):
        return None


def collect_pins(roots: tuple[Path, ...]) -> list[Pin]:
    """Find every {"path", "sha256"} pair reachable in the artifacts under `roots`."""
    pins: list[Pin] = []

    def walk(node: Any, source: str) -> None:
        if isinstance(node, dict):
            path, digest = node.get("path"), node.get("sha256")
            if isinstance(path, str) and isinstance(digest, str) and len(digest) == 64:
                pins.append(Pin(path=path, sha256=digest, declared_by=source))
            for value in node.values():
                walk(value, source)
        elif isinstance(node, list):
            for value in node:
                walk(value, source)

    for root in roots:
        # Resolve against the repository so a relative `--root configs` behaves like the absolute
        # defaults; `relative_to(REPO)` below needs an absolute path and raises otherwise.
        root = root if root.is_absolute() else (REPO / root)
        if not root.exists():
            continue
        for candidate in sorted(root.rglob("*")):
            if not candidate.is_file():
                continue
            if candidate.suffix != ".json" and not candidate.name.endswith(".json.gz"):
                continue
            document = _load(candidate)
            if document is not None:
                walk(document, str(candidate.relative_to(REPO)))
    return pins


def verify(
    pins: list[Pin],
    baseline: Set[tuple[str, str]] | None = None,
    moves: dict[str, str] | None = None,
    archive: HistoricalPinArchive | None = None,
) -> Report:
    accepted_baseline: Set[tuple[str, str]] = baseline or frozenset()
    moves = moves or {}
    report = Report()
    # One file is typically pinned by several artifacts. Hash it once.
    by_path: dict[str, list[Pin]] = defaultdict(list)
    for pin in pins:
        by_path[pin.path].append(pin)

    for path, declarations in sorted(by_path.items()):
        if path.startswith(FOREIGN_PREFIXES):
            report.foreign.extend(declarations)
            continue
        resolved = REPO / moves.get(path, path)
        actual = sha256_file(resolved) if resolved.is_file() else None
        for pin in declarations:
            if actual is not None and actual == pin.sha256:
                report.verified.append(pin)
                continue
            archived = archive.resolve(path, pin.sha256) if archive is not None else None
            if archived is not None:
                report.archived.append(pin)
            elif (path, pin.sha256) in accepted_baseline:
                report.known_drift.append((pin, actual))
            elif actual is None:
                report.absent.append(pin)
            else:
                report.drift.append((pin, actual))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=None,
        help="directory to scan for artifacts (repeatable; default: results/ and docs/provenance/)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail when a pinned input is absent, not only when it has drifted",
    )
    parser.add_argument(
        "--expect-verified",
        type=int,
        default=None,
        help="fail if fewer than this many pins verify; use the recorded baseline to catch losses",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=REPO / "docs" / "known_artifact_drift.json",
        help="known-drifted pins to accept; anything not listed still fails",
    )
    parser.add_argument(
        "--moves",
        type=Path,
        default=REPO / "docs" / "artifact_path_moves.json",
        help="old-path to new-path map for files a restructure relocated",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=REPO / "provenance" / "frozen-code" / "manifest.json",
        help="content-addressed historical source/config archive",
    )
    parser.add_argument(
        "--allow-drift",
        type=int,
        default=0,
        help=(
            "burn-down ratchet: tolerate at most this many drifted pins, and fail if the count "
            "grows. This is NOT a way to accept drift -- accepted drift belongs in "
            "docs/known_artifact_drift.json, one reviewed entry per unrecoverable identity. Use "
            "this only to hold a known backlog flat while it is being retired, and never on a "
            "root whose drift count is already zero."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args(argv)
    if args.allow_drift < 0:
        parser.error("--allow-drift must not be negative")

    roots = tuple(args.root) if args.root else (REPO / "results", REPO / "docs" / "provenance")
    archive = HistoricalPinArchive.load(args.archive, REPO)
    report = verify(
        collect_pins(roots),
        load_baseline(args.baseline),
        load_moves(args.moves),
        archive,
    )

    verified_count = len(report.verified) + len(report.archived)
    distinct = len({pin.path for pin in (*report.verified, *report.archived)})
    if args.json:
        print(
            json.dumps(
                {
                    "verified": verified_count,
                    "verified_active": len(report.verified),
                    "verified_archived": len(report.archived),
                    "known_drift": sorted({p.path for p, _ in report.known_drift}),
                    "distinct_files_verified": distinct,
                    "drift": [
                        {
                            "path": p.path,
                            "expected": p.sha256,
                            "actual": a,
                            "declared_by": p.declared_by,
                        }
                        for p, a in report.drift
                    ],
                    "absent": sorted({p.path for p in report.absent}),
                    "foreign": sorted({p.path for p in report.foreign}),
                },
                indent=2,
            )
        )
    else:
        print(f"  verified   {verified_count:5d} pins over {distinct} distinct files")
        print(f"    active   {len(report.verified):5d}")
        print(f"    archive  {len(report.archived):5d}")
        print(f"  absent     {len({p.path for p in report.absent}):5d} files not on this machine")
        print(
            f"  foreign    {len({p.path for p in report.foreign}):5d} container-side paths (skipped)"
        )
        print(
            f"  known      {len({p.path for p, _ in report.known_drift}):5d} accepted pre-existing "
            f"drift (docs/known_artifact_drift.json)"
        )
        print(f"  DRIFT      {len(report.drift):5d}")
        for pin, actual in report.drift:
            print(f"\nDRIFT {pin.path}\n  declared by {pin.declared_by}")
            print(f"  expected {pin.sha256}\n  actual   {actual}")

    def note(message: str) -> None:
        # Keep --json output parseable: diagnostics go to stderr when it is on.
        print(message, file=sys.stderr if args.json else sys.stdout)

    if len(report.drift) > args.allow_drift:
        note(f"\n{len(report.drift)} pinned input(s) DRIFTED — a frozen byte moved. Stop.")
        if args.allow_drift:
            note(f"the ratchet tolerates {args.allow_drift}; this run exceeds it.")
        return 1
    if report.drift:
        note(
            f"\n{len(report.drift)} drifted pin(s) held under the --allow-drift {args.allow_drift} "
            f"ratchet. This is a backlog to retire, not an accepted state."
        )
    if args.strict and report.absent:
        note(f"\n{len({p.path for p in report.absent})} pinned input(s) absent under --strict.")
        return 1
    if args.expect_verified is not None and verified_count < args.expect_verified:
        note(
            f"\nonly {verified_count} pins verified, expected at least "
            f"{args.expect_verified} — an input went missing."
        )
        return 1
    note("\nno drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
