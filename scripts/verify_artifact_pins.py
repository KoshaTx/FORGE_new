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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

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
    drift: list[tuple[Pin, str]] = field(default_factory=list)
    known_drift: list[tuple[Pin, str]] = field(default_factory=list)
    absent: list[Pin] = field(default_factory=list)
    foreign: list[Pin] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.drift


def load_baseline(path: Path) -> dict[str, str]:
    """Known-drifted pins: path -> the current hash that is accepted.

    Some artifacts were computed against an input that has since moved on, and the original
    bytes no longer exist anywhere. Recording them here keeps the gate meaningful: inherited
    drift is acknowledged once, and anything new -- or any further change to an already-drifted
    file -- still fails. Mirrors how tests/unreproducible_pins.py quarantines test node ids.
    """
    if not path.is_file():
        return {}
    document = json.loads(path.read_text())
    return {entry["path"]: entry["actual"] for entry in document.get("known_drift", [])}


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


def verify(pins: list[Pin], baseline: dict[str, str] | None = None) -> Report:
    baseline = baseline or {}
    report = Report()
    # One file is typically pinned by several artifacts. Hash it once.
    by_path: dict[str, list[Pin]] = defaultdict(list)
    for pin in pins:
        by_path[pin.path].append(pin)

    for path, declarations in sorted(by_path.items()):
        if path.startswith(FOREIGN_PREFIXES):
            report.foreign.extend(declarations)
            continue
        resolved = REPO / path
        if not resolved.is_file():
            report.absent.extend(declarations)
            continue
        actual = sha256_file(resolved)
        for pin in declarations:
            if actual == pin.sha256:
                report.verified.append(pin)
            elif baseline.get(path) == actual:
                report.known_drift.append((pin, actual))
            else:
                report.drift.append((pin, actual))
    return report


def main() -> int:
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
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()

    roots = tuple(args.root) if args.root else (REPO / "results", REPO / "docs" / "provenance")
    report = verify(collect_pins(roots), load_baseline(args.baseline))

    distinct = len({pin.path for pin in report.verified})
    if args.json:
        print(
            json.dumps(
                {
                    "verified": len(report.verified),
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
        print(f"  verified   {len(report.verified):5d} pins over {distinct} distinct files")
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

    if report.drift:
        note(f"\n{len(report.drift)} pinned input(s) DRIFTED — a frozen byte moved. Stop.")
        return 1
    if args.strict and report.absent:
        note(f"\n{len({p.path for p in report.absent})} pinned input(s) absent under --strict.")
        return 1
    if args.expect_verified is not None and len(report.verified) < args.expect_verified:
        note(
            f"\nonly {len(report.verified)} pins verified, expected at least "
            f"{args.expect_verified} — an input went missing."
        )
        return 1
    note("\nno drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
