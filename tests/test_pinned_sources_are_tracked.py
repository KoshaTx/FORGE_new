"""Every source file pinned by a config must be under version control.

A config that pins a file by SHA-256 asserts "this artifact was produced by
exactly this code".  If that file is not tracked by git, the assertion is
unfalsifiable and unrecoverable: once the file changes, the pinned content is
gone and no one can ever reproduce the artifact or even see what was lost.

This is not a style rule.  It is the difference between a pin that can be
audited and a pin that merely records a number nobody can check.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parents[1]

# Pinned paths that resolve outside this repository and therefore cannot be
# tracked here.  They are recorded rather than silently skipped: any artifact
# depending on them is reproducible only on a machine that also holds the
# sibling checkout at the pinned revision.
EXTERNAL_PINS = {
    "../electron_flow_lipids/modal_apps/flower_diagnostic.py",
    "../electron_flow_lipids/modal_apps/flower_probe.py",
}


def _pinned_python_paths() -> set[str]:
    """Collect every `.py` path pinned with a sha256 anywhere under configs/."""

    def walk(node: object) -> Iterator[str]:
        if isinstance(node, dict):
            path, digest = node.get("path"), node.get("sha256")
            if isinstance(path, str) and isinstance(digest, str) and path.endswith(".py"):
                yield path
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    pinned: set[str] = set()
    for config in (REPO / "configs").rglob("*.json"):
        try:
            payload = json.loads(config.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        pinned.update(walk(payload))
    return pinned


def _tracked_paths() -> set[str]:
    listing = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return set(listing.stdout.splitlines())


def test_every_pinned_source_file_is_tracked_by_git() -> None:
    pinned = _pinned_python_paths()
    assert pinned, "no pinned .py paths found; the collector is probably broken"

    tracked = _tracked_paths()
    untracked = sorted(p for p in pinned if p not in tracked and p not in EXTERNAL_PINS)

    assert not untracked, (
        f"{len(untracked)} source files are pinned by a config but are not tracked by git, "
        "so their pinned content cannot be recovered or audited. Commit them:\n  "
        + "\n  ".join(untracked)
    )


def test_pinned_source_files_exist_on_disk() -> None:
    """A pin naming a file that is absent is a dangling provenance claim."""

    missing = sorted(
        path
        for path in _pinned_python_paths()
        if path not in EXTERNAL_PINS and not (REPO / path).is_file()
    )
    assert not missing, "configs pin source files that do not exist:\n  " + "\n  ".join(missing)


def test_quarantined_pin_tests_still_exist() -> None:
    """The unreproducible-pin quarantine must not outlive the tests it names.

    Without this, a quarantined test that is renamed or deleted leaves a dead
    entry behind, and the list slowly stops describing reality.
    """

    from unreproducible_pins import UNREPRODUCIBLE_PIN_SETUP_ERRORS, UNREPRODUCIBLE_PIN_TESTS

    dangling: list[str] = []
    for node_id in sorted(UNREPRODUCIBLE_PIN_TESTS | UNREPRODUCIBLE_PIN_SETUP_ERRORS):
        file_part, _, test_part = node_id.partition("::")
        path = REPO / file_part
        if not path.is_file():
            dangling.append(f"{node_id}  (file missing)")
            continue
        # Strip any parametrisation suffix before looking for the definition.
        name = test_part.split("[", 1)[0]
        if name and f"def {name}(" not in path.read_text():
            dangling.append(f"{node_id}  (test not defined)")

    assert not dangling, (
        "quarantined pin tests no longer exist; remove them from "
        "tests/unreproducible_pins.py:\n  " + "\n  ".join(dangling)
    )
