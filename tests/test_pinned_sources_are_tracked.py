"""Every source identity pinned by a config must be recoverable.

A config that pins a file by SHA-256 asserts "this artifact was produced by
exactly this code". Active code must be tracked. Retired code may instead resolve through the
content-addressed historical archive, which preserves the exact path/hash identity without keeping
dead code importable.

This is not a style rule.  It is the difference between a pin that can be
audited and a pin that merely records a number nobody can check.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Pinned paths that resolve outside this repository and therefore cannot be
# tracked here.  They are recorded rather than silently skipped: any artifact
# depending on them is reproducible only on a machine that also holds the
# sibling checkout at the pinned revision.
EXTERNAL_PINS = {
    "../electron_flow_lipids/modal_apps/flower_diagnostic.py",
    "../electron_flow_lipids/modal_apps/flower_probe.py",
}


def _pinned_python_identities() -> set[tuple[str, str]]:
    """Collect every ``(path, sha256)`` Python identity pinned under configs/."""

    def walk(node: object) -> Iterator[tuple[str, str]]:
        if isinstance(node, dict):
            path, digest = node.get("path"), node.get("sha256")
            if isinstance(path, str) and isinstance(digest, str) and path.endswith(".py"):
                yield path, digest
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    pinned: set[tuple[str, str]] = set()
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


def _unrecoverable_pins() -> list[str]:
    from forge_provenance.pins import load_moves
    from forge_provenance.resolver import HistoricalPinArchive

    pinned = _pinned_python_identities()
    assert pinned, "no pinned .py paths found; the collector is probably broken"
    tracked = _tracked_paths()
    moves = load_moves(REPO / "docs/artifact_path_moves.json")
    archive = HistoricalPinArchive.load(REPO / "provenance/frozen-code/manifest.json", REPO)
    unresolved = []
    for path, digest in sorted(pinned):
        if path in EXTERNAL_PINS:
            continue
        active_path = moves.get(path, path)
        active = REPO / active_path
        # Active drift is adjudicated by the separate provenance ratchet. This test asks only
        # whether the path is tracked or, once retired, its exact historical identity is archived.
        if active.is_file() and active_path in tracked:
            continue
        if archive.resolve(path, digest) is not None:
            continue
        unresolved.append(f"{path} {digest}")
    return unresolved


def test_every_active_pinned_source_file_is_tracked_or_archived() -> None:
    unresolved = _unrecoverable_pins()

    assert not unresolved, (
        f"{len(unresolved)} pinned source identities are neither tracked at their exact bytes nor "
        "present in provenance/frozen-code/:\n  " + "\n  ".join(unresolved)
    )


def test_retired_pinned_source_files_resolve_from_the_archive() -> None:
    """Deleting an active historical path is valid only when its exact bytes remain resolvable."""

    assert not _unrecoverable_pins()


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


def test_collect_pins_reads_the_path_keyed_declaration_shape(tmp_path: Path) -> None:
    """Pins keyed by path, with no `path` field, must be collected too.

    Artifacts declare pins two ways. The common shape names the path in a field; the other keys the
    mapping by the path itself::

        {"source_files": {"experiments/archive/producers/x.py": {"bytes": 1, "sha256": "..."}}}

    Only the first was recognised, so the second went unverified -- 32 pins that appear nowhere
    else, and the gate stayed green while `experiments/archive/producers/m0_07_oracle_graph_pretraining.py` drifted. That
    drift was introduced by a one-line import rewrite during the bio-to-potency move and nothing
    reported it, which is precisely the failure a provenance gate exists to prevent.
    """
    from forge_provenance.pins import collect_pins

    digest = "a" * 64
    other = "b" * 64
    (tmp_path / "artifact.json").write_text(
        json.dumps(
            {
                "inputs": {"declared": {"path": "src/forge/named.py", "sha256": other}},
                "source_files": {
                    "experiments/archive/producers/keyed.py": {"bytes": 12, "sha256": digest}
                },
                # A plain field whose value happens to carry a digest is not a pin: the key is not
                # a path, so it must not be collected.
                "summary": {"checksum": {"sha256": "c" * 64}},
            }
        )
    )
    found = {(pin.path, pin.sha256) for pin in collect_pins((tmp_path,))}

    assert (
        "experiments/archive/producers/keyed.py",
        digest,
    ) in found, "path-keyed pin was not collected"
    assert ("src/forge/named.py", other) in found, "field-named pin regressed"
    assert not any(path == "checksum" for path, _ in found), "a field name was mistaken for a path"


def test_the_path_keyed_shape_is_actually_present_in_this_repository() -> None:
    """Guard the guard: the shape above must still occur, or the test above proves nothing."""
    from forge_provenance.pins import collect_pins

    keyed = [
        pin
        for pin in collect_pins((REPO / "results",))
        if pin.path.endswith(".py") and "source_files" in (REPO / pin.declared_by).read_text()
    ]
    assert keyed, "no artifact declares a path-keyed source pin any more; retire the branch"
