from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

# The tests directory is not importable by default under every invocation
# (`make test` sets only PYTHONPATH=src), so make the quarantine registry
# reachable regardless of how pytest was started.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from unreproducible_pins import (  # noqa: E402
    UNREPRODUCIBLE_PIN_REASON,
    UNREPRODUCIBLE_PIN_SETUP_ERRORS,
    UNREPRODUCIBLE_PIN_TESTS,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-external-artifacts",
        action="store_true",
        help="Fail instead of skipping tests whose declared historical inputs are unavailable.",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    repo = Path(__file__).resolve().parents[1]
    missing = sorted(
        {
            str(relative)
            for marker in item.iter_markers("requires_artifacts")
            for relative in marker.args
            if not (repo / relative).is_file()
        }
    )
    if missing:
        message = "External historical artifacts unavailable: " + ", ".join(missing)
        if item.config.getoption("--require-external-artifacts"):
            pytest.fail(message)
        pytest.skip(message)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark tests with unrecoverable pinned inputs as expected failures.

    Applied by exact node id only, never by file or pattern, so a genuine
    regression in a neighbouring test in the same module still fails loudly.
    The marker is non-strict: if a quarantined test is repaired it reports
    XPASS rather than failing, which is the signal to remove it from the list.
    """

    for item in items:
        if item.nodeid in UNREPRODUCIBLE_PIN_TESTS:
            item.add_marker(pytest.mark.xfail(reason=UNREPRODUCIBLE_PIN_REASON, strict=False))
        elif item.nodeid in UNREPRODUCIBLE_PIN_SETUP_ERRORS:
            # The pin check fires in a module-scoped fixture; a setup error
            # cannot be expressed as xfail, so it is skipped with the same reason.
            item.add_marker(pytest.mark.skip(reason=UNREPRODUCIBLE_PIN_REASON))


@dataclass(frozen=True)
class PrerevealRegistryView:
    """Minimal filesystem view in which the frozen prereveal gate remains testable."""

    repo: Path
    protocol: Path


@pytest.fixture
def prereveal_registry_view(monkeypatch: pytest.MonkeyPatch) -> PrerevealRegistryView:
    """Emulate only the historical sample-absence predicate for unit tests.

    The live repository is postreveal and must now fail the historical
    sample-absence gate. Synthetic registry-pair tests still need to exercise
    every prereveal validation branch using the original hash-pinned absolute
    paths. This fixture masks exactly the sealed sample path's ``exists`` call;
    a separate test without this fixture proves that the live validator fails
    closed after reveal. No production artifact is modified.
    """

    source_repo = Path(__file__).resolve().parents[1]
    source_protocol = source_repo / "configs/route/phase1_ugi3_route_registry_pair_protocol_v1.json"
    protocol = json.loads(source_protocol.read_text())
    anchor = protocol["sealed_holdout_anchor"]
    sealed_sample = source_repo / anchor["sealed_sample_output"]
    assert sealed_sample.exists()
    unpatched_exists = Path.exists

    def prereveal_exists(path: Path) -> bool:
        if path == sealed_sample:
            return False
        return unpatched_exists(path)

    monkeypatch.setattr(Path, "exists", prereveal_exists)
    return PrerevealRegistryView(
        repo=source_repo,
        protocol=source_protocol,
    )
