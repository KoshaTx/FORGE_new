"""The CLI must report domain failures cleanly and let genuine defects keep their traceback."""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from forge.cli import main
from forge_experiment.errors import ExperimentError


def _run_raising(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> int:
    """Drive `main` so the parsed command raises `error`, bypassing real experiment plumbing."""

    def fake_parse(self: argparse.ArgumentParser, argv: Any = None) -> argparse.Namespace:
        def command(_: argparse.Namespace) -> int:
            raise error

        return argparse.Namespace(function=command)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fake_parse)
    return main([])


@pytest.mark.parametrize(
    "error",
    [
        ExperimentError("stage contract changed"),
        FileNotFoundError("configs/experiments/missing.json"),
        RuntimeError("planner cache is stale"),
        ValueError("seed must be a positive integer"),
    ],
    ids=["experiment", "missing-file", "runtime", "value"],
)
def test_domain_failures_are_reported_and_exit_two(
    error: Exception, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run_raising(monkeypatch, error) == 2
    assert capsys.readouterr().err.startswith("forge: ")


@pytest.mark.parametrize(
    "error",
    [
        RecursionError("maximum recursion depth exceeded"),
        NotImplementedError("backend is not wired up"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    ],
    ids=["recursion", "not-implemented", "unicode"],
)
def test_defects_keep_their_traceback(error: Exception, monkeypatch: pytest.MonkeyPatch) -> None:
    """These are subclasses of the two bases every domain error uses, and must not be swallowed.

    `RecursionError` and `NotImplementedError` are `RuntimeError`s and the `UnicodeError` family
    are `ValueError`s, so the broad `except` reached them and printed a tidy `forge: ...` with exit
    2 -- a runaway recursion or a mis-encoded input was indistinguishable from a failed validation,
    and the stack that would have explained it was gone.
    """
    with pytest.raises(type(error)):
        _run_raising(monkeypatch, error)


def test_keyboard_interrupt_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancelled run must stay cancelled rather than being reported as a validation failure."""
    with pytest.raises(KeyboardInterrupt):
        _run_raising(monkeypatch, KeyboardInterrupt())
