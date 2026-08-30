"""In-container execution identity for the generic Modal launcher."""

from __future__ import annotations

from collections.abc import Callable

from experiments._runtime.backends.local import LocalBackend
from experiments._runtime.stage import RunContext, StageCallable, StageResult


class ModalRuntimeBackend(LocalBackend):
    """Execute a stage inside Modal while preserving the ordinary stage callable.

    Resource allocation and transport happen in ``experiments._runtime.modal_app``.  Once the
    container starts, seeding and deterministic-kernel handling are identical to local execution.
    """

    name = "modal"

    def __init__(self, *, progress_commit: Callable[[], None] | None = None) -> None:
        self._progress_commit = progress_commit

    def execute(self, function: StageCallable, context: RunContext) -> StageResult:
        context._set_progress_committer(self._progress_commit)
        return super().execute(function, context)


__all__ = ["ModalRuntimeBackend"]
