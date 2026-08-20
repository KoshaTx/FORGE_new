"""Compatibility import for the repository provenance tools.

The resolver is runtime infrastructure because active experiments consume historical pins. The
maintenance package re-exports it so existing verification commands keep a stable API.
"""

from experiments._runtime.historical import (
    HISTORICAL_PIN_ARCHIVE_SCHEMA,
    HistoricalPinArchive,
    HistoricalPinArchiveError,
    HistoricalPinEntry,
    resolve_current_path,
    resolve_pinned_input,
)

__all__ = [
    "HISTORICAL_PIN_ARCHIVE_SCHEMA",
    "HistoricalPinArchive",
    "HistoricalPinArchiveError",
    "HistoricalPinEntry",
    "resolve_current_path",
    "resolve_pinned_input",
]
