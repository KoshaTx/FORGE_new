"""Serialization and atomic writes.

Every function here is byte-compatible with the copies it replaces. That is the whole requirement:
artifacts already on disk are hash-pinned, and those pins are the paper's evidence chain, so a
serializer that emits one different byte silently invalidates every result it touches.

Two details carry that compatibility and are easy to lose:

  `lineterminator="\\n"` -- Python's csv module defaults to CRLF. Every existing FORGE ledger was
  written with LF, so the default would change every line of every CSV.

  `mtime=0` on gzip -- gzip embeds a modification timestamp by default, so the same rows compressed
  twice produce different bytes. Every existing `.csv.gz` was written with the timestamp zeroed,
  which is what makes them reproducible at all.

Where the existing copies disagreed, the strictest behavior was adopted deliberately. `stable_json`
takes `allow_nan=False` from the 29-copy variant rather than the 32-copy one: the loose variant
emits bare `NaN`, which is not valid JSON and which a strict reader cannot load back. That can now
raise where it previously wrote a broken artifact, and raising is the correct outcome.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------- JSON


def stable_json(value: Any) -> str:
    """Serialize deterministically: sorted keys, no incidental whitespace, no NaN.

    Sorted so a dict's iteration order cannot leak into an artifact's hash, and compact so
    formatting changes cannot either.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_json_bytes(value: Any) -> bytes:
    return stable_json(value).encode()


def pretty_json_bytes(value: Any) -> bytes:
    """Indented, sorted, newline-terminated -- the shape used for human-read `result.json` files."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


# --------------------------------------------------------------------------------- CSV


def csv_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    """Encode rows as CSV with LF line endings.

    `lineterminator="\\n"` is not a style choice: the csv module defaults to CRLF, and every
    ledger in this repository was written with LF.
    """
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def gzip_bytes(payload: bytes) -> bytes:
    """Gzip with the timestamp zeroed, so identical input always gives identical output."""
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(payload)
    return output.getvalue()


def csv_gz_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    return gzip_bytes(csv_bytes(rows, fieldnames))


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV, transparently handling gzip by extension.

    Returns row dicts of strings, matching what the ~60 local readers return -- callers parse their
    own numerics. Deliberately not pandas: the return type is what existing call sites expect.
    """
    if path.suffix == ".gz":
        with gzip.open(path, "rt", newline="") as handle:
            return list(csv.DictReader(handle))
    with path.open("rt", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_csv(path: Path) -> Iterator[dict[str, str]]:
    """Stream a CSV row by row, for ledgers too large to hold in memory."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", newline="") as handle:
            yield from csv.DictReader(handle)
    else:
        with path.open("rt", newline="") as handle:
            yield from csv.DictReader(handle)


# --------------------------------------------------------------------------------- JSON lines


def read_json(path: Path) -> Any:
    """Read a JSON document, transparently handling gzip by extension."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[Any]:
    return list(iter_jsonl(path))


def iter_jsonl(path: Path) -> Iterator[Any]:
    """Stream JSON lines, skipping blank lines. Handles gzip by extension."""
    opener = gzip.open(path, "rt") if path.suffix == ".gz" else path.open("rt")
    with opener as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def jsonl_bytes(records: Iterable[Any]) -> bytes:
    """Encode records as newline-delimited JSON, each line deterministically serialized."""
    return "".join(f"{stable_json(record)}\n" for record in records).encode()


# --------------------------------------------------------------------------------- writing


def atomic_write(path: Path, payload: bytes) -> None:
    """Write bytes so the destination is never observed partially written.

    Writes to a temporary file in the same directory, fsyncs it, then renames over the target --
    rename within a directory is atomic. The repository's engineering contract requires this:
    "calculate first, then write complete artifacts. Do not leave a plausible-looking result after
    an exception." A half-written ledger that still parses is the failure mode this prevents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    atomic_write(path, pretty_json_bytes(value) if pretty else stable_json_bytes(value))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write a CSV, gzipping when the path ends in `.gz`."""
    payload = csv_bytes(rows, fieldnames)
    atomic_write(path, gzip_bytes(payload) if path.suffix == ".gz" else payload)


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    payload = jsonl_bytes(records)
    atomic_write(path, gzip_bytes(payload) if path.suffix == ".gz" else payload)


__all__ = [
    "atomic_write",
    "csv_bytes",
    "csv_gz_bytes",
    "gzip_bytes",
    "iter_csv",
    "iter_jsonl",
    "jsonl_bytes",
    "pretty_json_bytes",
    "read_csv",
    "read_json",
    "read_jsonl",
    "stable_json",
    "stable_json_bytes",
    "write_csv",
    "write_json",
    "write_jsonl",
]
