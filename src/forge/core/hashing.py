"""Content hashing, and the pinned-input check built on it.

Provenance is the load-bearing concern in this repository -- a result is only usable if you can
say which bytes produced it -- and it is also the most duplicated: roughly 238 local hash helpers
and 50 separate `_pin` implementations. The copies are not equivalent. Some re-hash the file and
reject symlinks; others only check that the path sits inside the repository. Which guarantee you
got depended on which module you were in.

`verify_pin` adopts the strictest behavior found among them, deliberately. Loosening a provenance
check to make a call site pass would be a silent downgrade of the evidence chain.

The signatures match the existing `sha256_file(path, chunk_size=1 << 20)` and `sha256_bytes(payload)`
in `forge.data.r0_splits` and `forge.data.r1_prime_audit`, so migrating a module is an import
change and nothing else.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.types import Sha256

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

DEFAULT_CHUNK_SIZE = 1 << 20


class PinError(ValueError):
    """A declared input pin is malformed, unreachable, or does not match its recorded digest."""


def is_sha256(value: object) -> bool:
    """True when `value` is a lowercase 64-character hex digest.

    Uppercase is rejected on purpose: digests are compared by string equality throughout the
    codebase, so accepting both cases would let two spellings of the same digest miss each other.
    """
    return isinstance(value, str) and bool(_SHA256_RE.match(value))


def sha256_bytes(payload: bytes) -> Sha256:
    """Return a byte payload's SHA-256 digest."""
    return Sha256(hashlib.sha256(payload).hexdigest())


def sha256_file(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Sha256:
    """Return a file's SHA-256 digest, read in bounded chunks.

    Chunked so that hashing the 96 MB R1 corpus or a 768 MB tensor cache does not depend on
    holding the whole file in memory.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return Sha256(digest.hexdigest())


def resolve_pin(record: Mapping[str, Any], repo: Path, *, label: str) -> Path:
    """Validate one `{"path", "sha256"}` record and return the file it names.

    Fails closed on every count: the record must have exactly those two keys, the digest must be
    well formed, the path must resolve inside `repo`, it must be a real file rather than a symlink
    pointing elsewhere, and its contents must hash to the recorded digest.

    The symlink and containment checks matter together. Without both, a pinned input could be made
    to satisfy its own hash while actually reading bytes from outside the repository.
    """
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise PinError(f"malformed input pin for {label}: expected exactly path and sha256")

    declared = record["sha256"]
    if not is_sha256(declared):
        raise PinError(f"malformed sha256 for {label}: {declared!r}")

    path = Path(record["path"])
    candidate = path if path.is_absolute() else repo / path

    # Check for a symlink before resolving, not after: resolve() follows the link, so asking a
    # resolved path whether it is a symlink always answers no.
    if candidate.is_symlink():
        raise PinError(f"pinned input for {label} is a symlink: {candidate}")

    resolved = candidate.resolve()
    try:
        resolved.relative_to(repo.resolve())
    except ValueError as error:
        raise PinError(
            f"pinned input for {label} resolves outside the repository: {resolved}"
        ) from error

    if not resolved.is_file():
        raise PinError(f"pinned input for {label} is missing: {resolved}")

    observed = sha256_file(resolved)
    if observed != declared:
        raise PinError(
            f"pinned input for {label} changed: expected {declared}, found {observed} at {resolved}"
        )
    return resolved


def pin_record(path: Path, repo: Path) -> dict[str, Any]:
    """Build the `{"path", "sha256", "bytes"}` record an artifact writes for one of its inputs.

    The path is recorded repository-relative so an artifact stays portable between checkouts --
    an absolute path baked into a result is what makes it unverifiable on another machine, which
    is exactly how this project lost track of its inputs once already.
    """
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(repo.resolve())),
        "sha256": str(sha256_file(resolved)),
        "bytes": resolved.stat().st_size,
    }


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "PinError",
    "is_sha256",
    "pin_record",
    "resolve_pin",
    "sha256_bytes",
    "sha256_file",
]
