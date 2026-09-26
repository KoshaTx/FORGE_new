"""Shared filesystem conventions for primary-source acquisition."""

from pathlib import Path

from forge.core.hashing import sha256_bytes
from forge.core.io import atomic_write


def content_sha256(content: bytes) -> str:
    """Keep the acquisition helper's content keyword while using the shared hasher."""
    return sha256_bytes(content)


def portable_path(path: Path) -> str:
    """Record a resolved path relative to the working directory when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Publish source evidence with the acquisition modules' existing permissions."""
    atomic_write(path, content, mode=0o644)
