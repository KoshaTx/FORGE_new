"""Authenticate explicitly admitted raw files without materializing their contents."""

import hashlib
from pathlib import Path


def local_path(root, record):
    location = record.get("path", record.get("location"))
    if not isinstance(location, str) or "#" in location:
        raise ValueError("expected a whole local source path")
    path = (Path(root) / location).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("source path escapes repository")
    return path


def signature(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def stream_verify(root, record, allowed_sources, cache):
    """Caller supplies authenticated completion descriptors; cache stores no file bytes."""
    path = local_path(root, record)
    matches = [
        source
        for source in allowed_sources
        if local_path(root, source) == path and source["sha256"] == record["sha256"]
    ]
    if len(matches) != 1:
        raise ValueError("source is not uniquely allowlisted")
    expected = matches[0]
    size = expected.get("bytes")
    if type(size) is not int or size <= 0 or expected.get("complete") is not True:
        raise ValueError("allowlist requires a complete positive-size source")
    if path.name.endswith(".part"):
        raise ValueError("partial source prohibited")
    before = signature(path)
    if before[2] != size:
        raise ValueError("source byte-size mismatch")
    key = (str(path), record["sha256"], before)
    if key in cache:
        return dict(cache[key])
    hasher = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
            count += len(chunk)
    if signature(path) != before:
        raise ValueError("source changed while hashing")
    if count != size or hasher.hexdigest() != record["sha256"]:
        raise ValueError("source digest mismatch")
    result = {
        "path": str(path.relative_to(Path(root).resolve())),
        "sha256": record["sha256"],
        "bytes": count,
        "stat_signature": list(before),
        "verification": "streamed_SHA256_1MiB_chunks",
    }
    cache[key] = result
    return dict(result)
