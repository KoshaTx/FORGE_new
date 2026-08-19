"""Tests for forge.core.hashing.

The point of these is not that hashing works -- it is that the new shared implementation agrees
exactly with the copies it replaces. If `forge.core.hashing.sha256_file` returned anything other
than what `forge.data.r0_splits.sha256_file` returns, migrating a module onto core would silently
rewrite the pins in every artifact that module produces.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from forge.core.hashing import (
    PinError,
    is_sha256,
    pin_record,
    resolve_pin,
    sha256_bytes,
    sha256_file,
    sha256_json,
)

REPO = Path(__file__).resolve().parents[1]


def test_sha256_bytes_matches_hashlib() -> None:
    payload = b"forge"
    assert sha256_bytes(payload) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(b"x" * 5_000_000)
    assert sha256_file(target) == hashlib.sha256(target.read_bytes()).hexdigest()


def test_sha256_file_is_chunk_size_invariant(tmp_path: Path) -> None:
    """A file spanning several chunks must hash identically at any chunk size."""
    target = tmp_path / "payload.bin"
    target.write_bytes(bytes(range(256)) * 9_000)
    assert sha256_file(target, chunk_size=7) == sha256_file(target, chunk_size=1 << 20)


def test_agrees_with_the_implementation_it_replaces(tmp_path: Path) -> None:
    """Byte-compatibility with the pre-existing helper. This is the one that matters."""
    legacy = pytest.importorskip("forge.data.r0_splits")
    target = tmp_path / "payload.bin"
    target.write_bytes(b"lipid" * 100_000)
    assert sha256_file(target) == legacy.sha256_file(target)
    assert sha256_bytes(b"lipid") == legacy.sha256_bytes(b"lipid")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a" * 64, True),
        ("A" * 64, False),  # uppercase rejected: digests are compared as strings
        ("a" * 63, False),
        ("g" * 64, False),
        (None, False),
        (b"a" * 64, False),
    ],
)
def test_is_sha256(value: object, expected: bool) -> None:
    assert is_sha256(value) is expected


def test_resolve_pin_accepts_a_matching_input(tmp_path: Path) -> None:
    target = tmp_path / "input.json"
    target.write_bytes(b"{}")
    record = {"path": "input.json", "sha256": sha256_file(target)}
    assert resolve_pin(record, tmp_path, label="input") == target.resolve()


def test_resolve_pin_rejects_a_changed_input(tmp_path: Path) -> None:
    target = tmp_path / "input.json"
    target.write_bytes(b"{}")
    record = {"path": "input.json", "sha256": sha256_file(target)}
    target.write_bytes(b'{"changed": true}')
    with pytest.raises(PinError, match="changed"):
        resolve_pin(record, tmp_path, label="input")


@pytest.mark.parametrize(
    "record",
    [
        {"path": "input.json"},
        {"sha256": "a" * 64},
        {"path": "input.json", "sha256": "a" * 64, "bytes": 2},
        {"path": "input.json", "sha256": "not-a-digest"},
    ],
)
def test_resolve_pin_rejects_malformed_records(record: dict[str, object], tmp_path: Path) -> None:
    (tmp_path / "input.json").write_bytes(b"{}")
    with pytest.raises(PinError):
        resolve_pin(record, tmp_path, label="input")


def test_resolve_pin_rejects_escape_from_the_repository(tmp_path: Path) -> None:
    """The containment check: a pin must not be able to read bytes from outside the repo."""
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}")
    inside = tmp_path / "repo"
    inside.mkdir()
    record = {"path": "../outside.json", "sha256": sha256_file(outside)}
    with pytest.raises(PinError, match="outside the repository"):
        resolve_pin(record, inside, label="input")


def test_resolve_pin_rejects_a_symlink(tmp_path: Path) -> None:
    """Without this, a symlink could satisfy its own hash while reading foreign bytes."""
    real = tmp_path / "real.json"
    real.write_bytes(b"{}")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    record = {"path": "link.json", "sha256": sha256_file(real)}
    with pytest.raises(PinError, match="symlink"):
        resolve_pin(record, tmp_path, label="input")


def test_pin_record_round_trips(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "input.json"
    target.parent.mkdir()
    target.write_bytes(b'{"a": 1}')
    record = pin_record(target, tmp_path)
    assert record == {"path": "nested/input.json", "sha256": sha256_file(target), "bytes": 8}
    assert (
        resolve_pin({"path": record["path"], "sha256": record["sha256"]}, tmp_path, label="input")
        == target.resolve()
    )


def test_pin_record_paths_are_relative_for_portability(tmp_path: Path) -> None:
    target = tmp_path / "input.json"
    target.write_bytes(b"{}")
    assert not Path(pin_record(target, tmp_path)["path"]).is_absolute()


def test_sha256_json_is_order_independent() -> None:
    """Structurally equal documents must hash the same, whatever order they were built in."""
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_sha256_json_distinguishes_different_documents() -> None:
    assert sha256_json({"a": 1}) != sha256_json({"a": 2})


def test_sha256_json_matches_the_composition_it_replaces() -> None:
    """The 55 local _sha256_payload copies are sha256 over the canonical JSON."""
    import hashlib

    from forge.core.io import stable_json

    value = {"b": [3, 2], "a": {"z": None}}
    assert sha256_json(value) == hashlib.sha256(stable_json(value).encode()).hexdigest()
