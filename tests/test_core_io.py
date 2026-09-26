"""Tests for forge.core.io.

These are byte-compatibility tests, not behavior tests. Every artifact under results/ is
hash-pinned, so the only thing that makes this module safe to migrate onto is proof that it emits
exactly what the code it replaces emitted. The strongest evidence is the round-trip against real
committed ledgers: read one, write it back, and require the bytes to be identical.
"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import pytest

from forge.core.io import (
    atomic_write,
    csv_bytes,
    csv_gz_bytes,
    gzip_bytes,
    iter_csv,
    jsonl_bytes,
    read_csv,
    read_csv_rows,
    read_json,
    read_json_object,
    read_jsonl,
    stable_json,
    write_csv,
    write_csv_iter,
    write_json,
    write_jsonl,
)

REPO = Path(__file__).resolve().parents[1]

ROWS = [
    {"id": "p1", "smiles": "CCO", "note": 'has,comma and "quote"'},
    {"id": "p2", "smiles": "CCN", "note": "plain"},
    {"id": "p3", "smiles": "C(=O)O", "note": "line\nbreak"},
]
FIELDS = ["id", "smiles", "note"]


# ------------------------------------------------------------------ agreement with what it replaces


def test_csv_bytes_matches_the_legacy_writer() -> None:
    legacy = pytest.importorskip("forge.corpus.r1_prime_audit")
    assert csv_bytes(ROWS, FIELDS) == legacy._plain_csv_bytes(ROWS, FIELDS)


def test_csv_gz_bytes_matches_the_legacy_writer() -> None:
    legacy = pytest.importorskip("forge.corpus.r1_prime_audit")
    assert csv_gz_bytes(ROWS, FIELDS) == legacy._gzip_csv_bytes(ROWS, FIELDS)


def test_csv_uses_lf_not_crlf() -> None:
    """The csv module defaults to CRLF; every ledger here was written with LF."""
    payload = csv_bytes(ROWS, FIELDS)
    assert b"\r\n" not in payload
    assert payload.startswith(b"id,smiles,note\n")


def test_gzip_is_deterministic() -> None:
    """gzip embeds an mtime by default, which would make identical rows hash differently."""
    assert gzip_bytes(b"payload") == gzip_bytes(b"payload")


def test_gzip_records_no_timestamp() -> None:
    """Bytes 4-8 of a gzip member are the mtime field. They must be zero."""
    assert gzip_bytes(b"payload")[4:8] == b"\x00\x00\x00\x00"


def test_stable_json_agrees_with_the_loose_variant_for_valid_input() -> None:
    """The three existing _stable_json variants differ only in NaN handling."""
    value = {"b": 1, "a": [3, 2], "c": {"z": None, "y": True}}
    assert stable_json(value) == json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_stable_json_rejects_nan() -> None:
    """The strictest variant wins: bare NaN is not valid JSON and cannot be read back."""
    with pytest.raises(ValueError):
        stable_json({"value": float("nan")})


# ------------------------------------------------------------------ round-trip against real artifacts

REAL_LEDGERS = [
    p
    for p in [
        REPO / "results/m0_03/r0_constitutional.csv.gz",
        REPO / "results/m0_04/component_pool.csv.gz",
        REPO / "results/m0_07/agile_oracle_curated.csv.gz",
        REPO / "results/m0_09/lnpdb_lipid_route_ledger.csv.gz",
    ]
    if p.is_file()
]


@pytest.mark.skipif(not REAL_LEDGERS, reason="no committed .csv.gz ledgers present")
@pytest.mark.parametrize("ledger", REAL_LEDGERS, ids=lambda p: p.name)
def test_real_ledger_round_trips_byte_identically(ledger: Path, tmp_path: Path) -> None:
    """Read a committed, hash-pinned ledger and write it back. The bytes must not move.

    This is the test that licenses migrating any module onto core.io. If it fails, every artifact
    that module produces would be silently re-hashed.
    """
    rows = read_csv(ledger)
    fieldnames = list(rows[0]) if rows else []
    rewritten = csv_gz_bytes(rows, fieldnames)
    assert gzip.decompress(rewritten) == gzip.decompress(ledger.read_bytes())


@pytest.mark.skipif(not REAL_LEDGERS, reason="no committed .csv.gz ledgers present")
def test_iter_csv_agrees_with_read_csv() -> None:
    ledger = REAL_LEDGERS[0]
    streamed = list(iter_csv(ledger))
    assert streamed == read_csv(ledger)


def test_read_json_handles_gzip(tmp_path: Path) -> None:
    plain, packed = tmp_path / "a.json", tmp_path / "b.json.gz"
    plain.write_text('{"k": 1}')
    packed.write_bytes(gzip_bytes(b'{"k": 1}'))
    assert read_json(plain) == read_json(packed) == {"k": 1}


# ------------------------------------------------------------------ writing


def test_write_csv_gzips_by_extension(tmp_path: Path) -> None:
    packed, plain = tmp_path / "l.csv.gz", tmp_path / "l.csv"
    write_csv(packed, ROWS, FIELDS)
    write_csv(plain, ROWS, FIELDS)
    assert gzip.decompress(packed.read_bytes()) == plain.read_bytes()
    assert read_csv(packed) == read_csv(plain)


@pytest.mark.parametrize("suffix", (".csv", ".csv.gz"))
def test_streaming_csv_writer_is_deterministic_and_content_identical(
    tmp_path: Path, suffix: str
) -> None:
    buffered = tmp_path / f"buffered{suffix}"
    streamed = tmp_path / f"streamed{suffix}"
    repeated = tmp_path / f"repeated{suffix}"
    write_csv(buffered, ROWS, FIELDS)
    write_csv_iter(streamed, iter(ROWS), FIELDS)
    write_csv_iter(repeated, iter(ROWS), FIELDS)
    assert read_csv(streamed) == read_csv(buffered)
    assert streamed.read_bytes() == repeated.read_bytes()


def test_jsonl_round_trips(tmp_path: Path) -> None:
    records = [{"b": 1, "a": 2}, {"x": [1, 2, 3]}]
    for name in ("r.jsonl", "r.jsonl.gz"):
        target = tmp_path / name
        write_jsonl(target, records)
        assert read_jsonl(target) == records


def test_jsonl_lines_are_deterministic() -> None:
    assert jsonl_bytes([{"b": 1, "a": 2}]) == b'{"a":2,"b":1}\n'


def test_write_json_pretty_is_sorted_and_newline_terminated(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    write_json(target, {"b": 1, "a": 2})
    assert target.read_text() == '{\n  "a": 2,\n  "b": 1\n}\n'


def test_atomic_write_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "out.bin"
    atomic_write(target, b"payload")
    assert target.read_bytes() == b"payload"


def test_atomic_write_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    atomic_write(tmp_path / "out.bin", b"payload")
    assert [p.name for p in tmp_path.iterdir()] == ["out.bin"]


@pytest.mark.parametrize("mode", [None, 0o644])
def test_atomic_write_does_not_clobber_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: int | None
) -> None:
    """A failed write must leave the previous artifact intact, not a truncated one.

    The failure is injected at the rename, which is the last step and the only one that can
    expose a partial result. The previous bytes must survive and the temporary must be cleaned up.
    """
    target = tmp_path / "out.bin"
    atomic_write(target, b"original")

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("forge.core.io.os.replace", explode)
    with pytest.raises(RuntimeError):
        atomic_write(target, b"replacement", mode=mode)

    assert target.read_bytes() == b"original"
    assert [p.name for p in tmp_path.iterdir()] == ["out.bin"]


@pytest.mark.parametrize("mode,expected", [(None, 0o600), (0o644, 0o644)])
def test_atomic_write_publishes_requested_permissions(
    tmp_path: Path, mode: int | None, expected: int
) -> None:
    target = tmp_path / "out.bin"
    atomic_write(target, b"payload", mode=mode)
    assert target.stat().st_mode & 0o777 == expected
    assert target.read_bytes() == b"payload"


def test_atomic_write_cleans_up_when_permission_change_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.bin"
    atomic_write(target, b"original")

    def fail(*args: object) -> None:
        raise OSError("permission failure")

    monkeypatch.setattr("forge.core.io.os.fchmod", fail)
    with pytest.raises(OSError, match="permission failure"):
        atomic_write(target, b"replacement", mode=0o644)
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_replaces_existing_content(tmp_path: Path) -> None:
    target = tmp_path / "out.bin"
    atomic_write(target, b"first")
    atomic_write(target, b"second")
    assert target.read_bytes() == b"second"


def test_csv_bytes_preserves_embedded_commas_quotes_and_newlines(tmp_path: Path) -> None:
    """Quoting must survive a round trip, or ledger columns silently shift."""
    target = tmp_path / "l.csv.gz"
    write_csv(target, ROWS, FIELDS)
    assert read_csv(target) == [dict(row) for row in ROWS]


def test_gzip_bytes_matches_stdlib_decompression() -> None:
    assert gzip.decompress(gzip_bytes(b"payload")) == b"payload"
    with gzip.GzipFile(fileobj=io.BytesIO(gzip_bytes(b"payload"))) as handle:
        assert handle.read() == b"payload"


# ------------------------------------------------------------------ reading with a domain error


class DomainError(ValueError):
    """Stands in for the ~120 module-specific error classes the local readers raise."""


def test_read_json_object_returns_the_document(tmp_path: Path) -> None:
    target = tmp_path / "config.json"
    target.write_text('{"schema_version": "v1", "n": 2}')
    assert read_json_object(target, error=DomainError) == {"schema_version": "v1", "n": 2}


def test_read_json_object_handles_gzip(tmp_path: Path) -> None:
    target = tmp_path / "config.json.gz"
    target.write_bytes(gzip_bytes(b'{"a": 1}'))
    assert read_json_object(target, error=DomainError) == {"a": 1}


@pytest.mark.parametrize(
    ("name", "payload"),
    [("missing.json", None), ("bad.json", b"{not json"), ("empty.json", b"")],
)
def test_read_json_object_raises_the_declared_error(
    tmp_path: Path, name: str, payload: bytes | None
) -> None:
    target = tmp_path / name
    if payload is not None:
        target.write_bytes(payload)
    with pytest.raises(DomainError):
        read_json_object(target, error=DomainError)


@pytest.mark.parametrize("payload", [b"[1, 2, 3]", b'"a string"', b"7", b"null"])
def test_read_json_object_rejects_a_non_object(tmp_path: Path, payload: bytes) -> None:
    """121 of 122 local copies assert the parsed value is an object; that is the contract."""
    target = tmp_path / "doc.json"
    target.write_bytes(payload)
    with pytest.raises(DomainError, match="must contain a JSON object"):
        read_json_object(target, error=DomainError)


def test_read_json_object_preserves_the_original_exception(tmp_path: Path) -> None:
    """Losing __cause__ turns 'the config is invalid' into a dead end."""
    with pytest.raises(DomainError) as caught:
        read_json_object(tmp_path / "absent.json", error=DomainError)
    assert isinstance(caught.value.__cause__, FileNotFoundError)

    bad = tmp_path / "bad.json"
    bad.write_bytes(b"{not json")
    with pytest.raises(DomainError) as caught:
        read_json_object(bad, error=DomainError)
    assert isinstance(caught.value.__cause__, json.JSONDecodeError)


def test_read_json_object_label_names_the_input(tmp_path: Path) -> None:
    with pytest.raises(DomainError, match="frozen split manifest"):
        read_json_object(tmp_path / "absent.json", error=DomainError, label="frozen split manifest")


def test_read_json_object_defaults_to_valueerror(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        read_json_object(tmp_path / "absent.json")


def test_read_csv_rows_returns_rows(tmp_path: Path) -> None:
    target = tmp_path / "ledger.csv.gz"
    write_csv(target, ROWS, FIELDS)
    assert read_csv_rows(target, error=DomainError) == [dict(row) for row in ROWS]


def test_read_csv_rows_raises_the_declared_error_when_absent(tmp_path: Path) -> None:
    with pytest.raises(DomainError) as caught:
        read_csv_rows(tmp_path / "absent.csv", error=DomainError)
    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_read_csv_rows_reports_every_missing_field_at_once(tmp_path: Path) -> None:
    """A caller fixing a schema mismatch should see the whole gap, not the first item."""
    target = tmp_path / "ledger.csv"
    write_csv(target, ROWS, FIELDS)
    with pytest.raises(DomainError, match=r"missing fields \['role', 'sha256'\]"):
        read_csv_rows(target, error=DomainError, required_fields=["id", "role", "sha256"])


def test_read_csv_rows_accepts_present_fields(tmp_path: Path) -> None:
    target = tmp_path / "ledger.csv"
    write_csv(target, ROWS, FIELDS)
    assert len(read_csv_rows(target, error=DomainError, required_fields=["id", "smiles"])) == 3


def test_read_csv_rows_treats_an_empty_ledger_as_missing_every_field(tmp_path: Path) -> None:
    target = tmp_path / "empty.csv"
    target.write_text("")
    with pytest.raises(DomainError, match="missing fields"):
        read_csv_rows(target, error=DomainError, required_fields=["id"])
