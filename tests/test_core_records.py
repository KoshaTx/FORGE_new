"""Tests for forge.core.records.

The load-bearing test is `test_every_real_pin_round_trips_byte_identically`. These records appear
in 91 tracked artifacts whose hashes are pinned, so a round-trip that drops or reorders a key would
not raise -- it would silently corrupt an artifact the next time a stage rewrote it. Everything
else here is a corollary of that.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from forge.core.hashing import PinError, sha256_file
from forge.core.records import ArtifactRef, PinnedInput, RecordError, is_pin

REPO = Path(__file__).resolve().parents[1]


def _load(path: Path) -> Any:
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def _collect_pins(node: Any, found: list[dict[str, Any]]) -> None:
    if isinstance(node, dict):
        if isinstance(node.get("sha256"), str) and "path" in node:
            found.append(node)
        for value in node.values():
            _collect_pins(value, found)
    elif isinstance(node, list):
        for value in node:
            _collect_pins(value, found)


def real_pins() -> list[dict[str, Any]]:
    """Every path-bearing digest record in the tracked artifacts."""
    found: list[dict[str, Any]] = []
    results = REPO / "results"
    if not results.exists():
        return found
    for path in sorted(results.rglob("*.json")) + sorted(results.rglob("*.json.gz")):
        try:
            document = _load(path)
        except (OSError, ValueError, EOFError):
            continue
        _collect_pins(document, found)
    return found


REAL_PINS = real_pins()


# ------------------------------------------------------------------ the test that licenses this


@pytest.mark.skipif(not REAL_PINS, reason="no tracked artifacts present")
def test_every_real_pin_round_trips_byte_identically() -> None:
    """Parse and re-serialize every pin on disk; the JSON bytes must not move.

    Compared under `sort_keys=True`, which is how the repository's writers emit JSON, so this
    checks content rather than incidental ordering. A dropped extension key fails here.
    """
    for record in REAL_PINS:
        parsed = PinnedInput.from_mapping(record)
        before = json.dumps(record, sort_keys=True)
        after = json.dumps(parsed.to_mapping(), sort_keys=True)
        assert after == before, f"round trip changed {record.get('path')!r}"


@pytest.mark.skipif(not REAL_PINS, reason="no tracked artifacts present")
def test_the_corpus_of_real_pins_is_large_and_varied() -> None:
    """Guard the guard: if this suddenly matched two records, the test above would prove nothing."""
    assert len(REAL_PINS) > 500
    shapes = {tuple(sorted(record)) for record in REAL_PINS}
    assert len(shapes) > 10


@pytest.mark.skipif(not REAL_PINS, reason="no tracked artifacts present")
def test_no_real_pin_loses_an_extension_key() -> None:
    for record in REAL_PINS:
        assert set(PinnedInput.from_mapping(record).to_mapping()) == set(record)


# ------------------------------------------------------------------ preserving the tail


def test_unknown_keys_survive() -> None:
    """32 distinct extra-key combinations exist; enumerating them would drop the next one."""
    record = {"path": "a.csv", "sha256": "a" * 64, "job_id": "j1", "note": "why", "seed": 7}
    assert PinnedInput.from_mapping(record).to_mapping() == record


def test_the_most_common_real_shape_round_trips() -> None:
    """`{job_id, path, sha256}` is the single most common shape at 960 instances."""
    record = {"job_id": "fold-3", "path": "x.json", "sha256": "b" * 64}
    assert PinnedInput.from_mapping(record).to_mapping() == record


def test_size_is_absent_when_it_was_absent() -> None:
    """Emitting `bytes` unprompted would add a key the artifact never had."""
    parsed = PinnedInput.from_mapping({"path": "a.csv", "sha256": "a" * 64})
    assert parsed.size is None
    assert "bytes" not in parsed.to_mapping()


def test_size_is_kept_when_present() -> None:
    record = {"path": "a.csv", "sha256": "a" * 64, "bytes": 12}
    assert PinnedInput.from_mapping(record).to_mapping() == record


@pytest.mark.parametrize("missing", [{"path": "a"}, {"sha256": "a" * 64}, {}])
def test_a_pin_without_its_core_pair_is_rejected(missing: dict[str, Any]) -> None:
    with pytest.raises(RecordError):
        PinnedInput.from_mapping(missing)


# ------------------------------------------------------------------ immutability


def test_records_are_frozen() -> None:
    pin = PinnedInput.from_mapping({"path": "a", "sha256": "a" * 64})
    with pytest.raises(Exception):
        pin.path = "b"  # type: ignore[misc]


def test_extras_cannot_be_mutated_through_the_record() -> None:
    """A frozen record whose mapping was editable would only look immutable."""
    pin = PinnedInput.from_mapping({"path": "a", "sha256": "a" * 64, "job_id": "j"})
    with pytest.raises(TypeError):
        pin.extra["job_id"] = "other"  # type: ignore[index]


def test_mutating_the_source_mapping_does_not_affect_the_record() -> None:
    source = {"path": "a", "sha256": "a" * 64, "job_id": "j"}
    pin = PinnedInput.from_mapping(source)
    source["job_id"] = "changed"
    assert pin.extra["job_id"] == "j"


# ------------------------------------------------------------------ verification stays strict


def test_resolve_verifies_against_the_real_file(tmp_path: Path) -> None:
    target = tmp_path / "input.json"
    target.write_bytes(b"{}")
    pin = PinnedInput(path="input.json", sha256=sha256_file(target))
    assert pin.resolve(tmp_path) == target.resolve()


def test_resolve_rejects_a_changed_file(tmp_path: Path) -> None:
    """The whole point of the record: it must still fail closed."""
    target = tmp_path / "input.json"
    target.write_bytes(b"{}")
    pin = PinnedInput(path="input.json", sha256=sha256_file(target))
    target.write_bytes(b'{"changed": true}')
    with pytest.raises(PinError):
        pin.resolve(tmp_path)


def test_resolve_still_rejects_escape_from_the_repository(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}")
    inside = tmp_path / "repo"
    inside.mkdir()
    pin = PinnedInput(path="../outside.json", sha256=sha256_file(outside))
    with pytest.raises(PinError, match="outside the repository"):
        pin.resolve(inside)


def test_extras_do_not_weaken_verification(tmp_path: Path) -> None:
    """A record carrying metadata is still checked on its path and digest."""
    target = tmp_path / "input.json"
    target.write_bytes(b"{}")
    pin = PinnedInput.from_mapping(
        {"path": "input.json", "sha256": str(sha256_file(target)), "job_id": "j"}
    )
    assert pin.resolve(tmp_path) == target.resolve()
    target.write_bytes(b"tampered")
    with pytest.raises(PinError):
        pin.resolve(tmp_path)


# ------------------------------------------------------------------ discriminating the concepts


def test_is_pin_accepts_path_bearing_records() -> None:
    assert is_pin({"path": "a", "sha256": "a" * 64})
    assert is_pin({"path": "a", "sha256": "a" * 64, "job_id": "j"})


def test_is_pin_rejects_the_other_digest_records() -> None:
    """Downloads, vendored assets and tensor refs also carry sha256 but name no path."""
    assert not is_pin({"url": "https://x", "sha256": "a" * 64, "cache_asset": "c"})
    assert not is_pin({"asset": "uspto.csv", "bytes": 4, "sha256": "a" * 64})
    assert not is_pin({"dtype": "float32", "shape": [4], "sha256": "a" * 64})


# ------------------------------------------------------------------ ArtifactRef


def test_artifact_ref_round_trips() -> None:
    record = {"schema_version": "forge.x.v1", "sha256": "a" * 64, "rows": 12}
    assert ArtifactRef.from_mapping(record).to_mapping() == record


def test_artifact_ref_omits_rows_when_absent() -> None:
    parsed = ArtifactRef.from_mapping({"schema_version": "forge.x.v1", "sha256": "a" * 64})
    assert parsed.rows is None
    assert "rows" not in parsed.to_mapping()


def test_artifact_ref_keeps_unknown_keys() -> None:
    record = {"schema_version": "v1", "sha256": "a" * 64, "columns": ["a", "b"]}
    assert ArtifactRef.from_mapping(record).to_mapping() == record


@pytest.mark.parametrize("missing", [{"sha256": "a" * 64}, {"schema_version": "v1"}, {}])
def test_artifact_ref_requires_its_core_pair(missing: dict[str, Any]) -> None:
    with pytest.raises(RecordError):
        ArtifactRef.from_mapping(missing)


def test_a_produced_ref_is_not_a_consumed_pin() -> None:
    """They are different records, not one with optional fields."""
    ref = {"schema_version": "v1", "sha256": "a" * 64, "rows": 3}
    assert not is_pin(ref)
    with pytest.raises(RecordError):
        PinnedInput.from_mapping(ref)
