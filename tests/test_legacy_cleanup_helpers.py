"""Observable contracts of consolidated historical infrastructure helpers."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module,keyword,error",
    [
        ("forge.corpus.r0_splits", "chunk_size", "SplitError"),
        ("forge.corpus.r1_prime_audit", "chunk_size", "AuditError"),
        ("forge_provenance.pins", "chunk", "FileNotFoundError"),
    ],
)
@pytest.mark.parametrize("payload", [b"", b"abcdef" * 50])
def test_historical_hashing_keywords_and_missing_input(
    module: str, keyword: str, error: str, payload: bytes, tmp_path: Path
) -> None:
    owner = importlib.import_module(module)
    path = tmp_path / "input"
    path.write_bytes(payload)
    assert owner.sha256_file(path, **{keyword: 7}) == hashlib.sha256(payload).hexdigest()
    expected = FileNotFoundError if error == "FileNotFoundError" else getattr(owner, error)
    with pytest.raises(expected):
        owner.sha256_file(tmp_path / "absent", **{keyword: 7})


@pytest.mark.parametrize(
    "module",
    ["forge.synthesis.evidence.ugi3_capability", "forge.synthesis.assessment.route_awareness"],
)
def test_evidence_writer_preserves_bytes_mode_and_target_on_failure(
    module: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import forge.core.io as core_io

    owner = importlib.import_module(module)
    path = tmp_path / "nested/output"
    owner._atomic_write_bytes(path=path, content=b"original\n")
    assert path.read_bytes() == b"original\n"
    assert path.stat().st_mode & 0o777 == 0o644

    def fail(*args):
        raise OSError("publication failed")

    monkeypatch.setattr(core_io.os, "replace", fail)
    with pytest.raises(OSError, match="publication failed"):
        owner._atomic_write_bytes(path=path, content=b"replacement")
    assert path.read_bytes() == b"original\n"
    assert list(path.parent.iterdir()) == [path]


def test_historical_audit_diameter_helper_remains_available() -> None:
    from forge.model._synthesis_sampling.constraints import _carbon_skeleton_diameter_from_states
    from forge.model.synthesis_program_sampling import (
        _carbon_skeleton_diameter_from_states as historical_helper,
    )

    assert historical_helper is _carbon_skeleton_diameter_from_states
