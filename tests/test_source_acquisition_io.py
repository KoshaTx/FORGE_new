"""Acquisition keeps published evidence bytes, hashes, modes, and portable paths."""

from pathlib import Path

import pytest

from forge.synthesis.sources import pmc_sources, publisher_sources


@pytest.mark.parametrize("source", [pmc_sources, publisher_sources])
def test_acquisition_io_contract(source, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "cache" / "source.bin"
    source._atomic_write_bytes(target, b"abc")
    assert target.read_bytes() == b"abc"
    assert source._sha256_bytes(content=target.read_bytes()) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert target.stat().st_mode & 0o777 == 0o644
    assert source._portable_path(target) == "cache/source.bin"
    assert source._portable_path(tmp_path.parent) == str(tmp_path.parent.resolve())
    assert list(target.parent.iterdir()) == [target]
