from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from forge_data.fetch import fetch


@pytest.fixture
def public_asset(monkeypatch: pytest.MonkeyPatch) -> bytes:
    payload = b"exact published fixture bytes"
    monkeypatch.setattr(
        "forge_data.fetch.REMOTE_ASSETS",
        {
            "fixture.csv": (
                "https://example.invalid/pinned.csv",
                hashlib.sha256(payload).hexdigest(),
            ),
        },
    )
    return payload


def test_fetch_checks_bytes_and_reuses_verified_file(tmp_path: Path, public_asset: bytes) -> None:
    with patch("urllib.request.urlopen", return_value=BytesIO(public_asset)) as download:
        path = fetch("fixture.csv", tmp_path)
        assert path.read_bytes() == public_asset
        assert fetch("fixture.csv", tmp_path) == path
        assert download.call_count == 1
    assert not (tmp_path / "MANIFEST.json").exists()


def test_wrong_download_never_becomes_an_asset(tmp_path: Path, public_asset: bytes) -> None:
    with patch("urllib.request.urlopen", return_value=BytesIO(b"wrong")):
        with pytest.raises(ValueError, match="checksum mismatch"):
            fetch("fixture.csv", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_corrupt_local_file_is_not_silently_replaced(tmp_path: Path, public_asset: bytes) -> None:
    (tmp_path / "fixture.csv").write_bytes(b"changed")
    with patch("urllib.request.urlopen") as download:
        with pytest.raises(ValueError, match="existing asset checksum mismatch"):
            fetch("fixture.csv", tmp_path)
        download.assert_not_called()
    assert (tmp_path / "fixture.csv").read_bytes() == b"changed"


def test_fetch_failure_cleans_temporary_files(tmp_path: Path, public_asset: bytes) -> None:
    with patch("urllib.request.urlopen", side_effect=OSError("connection lost")):
        with pytest.raises(OSError, match="connection lost"):
            fetch("fixture.csv", tmp_path)
    assert list(tmp_path.iterdir()) == []
