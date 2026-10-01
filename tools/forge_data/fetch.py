"""Fetch selected, already-pinned public assets without rewriting the vendor manifest."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
from pathlib import Path

from forge_data.vendor import REMOTE_ASSETS, VENDOR, sha256


def fetch(name: str, destination: Path = VENDOR) -> Path:
    url, expected = REMOTE_ASSETS[name]
    target = destination / name
    if target.exists():
        if sha256(target) != expected:
            raise ValueError(f"existing asset checksum mismatch: {target}; expected {expected}")
        return target
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fetch-", dir=destination) as directory:
        temporary = Path(directory) / "asset"
        with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        actual = sha256(temporary)
        if actual != expected:
            raise ValueError(
                f"download checksum mismatch for {name}: expected {expected}, got {actual}"
            )
        temporary.replace(target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", choices=sorted(REMOTE_ASSETS), nargs="+")
    args = parser.parse_args(argv)
    for name in args.assets:
        print(f"Verified: {fetch(name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
