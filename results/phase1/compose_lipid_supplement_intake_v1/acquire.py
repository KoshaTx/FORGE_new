"""Acquire the explicitly supplied public Drive supplement; never execute its contents."""

from __future__ import annotations

import concurrent.futures
import hashlib
import html
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data/source_cache/compose_lipid_supplement_2026-09-19"
ACQ = CACHE / "_acquisition"
FOLDER = "1WRV6YjOvRHcuPVvuENd6AlcNpEGk6ZPx"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def request(url: str):
    return urllib.request.urlopen(url, timeout=120)


def listing(folder: tuple[str, str]) -> list[dict]:
    folder_id, parent = folder
    saved = ACQ / f"listing-{folder_id}.json"
    if saved.exists():
        return json.loads(saved.read_text())
    with request("https://drive.google.com/drive/folders/" + folder_id) as response:
        raw = response.read().decode()
    match = re.search(r"window\['_DRIVE_ivd'\]\s*=\s*'((?:\\.|[^'\\])*)';", raw)
    if not match:
        raise ValueError(f"No Drive listing: {folder_id}")
    payload = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m[1], 16)), match[1])
    records = []
    for row in json.loads(payload)[0] or []:
        name = row[2]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Unsafe Drive filename: {name!r}")
        records.append(
            dict(
                id=row[0],
                path=str(Path(parent) / name),
                mime_type=row[3],
                size_bytes=row[13],
                modified_ms=row[10],
            )
        )
    saved.write_text(json.dumps(records, indent=2) + "\n")
    return records


def download(item: dict, expected: dict[str, str]) -> dict:
    path = CACHE / item["path"]
    pin = expected.get(item["path"])
    if path.exists() and path.stat().st_size == item["size_bytes"]:
        actual = digest(path)
        if pin is None or actual == pin:
            return {**item, "sha256": actual}
    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://drive.google.com/uc?export=download&id=" + item["id"]
    for attempt in range(3):
        try:
            response = request(url)
            if "text/html" in response.headers.get("Content-Type", ""):
                page = response.read().decode()
                response.close()
                action = re.search(r'<form[^>]*action="([^"]+)"', page)
                if not action:
                    raise ValueError(f"Download returned HTML: {item['path']}")
                params = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', page))
                destination = html.unescape(action[1])
                if urllib.parse.urlparse(destination).hostname != "drive.usercontent.google.com":
                    raise ValueError("Unexpected confirmation host")
                response = request(destination + "?" + urllib.parse.urlencode(params))
            if "text/html" in response.headers.get("Content-Type", ""):
                raise ValueError(f"Download confirmation failed: {item['path']}")
            partial = path.with_name(path.name + ".partial")
            with response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream, 4 << 20)
            if partial.stat().st_size != item["size_bytes"]:
                raise ValueError(f"Wrong size: {item['path']}")
            actual = digest(partial)
            if pin is not None and actual != pin:
                raise ValueError(f"SHA-256 mismatch: {item['path']}")
            partial.replace(path)
            return {**item, "sha256": actual}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    ACQ.mkdir(parents=True, exist_ok=True)
    expected = {}
    for line in (CACHE / "SHA256SUMS").read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        name = name.removeprefix("*")
        if not re.fullmatch("[0-9a-f]{64}", sha) or Path(name).is_absolute():
            raise ValueError("Unsafe checksum manifest")
        if ".." in Path(name).parts or name in expected:
            raise ValueError("Unsafe or repeated checksum path")
        expected[name] = sha
    files = []
    pending = [(FOLDER, "")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        while pending:
            next_pending = []
            for entries in pool.map(listing, pending):
                for item in entries:
                    if item["mime_type"] == "application/vnd.google-apps.folder":
                        next_pending.append((item["id"], item["path"]))
                    else:
                        files.append(item)
            pending = next_pending
        paths = {item["path"] for item in files}
        if len(paths) != len(files):
            raise ValueError("Repeated Drive listing paths")
        missing = sorted(set(expected) - paths)
        (ACQ / "missing_manifest_paths.json").write_text(json.dumps(missing, indent=2) + "\n")
        (ACQ / "recursive_listing.json").write_text(json.dumps(files, indent=2) + "\n")
        print(
            f"Listed {len(files)} files, {sum(f['size_bytes'] for f in files):,} bytes; missing {len(missing)} checksum paths",
            flush=True,
        )
        # Small manifests become available promptly while larger files stream independently.
        futures = [
            pool.submit(download, item, expected)
            for item in sorted(files, key=lambda f: f["size_bytes"])
        ]
        records = []
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
            if len(records) % 50 == 0 or len(records) == len(files):
                print(f"Verified downloads: {len(records)}/{len(files)}", flush=True)
    manifest = dict(
        source_url="https://drive.google.com/drive/folders/" + FOLDER,
        implementation_sha256=digest(Path(__file__)),
        checksum_manifest_sha256=digest(CACHE / "SHA256SUMS"),
        missing_manifest_paths=missing,
        files=sorted(records, key=lambda r: r["path"]),
    )
    (ACQ / "download_receipt.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
