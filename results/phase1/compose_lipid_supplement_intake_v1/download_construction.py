"""Resume the large public construction export using checked HTTP byte ranges."""

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
PARTS = CACHE / "_acquisition/construction_parts"
SIZE = 1827673665
EXPECTED = "c9f63f974d4b3d899bee7529c84be82cd4faa29b6f1c633fb49e23a0c2cf4afe"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 << 20), b""):
            h.update(data)
    return h.hexdigest()


def main():
    PARTS.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        "https://drive.google.com/uc?export=download&id=1UKgetRfb8Ee7yoOuNndw3sHRc0cdwJso",
        timeout=90,
    ) as response:
        if "text/html" in response.headers.get("Content-Type", ""):
            page = response.read().decode()
            action = html.unescape(re.search(r'<form[^>]*action="([^"]+)"', page)[1])
            if urllib.parse.urlparse(action).hostname != "drive.usercontent.google.com":
                raise ValueError("Unexpected download host")
            params = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', page))
            url = action + "?" + urllib.parse.urlencode(params)
        else:
            url = response.geturl()

    def fetch(start):
        end = min(SIZE, start + (16 << 20)) - 1
        part = PARTS / str(start)
        receipt = part.with_suffix(".json")
        if receipt.exists() and part.exists():
            previous = json.loads(receipt.read_text())
            if part.stat().st_size == end - start + 1 and sha(part) == previous["sha256"]:
                return previous
        for attempt in range(3):
            try:
                temp = part.with_suffix(".partial")
                offset = start + (temp.stat().st_size if temp.exists() else 0)
                if offset > end + 1:
                    raise ValueError("Oversized saved range")
                if offset <= end:
                    request = urllib.request.Request(
                        url, headers={"Range": f"bytes={offset}-{end}"}
                    )
                    with urllib.request.urlopen(request, timeout=180) as response:
                        if (
                            response.status != 206
                            or response.headers.get("Content-Range")
                            != f"bytes {offset}-{end}/{SIZE}"
                        ):
                            raise ValueError("Server did not honor the exact byte range")
                        with temp.open("ab") as stream:
                            shutil.copyfileobj(response, stream, 1 << 20)
                if temp.stat().st_size != end - start + 1:
                    raise ValueError("Incomplete range")
                digest = sha(temp)
                temp.replace(part)
                record = {"start": start, "end": end, "sha256": digest}
                receipt.write_text(json.dumps(record) + "\n")
                return record
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    starts = list(range(0, SIZE, 16 << 20))
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=96) as pool:
        futures = [pool.submit(fetch, start) for start in starts]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
            print(f"Verified ranges {len(records)}/{len(starts)}", flush=True)
    assembled = CACHE / "construction_records.jsonl.gz.ranged"
    with assembled.open("wb") as stream:
        for start in starts:
            with (PARTS / str(start)).open("rb") as part:
                shutil.copyfileobj(part, stream, 4 << 20)
    if assembled.stat().st_size != SIZE or sha(assembled) != EXPECTED:
        raise ValueError("Final construction export checksum mismatch")
    assembled.replace(CACHE / "construction_records.jsonl.gz")
    (PARTS.parent / "construction_range_receipt.json").write_text(
        json.dumps(
            {
                "implementation_sha256": sha(Path(__file__)),
                "sha256": EXPECTED,
                "size_bytes": SIZE,
                "ranges": sorted(records, key=lambda row: row["start"]),
            },
            indent=2,
        )
        + "\n"
    )
    print("Construction export checksum verified", flush=True)


if __name__ == "__main__":
    main()
