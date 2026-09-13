"""Save immutable public-source responses; this does not qualify evidence or restore pins."""

import argparse
import concurrent.futures
import hashlib
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def pin(path):
    payload = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def retrieve(item, directory):
    record = dict(item, requested_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        request = urllib.request.Request(item["url"], headers={"User-Agent": "Mozilla/5.0"})
        try:
            response = urllib.request.urlopen(request, timeout=35)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            body = response.read(50_000_001)
            record.update(
                http_status=response.status,
                resolved_url=response.url,
                headers={
                    k: response.headers.get(k)
                    for k in ("Content-Type", "Date", "Last-Modified", "ETag")
                },
            )
        if len(body) > 50_000_000:
            raise ValueError("Response exceeds 50 MB download bound")
        destination = directory / item["filename"]
        with destination.open("xb") as handle:
            handle.write(body)
        record.update(
            response=pin(destination),
            status="saved_response_requires_content_review",
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as error:
        record.update(status="retrieval_failed", error_type=type(error).__name__, error=str(error))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("requests", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    requests = json.loads(args.requests.read_text())
    names = [item["filename"] for item in requests]
    if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
        raise ValueError("Response filenames must be unique basenames")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda item: retrieve(item, args.output), requests))
    report = {
        "schema_version": "forge.public_source_acquisition.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"requests": pin(args.requests), "script": pin(Path(__file__))},
        "records": records,
        "historical_pins_changed": False,
        "evidence_admitted": False,
        "purchase_or_contact_actions": 0,
    }
    with (args.output / "receipt.json").open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            [
                {k: r[k] for k in ("id", "http_status", "status", "error") if k in r}
                for r in records
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
