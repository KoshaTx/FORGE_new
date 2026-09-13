"""Archive raw follow-up sources and logs without changing their original bytes."""

import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from forge.core.hashing import pin_record
from forge.core.io import atomic_write, write_json

repo = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
archive = repo / "provenance/recovery/ugi_guidance_evidence_v3/assets"
receipt_path = repo / "provenance/recovery/ugi_guidance_followup_archive_v3.json"
initial_path = repo / "provenance/recovery/ugi_guidance_evidence_acquisition_v3.json"


def verify_entry(item, restore=False):
    packed = repo / item["archive"]["path"]
    payload = packed.read_bytes()
    if hashlib.sha256(payload).hexdigest() != item["archive"]["sha256"]:
        raise ValueError(f"Archive changed: {packed}")
    raw = gzip.decompress(payload)
    if hashlib.sha256(raw).hexdigest() != item["original"]["sha256"]:
        raise ValueError("Archive does not reproduce original bytes")
    if restore:
        destination = repo / item["original"]["path"]
        allowed = (out, repo / "data/source_cache/ugi_guidance_evidence_20260913")
        if not any(destination.resolve().is_relative_to(root) for root in allowed):
            raise ValueError("Refusing restoration outside this acquisition's directories")
        if destination.exists():
            if destination.read_bytes() != raw:
                raise ValueError(f"Preserve different existing bytes: {destination}")
        else:
            atomic_write(destination, raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    if args.verify or args.restore:
        initial = json.loads(initial_path.read_text())
        followup = json.loads(receipt_path.read_text())
        for item in initial["archived_sources"] + followup["archived_sources"]:
            verify_entry(item, restore=args.restore)
        print(
            "All original source/log bytes authenticate"
            + (" and are restored" if args.restore else "")
        )
        return
    if receipt_path.exists():
        raise ValueError("Preserve the existing archive receipt")
    entries = []
    for source in sorted(out.rglob("*")):
        if not source.is_file() or source.suffix not in (".html", ".js", ".xml", ".log"):
            continue
        original = pin_record(source, repo)
        destination = archive / (original["sha256"] + ".gz")
        packed = gzip.compress(source.read_bytes(), mtime=0)
        if destination.exists() and destination.read_bytes() != packed:
            raise ValueError("Preserve different existing compressed bytes")
        if not destination.exists():
            atomic_write(destination, packed)
        entry = {"original": original, "archive": pin_record(destination, repo)}
        verify_entry(entry)
        entries.append(entry)
    write_json(
        receipt_path,
        {
            "schema_version": "forge.guidance_followup_source_archive.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "script": pin_record(Path(__file__), repo),
                "closeout": pin_record(out / "closeout.json", repo),
            },
            "archived_sources": entries,
            "source_bytes_modified": False,
            "compression": "gzip_mtime_zero",
        },
    )
    print(f"Archived {len(entries)} raw source and log files without changing their hashes")


if __name__ == "__main__":
    main()
