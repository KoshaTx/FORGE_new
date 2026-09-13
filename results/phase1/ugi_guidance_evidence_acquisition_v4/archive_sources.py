"""Preserve exact public response bytes and restore them without overwriting drift."""

import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from forge.core.hashing import pin_record
from forge.core.io import atomic_write, write_json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
ARCHIVE = ROOT / "provenance/recovery/ugi_guidance_evidence_v4/assets"
RECEIPT = ROOT / "provenance/recovery/ugi_guidance_evidence_acquisition_v4.json"


def authenticate(entry, restore=False):
    packed = (ROOT / entry["archive"]["path"]).read_bytes()
    if hashlib.sha256(packed).hexdigest() != entry["archive"]["sha256"]:
        raise ValueError("Compressed source changed")
    raw = gzip.decompress(packed)
    if hashlib.sha256(raw).hexdigest() != entry["original"]["sha256"]:
        raise ValueError("Archive does not reproduce the original source")
    destination = ROOT / entry["original"]["path"]
    if not destination.resolve().is_relative_to(OUT):
        raise ValueError("Archive destination is outside this acquisition")
    if destination.exists() and destination.read_bytes() != raw:
        raise ValueError(f"Preserve changed source: {destination}")
    if restore and not destination.exists():
        atomic_write(destination, raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    if args.verify or args.restore:
        receipt = json.loads(RECEIPT.read_text())
        for entry in receipt["archived_sources"]:
            authenticate(entry, args.restore)
        print(f"Verified {len(receipt['archived_sources'])} exact archived source/log assets")
        return
    if RECEIPT.exists():
        raise ValueError("Preserve the existing archive receipt")
    entries = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.suffix not in {
            ".html",
            ".js",
            ".pdf",
            ".jpg",
            ".png",
            ".txt",
            ".xml",
            ".log",
        }:
            continue
        original = pin_record(path, ROOT)
        packed = gzip.compress(path.read_bytes(), mtime=0)
        destination = ARCHIVE / (original["sha256"] + ".gz")
        if destination.exists() and destination.read_bytes() != packed:
            raise ValueError("Preserve changed archive asset")
        if not destination.exists():
            atomic_write(destination, packed)
        entry = {"original": original, "archive": pin_record(destination, ROOT)}
        authenticate(entry)
        entries.append(entry)
    write_json(
        RECEIPT,
        {
            "schema_version": "forge.guidance_followup_source_archive.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {"script": pin_record(Path(__file__), ROOT)},
            "archived_sources": entries,
            "source_bytes_modified": False,
            "compression": "gzip_mtime_zero",
        },
    )
    print(f"Archived {len(entries)} exact source/log assets")


if __name__ == "__main__":
    main()
