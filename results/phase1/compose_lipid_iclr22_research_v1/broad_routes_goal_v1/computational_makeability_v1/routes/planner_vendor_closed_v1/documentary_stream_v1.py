"""Bounded documentary authentication; no change to chemistry or listing predicates."""

import gzip
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "evaluation/bulk_supplier_intervention_v1"))
from stream_evidence_v1 import local_path, signature, stream_verify  # noqa: E402


class DocumentaryAuthenticator:
    def __init__(self, root, allowed_sources, maximum_document_bytes=128 * 1024**2):
        self.root = Path(root)
        self.allowed = allowed_sources
        self.maximum = maximum_document_bytes
        self.cache = {}
        self.parsed_documents = {}

    def verify(self, record):
        # Documentary JSON pointers authenticate the complete underlying bytes.
        location = record.get("path", record.get("location"))
        if not isinstance(location, str) or location.startswith(("https://", "http://")):
            raise ValueError("missing local documentary path")
        whole = {"path": location.split("#", 1)[0], "sha256": record["sha256"]}
        path = local_path(self.root, whole)
        allowed = [source for source in self.allowed if local_path(self.root, source) == path]
        if allowed:
            attestation = stream_verify(self.root, whole, allowed, self.cache)
            return path, True, tuple(attestation["stat_signature"])
        before = signature(path)
        if before[2] > self.maximum:
            raise ValueError("unallowlisted oversized documentary source")
        if before[2] == 0:
            if whole["sha256"] != hashlib.sha256(b"").hexdigest() or signature(path) != before:
                raise ValueError("empty documentary digest mismatch")
        else:
            stream_verify(
                self.root,
                whole,
                [{**whole, "bytes": before[2], "complete": True}],
                self.cache,
            )
        return path, False, before

    def authenticate(self, value):
        active = set()
        evidence_keys = {
            "source_receipts",
            "receipt",
            "source",
            "supporting_receipts",
            "prior_observation",
            "body",
            "headers",
            "response",
            "response_body",
            "HTTP_receipt",
            "checksum_receipt",
        }

        def walk(item, follow=False):
            if isinstance(item, dict):
                location = item.get("path", item.get("location"))
                digest = item.get("sha256")
                if isinstance(location, str) and isinstance(digest, str):
                    path, raw_bulk, verified_signature = self.verify(item)
                    key = (location, digest, verified_signature)
                    if follow and not raw_bulk and key not in active:
                        active.add(key)
                        if key not in self.parsed_documents:
                            before = verified_signature
                            if signature(path) != before:
                                raise ValueError("source changed after hash verification")
                            with path.open("rb") as handle:
                                magic = handle.read(2)
                                handle.seek(0)
                                if magic == b"\x1f\x8b":
                                    with gzip.GzipFile(fileobj=handle) as compressed:
                                        data = compressed.read(self.maximum + 1)
                                else:
                                    data = handle.read(self.maximum + 1)
                            if len(data) > self.maximum or signature(path) != before:
                                raise ValueError("document parse cap or source mutation")
                            self.parsed_documents[key] = (
                                json.loads(data) if data.lstrip().startswith((b"{", b"[")) else None
                            )
                        walk(self.parsed_documents[key])
                        active.remove(key)
                for name, child in item.items():
                    walk(child, name in evidence_keys)
            elif isinstance(item, list):
                for child in item:
                    walk(child, follow)

        walk(value)
