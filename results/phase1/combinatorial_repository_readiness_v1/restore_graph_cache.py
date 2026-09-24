"""Restore the original cache after authenticating a gzip-header-only difference."""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
AUTHORITY = ROOT / "results/m0_07/oracle_graph_tensor_cache_result.json"


def pin(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    replay_path = OUTPUT / "graph_cache_recovery.json"
    replay = json.loads(replay_path.read_text())
    request_path = ROOT / replay["request"]["path"]
    if pin(request_path) != replay["request"]:
        raise ValueError("replay request changed")
    request = json.loads(request_path.read_text())
    for value in (*request["inputs"].values(), *request["source_files"]):
        if pin(ROOT / value["path"]) != value:
            raise ValueError(f"replay input or source changed: {value['path']}")
    authority = json.loads(AUTHORITY.read_text())
    restored = []
    payloads = {}
    normalization = None
    for comparison in replay["comparisons"]:
        path = ROOT / comparison["candidate"]["path"]
        if pin(path) != comparison["candidate"]:
            raise ValueError(f"replay candidate changed: {path}")
        name = comparison["filename"]
        payload = path.read_bytes()
        if name.endswith(".csv.gz"):
            if payload[:9] != bytes.fromhex("1f8b08000000000002"):
                raise ValueError("unexpected gzip header; no normalization permitted")
            original = payload
            # Preserve every compressed payload byte, CRC, length and timestamp.
            # Only the gzip OS byte differs in this environment's replay.
            payload = payload[:9] + b"\xff" + payload[10:]
            if gzip.decompress(original) != gzip.decompress(payload):
                raise ValueError("metadata content changed")
            normalization = {
                "candidate": pin(path),
                "byte_offset": 9,
                "before": original[9],
                "after": payload[9],
                "all_other_bytes_identical": original[:9] + original[10:]
                == payload[:9] + payload[10:],
                "uncompressed_sha256": hashlib.sha256(gzip.decompress(payload)).hexdigest(),
                "final_sha256": hashlib.sha256(payload).hexdigest(),
            }
        expected = authority["artifacts"][name]
        if (
            hashlib.sha256(payload).hexdigest() != expected["sha256"]
            or len(payload) != expected["bytes"]
        ):
            raise ValueError(f"original artifact identity not recovered: {name}")
        target = AUTHORITY.parent / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refuse overwrite: {target}")
        payloads[target] = payload
    if {path.name for path in payloads} != set(authority["artifacts"]):
        raise ValueError("complete artifact set required")
    for target, payload in payloads.items():
        with target.open("xb") as handle:
            handle.write(payload)
        restored.append(pin(target))
    result = {
        "schema_version": "forge.graph_cache_exact_restoration.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "all_original_artifact_bytes_restored",
        "inputs": {
            "original_result": pin(AUTHORITY),
            "replay_receipt": pin(replay_path),
            "script": pin(Path(__file__).resolve()),
        },
        "normalization": normalization,
        "restored": restored,
        "original_result_unchanged": pin(AUTHORITY) == request["inputs"]["original_result"],
        "original_sha256s_changed": False,
        "new_scientific_result": False,
        "training_or_generation_calls": 0,
    }
    with (OUTPUT / "graph_cache_restoration.json").open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
