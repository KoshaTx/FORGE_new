"""Replay historical L1 annotations and authenticate the two frozen training-table pins."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
REPLAY = ROOT / "results/phase1/ugi_guidance_artifact_recovery_v2/replay_tree"
TREE = ROOT / "results/phase1/ugi_guidance_artifact_recovery_v2/original_code_tree.json"
CONFIG = ROOT / "configs/model/phase1_ugi_l1_semantics.json"
AUTHORITY = ROOT / "configs/model/phase1_ugi_chemistry.json"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def pin(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path.read_bytes())}


def write(path: Path, document: dict) -> None:
    with path.open("x") as handle:
        handle.write(json.dumps(document, indent=2, sort_keys=True) + "\n")


def source_pins() -> list[dict]:
    original = {row["path"]: row for row in json.loads(TREE.read_text())["files"]}
    sources = []
    for name, module in sorted(tuple(sys.modules.items())):
        if name == "forge" or name.startswith("forge."):
            path = Path(module.__file__).resolve()
            relative = str(path.relative_to(REPLAY))
            if sha256(path.read_bytes()) != original[relative]["materialized_sha256"]:
                raise ValueError(f"historical replay source changed: {relative}")
            sources.append(pin(path))
    return sources


def candidate() -> None:
    destination = OUTPUT / "l1_annotations_candidate"
    if destination.exists():
        raise FileExistsError(destination)
    sys.path.insert(0, str(REPLAY / "src"))
    from forge.product.ugi_l1_origin_annotations import build_phase1_ugi_l1_semantics

    config = json.loads(CONFIG.read_text())
    for value in config["inputs"].values():
        if pin(ROOT / value["path"]) != value:
            raise ValueError(f"original input changed: {value['path']}")
    request = {
        "schema_version": "forge.l1_annotation_recovery_request.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": pin(CONFIG),
            "authority": pin(AUTHORITY),
            "historical_tree": pin(TREE),
            "script": pin(Path(__file__).resolve()),
            **config["inputs"],
        },
        "historical_sources": source_pins(),
        "python": platform.python_version(),
        "training_calls": 0,
        "new_product_generation_calls": 0,
        "purpose": "exact recovery of existing annotations, with all frozen annotation gates",
    }
    request_path = OUTPUT / "l1_annotations_request.json"
    write(request_path, request)
    result = build_phase1_ugi_l1_semantics(CONFIG, destination, ROOT)
    if result["status"] != "pass" or not all(result["gates"].values()):
        raise ValueError("historical annotation gates failed")
    comparisons = []
    for label in ("semantic_products", "semantic_atoms"):
        expected = json.loads(AUTHORITY.read_text())["inputs"][label]
        path = destination / Path(expected["path"]).name
        original = path.read_bytes()
        if original[:3] != b"\x1f\x8b\x08":
            raise ValueError("expected gzip artifact")
        uncompressed = gzip.decompress(original)
        candidates = [("unchanged", original)]
        candidates.append(("gzip_os_byte_255", original[:9] + b"\xff" + original[10:]))
        # Re-encode the unchanged CSV for the two Python default compression levels.
        for level in (6, 9):
            compressed = gzip.compress(uncompressed, compresslevel=level, mtime=0)
            candidates.append(
                (f"gzip_level_{level}_os_255", compressed[:9] + b"\xff" + compressed[10:])
            )
        exact = next(
            ((method, data) for method, data in candidates if sha256(data) == expected["sha256"]),
            None,
        )
        record = {
            "expected": expected,
            "replayed": pin(path),
            "csv_sha256": sha256(uncompressed),
            "exact_original_recovered": exact is not None,
            "encoding_checks": [
                {"method": method, "sha256": sha256(data)} for method, data in candidates
            ],
        }
        if exact is not None:
            method, data = exact
            if gzip.decompress(data) != uncompressed:
                raise ValueError("CSV changed while re-encoding")
            authenticated = destination / f"authenticated_{path.name}"
            with authenticated.open("xb") as handle:
                handle.write(data)
            record.update({"method": method, "authenticated": pin(authenticated)})
        comparisons.append(record)
    for value in (*request["inputs"].values(), *request["historical_sources"]):
        if pin(ROOT / value["path"]) != value:
            raise ValueError(f"input changed during replay: {value['path']}")
    receipt = {
        "schema_version": "forge.l1_annotation_recovery.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": pin(request_path),
        "replay_result": pin(destination / "ugi_l1_semantics_result.json"),
        "comparisons": comparisons,
        "historical_annotation_gates": result["gates"],
        "ready_for_restoration": all(row["exact_original_recovered"] for row in comparisons),
        "installed": False,
    }
    write(OUTPUT / "l1_annotations_recovery.json", receipt)
    print("ready_for_restoration", receipt["ready_for_restoration"])


def restore() -> None:
    receipt_path = OUTPUT / "l1_annotations_recovery.json"
    receipt = json.loads(receipt_path.read_text())
    request_path = ROOT / receipt["request"]["path"]
    if pin(request_path) != receipt["request"] or not receipt["ready_for_restoration"]:
        raise ValueError("recovery is not authenticated")
    request = json.loads(request_path.read_text())
    for value in (*request["inputs"].values(), *request["historical_sources"]):
        if pin(ROOT / value["path"]) != value:
            raise ValueError(f"recovery input changed: {value['path']}")
    pending = []
    for row in receipt["comparisons"]:
        source = ROOT / row["authenticated"]["path"]
        if pin(source) != row["authenticated"]:
            raise ValueError("authenticated candidate changed")
        payload = source.read_bytes()
        if sha256(payload) != row["expected"]["sha256"]:
            raise ValueError("original hash mismatch")
        target = ROOT / row["expected"]["path"]
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        pending.append((target, payload))
    for target, payload in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(payload)
    write(
        OUTPUT / "l1_annotations_restoration.json",
        {
            "schema_version": "forge.l1_annotation_exact_restoration.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {"recovery_receipt": pin(receipt_path)},
            "restored": [pin(path) for path, _ in pending],
            "original_sha256s_changed": False,
            "historical_result_json_recreated": False,
        },
    )
    print("restored exact original training-table bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    restore() if args.restore else candidate()
