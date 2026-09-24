"""Verify both manifests and restore only the previously absent, exactly matching task files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CACHE = ROOT / "data/source_cache/compose_lipid_original_tasks_2026-09-20"
PREVIOUS = ROOT / "data/source_cache/compose_lipid_supplement_2026-09-19"
MISSING = ROOT / "results/phase1/compose_lipid_supplement_intake_v1/missing_source_files.json"


def pin(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest.hexdigest()}


def safe_path(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"Unsafe manifest path: {name}")
    return path


def checksums(path):
    result = {}
    for line in path.read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        name = name.removeprefix("*")
        safe_path(name)
        if name in result or not re.fullmatch("[a-f0-9]{64}", sha):
            raise ValueError(f"Invalid checksum entry: {name}")
        result[name] = sha
    return result


def check_package(cache, log_name):
    expected = checksums(cache / "SHA256SUMS")
    command = ["shasum", "-a", "256", "-c", "SHA256SUMS"]
    with (OUT / log_name).open("w") as stream:
        completed = subprocess.run(command, cwd=cache, stdout=stream, stderr=stream)
    if completed.returncode:
        raise ValueError(f"Checksum verification failed: {log_name}")
    return {
        "command": command,
        "working_directory": str(cache.relative_to(ROOT)),
        "exit_code": completed.returncode,
        "verified_files": len(expected),
        "checksum_manifest": pin(cache / "SHA256SUMS"),
        "log": pin(OUT / log_name),
    }


def main():
    new_expected = checksums(CACHE / "SHA256SUMS")
    old_expected = checksums(PREVIOUS / "SHA256SUMS")
    new_check = check_package(CACHE, "bundle-shasum.log")
    manifest = json.loads((CACHE / "MANIFEST.json").read_text())
    files = {row["path"]: row for row in manifest["files"]}
    if len(files) != len(manifest["files"]):
        raise ValueError("Repeated manifest path")
    if set(new_expected) != set(files) | {"MANIFEST.json"}:
        raise ValueError("Manifest and checksums cover different files")
    for name, row in files.items():
        path = CACHE / safe_path(name)
        if row["sha256"] != new_expected[name] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Manifest identity or size mismatch: {name}")
    requested = {name for name in files if name.startswith("original_generator_tasks/")}
    missing = json.loads(MISSING.read_text())
    if requested != set(missing["missing_paths"]):
        raise ValueError("Delivered tasks do not exactly cover the previous missing-file inventory")
    if len(requested) != manifest["requested_task_files"]:
        raise ValueError("Task count disagrees with manifest")
    # Verify every source binding before changing the earlier cache.
    source_files = {row["source"]: row for row in files.values()}
    bindings = {}
    for name in sorted(files):
        if not name.startswith("provenance/"):
            continue
        receipt = json.loads((CACHE / name).read_text())
        if receipt.get("complete") is not True:
            raise ValueError(f"Incomplete source receipt: {name}")
        references = dict(receipt["inputs"])
        references.update({r["path"]: r["file_sha256"] for r in receipt.get("task_shards", [])})
        for source, expected in references.items():
            if source not in source_files:
                continue
            row = source_files[source]
            if row["sha256"] != expected:
                raise ValueError(f"Source receipt disagrees with supplied file: {source}")
            bindings[row["path"]] = {"receipt": pin(CACHE / name), "source": source}
    if set(bindings) != requested:
        raise ValueError("Some requested tasks are not bound by a family receipt")
    restored = []
    for name in sorted(requested):
        if old_expected.get(name) != new_expected[name]:
            raise ValueError(f"Old and new package disagree: {name}")
    for name in sorted(requested):
        target = PREVIOUS / safe_path(name)
        existed = target.exists()
        if existed:
            if pin(target)["sha256"] != new_expected[name]:
                raise ValueError(f"Refusing to replace a different existing file: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
                with (CACHE / name).open("rb") as source:
                    shutil.copyfileobj(source, temporary, 4 << 20)
                temporary_path = Path(temporary.name)
            try:
                if pin(temporary_path)["sha256"] != new_expected[name]:
                    raise ValueError(f"Copy changed bytes: {name}")
                os.link(temporary_path, target)
            finally:
                temporary_path.unlink()
        restored.append({**pin(target), "already_present": existed, "binding": bindings[name]})
    old_check = check_package(PREVIOUS, "completed-package-shasum.log")
    result = {
        "schema_version": "forge.compose_lipid_original_tasks_package_check.v1",
        "status": "both_checksum_manifests_verified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": 0,
        "implementation": pin(Path(__file__)),
        "inputs": {
            "new_manifest": pin(CACHE / "MANIFEST.json"),
            "new_readme": pin(CACHE / "README.md"),
            "historical_missing_inventory": pin(MISSING),
            "acquisition": pin(OUT / "acquisition.json"),
        },
        "new_bundle": new_check,
        "completed_previous_package": old_check,
        "previously_missing_files_resolved": len(restored),
        "remaining_missing_manifest_files": [],
        "restored_files": restored,
        "training_admitted": False,
        "training_calls": 0,
        "scope": "Exact file restoration and receipt binding; no reaction or training admission.",
    }
    (OUT / "package_check.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "previously_missing_files_resolved",
                    "remaining_missing_manifest_files",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
