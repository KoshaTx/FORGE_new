#!/usr/bin/env python3
"""Train the widened chemistry-aware joint sparse Ugi flow on Modal."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src" / "forge"
REMOTE_SOURCE_ROOT = Path("/root")
REMOTE_REPO_ROOT = Path("/root/forge_repo")
VOLUME_MOUNT = Path("/forge-output")
CONFIG_RELATIVE = "configs/model/phase1_ugi_joint_sparse_balanced_v2.json"
CONFIG_PATH = REPO_ROOT / CONFIG_RELATIVE
GPU_CLASS = "L4"
VOLUME_NAME = "forge-phase1-training"
INPUT_FILES = (
    CONFIG_RELATIVE,
    "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz",
    "results/phase1/ugi_balanced_chemistry_corpus_v2/semantic_products.csv.gz",
    "results/phase1/ugi_balanced_chemistry_corpus_v2/semantic_atoms.csv.gz",
    "results/phase1/product_v3_atom_vocabulary.json",
    "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt",
)
STATIC_OUTPUT_FILES = (
    "result.json",
    "checkpoint_best.pt",
    "checkpoint_latest.pt",
    "progress.json",
)

app = modal.App("forge-phase1-ugi-joint-sparse-v2")
output_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.0",
        "numpy==2.4.2",
        "rdkit==2025.9.6",
        "PyYAML==6.0.3",
    )
    .add_local_dir(SOURCE_ROOT, remote_path=str(REMOTE_SOURCE_ROOT / "forge"))
)
for relative in INPUT_FILES:
    image = image.add_local_file(
        REPO_ROOT / relative,
        remote_path=str(REMOTE_REPO_ROOT / relative),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_python_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*.py") if candidate.is_file())
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(candidate)))
    return digest.hexdigest()


def _run_fingerprint(config_path: Path, source_root: Path, launcher_path: Path) -> str:
    digest = hashlib.sha256()
    for value in (
        _sha256_file(config_path),
        _sha256_python_tree(source_root),
        _sha256_file(launcher_path),
    ):
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _run_name() -> str:
    fingerprint = _run_fingerprint(CONFIG_PATH, SOURCE_ROOT, Path(__file__).resolve())
    return f"ugi_joint_sparse_balanced_v2_{fingerprint[:12]}"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_volume_download(remote_path: str, local_path: Path) -> int:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{local_path.name}.", dir=local_path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            count = output_volume.read_file_into_fileobj(remote_path, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, local_path)
        return int(count)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@app.function(
    image=image,
    gpu=GPU_CLASS,
    cpu=4.0,
    memory=24576,
    timeout=14_400,
    volumes={str(VOLUME_MOUNT): output_volume},
)
def train_balanced_joint_sparse_v2(run_name: str) -> dict[str, Any]:
    """Run the immutable widened full-training contract."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(REMOTE_SOURCE_ROOT))

    import numpy as np
    import rdkit
    import torch

    from forge.product.defog_feasibility import sha256_file
    from forge.product.ugi_joint_sparse_training import train_ugi_joint_sparse

    config_path = REMOTE_REPO_ROOT / CONFIG_RELATIVE
    launcher_path = Path(__file__).resolve()
    source_sha256 = _sha256_python_tree(REMOTE_SOURCE_ROOT / "forge")
    launcher_sha256 = sha256_file(launcher_path)
    fingerprint = _run_fingerprint(
        config_path,
        REMOTE_SOURCE_ROOT / "forge",
        launcher_path,
    )
    expected_name = f"ugi_joint_sparse_balanced_v2_{fingerprint[:12]}"
    if run_name != expected_name:
        raise RuntimeError("run name does not match the frozen v2 joint config")
    output_dir = VOLUME_MOUNT / run_name
    result_path = output_dir / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if result.get("status") != "complete":
            raise RuntimeError("existing v2 joint result is incomplete")
        return {"status": "already_complete", "run_name": run_name, "result": result}

    restart_partial = output_dir.exists() and any(output_dir.iterdir())

    def persist_progress(_step: int, _: Path) -> None:
        output_volume.commit()

    result = train_ugi_joint_sparse(
        config_path,
        REMOTE_REPO_ROOT,
        output_dir,
        smoke=False,
        overwrite=restart_partial,
        progress_callback=persist_progress,
    )
    result["cloud_execution"] = {
        "run_name": run_name,
        "gpu_class": GPU_CLASS,
        "volume": VOLUME_NAME,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "rdkit": rdkit.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": int(torch.backends.cudnn.version()),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "precision": "float32",
            "mixed_precision": False,
        },
        "mounted_inputs": {
            relative: sha256_file(REMOTE_REPO_ROOT / relative) for relative in INPUT_FILES
        },
        "source": {
            "python_tree_sha256": source_sha256,
            "launcher_sha256": launcher_sha256,
            "run_fingerprint_sha256": fingerprint,
        },
        "restarted_uncheckpointed_partial": restart_partial,
        "progress_committed_at_each_evaluation": True,
    }
    _atomic_json(result_path, result)
    output_volume.commit()
    return {"status": "complete", "run_name": run_name, "result": result}


@app.local_entrypoint()
def main(local_output: str = "") -> None:
    run_name = _run_name()
    remote = train_balanced_joint_sparse_v2.remote(run_name)
    result = remote["result"]
    if result.get("status") != "complete":
        raise RuntimeError("remote v2 joint training did not complete")
    local_dir = (
        Path(local_output).resolve()
        if local_output
        else (REPO_ROOT / "results/phase1/ugi_joint_sparse_balanced_v2_full").resolve()
    )
    snapshot_files = tuple(
        Path(record["path"]).name for record in result.get("checkpoint_snapshots", ())
    )
    downloaded = {}
    for filename in (*STATIC_OUTPUT_FILES, *snapshot_files):
        local_path = local_dir / filename
        downloaded[filename] = {
            "bytes": _atomic_volume_download(f"{run_name}/{filename}", local_path),
            "sha256": _sha256_file(local_path),
        }
    if downloaded["checkpoint_best.pt"]["sha256"] != result["checkpoint"]["sha256"]:
        raise RuntimeError("downloaded best v2 joint checkpoint hash mismatch")
    if downloaded["checkpoint_latest.pt"]["sha256"] != result["checkpoint_latest"]["sha256"]:
        raise RuntimeError("downloaded latest v2 joint checkpoint hash mismatch")
    for record in result.get("checkpoint_snapshots", ()):
        filename = Path(record["path"]).name
        if downloaded[filename]["sha256"] != record["sha256"]:
            raise RuntimeError(f"downloaded serial checkpoint hash mismatch: {filename}")
    if json.loads((local_dir / "result.json").read_text()) != result:
        raise RuntimeError("downloaded v2 joint result differs from remote result")
    print(
        json.dumps(
            {
                "status": remote["status"],
                "run_name": run_name,
                "local_output": str(local_dir),
                "downloaded": downloaded,
                "selection": result["selection"],
                "final_evaluation": result["evaluations"][-1],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    raise SystemExit("Use `modal run scripts/modal_phase1_ugi_joint_sparse_v2.py`.")
