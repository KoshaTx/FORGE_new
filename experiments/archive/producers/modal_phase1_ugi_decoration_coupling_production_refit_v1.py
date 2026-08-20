#!/usr/bin/env python3
"""Run the frozen all-fold decoration-coupled Ugi production refit on Modal."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import modal

_LOCAL_REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = (
    Path("/root/forge_repo") if not (_LOCAL_REPO_ROOT / "configs").is_dir() else _LOCAL_REPO_ROOT
)
SOURCE_ROOT = REPO_ROOT / "src" / "forge"
REMOTE_SOURCE_ROOT = Path("/root")
REMOTE_REPO_ROOT = Path("/root/forge_repo")
VOLUME_MOUNT = Path("/forge-output")
VOLUME_NAME = "forge-phase1-training"
CONFIG_RELATIVE = "configs/model/phase1_ugi_decoration_coupling_production_refit_v1.json"
CONFIG_PATH = REPO_ROOT / CONFIG_RELATIVE
GPU_CLASS = "L4"
SHARED_INPUT_FILES = (
    "configs/model/phase1_ugi_decoration_coupling_challenger_v1.json",
    "results/phase1/ugi_decoration_coupling_checkpoint_confirmation_v1/evaluation_v1.json",
    "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz",
    "results/phase1/ugi_balanced_chemistry_corpus_v2/semantic_products.csv.gz",
    "results/phase1/ugi_balanced_chemistry_corpus_v2/semantic_atoms.csv.gz",
    "results/phase1/product_v3_atom_vocabulary.json",
    "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt",
)
INPUT_FILES = (CONFIG_RELATIVE, *SHARED_INPUT_FILES)
STATIC_OUTPUT_FILES = (
    "result.json",
    "checkpoint_best.pt",
    "checkpoint_latest.pt",
    "progress.json",
)

app = modal.App("forge-phase1-ugi-decoration-coupling-production-refit-v1")
output_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.0",
        "numpy==2.4.2",
        "rdkit==2025.9.6",
        "PyYAML==6.0.3",
    )
    .add_local_dir(SOURCE_ROOT, remote_path=str(REMOTE_SOURCE_ROOT / "forge"), copy=True)
)
for relative in INPUT_FILES:
    image = image.add_local_file(
        REPO_ROOT / relative,
        remote_path=str(REMOTE_REPO_ROOT / relative),
        copy=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_python_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for candidate in sorted(value for value in path.rglob("*.py") if value.is_file()):
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
    return f"ugi_decoration_coupling_production_refit_v1_{fingerprint[:12]}"


def _verify_record(repo_root: Path, record: Mapping[str, Any]) -> None:
    path = repo_root / str(record["path"])
    if _sha256_file(path) != str(record["sha256"]):
        raise RuntimeError(f"frozen production-refit input changed: {path}")


def _verify_contract(repo_root: Path, config: Mapping[str, Any]) -> None:
    for record in config["inputs"].values():
        _verify_record(repo_root, record)
    for record in config["selection_evidence"].values():
        _verify_record(repo_root, record)
    duration = config["duration_contract"]
    expected = (
        round(
            int(duration["selected_development_checkpoint_step"])
            * int(duration["production_training_records"])
            / int(duration["development_training_records"])
            / 100
        )
        * 100
    )
    if expected != int(duration["fixed_production_steps"]):
        raise RuntimeError("production duration no longer matches the frozen exposure rule")
    if int(config["full"]["steps"]) != expected:
        raise RuntimeError("full training steps differ from the frozen duration contract")
    if config["training_partition"]["selection_mode"] != "fixed_final_step":
        raise RuntimeError("production refit must use fixed-final-step selection")
    if config["model"].get("decoration_state_conditioning") != "bidirectional_anchor_local":
        raise RuntimeError("production refit lost the selected decoration architecture")


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
def train_production_refit(run_name: str) -> dict[str, Any]:
    """Refit the selected architecture from scratch on all structural folds."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(REMOTE_SOURCE_ROOT))

    import numpy as np
    import rdkit
    import torch

    from experiments.phase1.product_l1.training.ugi_joint_sparse_training import train_ugi_joint_sparse

    config_path = REMOTE_REPO_ROOT / CONFIG_RELATIVE
    config = json.loads(config_path.read_text())
    _verify_contract(REMOTE_REPO_ROOT, config)
    launcher_path = Path(__file__).resolve()
    source_sha256 = _sha256_python_tree(REMOTE_SOURCE_ROOT / "forge")
    launcher_sha256 = _sha256_file(launcher_path)
    fingerprint = _run_fingerprint(config_path, REMOTE_SOURCE_ROOT / "forge", launcher_path)
    expected_name = f"ugi_decoration_coupling_production_refit_v1_{fingerprint[:12]}"
    if run_name != expected_name:
        raise RuntimeError("run name does not match the frozen production-refit contract")
    output_dir = VOLUME_MOUNT / run_name
    result_path = output_dir / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if result.get("status") != "complete":
            raise RuntimeError("existing production-refit result is incomplete")
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
            relative: _sha256_file(REMOTE_REPO_ROOT / relative) for relative in INPUT_FILES
        },
        "source": {
            "python_tree_sha256": source_sha256,
            "launcher_sha256": launcher_sha256,
            "run_fingerprint_sha256": fingerprint,
        },
        "random_initialization": True,
        "restarted_uncheckpointed_partial": restart_partial,
        "progress_committed_at_each_evaluation": True,
    }
    _atomic_json(result_path, result)
    output_volume.commit()
    return {"status": "complete", "run_name": run_name, "result": result}


@app.local_entrypoint()
def main(local_output: str = "") -> None:
    """Launch the immutable refit and download every integrity-checked artifact."""

    config = json.loads(CONFIG_PATH.read_text())
    _verify_contract(REPO_ROOT, config)
    run_name = _run_name()
    remote = train_production_refit.remote(run_name)
    result = remote["result"]
    if result.get("status") != "complete":
        raise RuntimeError("remote production refit did not complete")
    local_dir = (
        Path(local_output).resolve()
        if local_output
        else (REPO_ROOT / "results/phase1/ugi_decoration_coupling_production_refit_v1").resolve()
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
        raise RuntimeError("downloaded final production-refit checkpoint hash mismatch")
    if downloaded["checkpoint_latest.pt"]["sha256"] != result["checkpoint_latest"]["sha256"]:
        raise RuntimeError("downloaded latest production-refit checkpoint hash mismatch")
    for record in result.get("checkpoint_snapshots", ()):
        filename = Path(record["path"]).name
        if downloaded[filename]["sha256"] != record["sha256"]:
            raise RuntimeError(f"downloaded serial checkpoint hash mismatch: {filename}")
    if json.loads((local_dir / "result.json").read_text()) != result:
        raise RuntimeError("downloaded production-refit result differs from remote result")
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
    raise SystemExit(
        "Use `modal run scripts/modal_phase1_ugi_decoration_coupling_production_refit_v1.py`."
    )
