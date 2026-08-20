#!/usr/bin/env python3
"""Two more paired seeds for the semantics pilot, s4 and s5.

The v1 lanes are frozen at docs/FORGE_EVIDENCE_CONTRACT_v1.md amendments 11 and 12 and must not be
edited after launch, so this is a separate launcher rather than an edit to that table. Everything
that defines the experiment is inherited from v1 unchanged: same image, same L4 class, same
deterministic settings, same mounts, same volume, same sealed split, same training entry point. The
new configs differ from their s3 counterparts in exactly two fields, `seed` and the `task` string.

Why: delta_sem is reported over three paired seeds, which leaves the SD estimate on two degrees of
freedom. Two more paired seeds do not change the sign, which is positive in every head in every
seed already; they make the reported dispersion an honest number rather than a badly determined one.
The pilot track is separate from the production generator and feeds no novelty, yield, route or
assembly figure, so nothing frozen depends on the outcome.
"""

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
VOLUME_NAME = "forge-phase1-training"
GPU_CLASS = "L4"
LANES = (
    "flat_true_role_a100_s4",
    "flat_no_role_a100_s4",
    "flat_true_role_a100_s5",
    "flat_no_role_a100_s5",
)
CONFIGS = {
    "flat_true_role_a100_s4": "configs/model/phase1_ugi_semantics_closure_flat_true_role_a100_s4_v1.json",
    "flat_no_role_a100_s4": "configs/model/phase1_ugi_semantics_closure_flat_no_role_a100_s4_v1.json",
    "flat_true_role_a100_s5": "configs/model/phase1_ugi_semantics_closure_flat_true_role_a100_s5_v1.json",
    "flat_no_role_a100_s5": "configs/model/phase1_ugi_semantics_closure_flat_no_role_a100_s5_v1.json",
}
SEALED_SPLIT = "results/phase1/forge_coverage_splits_v1/splits.json"
SHARED_INPUT_FILES = (
    SEALED_SPLIT,
    "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz",
    "results/phase1/ugi_balanced_chemistry_corpus_v2/semantic_products.csv.gz",
    "results/phase1/ugi_balanced_chemistry_corpus_v2/semantic_atoms.csv.gz",
    "results/phase1/product_v3_atom_vocabulary.json",
    "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt",
)
INPUT_FILES = (*CONFIGS.values(), *SHARED_INPUT_FILES)
STATIC_OUTPUT_FILES = (
    "result.json",
    "checkpoint_best.pt",
    "checkpoint_latest.pt",
    "progress.json",
)

app = modal.App("forge-phase1-ugi-semantics-pilot-v2")
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


def _run_name(lane: str) -> str:
    if lane not in CONFIGS:
        raise ValueError(f"unknown semantics-pilot arm: {lane}")
    fingerprint = _run_fingerprint(REPO_ROOT / CONFIGS[lane], SOURCE_ROOT, Path(__file__).resolve())
    return f"ugi_semantics_pilot_v1_{lane}_{fingerprint[:12]}"


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
def train_lane(lane: str) -> dict[str, Any]:
    """Train one hash-pinned matched architecture lane."""

    if lane not in CONFIGS:
        raise RuntimeError(f"unknown semantics-pilot arm: {lane}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(REMOTE_SOURCE_ROOT))

    import numpy as np
    import rdkit
    import torch

    from forge.model.defog_feasibility import sha256_file
    from experiments.phase1.product_l1.training.ugi_joint_sparse_training import train_ugi_joint_sparse

    config_relative = CONFIGS[lane]
    config_path = REMOTE_REPO_ROOT / config_relative
    launcher_path = Path(__file__).resolve()
    source_sha256 = _sha256_python_tree(REMOTE_SOURCE_ROOT / "forge")
    launcher_sha256 = sha256_file(launcher_path)
    fingerprint = _run_fingerprint(
        config_path,
        REMOTE_SOURCE_ROOT / "forge",
        launcher_path,
    )
    expected_name = f"ugi_semantics_pilot_v1_{lane}_{fingerprint[:12]}"
    output_dir = VOLUME_MOUNT / expected_name
    result_path = output_dir / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if result.get("status") != "complete":
            raise RuntimeError("existing semantics-pilot result is incomplete")
        return {
            "status": "already_complete",
            "lane": lane,
            "run_name": expected_name,
            "result": result,
        }

    restart_partial = output_dir.exists() and any(output_dir.iterdir())

    def persist_progress(_step: int, _: Path) -> None:
        output_volume.commit()

    import time as _time

    started = _time.time()
    result = train_ugi_joint_sparse(
        config_path,
        REMOTE_REPO_ROOT,
        output_dir,
        smoke=False,
        overwrite=restart_partial,
        progress_callback=persist_progress,
    )
    wall_seconds = _time.time() - started
    arm_config = json.loads(config_path.read_text())
    result["semantics_pilot"] = {
        "arm": lane,
        "semantic_organization": arm_config["semantic_organization"],
        "coverage_alpha": arm_config["coverage_split"]["alpha"],
        "coverage_split_sha256": arm_config["coverage_split"]["sha256"],
        "sealed_split_sha256_as_mounted": sha256_file(REMOTE_REPO_ROOT / SEALED_SPLIT),
        "seed": arm_config["seed"],
        "steps": arm_config["full"]["steps"],
        "config_sha256": sha256_file(config_path),
        "training_wall_seconds": round(wall_seconds, 1),
        "gpu_seconds_billed_estimate": round(wall_seconds, 1),
        "contract": "docs/FORGE_EVIDENCE_CONTRACT_v1.md amendments 11 and 12",
    }
    result["cloud_execution"] = {
        "lane": lane,
        "run_name": expected_name,
        "gpu_class": GPU_CLASS,
        "volume": VOLUME_NAME,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "rdkit": rdkit.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else "cpu"),
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
    return {
        "status": "complete",
        "lane": lane,
        "run_name": expected_name,
        "result": result,
    }


def _git_provenance() -> dict[str, Any]:
    import subprocess

    def run(*args: str) -> str:
        return subprocess.run(args, cwd=REPO_ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()

    dirty = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty),
        "dirty_paths": dirty.splitlines()[:20],
    }


@app.local_entrypoint()
def main(local_output: str = "", arms: str = "") -> None:
    """arms: optional comma-separated subset, so one arm can be run first as a smoke check."""
    provenance = _git_provenance()
    if provenance["dirty"]:
        print("WARNING: launching from a dirty worktree; recorded in the summary")
    selected = tuple(value.strip() for value in arms.split(",") if value.strip()) or LANES
    unknown = [value for value in selected if value not in CONFIGS]
    if unknown:
        raise RuntimeError(f"unknown semantics-pilot arms: {unknown}")
    print(f"launching {len(selected)} arm(s): {', '.join(selected)}")
    remote_results = list(train_lane.map(selected))
    root = (
        Path(local_output).resolve()
        if local_output
        else (REPO_ROOT / "results/phase1/forge_semantics_pilot_v2").resolve()
    )
    summary = {}
    for remote in remote_results:
        lane = str(remote["lane"])
        run_name = str(remote["run_name"])
        result = remote["result"]
        if result.get("status") != "complete":
            raise RuntimeError(f"remote {lane} training did not complete")
        pilot = result["semantics_pilot"]
        if pilot["coverage_split_sha256"] != pilot["sealed_split_sha256_as_mounted"]:
            raise RuntimeError(f"{lane} trained against a different sealed split than its config pins")
        local_dir = root / lane
        snapshots = tuple(
            Path(record["path"]).name for record in result.get("checkpoint_snapshots", ())
        )
        downloaded = {}
        for filename in (*STATIC_OUTPUT_FILES, *snapshots):
            local_path = local_dir / filename
            downloaded[filename] = {
                "bytes": _atomic_volume_download(f"{run_name}/{filename}", local_path),
                "sha256": _sha256_file(local_path),
            }
        if downloaded["checkpoint_best.pt"]["sha256"] != result["checkpoint"]["sha256"]:
            raise RuntimeError(f"downloaded {lane} best checkpoint hash mismatch")
        if downloaded["checkpoint_latest.pt"]["sha256"] != result["checkpoint_latest"]["sha256"]:
            raise RuntimeError(f"downloaded {lane} latest checkpoint hash mismatch")
        for record in result.get("checkpoint_snapshots", ()):
            filename = Path(record["path"]).name
            if downloaded[filename]["sha256"] != record["sha256"]:
                raise RuntimeError(f"downloaded {lane} serial checkpoint hash mismatch")
        if json.loads((local_dir / "result.json").read_text()) != result:
            raise RuntimeError(f"downloaded {lane} result differs from remote result")
        summary[lane] = {
            "remote_status": remote["status"],
            "semantics_pilot": result["semantics_pilot"],
            "run_name": run_name,
            "local_output": str(local_dir),
            "downloaded": downloaded,
            "selection": result["selection"],
            "final_evaluation": result["evaluations"][-1],
        }
    _atomic_json(root / "launch_summary.json",
                 {"git": provenance, "arms": summary,
                  "gpu_class": GPU_CLASS, "volume": VOLUME_NAME})
    print(json.dumps({"git": provenance, "arms": summary}, indent=2, sort_keys=True))
