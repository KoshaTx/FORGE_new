#!/usr/bin/env python3
"""Run the decoder-matched all-fold checkpoint audit on Modal CPUs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO_ROOT = Path("/root/forge_repo")
REMOTE_CHECKPOINT_ROOT = Path("/forge-training/ugi_joint_sparse_production_refit_67bd5152b3a3")
REMOTE_OUTPUT_ROOT = Path("/forge-output/ugi_production_refit_decoder_checkpoint_selection_v1")
LOCAL_OUTPUT_ROOT = (
    REPO_ROOT / "results/phase1/ugi_production_refit_decoder_checkpoint_selection_v1"
)
TRAINING_CACHE_MOUNT = REMOTE_REPO_ROOT / "results/phase1/ugi_balanced_training_cache_v2"
LOCAL_CHECKPOINT_ROOT = REPO_ROOT / "results/phase1/ugi_joint_sparse_production_refit_full"
REMOTE_LOCAL_CHECKPOINT_ROOT = (
    REMOTE_REPO_ROOT / "results/phase1/ugi_joint_sparse_production_refit_full"
)
STEPS = (100, 500, 1000, 1500, 1700)
CHECKPOINT_HASHES = {
    100: "e8b5ef5e0e5dce6258c82e7d13f009fa01cdbb29edba978b7b22ab6cdb826aa2",
    500: "6fea3b259f565b2499720e8b51438ac2dc97599d68de942bb0e52d7d6209db9a",
    1000: "7595b650745df1194c31eb0293ef17b292d57a655e61433c75f307074aa174a4",
    1500: "011f7a205014eb8ee49cca2014d8c16411e476423d2ad9bfd52d0f4c2edb258e",
    1700: "997dee999349045fc306e25684aabe6f9ba58f195cbba7f67e33e1ff2011c5a5",
}
FLOW_SEED = 20260814
TERMINAL_SEED = 20260815
PINNED_FILES = (
    "configs/assembly/ugi_variant.yaml",
    "data/vendor/qualified_reactions_v1.json",
    "results/phase1/product_v3_atom_vocabulary.json",
    "results/phase1/ugi_closure_expanded_full/checkpoint_best.pt",
    "results/phase1/ugi_production_refit_decoder_checkpoint_audit_probe_1024.json",
)

app = modal.App("forge-phase1-ugi-production-refit-checkpoint-audit")
training_volume = modal.Volume.from_name("forge-phase1-training")
input_volume = modal.Volume.from_name("forge-phase1-production-inputs")
output_volume = modal.Volume.from_name(
    "forge-phase1-production-refit-checkpoint-audit", create_if_missing=True
)
image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("libxrender1", "libxext6")
    .pip_install(
        "torch==2.13.0",
        "numpy==2.5.1",
        "rdkit==2026.3.4",
        "PyYAML==6.0.3",
    )
    .add_local_dir(REPO_ROOT / "src", remote_path=str(REMOTE_REPO_ROOT / "src"), copy=True)
)
for relative in PINNED_FILES:
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
    cpu=4.0,
    memory=12288,
    timeout=1800,
    max_containers=5,
    single_use_containers=True,
    volumes={
        "/forge-training": training_volume,
        str(TRAINING_CACHE_MOUNT): input_volume,
        "/forge-output": output_volume,
    },
)
def run_checkpoint(step: int) -> dict[str, Any]:
    """Sample one checkpoint under the frozen production decoder."""

    import platform
    import sys

    sys.path.insert(0, str(REMOTE_REPO_ROOT / "src"))
    import numpy
    import torch
    from rdkit import rdBase

    from forge.product.ugi_joint_end_to_end_sampling import sample_ugi_joint_end_to_end

    if int(step) not in STEPS:
        raise ValueError(f"unsupported checkpoint step: {step}")
    expected_runtime = {
        "python": "3.14.2",
        "numpy": "2.5.1",
        "torch": "2.13.0",
        "rdkit": "2026.03.4",
    }
    observed_runtime = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__.split("+", 1)[0],
        "rdkit": rdBase.rdkitVersion,
    }
    if observed_runtime != expected_runtime:
        raise RuntimeError(f"unexpected checkpoint-audit runtime: {observed_runtime}")

    checkpoint_name = f"checkpoint_step_{step}.pt"
    mounted_checkpoint = REMOTE_CHECKPOINT_ROOT / checkpoint_name
    expected_hash = CHECKPOINT_HASHES[int(step)]
    if _sha256_file(mounted_checkpoint) != expected_hash:
        raise RuntimeError(f"mounted checkpoint hash changed at step {step}")
    REMOTE_LOCAL_CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = REMOTE_LOCAL_CHECKPOINT_ROOT / checkpoint_name
    shutil.copyfile(mounted_checkpoint, checkpoint_path)

    output_dir = REMOTE_OUTPUT_ROOT / f"step_{step}"
    result = sample_ugi_joint_end_to_end(
        REMOTE_REPO_ROOT,
        output_dir,
        joint_checkpoint_path=checkpoint_path,
        closure_checkpoint_path=(
            REMOTE_REPO_ROOT / "results/phase1/ugi_closure_expanded_full/checkpoint_best.pt"
        ),
        matched_staged_result_path=(
            REMOTE_REPO_ROOT
            / "results/phase1/ugi_production_refit_decoder_checkpoint_audit_probe_1024.json"
        ),
        sample_steps=8,
        batch_size=16,
        seed=FLOW_SEED,
        overwrite=True,
        maximum_adjacent_branch_runs=(2, 1, 1),
        qualified_reactions_path=(REMOTE_REPO_ROOT / "data/vendor/qualified_reactions_v1.json"),
        evaluate_exact_l1_terminal_admission=True,
        terminal_decoder_mode="bond_stochastic",
        terminal_decoder_seed=TERMINAL_SEED,
        terminal_temperature=1.0,
    )
    output_volume.commit()
    statistics = result["statistics"]
    return {
        "step": int(step),
        "runtime": observed_runtime,
        "result_sha256": _sha256_file(output_dir / "result.json"),
        "valid_fraction": statistics["valid_fraction"],
        "unique_valid_molecules": statistics["unique_valid_molecules"],
    }


@app.local_entrypoint()
def main() -> None:
    """Run all saved checkpoints concurrently and download their artifacts."""

    expected_hashes = dict(CHECKPOINT_HASHES)
    for step, expected_hash in expected_hashes.items():
        local_checkpoint = LOCAL_CHECKPOINT_ROOT / f"checkpoint_step_{step}.pt"
        if _sha256_file(local_checkpoint) != expected_hash:
            raise RuntimeError(f"local checkpoint hash changed before launch at step {step}")
    remote_results = list(run_checkpoint.map(STEPS))
    downloaded: dict[int, dict[str, Any]] = {}
    for remote in remote_results:
        step = int(remote["step"])
        local_dir = LOCAL_OUTPUT_ROOT / f"step_{step}"
        result_path = local_dir / "result.json"
        image_path = local_dir / "samples.png"
        result_bytes = _atomic_volume_download(
            f"ugi_production_refit_decoder_checkpoint_selection_v1/step_{step}/result.json",
            result_path,
        )
        image_bytes = _atomic_volume_download(
            f"ugi_production_refit_decoder_checkpoint_selection_v1/step_{step}/samples.png",
            image_path,
        )
        if _sha256_file(result_path) != remote["result_sha256"]:
            raise RuntimeError(f"downloaded result hash changed at step {step}")
        result = json.loads(result_path.read_text())
        local_checkpoint = LOCAL_CHECKPOINT_ROOT / f"checkpoint_step_{step}.pt"
        if _sha256_file(local_checkpoint) != expected_hashes[step]:
            raise RuntimeError(f"local checkpoint hash changed at step {step}")
        downloaded[step] = {
            "result_bytes": result_bytes,
            "image_bytes": image_bytes,
            "result_sha256": remote["result_sha256"],
            "valid_fraction": result["statistics"]["valid_fraction"],
            "runtime": remote["runtime"],
        }
    print(json.dumps({"status": "complete", "checkpoints": downloaded}, indent=2))


if __name__ == "__main__":
    raise SystemExit(
        "Use `modal run scripts/modal_phase1_ugi_production_refit_checkpoint_audit.py`."
    )
