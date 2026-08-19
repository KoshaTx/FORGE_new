#!/usr/bin/env python3
"""Run the independent all-fold-refit generator census on Modal CPUs."""

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
REMOTE_CHECKPOINT = Path(
    "/forge-training/ugi_joint_sparse_production_refit_67bd5152b3a3/checkpoint_step_1000.pt"
)
LOCAL_CHECKPOINT = (
    REPO_ROOT / "results/phase1/ugi_joint_sparse_production_refit_full/checkpoint_step_1000.pt"
)
REMOTE_LOCAL_CHECKPOINT = (
    REMOTE_REPO_ROOT
    / "results/phase1/ugi_joint_sparse_production_refit_full/checkpoint_step_1000.pt"
)
REMOTE_OUTPUT_ROOT = Path("/forge-output/ugi_production_refit_fresh_census_v1")
LOCAL_OUTPUT_ROOT = REPO_ROOT / "results/phase1/ugi_production_refit_fresh_census_v1"
TRAINING_CACHE_MOUNT = REMOTE_REPO_ROOT / "results/phase1/ugi_balanced_training_cache_v2"
CHECKPOINT_SHA256 = "7595b650745df1194c31eb0293ef17b292d57a655e61433c75f307074aa174a4"
SHARDS = (0, 1, 2, 3)
FLOW_SEEDS = (20260824, 20260825, 20260826, 20260827)
TERMINAL_SEEDS = (20260828, 20260829, 20260830, 20260831)
PROGRAM_HASHES = (
    "87ae8e1066cfe1afb1042d77258b29cb6f1b04b8c0b9156c7966fde901008c1d",
    "904881d1c14ed378cca11aa523c6737827045c26620f21afe363750f408b67ea",
    "31ddea01cdada991aa9db3cab2c4a4b1a0b80c31a608348028872db44c5e246c",
    "bfd18bc58ead4f38b6e2b822ce21a419bb632eb1703e72aeb9cd03a024335039",
)
PINNED_FILES = (
    "configs/assembly/ugi_variant.yaml",
    "configs/model/phase1_ugi_production_refit_fresh_census_v1.json",
    "data/vendor/qualified_reactions_v1.json",
    "results/phase1/product_v3_atom_vocabulary.json",
    "results/phase1/ugi_closure_expanded_full/checkpoint_best.pt",
    "results/phase1/ugi_product_l1_production_generator_v3.json",
    "results/phase1/ugi_program_prior_v3_all_fold.json",
    "results/phase1/ugi_production_refit_fresh_census_v1/programs_shard_00.json",
    "results/phase1/ugi_production_refit_fresh_census_v1/programs_shard_01.json",
    "results/phase1/ugi_production_refit_fresh_census_v1/programs_shard_02.json",
    "results/phase1/ugi_production_refit_fresh_census_v1/programs_shard_03.json",
)

app = modal.App("forge-phase1-ugi-production-refit-fresh-census")
training_volume = modal.Volume.from_name("forge-phase1-training")
input_volume = modal.Volume.from_name("forge-phase1-production-inputs")
output_volume = modal.Volume.from_name(
    "forge-phase1-production-refit-fresh-census", create_if_missing=True
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
    max_containers=4,
    single_use_containers=True,
    volumes={
        "/forge-training": training_volume,
        str(TRAINING_CACHE_MOUNT): input_volume,
        "/forge-output": output_volume,
    },
)
def run_shard(shard: int) -> dict[str, Any]:
    """Generate one independent 1,024-program census shard."""

    import platform
    import sys

    sys.path.insert(0, str(REMOTE_REPO_ROOT / "src"))
    import numpy
    import torch
    from rdkit import rdBase

    from forge.product.ugi_joint_end_to_end_sampling import sample_ugi_joint_end_to_end

    if int(shard) not in SHARDS:
        raise ValueError(f"unsupported census shard: {shard}")
    observed_runtime = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__.split("+", 1)[0],
        "rdkit": rdBase.rdkitVersion,
    }
    expected_runtime = {
        "python": "3.14.2",
        "numpy": "2.5.1",
        "torch": "2.13.0",
        "rdkit": "2026.03.4",
    }
    if observed_runtime != expected_runtime:
        raise RuntimeError(f"unexpected fresh-census runtime: {observed_runtime}")
    if _sha256_file(REMOTE_CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("mounted selected checkpoint changed")
    program_path = (
        REMOTE_REPO_ROOT
        / f"results/phase1/ugi_production_refit_fresh_census_v1/programs_shard_{shard:02d}.json"
    )
    if _sha256_file(program_path) != PROGRAM_HASHES[shard]:
        raise RuntimeError(f"census program shard changed: {shard}")

    REMOTE_LOCAL_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REMOTE_CHECKPOINT, REMOTE_LOCAL_CHECKPOINT)
    output_dir = REMOTE_OUTPUT_ROOT / f"shard_{shard:02d}"
    result = sample_ugi_joint_end_to_end(
        REMOTE_REPO_ROOT,
        output_dir,
        joint_checkpoint_path=REMOTE_LOCAL_CHECKPOINT,
        closure_checkpoint_path=(
            REMOTE_REPO_ROOT / "results/phase1/ugi_closure_expanded_full/checkpoint_best.pt"
        ),
        matched_staged_result_path=program_path,
        sample_steps=8,
        batch_size=16,
        seed=FLOW_SEEDS[shard],
        overwrite=True,
        maximum_adjacent_branch_runs=(2, 1, 1),
        qualified_reactions_path=(REMOTE_REPO_ROOT / "data/vendor/qualified_reactions_v1.json"),
        evaluate_exact_l1_terminal_admission=True,
        terminal_decoder_mode="bond_stochastic",
        terminal_decoder_seed=TERMINAL_SEEDS[shard],
        terminal_temperature=1.0,
    )
    output_volume.commit()
    return {
        "shard": int(shard),
        "runtime": observed_runtime,
        "result_sha256": _sha256_file(output_dir / "result.json"),
        "valid_fraction": result["statistics"]["valid_fraction"],
    }


@app.local_entrypoint()
def main() -> None:
    """Run all four shards concurrently and download immutable artifacts."""

    if _sha256_file(LOCAL_CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("local selected checkpoint changed before launch")
    remote_results = list(run_shard.map(SHARDS))
    downloaded: dict[int, dict[str, Any]] = {}
    for remote in remote_results:
        shard = int(remote["shard"])
        local_dir = LOCAL_OUTPUT_ROOT / f"shard_{shard:02d}"
        result_path = local_dir / "result.json"
        image_path = local_dir / "samples.png"
        result_bytes = _atomic_volume_download(
            f"ugi_production_refit_fresh_census_v1/shard_{shard:02d}/result.json",
            result_path,
        )
        image_bytes = _atomic_volume_download(
            f"ugi_production_refit_fresh_census_v1/shard_{shard:02d}/samples.png",
            image_path,
        )
        if _sha256_file(result_path) != remote["result_sha256"]:
            raise RuntimeError(f"downloaded census result changed: {shard}")
        downloaded[shard] = {
            "result_bytes": result_bytes,
            "image_bytes": image_bytes,
            **remote,
        }
    print(json.dumps({"status": "complete", "shards": downloaded}, indent=2))


if __name__ == "__main__":
    raise SystemExit("Use `modal run scripts/modal_phase1_ugi_production_refit_fresh_census.py`.")
