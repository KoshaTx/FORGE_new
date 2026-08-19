#!/usr/bin/env python3
"""Run the matched Ugi numerical sampler convergence sweep on Modal."""

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
REMOTE_LOCAL_CHECKPOINT = (
    REMOTE_REPO_ROOT
    / "results/phase1/ugi_joint_sparse_production_refit_full/checkpoint_step_1000.pt"
)
LOCAL_OUTPUT_ROOT = REPO_ROOT / "results/phase1/ugi_numerical_sampler_convergence_v1"
REMOTE_OUTPUT_ROOT = Path("/forge-output/ugi_numerical_sampler_convergence_v1")
TRAINING_CACHE_MOUNT = REMOTE_REPO_ROOT / "results/phase1/ugi_balanced_training_cache_v2"
CHECKPOINT_SHA256 = "7595b650745df1194c31eb0293ef17b292d57a655e61433c75f307074aa174a4"
PROGRAM_SHA256 = "87ae8e1066cfe1afb1042d77258b29cb6f1b04b8c0b9156c7966fde901008c1d"
FLOW_SEED = 20260824
TERMINAL_SEED = 20260828
PROGRAM_COUNT = 512
SAMPLE_STEPS = (8, 16, 32, 64)
PINNED_FILES = (
    "configs/assembly/ugi_variant.yaml",
    "configs/model/phase1_ugi_numerical_sampler_convergence_v1.json",
    "data/vendor/qualified_reactions_v1.json",
    "results/phase1/product_v3_atom_vocabulary.json",
    "results/phase1/ugi_closure_expanded_full/checkpoint_best.pt",
    "results/phase1/ugi_production_refit_fresh_census_v1/programs_shard_00.json",
)

app = modal.App("forge-phase1-ugi-numerical-sampler-convergence")
training_volume = modal.Volume.from_name("forge-phase1-training")
input_volume = modal.Volume.from_name("forge-phase1-production-inputs")
output_volume = modal.Volume.from_name(
    "forge-phase1-numerical-sampler-convergence", create_if_missing=True
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


def _program_subset(source: Path, destination: Path) -> None:
    value = json.loads(source.read_text())
    samples = value.get("samples")
    if not isinstance(samples, list) or len(samples) < PROGRAM_COUNT:
        raise RuntimeError("frozen program file does not contain the requested subset")
    value["samples"] = samples[:PROGRAM_COUNT]
    value["subset_contract"] = {
        "source_sha256": PROGRAM_SHA256,
        "selection": f"first_{PROGRAM_COUNT}_rows",
    }
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


@app.function(
    image=image,
    cpu=8.0,
    memory=16384,
    timeout=7200,
    max_containers=4,
    single_use_containers=True,
    volumes={
        "/forge-training": training_volume,
        str(TRAINING_CACHE_MOUNT): input_volume,
        "/forge-output": output_volume,
    },
)
def run_step_count(sample_steps: int) -> dict[str, Any]:
    """Generate one integration-grid arm on matched programs and random seeds."""

    import sys

    sys.path.insert(0, str(REMOTE_REPO_ROOT / "src"))
    from forge.product.ugi_joint_end_to_end_sampling import sample_ugi_joint_end_to_end

    if sample_steps not in SAMPLE_STEPS:
        raise ValueError(f"unsupported numerical grid: {sample_steps}")
    if _sha256_file(REMOTE_CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("mounted selected checkpoint changed")
    full_program_path = (
        REMOTE_REPO_ROOT
        / "results/phase1/ugi_production_refit_fresh_census_v1/programs_shard_00.json"
    )
    if _sha256_file(full_program_path) != PROGRAM_SHA256:
        raise RuntimeError("matched program file changed")

    REMOTE_LOCAL_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REMOTE_CHECKPOINT, REMOTE_LOCAL_CHECKPOINT)
    subset_path = Path(f"/tmp/program_subset_{PROGRAM_COUNT}.json")
    _program_subset(full_program_path, subset_path)
    arm = f"steps_{sample_steps:02d}"
    output_dir = REMOTE_OUTPUT_ROOT / arm
    result = sample_ugi_joint_end_to_end(
        REMOTE_REPO_ROOT,
        output_dir,
        joint_checkpoint_path=REMOTE_LOCAL_CHECKPOINT,
        closure_checkpoint_path=(
            REMOTE_REPO_ROOT / "results/phase1/ugi_closure_expanded_full/checkpoint_best.pt"
        ),
        matched_staged_result_path=subset_path,
        sample_steps=sample_steps,
        batch_size=32,
        seed=FLOW_SEED,
        overwrite=True,
        maximum_adjacent_branch_runs=(2, 1, 1),
        qualified_reactions_path=(REMOTE_REPO_ROOT / "data/vendor/qualified_reactions_v1.json"),
        evaluate_exact_l1_terminal_admission=True,
        terminal_decoder_mode="stochastic",
        terminal_decoder_seed=TERMINAL_SEED,
        terminal_temperature=1.0,
    )
    output_volume.commit()
    return {
        "sample_steps": sample_steps,
        "arm": arm,
        "result_sha256": _sha256_file(output_dir / "result.json"),
        "valid_fraction": result["statistics"]["valid_fraction"],
        "total_seconds": result["sampling"]["total_seconds"],
    }


@app.local_entrypoint()
def main() -> None:
    """Run all numerical grids concurrently and download their artifacts."""

    remote_results = list(run_step_count.map(SAMPLE_STEPS))
    downloaded: dict[str, dict[str, Any]] = {}
    for remote in remote_results:
        arm = str(remote["arm"])
        local_dir = LOCAL_OUTPUT_ROOT / arm
        result_path = local_dir / "result.json"
        image_path = local_dir / "samples.png"
        result_bytes = _atomic_volume_download(
            f"ugi_numerical_sampler_convergence_v1/{arm}/result.json", result_path
        )
        image_bytes = _atomic_volume_download(
            f"ugi_numerical_sampler_convergence_v1/{arm}/samples.png", image_path
        )
        if _sha256_file(result_path) != remote["result_sha256"]:
            raise RuntimeError(f"downloaded numerical result changed: {arm}")
        downloaded[arm] = {
            "result_bytes": result_bytes,
            "image_bytes": image_bytes,
            **remote,
        }
    print(json.dumps({"status": "complete", "arms": downloaded}, indent=2))


if __name__ == "__main__":
    raise SystemExit("Use `modal run scripts/modal_phase1_ugi_numerical_sampler_convergence.py`.")
