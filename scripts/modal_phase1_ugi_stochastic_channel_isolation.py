#!/usr/bin/env python3
"""Run the matched Ugi terminal-channel isolation diagnostic on Modal."""

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
LOCAL_OUTPUT_ROOT = REPO_ROOT / "results/phase1/ugi_stochastic_channel_isolation_v1"
REMOTE_OUTPUT_ROOT = Path("/forge-output/ugi_stochastic_channel_isolation_v1")
TRAINING_CACHE_MOUNT = REMOTE_REPO_ROOT / "results/phase1/ugi_balanced_training_cache_v2"
CHECKPOINT_SHA256 = "7595b650745df1194c31eb0293ef17b292d57a655e61433c75f307074aa174a4"
FLOW_SEED = 20260824
TERMINAL_SEED = 20260828
PROGRAM_SHA256 = "87ae8e1066cfe1afb1042d77258b29cb6f1b04b8c0b9156c7966fde901008c1d"
MODES = (
    "bond_stochastic",
    "atom_bond_stochastic",
    "decoration_bond_stochastic",
    "stochastic",
)
PINNED_FILES = (
    "configs/assembly/ugi_variant.yaml",
    "configs/model/phase1_ugi_stochastic_channel_isolation_v1.json",
    "data/vendor/qualified_reactions_v1.json",
    "results/phase1/product_v3_atom_vocabulary.json",
    "results/phase1/ugi_closure_expanded_full/checkpoint_best.pt",
    "results/phase1/ugi_production_refit_fresh_census_v1/programs_shard_00.json",
)

app = modal.App("forge-phase1-ugi-stochastic-channel-isolation")
training_volume = modal.Volume.from_name("forge-phase1-training")
input_volume = modal.Volume.from_name("forge-phase1-production-inputs")
output_volume = modal.Volume.from_name(
    "forge-phase1-stochastic-channel-isolation", create_if_missing=True
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
def run_mode(mode: str) -> dict[str, Any]:
    """Generate one decoder arm on exactly matched programs and flow seed."""

    import sys

    sys.path.insert(0, str(REMOTE_REPO_ROOT / "src"))
    from forge.product.ugi_joint_end_to_end_sampling import sample_ugi_joint_end_to_end

    if mode not in MODES:
        raise ValueError(f"unsupported decoder arm: {mode}")
    if _sha256_file(REMOTE_CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("mounted selected checkpoint changed")
    program_path = (
        REMOTE_REPO_ROOT
        / "results/phase1/ugi_production_refit_fresh_census_v1/programs_shard_00.json"
    )
    if _sha256_file(program_path) != PROGRAM_SHA256:
        raise RuntimeError("matched program file changed")

    REMOTE_LOCAL_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REMOTE_CHECKPOINT, REMOTE_LOCAL_CHECKPOINT)
    output_dir = REMOTE_OUTPUT_ROOT / mode
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
        seed=FLOW_SEED,
        overwrite=True,
        maximum_adjacent_branch_runs=(2, 1, 1),
        qualified_reactions_path=(REMOTE_REPO_ROOT / "data/vendor/qualified_reactions_v1.json"),
        evaluate_exact_l1_terminal_admission=True,
        terminal_decoder_mode=mode,
        terminal_decoder_seed=TERMINAL_SEED,
        terminal_temperature=1.0,
    )
    output_volume.commit()
    return {
        "mode": mode,
        "result_sha256": _sha256_file(output_dir / "result.json"),
        "valid_fraction": result["statistics"]["valid_fraction"],
    }


@app.local_entrypoint()
def main() -> None:
    """Run all four matched arms concurrently and download their artifacts."""

    remote_results = list(run_mode.map(MODES))
    downloaded: dict[str, dict[str, Any]] = {}
    for remote in remote_results:
        mode = str(remote["mode"])
        local_dir = LOCAL_OUTPUT_ROOT / mode
        result_path = local_dir / "result.json"
        image_path = local_dir / "samples.png"
        result_bytes = _atomic_volume_download(
            f"ugi_stochastic_channel_isolation_v1/{mode}/result.json", result_path
        )
        image_bytes = _atomic_volume_download(
            f"ugi_stochastic_channel_isolation_v1/{mode}/samples.png", image_path
        )
        if _sha256_file(result_path) != remote["result_sha256"]:
            raise RuntimeError(f"downloaded decoder result changed: {mode}")
        downloaded[mode] = {
            "result_bytes": result_bytes,
            "image_bytes": image_bytes,
            **remote,
        }
    print(json.dumps({"status": "complete", "arms": downloaded}, indent=2))


if __name__ == "__main__":
    raise SystemExit("Use `modal run scripts/modal_phase1_ugi_stochastic_channel_isolation.py`.")
