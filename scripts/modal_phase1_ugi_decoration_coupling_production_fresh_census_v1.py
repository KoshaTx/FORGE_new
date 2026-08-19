#!/usr/bin/env python3
"""Run the independent final-generator census on four Modal CPU workers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import modal

_LOCAL_REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = (
    Path("/root/forge_repo") if not (_LOCAL_REPO_ROOT / "configs").is_dir() else _LOCAL_REPO_ROOT
)
REMOTE_REPO_ROOT = Path("/root/forge_repo")
REMOTE_SOURCE_ROOT = Path("/root")
TRAINING_MOUNT = Path("/forge-training")
OUTPUT_MOUNT = Path("/forge-census")
REMOTE_CACHE_DIR = REMOTE_REPO_ROOT / "results/phase1/ugi_balanced_training_cache_v2"
CONFIG_RELATIVE = "configs/model/phase1_ugi_decoration_coupling_production_fresh_census_v1.json"
CONFIG_PATH = REPO_ROOT / CONFIG_RELATIVE
CONFIG = json.loads(CONFIG_PATH.read_text())
DESIGN = CONFIG["design"]
INPUTS = CONFIG["inputs"]
RUN_NAME = "ugi_decoration_coupling_production_refit_v1_d4a23a76b796"
REMOTE_CHECKPOINT = TRAINING_MOUNT / RUN_NAME / "checkpoint_best.pt"
LOCAL_OUTPUT_ROOT = REPO_ROOT / "results/phase1/ugi_decoration_coupling_production_fresh_census_v1"

app = modal.App("forge-phase1-ugi-decoration-production-fresh-census-v1")
training_volume = modal.Volume.from_name("forge-phase1-training")
input_volume = modal.Volume.from_name("forge-phase1-production-inputs")
output_volume = modal.Volume.from_name(
    "forge-phase1-decoration-production-fresh-census-v1", create_if_missing=True
)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libxrender1", "libxext6")
    .pip_install(
        "torch==2.11.0",
        "numpy==2.4.2",
        "rdkit==2025.9.6",
        "PyYAML==6.0.3",
    )
    .add_local_dir(REPO_ROOT / "src" / "forge", remote_path="/root/forge", copy=True)
)
for relative in (
    CONFIG_RELATIVE,
    str(INPUTS["qualified_reactions"]["path"]),
    str(INPUTS["closure_checkpoint"]["path"]),
    *(str(INPUTS[f"program_shard_{shard:02d}"]["path"]) for shard in range(4)),
):
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


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version")
        != "phase1_ugi_decoration_coupling_production_fresh_census_config.v1"
        or config.get("status") != "frozen_before_independent_production_census"
    ):
        raise RuntimeError("production-census config is not frozen")
    design = config["design"]
    if (
        int(design["shards"]) != 4
        or int(design["programs_per_shard"]) != 1024
        or int(design["attempted_draws"]) != 4096
        or design["terminal_decoder_mode"] != "stochastic"
        or bool(design["retry_or_repair"])
    ):
        raise RuntimeError("production-census design changed")


@app.function(
    image=image,
    cpu=8.0,
    memory=24576,
    timeout=3600,
    max_containers=4,
    single_use_containers=True,
    volumes={
        str(TRAINING_MOUNT): training_volume,
        str(REMOTE_CACHE_DIR): input_volume,
        str(OUTPUT_MOUNT): output_volume,
    },
)
def sample_shard(shard: int) -> dict[str, Any]:
    """Sample one immutable 1,024-program production-census shard."""

    import platform
    import sys

    sys.path.insert(0, str(REMOTE_SOURCE_ROOT))
    import numpy
    import torch
    from rdkit import rdBase

    from forge.product.ugi_joint_end_to_end_sampling import (
        joint_sampling_result_matches_request,
        sample_ugi_joint_end_to_end,
    )

    config = json.loads((REMOTE_REPO_ROOT / CONFIG_RELATIVE).read_text())
    _validate_config(config)
    if shard not in range(int(config["design"]["shards"])):
        raise ValueError(f"unsupported census shard: {shard}")
    checkpoint_hash = str(config["inputs"]["joint_checkpoint"]["sha256"])
    if _sha256_file(REMOTE_CHECKPOINT) != checkpoint_hash:
        raise RuntimeError("mounted production checkpoint changed")
    program_record = config["inputs"][f"program_shard_{shard:02d}"]
    program_path = REMOTE_REPO_ROOT / str(program_record["path"])
    if _sha256_file(program_path) != str(program_record["sha256"]):
        raise RuntimeError(f"production-census program shard changed: {shard}")

    local_checkpoint = Path("/tmp/forge-checkpoints/checkpoint_best.pt")
    local_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REMOTE_CHECKPOINT, local_checkpoint)
    output_dir = OUTPUT_MOUNT / f"shard_{shard:02d}"
    result_path = output_dir / "result.json"
    render_path = output_dir / "samples.png"
    flow_seed = int(config["design"]["flow_seeds"][shard])
    terminal_seed = int(config["design"]["terminal_seeds"][shard])
    if result_path.is_file() and render_path.is_file():
        existing = json.loads(result_path.read_text())
        if joint_sampling_result_matches_request(
            existing,
            seed=flow_seed,
            program_offset=0,
            program_limit=1024,
            terminal_decoder_mode="stochastic",
            terminal_decoder_seed=terminal_seed,
            terminal_temperature=1.0,
            checkpoint_filename="checkpoint_best.pt",
        ):
            return {
                "shard": shard,
                "result_sha256": _sha256_file(result_path),
                "samples_png_sha256": _sha256_file(render_path),
                "valid_fraction": existing["statistics"]["valid_fraction"],
                "reused_compatible_result": True,
            }

    result = sample_ugi_joint_end_to_end(
        REMOTE_REPO_ROOT,
        output_dir,
        joint_checkpoint_path=local_checkpoint,
        closure_checkpoint_path=(
            REMOTE_REPO_ROOT / str(config["inputs"]["closure_checkpoint"]["path"])
        ),
        matched_staged_result_path=program_path,
        sample_steps=int(config["design"]["sample_steps"]),
        batch_size=int(config["design"]["batch_size"]),
        seed=flow_seed,
        overwrite=True,
        maximum_adjacent_branch_runs=tuple(
            int(value) for value in config["design"]["maximum_adjacent_branch_runs"]
        ),
        qualified_reactions_path=(
            REMOTE_REPO_ROOT / str(config["inputs"]["qualified_reactions"]["path"])
        ),
        evaluate_exact_l1_terminal_admission=True,
        terminal_decoder_mode="stochastic",
        terminal_decoder_seed=terminal_seed,
        terminal_temperature=float(config["design"]["terminal_temperature"]),
        program_offset=0,
        program_limit=1024,
        reference_comparison_mode="deferred",
    )
    output_volume.commit()
    return {
        "shard": shard,
        "checkpoint_sha256": checkpoint_hash,
        "result_sha256": _sha256_file(result_path),
        "samples_png_sha256": _sha256_file(render_path),
        "valid_fraction": result["statistics"]["valid_fraction"],
        "reused_compatible_result": False,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
    }


@app.function(image=image, cpu=0.25, memory=512, timeout=4000)
def orchestrate(shards: tuple[int, ...]) -> list[dict[str, Any]]:
    """Keep every restartable census shard alive independently."""

    calls = [sample_shard.spawn(shard) for shard in shards]
    return [call.get() for call in calls]


@app.local_entrypoint()
def main() -> None:
    """Run four shards and download hash-verified results atomically."""

    _validate_config(CONFIG)
    implementation = CONFIG["implementation"]["runner"]
    if _sha256_file(Path(__file__)) != str(implementation["sha256"]):
        raise RuntimeError("production-census runner changed after freeze")
    if _sha256_file(REPO_ROOT / INPUTS["joint_checkpoint"]["path"]) != str(
        INPUTS["joint_checkpoint"]["sha256"]
    ):
        raise RuntimeError("local production checkpoint changed before launch")
    remote_results = orchestrate.remote(tuple(range(int(DESIGN["shards"]))))
    summary = {}
    for remote in remote_results:
        shard = int(remote["shard"])
        local_dir = LOCAL_OUTPUT_ROOT / f"shard_{shard:02d}"
        downloaded = {}
        for filename, digest_key in (
            ("result.json", "result_sha256"),
            ("samples.png", "samples_png_sha256"),
        ):
            local_path = local_dir / filename
            downloaded[filename] = _atomic_volume_download(
                f"shard_{shard:02d}/{filename}", local_path
            )
            if _sha256_file(local_path) != str(remote[digest_key]):
                raise RuntimeError(f"downloaded census artifact changed: {shard}/{filename}")
        summary[f"shard_{shard:02d}"] = {**remote, "downloaded_bytes": downloaded}
    print(
        json.dumps(
            {
                "status": "complete",
                "config_sha256": _sha256_file(CONFIG_PATH),
                "shards": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    raise SystemExit(
        "Use `modal run scripts/modal_phase1_ugi_decoration_coupling_production_fresh_census_v1.py`."
    )
