#!/usr/bin/env python3
"""Generate the fresh matched stochastic Ugi production candidate pool."""

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
OUTPUT_MOUNT = Path("/forge-candidates-v2")
REMOTE_CACHE_DIR = REMOTE_REPO_ROOT / "results/phase1/ugi_balanced_training_cache_v2"
CONFIG_RELATIVE = "configs/model/phase1_ugi_constrained_stochastic_production_candidates_v2.json"
CONFIG_PATH = REPO_ROOT / CONFIG_RELATIVE
CONFIG = json.loads(CONFIG_PATH.read_text())
DESIGN = CONFIG["design"]
INPUTS = CONFIG["inputs"]
RUN_NAME = "ugi_decoration_coupling_production_refit_v1_d4a23a76b796"
REMOTE_CHECKPOINT = TRAINING_MOUNT / RUN_NAME / "checkpoint_best.pt"
LOCAL_OUTPUT_ROOT = (
    REPO_ROOT / "results/phase1/ugi_constrained_stochastic_production_candidates_v2/raw"
)

app = modal.App("forge-phase1-ugi-constrained-stochastic-production-candidates-v2")
training_volume = modal.Volume.from_name("forge-phase1-training")
input_volume = modal.Volume.from_name("forge-phase1-production-inputs")
output_volume = modal.Volume.from_name(
    "forge-phase1-constrained-stochastic-production-candidates-v2",
    create_if_missing=True,
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
    str(INPUTS["schedule"]["path"]),
    str(INPUTS["qualified_reactions"]["path"]),
    str(INPUTS["closure_checkpoint"]["path"]),
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
        != "phase1_ugi_constrained_stochastic_production_candidates_config.v2"
        or config.get("status") != "frozen_before_fresh_matched_terminal_generation"
    ):
        raise RuntimeError("stochastic production-candidate config is not frozen")
    design = config["design"]
    if (
        tuple(design["arms"]) != ("broad_prior", "support_enriched")
        or int(design["draws_per_arm"]) != 16384
        or int(design["shard_draws"]) != 1024
        or int(design["shards_per_arm"]) != 16
        or design["terminal_decoder_mode"] != "stochastic"
        or float(design["terminal_temperature"]) != 1.0
        or bool(design["retry_or_repair"])
    ):
        raise RuntimeError("stochastic production-candidate design changed")


def _seeds(config: dict[str, Any], arm: str, shard: int) -> tuple[int, int]:
    arm_index = tuple(config["design"]["arms"]).index(arm)
    stream_index = arm_index * int(config["design"]["shards_per_arm"]) + shard
    return (
        int(config["design"]["flow_seed_base"]) + stream_index,
        int(config["design"]["terminal_seed_base"]) + stream_index,
    )


@app.function(
    image=image,
    cpu=8.0,
    memory=24576,
    timeout=3600,
    max_containers=20,
    single_use_containers=True,
    volumes={
        str(TRAINING_MOUNT): training_volume,
        str(REMOTE_CACHE_DIR): input_volume,
        str(OUTPUT_MOUNT): output_volume,
    },
)
def sample_shard(arm: str, shard: int) -> dict[str, Any]:
    """Sample one immutable 1,024-coordinate arm shard."""

    import platform
    import sys

    sys.path.insert(0, str(REMOTE_SOURCE_ROOT))
    import numpy
    import torch
    from rdkit import rdBase

    from forge.product.ugi_constrained_stochastic_production_candidates import (
        load_json,
        program_shard,
    )
    from forge.product.ugi_joint_end_to_end_sampling import (
        joint_sampling_result_matches_request,
        sample_ugi_joint_end_to_end,
    )

    config = load_json(REMOTE_REPO_ROOT / CONFIG_RELATIVE, label="production config")
    _validate_config(config)
    if arm not in tuple(config["design"]["arms"]) or shard not in range(
        int(config["design"]["shards_per_arm"])
    ):
        raise ValueError(f"unsupported production shard: {arm}/{shard}")
    if _sha256_file(REMOTE_CHECKPOINT) != str(config["inputs"]["joint_checkpoint"]["sha256"]):
        raise RuntimeError("mounted stochastic production checkpoint changed")
    schedule_path = REMOTE_REPO_ROOT / str(config["inputs"]["schedule"]["path"])
    if _sha256_file(schedule_path) != str(config["inputs"]["schedule"]["sha256"]):
        raise RuntimeError("frozen production schedule changed")
    schedule = load_json(schedule_path, label="production schedule")
    materialized = program_shard(
        schedule,
        arm=arm,
        shard_index=shard,
        shard_draws=int(config["design"]["shard_draws"]),
    )
    program_path = Path(f"/tmp/forge-programs/{arm}_shard_{shard:02d}.json")
    program_path.parent.mkdir(parents=True, exist_ok=True)
    program_path.write_text(json.dumps(materialized, sort_keys=True) + "\n")

    local_checkpoint = Path("/tmp/forge-checkpoints/checkpoint_best.pt")
    local_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REMOTE_CHECKPOINT, local_checkpoint)
    output_dir = OUTPUT_MOUNT / arm / f"shard_{shard:02d}"
    result_path = output_dir / "result.json"
    render_path = output_dir / "samples.png"
    flow_seed, terminal_seed = _seeds(config, arm, shard)
    if result_path.is_file() and render_path.is_file():
        existing = json.loads(result_path.read_text())
        if joint_sampling_result_matches_request(
            existing,
            seed=flow_seed,
            program_offset=0,
            program_limit=int(config["design"]["shard_draws"]),
            terminal_decoder_mode="stochastic",
            terminal_decoder_seed=terminal_seed,
            terminal_temperature=1.0,
            checkpoint_filename="checkpoint_best.pt",
        ):
            return {
                "arm": arm,
                "shard": shard,
                "result_sha256": _sha256_file(result_path),
                "samples_png_sha256": _sha256_file(render_path),
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
        program_limit=int(config["design"]["shard_draws"]),
        reference_comparison_mode="deferred",
    )
    output_volume.commit()
    return {
        "arm": arm,
        "shard": shard,
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


@app.function(image=image, cpu=0.25, memory=512, timeout=7200)
def orchestrate(tasks: tuple[tuple[str, int], ...]) -> list[dict[str, Any]]:
    """Keep every independent production shard alive until completion."""

    calls = [sample_shard.spawn(arm, shard) for arm, shard in tasks]
    return [call.get() for call in calls]


@app.local_entrypoint()
def main() -> None:
    """Launch every shard and download hash-verified raw results."""

    _validate_config(CONFIG)
    if _sha256_file(Path(__file__)) != str(CONFIG["implementation"]["runner"]["sha256"]):
        raise RuntimeError("production-candidate runner changed after freeze")
    if _sha256_file(REPO_ROOT / INPUTS["joint_checkpoint"]["path"]) != str(
        INPUTS["joint_checkpoint"]["sha256"]
    ):
        raise RuntimeError("local production checkpoint changed before launch")
    tasks = tuple(
        (arm, shard) for arm in DESIGN["arms"] for shard in range(int(DESIGN["shards_per_arm"]))
    )
    remote_results = orchestrate.remote(tasks)
    receipts = {}
    for remote in remote_results:
        arm = str(remote["arm"])
        shard = int(remote["shard"])
        local_dir = LOCAL_OUTPUT_ROOT / arm / f"shard_{shard:02d}"
        downloaded = {}
        for filename, digest_key in (
            ("result.json", "result_sha256"),
            ("samples.png", "samples_png_sha256"),
        ):
            local_path = local_dir / filename
            downloaded[filename] = _atomic_volume_download(
                f"{arm}/shard_{shard:02d}/{filename}", local_path
            )
            if _sha256_file(local_path) != str(remote[digest_key]):
                raise RuntimeError(f"downloaded artifact changed: {arm}/{shard}/{filename}")
        receipts[f"{arm}/shard_{shard:02d}"] = {
            **remote,
            "downloaded_bytes": downloaded,
        }
    print(
        json.dumps(
            {
                "status": "complete",
                "config_sha256": _sha256_file(CONFIG_PATH),
                "shards": receipts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    raise SystemExit(
        "Use `modal run scripts/modal_phase1_ugi_constrained_stochastic_production_candidates_v2.py`."
    )
