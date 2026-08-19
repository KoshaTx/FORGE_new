#!/usr/bin/env python3
"""Run the frozen multi-seed confirmation for Ugi decoration checkpoints."""

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
POLICY_PATH = (
    REPO_ROOT / "configs/model/phase1_ugi_decoration_checkpoint_confirmation_policy_v1.json"
)
POLICY = json.loads(POLICY_PATH.read_text())
REMOTE_REPO_ROOT = Path("/root/forge_repo")
REMOTE_SOURCE_ROOT = Path("/root")
TRAINING_MOUNT = Path("/forge-training")
OUTPUT_MOUNT = Path("/forge-confirmation")
LOCAL_OUTPUT_ROOT = REPO_ROOT / "results/phase1/ugi_decoration_coupling_checkpoint_confirmation_v1"
REMOTE_CACHE_DIR = REMOTE_REPO_ROOT / "results/phase1/ugi_balanced_training_cache_v2"
RUN_NAME = "ugi_decoration_coupling_v1_challenger_ba78c781ba17"
FINALISTS = {int(step): specification for step, specification in POLICY["finalists"].items()}
REPLICATES = tuple(POLICY["confirmation_design"]["paired_replicates"])
PROGRAM_SPECIFICATION = POLICY["confirmation_design"]["program_schedule"]
PROGRAM_RELATIVE_PATH = str(PROGRAM_SPECIFICATION["path"])
PROGRAM_LIMIT = int(PROGRAM_SPECIFICATION["program_limit"])
PROGRAM_OFFSET = int(PROGRAM_SPECIFICATION["program_offset"])

app = modal.App("forge-phase1-ugi-decoration-checkpoint-confirmation-v1")
training_volume = modal.Volume.from_name("forge-phase1-training")
input_volume = modal.Volume.from_name("forge-phase1-production-inputs")
output_volume = modal.Volume.from_name(
    "forge-phase1-decoration-checkpoint-confirmation-v1", create_if_missing=True
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
    "data/vendor/qualified_reactions_v1.json",
    "results/phase1/ugi_closure_expanded_full/checkpoint_best.pt",
    PROGRAM_RELATIVE_PATH,
    str(POLICY_PATH.relative_to(REPO_ROOT)),
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


def _checkpoint_filename(step: int) -> str:
    return f"checkpoint_step_{step}.pt"


def _compatible_existing_result(
    result: dict[str, Any],
    *,
    checkpoint_filename: str,
    flow_seed: int,
    terminal_seed: int,
) -> bool:
    from forge.product.ugi_joint_end_to_end_sampling import (
        joint_sampling_result_matches_request,
    )

    return str(result.get("matched_staged_result", "")).endswith(
        PROGRAM_RELATIVE_PATH
    ) and joint_sampling_result_matches_request(
        result,
        seed=flow_seed,
        program_offset=PROGRAM_OFFSET,
        program_limit=PROGRAM_LIMIT,
        terminal_decoder_mode="stochastic",
        terminal_decoder_seed=terminal_seed,
        terminal_temperature=1.0,
        checkpoint_filename=checkpoint_filename,
    )


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
    cpu=8.0,
    memory=24576,
    timeout=3600,
    max_containers=9,
    single_use_containers=True,
    volumes={
        str(TRAINING_MOUNT): training_volume,
        str(REMOTE_CACHE_DIR): input_volume,
        str(OUTPUT_MOUNT): output_volume,
    },
)
def sample_arm(specification: tuple[int, int, int, int]) -> dict[str, Any]:
    """Sample one finalist and replicate on the frozen confirmation programs."""

    import platform
    import sys

    sys.path.insert(0, str(REMOTE_SOURCE_ROOT))
    import numpy
    import torch
    from rdkit import rdBase

    from forge.product.ugi_joint_end_to_end_sampling import sample_ugi_joint_end_to_end

    step, replicate, flow_seed, terminal_seed = specification
    checkpoint_filename = _checkpoint_filename(step)
    mounted_checkpoint = TRAINING_MOUNT / RUN_NAME / checkpoint_filename
    expected_hash = str(FINALISTS[step]["sha256"])
    if _sha256_file(mounted_checkpoint) != expected_hash:
        raise RuntimeError(f"mounted checkpoint changed for step {step}")
    local_checkpoint = Path("/tmp/forge-checkpoints") / checkpoint_filename
    local_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(mounted_checkpoint, local_checkpoint)

    output_name = f"step_{step}/replicate_{replicate}"
    output_dir = OUTPUT_MOUNT / output_name
    result_path = output_dir / "result.json"
    render_path = output_dir / "samples.png"
    if result_path.is_file() and render_path.is_file():
        existing = json.loads(result_path.read_text())
        if _compatible_existing_result(
            existing,
            checkpoint_filename=checkpoint_filename,
            flow_seed=flow_seed,
            terminal_seed=terminal_seed,
        ):
            return {
                "step": step,
                "replicate": replicate,
                "checkpoint_sha256": expected_hash,
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
            REMOTE_REPO_ROOT / "results/phase1/ugi_closure_expanded_full/checkpoint_best.pt"
        ),
        matched_staged_result_path=REMOTE_REPO_ROOT / PROGRAM_RELATIVE_PATH,
        sample_steps=int(POLICY["confirmation_design"]["sample_steps"]),
        batch_size=16,
        seed=flow_seed,
        overwrite=True,
        maximum_adjacent_branch_runs=tuple(
            int(value) for value in POLICY["confirmation_design"]["maximum_adjacent_branch_runs"]
        ),
        qualified_reactions_path=(REMOTE_REPO_ROOT / "data/vendor/qualified_reactions_v1.json"),
        evaluate_exact_l1_terminal_admission=True,
        terminal_decoder_mode="stochastic",
        terminal_decoder_seed=terminal_seed,
        terminal_temperature=float(POLICY["confirmation_design"]["terminal_temperature"]),
        program_offset=PROGRAM_OFFSET,
        program_limit=PROGRAM_LIMIT,
        reference_comparison_mode="deferred",
    )
    output_volume.commit()
    return {
        "step": step,
        "replicate": replicate,
        "checkpoint_sha256": expected_hash,
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
def orchestrate_confirmation(
    specifications: tuple[tuple[int, int, int, int], ...],
) -> list[dict[str, Any]]:
    """Keep all independently restartable confirmation calls alive remotely."""

    calls = [sample_arm.spawn(specification) for specification in specifications]
    return [call.get() for call in calls]


@app.local_entrypoint()
def main() -> None:
    """Run all finalist replicates concurrently and download verified artifacts."""

    if _sha256_file(Path(__file__)) != POLICY["implementation"]["modal_runner"]["sha256"]:
        raise RuntimeError("confirmation runner changed after policy freeze")
    if _sha256_file(REPO_ROOT / PROGRAM_RELATIVE_PATH) != PROGRAM_SPECIFICATION["sha256"]:
        raise RuntimeError("confirmation program schedule changed")
    for step, checkpoint in FINALISTS.items():
        if _sha256_file(REPO_ROOT / checkpoint["path"]) != checkpoint["sha256"]:
            raise RuntimeError(f"local checkpoint changed for step {step}")
    specifications = tuple(
        (
            step,
            int(replicate["replicate"]),
            int(replicate["flow_seed"]),
            int(replicate["terminal_seed"]),
        )
        for step in sorted(FINALISTS)
        for replicate in REPLICATES
    )
    remote_results = orchestrate_confirmation.remote(specifications)
    summary = {}
    for remote in remote_results:
        step = int(remote["step"])
        replicate = int(remote["replicate"])
        output_name = f"step_{step}/replicate_{replicate}"
        local_dir = LOCAL_OUTPUT_ROOT / output_name
        downloaded = {}
        for filename, digest_key in (
            ("result.json", "result_sha256"),
            ("samples.png", "samples_png_sha256"),
        ):
            local_path = local_dir / filename
            downloaded[filename] = _atomic_volume_download(f"{output_name}/{filename}", local_path)
            if _sha256_file(local_path) != remote[digest_key]:
                raise RuntimeError(f"downloaded {output_name}/{filename} changed")
        summary[output_name] = {**remote, "downloaded_bytes": downloaded}
    print(
        json.dumps(
            {
                "status": "complete",
                "policy_path": str(POLICY_PATH),
                "policy_sha256": _sha256_file(POLICY_PATH),
                "program_schedule": PROGRAM_SPECIFICATION,
                "arms": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    raise SystemExit(
        "Use `modal run scripts/modal_phase1_ugi_decoration_checkpoint_confirmation_v1.py`."
    )
