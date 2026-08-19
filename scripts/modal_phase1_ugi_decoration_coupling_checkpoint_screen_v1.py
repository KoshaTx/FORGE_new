#!/usr/bin/env python3
"""Screen both decoration-coupling checkpoint series on untouched programs."""

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
REMOTE_SOURCE_ROOT = Path("/root")
TRAINING_MOUNT = Path("/forge-training")
OUTPUT_MOUNT = Path("/forge-screen")
LOCAL_OUTPUT_ROOT = REPO_ROOT / "results/phase1/ugi_decoration_coupling_checkpoint_screen_v1"
REMOTE_CACHE_DIR = REMOTE_REPO_ROOT / "results/phase1/ugi_balanced_training_cache_v2"
PROGRAM_OFFSET = 128
PROGRAM_LIMIT = 512
FLOW_SEED = 20260836
TERMINAL_SEED = 20260837
STEPS = ("best", 500, 1000, 1500, 2000, 3000, 4000)
RUN_NAMES = {
    "legacy": "ugi_decoration_coupling_v1_legacy_b826c843130d",
    "challenger": "ugi_decoration_coupling_v1_challenger_ba78c781ba17",
}
CHECKPOINT_HASHES = {
    "legacy": {
        "best": "cdca5339bb29aafb4ed835a3c9c7f99ad27a4066372e9a92fccd483a44260ce6",
        500: "cf5d987cb6718453d6317946a8c6c5697d5cc8063d9bb7076e59709b46810d80",
        1000: "21eb80365b7b1dbd4bf86cdb3f2a67d89e5a7526be0ba1bf3d5079825f3b24a1",
        1500: "d6fbc989511d7ffd35fb39df5df6ce950096da21214a0cf8b80d011fe28910d5",
        2000: "32a6d3bd4fceae4d9d75bc154f0e71cc0f880b02892b00deefc4041a2167b9a0",
        3000: "89f21787ea5834ac0a78b90aca4ff73a0fbb4951b05c82ae45432833d003940f",
        4000: "85a7d797d6a4fcdab332d55ed28fe4972736d78c822b8a0badb46664736e474b",
    },
    "challenger": {
        "best": "8998806ceeab96b3e6fcf5db05428c0c11c04a4b2198781c6d415bce7cf5b5b9",
        500: "6bc3f2aaab333c4505e70b4e7d2f0e7d0adfeb73044f10284439fa6c95d5b3f9",
        1000: "85859364a1130f3a439a1ce8729b932db0093d474941a8ca1f91b58c9d784bdb",
        1500: "420360d4f95d54abd4f7d0b5cb9b1caddabe6adfa6db77f3c7fa7fc5a96b2778",
        2000: "b34b14b4e2d0d0556b0448a692e7af8f43ac8e1f2f7fc87d4fa1a5b9c7297778",
        3000: "4a9dc01ec1bcc83cd18c257185b18fe2b66b2b19d93a12ed44f696b490bbad50",
        4000: "bb79c8d324dbff0d5651bff29acdee91f4888ff28335422384e325ace4e2a7d2",
    },
}
PINNED_FILES = (
    "data/vendor/qualified_reactions_v1.json",
    "results/phase1/ugi_closure_expanded_full/checkpoint_best.pt",
    "results/phase1/ugi_production_refit_fresh_census_v1/programs_shard_00.json",
)

app = modal.App("forge-phase1-ugi-decoration-coupling-checkpoint-screen-v1")
training_volume = modal.Volume.from_name("forge-phase1-training")
input_volume = modal.Volume.from_name("forge-phase1-production-inputs")
output_volume = modal.Volume.from_name(
    "forge-phase1-decoration-coupling-checkpoint-screen-v1", create_if_missing=True
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


def _checkpoint_filename(step: str | int) -> str:
    return "checkpoint_best.pt" if step == "best" else f"checkpoint_step_{step}.pt"


def _compatible_existing_result(
    result: dict[str, Any],
    *,
    checkpoint_filename: str,
) -> bool:
    from forge.product.ugi_joint_end_to_end_sampling import (
        joint_sampling_result_matches_request,
    )

    return joint_sampling_result_matches_request(
        result,
        seed=FLOW_SEED,
        program_offset=PROGRAM_OFFSET,
        program_limit=PROGRAM_LIMIT,
        terminal_decoder_mode="stochastic",
        terminal_decoder_seed=TERMINAL_SEED,
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
    max_containers=14,
    single_use_containers=True,
    volumes={
        str(TRAINING_MOUNT): training_volume,
        str(REMOTE_CACHE_DIR): input_volume,
        str(OUTPUT_MOUNT): output_volume,
    },
)
def sample_arm(specification: tuple[str, str | int]) -> dict[str, Any]:
    """Sample one checkpoint on the same untouched contiguous program block."""

    import platform
    import sys

    sys.path.insert(0, str(REMOTE_SOURCE_ROOT))
    import numpy
    import torch
    from rdkit import rdBase

    from forge.product.ugi_joint_end_to_end_sampling import sample_ugi_joint_end_to_end

    lane, step = specification
    checkpoint_filename = _checkpoint_filename(step)
    mounted_checkpoint = TRAINING_MOUNT / RUN_NAMES[lane] / checkpoint_filename
    expected_hash = CHECKPOINT_HASHES[lane][step]
    if _sha256_file(mounted_checkpoint) != expected_hash:
        raise RuntimeError(f"mounted checkpoint changed for {lane} {step}")
    local_checkpoint = Path("/tmp/forge-checkpoints") / lane / checkpoint_filename
    local_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(mounted_checkpoint, local_checkpoint)

    output_name = f"{lane}_{step}"
    output_dir = OUTPUT_MOUNT / output_name
    result_path = output_dir / "result.json"
    render_path = output_dir / "samples.png"
    if result_path.is_file() and render_path.is_file():
        existing = json.loads(result_path.read_text())
        if _compatible_existing_result(
            existing,
            checkpoint_filename=checkpoint_filename,
        ):
            return {
                "lane": lane,
                "step": step,
                "checkpoint_sha256": expected_hash,
                "result_sha256": _sha256_file(result_path),
                "samples_png_sha256": _sha256_file(render_path),
                "valid_fraction": existing["statistics"]["valid_fraction"],
                "reference_comparison_status": existing.get(
                    "reference_comparison_status",
                    "complete" if existing.get("reference_comparison") is not None else "unknown",
                ),
                "reused_compatible_result": True,
                "runtime": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "numpy": numpy.__version__,
                    "rdkit": rdBase.rdkitVersion,
                },
            }
    result = sample_ugi_joint_end_to_end(
        REMOTE_REPO_ROOT,
        output_dir,
        joint_checkpoint_path=local_checkpoint,
        closure_checkpoint_path=(
            REMOTE_REPO_ROOT / "results/phase1/ugi_closure_expanded_full/checkpoint_best.pt"
        ),
        matched_staged_result_path=(
            REMOTE_REPO_ROOT
            / "results/phase1/ugi_production_refit_fresh_census_v1/programs_shard_00.json"
        ),
        sample_steps=8,
        batch_size=16,
        seed=FLOW_SEED,
        overwrite=True,
        maximum_adjacent_branch_runs=(2, 1, 1),
        qualified_reactions_path=(REMOTE_REPO_ROOT / "data/vendor/qualified_reactions_v1.json"),
        evaluate_exact_l1_terminal_admission=True,
        terminal_decoder_mode="stochastic",
        terminal_decoder_seed=TERMINAL_SEED,
        terminal_temperature=1.0,
        program_offset=PROGRAM_OFFSET,
        program_limit=PROGRAM_LIMIT,
        reference_comparison_mode="deferred",
    )
    output_volume.commit()
    return {
        "lane": lane,
        "step": step,
        "checkpoint_sha256": expected_hash,
        "result_sha256": _sha256_file(output_dir / "result.json"),
        "samples_png_sha256": _sha256_file(output_dir / "samples.png"),
        "valid_fraction": result["statistics"]["valid_fraction"],
        "reference_comparison_status": result["reference_comparison_status"],
        "reused_compatible_result": False,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
    }


@app.function(image=image, cpu=0.25, memory=512, timeout=4000)
def orchestrate_screen(
    specifications: tuple[tuple[str, str | int], ...],
) -> list[dict[str, Any]]:
    """Keep all independently restartable checkpoint calls alive remotely."""

    calls = [sample_arm.spawn(specification) for specification in specifications]
    return [call.get() for call in calls]


@app.local_entrypoint()
def main() -> None:
    """Run all checkpoints concurrently and download verified artifacts."""

    specifications = tuple((lane, step) for lane in RUN_NAMES for step in STEPS)
    for lane, step in specifications:
        local_checkpoint = (
            REPO_ROOT
            / "results/phase1/ugi_decoration_coupling_v1"
            / lane
            / _checkpoint_filename(step)
        )
        if _sha256_file(local_checkpoint) != CHECKPOINT_HASHES[lane][step]:
            raise RuntimeError(f"local checkpoint changed for {lane} {step}")
    remote_results = orchestrate_screen.remote(specifications)
    summary = {}
    for remote in remote_results:
        lane = str(remote["lane"])
        step = remote["step"]
        output_name = f"{lane}_{step}"
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
                "program_offset": PROGRAM_OFFSET,
                "program_limit": PROGRAM_LIMIT,
                "flow_seed": FLOW_SEED,
                "terminal_seed": TERMINAL_SEED,
                "arms": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    raise SystemExit(
        "Use `modal run scripts/modal_phase1_ugi_decoration_coupling_checkpoint_screen_v1.py`."
    )
