#!/usr/bin/env python3
"""Run the frozen matched Ugi production-candidate shards on Modal CPUs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO_ROOT = Path("/root/forge_repo")
VOLUME_MOUNT = Path("/forge-output")
VOLUME_NAME = "forge-phase1-production-candidates"
INPUT_VOLUME_NAME = "forge-phase1-production-inputs"
TRAINING_CACHE_RELATIVE = "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt"
TRAINING_CACHE_MOUNT = REMOTE_REPO_ROOT / "results/phase1/ugi_balanced_training_cache_v2"
REMOTE_RUN_ROOT = "ugi_production_candidate_generation_v1"
LOCAL_OUTPUT = REPO_ROOT / "results/phase1/ugi_production_candidate_generation_v1"
CONFIG_RELATIVE = "configs/model/phase1_ugi_production_candidate_generation_v1.json"
CONFIG_PATH = REPO_ROOT / CONFIG_RELATIVE
PINNED_FILES = (
    "configs/assembly/ugi_variant.yaml",
    "configs/model/phase1_ugi_dynamic_frozen_prior_terminal_census_v1.json",
    "configs/model/phase1_ugi_product_l1_v2_fresh_pool_v1.json",
    "configs/model/phase1_ugi_production_candidate_generation_v1.json",
    "configs/model/phase1_ugi_selected_v2_pool_singleton_equivalence_v2.json",
    "data/vendor/qualified_reactions_v1.json",
    "results/phase1/product_v3_atom_vocabulary.json",
    "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt",
    "results/phase1/ugi_closure_expanded_full/checkpoint_best.pt",
    "results/phase1/ugi_grouped_smc_schedule_qualification_v1/result.json",
    "results/phase1/ugi_joint_sparse_balanced_v2_full/checkpoint_step_2000.pt",
    "results/phase1/ugi_product_l1_production_generator_v2.json",
    "results/phase1/ugi_product_l1_v2_fresh_pool_v1/programs.json",
    "results/phase1/ugi_product_l1_v2_fresh_pool_v1/sample/result.json",
    "results/phase1/ugi_production_candidate_schedule_v1/result.json",
    "results/phase1/ugi_production_candidate_schedule_v1/schedule.json",
    "results/phase1/ugi_restartable_sampler_equivalence_v2/result.json",
    "results/phase1/ugi_selected_v2_full_equivalence_v1/result.json",
    "results/phase1/ugi_selected_v2_pool_singleton_equivalence_v2/comparison_rows.json",
    "results/phase1/ugi_selected_v2_pool_singleton_equivalence_v2/result.json",
    "scripts/phase1_audit_ugi_selected_v2_pool_singleton_equivalence_v2.py",
    "scripts/phase1_collect_ugi_dynamic_frozen_prior_terminal_census.py",
    "scripts/phase1_generate_ugi_production_candidates.py",
    "src/forge/product/ugi_dynamic_frozen_prior_terminal_census.py",
    "src/forge/product/ugi_production_candidate_generation.py",
    "src/forge/product/ugi_selected_guidance_adapter_v2.py",
    "src/forge/product/ugi_selected_guidance_adapter_v3.py",
    "src/forge/product/ugi_selected_restartable_generator_v2.py",
    "src/forge/product/ugi_selected_v2_pool_singleton_equivalence_v2.py",
    "tests/test_ugi_dynamic_frozen_prior_terminal_census.py",
    "tests/test_ugi_production_candidate_generation.py",
    "tests/test_ugi_selected_v2_pool_singleton_equivalence_v2.py",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected one JSON object: {path}")
    return value


def _pinned_files() -> tuple[str, ...]:
    """Collect the frozen first- and second-order files validated by the runner."""

    relative = {CONFIG_RELATIVE}
    production = _json(CONFIG_PATH)
    for record in production["inputs"].values():
        relative.add(str(record["path"]))

    census_path = REPO_ROOT / production["inputs"]["base_census_config"]["path"]
    census = _json(census_path)
    for record in census["inputs"].values():
        relative.add(str(record["path"]))

    equivalence_path = (
        REPO_ROOT / "results/phase1/ugi_selected_v2_pool_singleton_equivalence_v2/result.json"
    )
    equivalence = _json(equivalence_path)
    relative.add(str(equivalence_path.relative_to(REPO_ROOT)))
    relative.add(str(equivalence["config"]["path"]))
    relative.add(str(equivalence["comparison_rows_artifact"]["path"]))
    for record in equivalence["inputs"].values():
        relative.add(str(record["path"]))

    relative.update(
        {
            "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt",
            "results/phase1/product_v3_atom_vocabulary.json",
            "data/vendor/qualified_reactions_v1.json",
            "configs/assembly/ugi_variant.yaml",
        }
    )
    missing = [path for path in sorted(relative) if not (REPO_ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"missing Modal production inputs: {missing}")
    discovered = tuple(sorted(relative))
    if discovered != PINNED_FILES:
        raise RuntimeError("explicit Modal input manifest differs from frozen pin discovery")
    return discovered


app = modal.App("forge-phase1-ugi-production-candidates")
output_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
input_volume = modal.Volume.from_name(INPUT_VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("libxrender1", "libxext6")
    .pip_install(
        "torch==2.13.0",
        "numpy==2.5.1",
        "rdkit==2026.3.4",
    )
    .add_local_dir(
        REPO_ROOT / "src",
        remote_path=str(REMOTE_REPO_ROOT / "src"),
        copy=True,
    )
)
for relative in PINNED_FILES:
    if relative.startswith("src/") or relative == TRAINING_CACHE_RELATIVE:
        continue
    image = image.add_local_file(
        REPO_ROOT / relative,
        remote_path=str(REMOTE_REPO_ROOT / relative),
        copy=True,
    )
image = image.pip_install("PyYAML==6.0.3")


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
    cpu=1.0,
    memory=4096,
    timeout=1_800,
    max_containers=20,
    single_use_containers=True,
    volumes={
        str(VOLUME_MOUNT): output_volume,
        str(TRAINING_CACHE_MOUNT): input_volume,
    },
)
def run_shard(shard_index: int) -> dict[str, Any]:
    """Execute one independent frozen shard and commit its two artifacts."""

    import platform
    import sys

    sys.path.insert(0, str(REMOTE_REPO_ROOT / "src"))

    import numpy
    import torch
    from rdkit import rdBase

    raw_torch_version = torch.__version__
    if raw_torch_version.split("+", 1)[0] != "2.13.0":
        raise RuntimeError(f"unexpected Modal torch distribution: {raw_torch_version}")
    # The frozen implementation receipt records the PEP 440 public release.
    # Linux wheels append a local CUDA build tag even for this CPU-only run.
    torch.__version__ = "2.13.0"

    from forge.product.ugi_production_candidate_generation import (
        run_one_production_candidate_shard,
    )

    expected_runtime = {
        "python": "3.14.2",
        "numpy": "2.5.1",
        "torch": "2.13.0",
        "rdkit": "2026.03.4",
    }
    observed_runtime = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "torch_distribution": raw_torch_version,
        "rdkit": rdBase.rdkitVersion,
    }
    if {key: observed_runtime[key] for key in expected_runtime} != expected_runtime:
        raise RuntimeError(
            f"Modal runtime differs from the frozen generator runtime: {observed_runtime}"
        )

    result = run_one_production_candidate_shard(
        REMOTE_REPO_ROOT,
        REMOTE_REPO_ROOT / CONFIG_RELATIVE,
        VOLUME_MOUNT / REMOTE_RUN_ROOT,
        shard_index=int(shard_index),
    )
    output_volume.commit()
    return {
        "shard_index": int(shard_index),
        "result_sha256": result["result_sha256"],
        "ledger_sha256": result["terminal_ledger"]["sha256"],
        "resumed": bool(result["resumed"]),
        "runtime": observed_runtime,
    }


@app.local_entrypoint()
def main(first_shard: int = 0, last_shard: int = 127) -> None:
    """Run missing shards in parallel and merge them into the local restartable tree."""

    from forge.product.ugi_production_candidate_generation import (
        load_production_candidate_contract,
    )

    _pinned_files()
    if not 0 <= first_shard <= last_shard < 128:
        raise ValueError("expected 0 <= first_shard <= last_shard < 128")
    contract = load_production_candidate_contract(REPO_ROOT, CONFIG_PATH)
    if contract.design.shard_count != 128:
        raise RuntimeError("frozen shard count changed")

    pending = [
        index
        for index in range(first_shard, last_shard + 1)
        if not (LOCAL_OUTPUT / "shards" / f"shard_{index:04d}" / "receipt.json").is_file()
    ]
    print(
        json.dumps(
            {
                "status": "launching_missing_modal_shards",
                "completed_local": 128 - len(pending),
                "pending": len(pending),
                "max_containers": 20,
            },
            sort_keys=True,
        )
    )
    results = list(run_shard.map(pending, order_outputs=False))

    downloaded = []
    for result in sorted(results, key=lambda value: value["shard_index"]):
        index = int(result["shard_index"])
        local_shard = LOCAL_OUTPUT / "shards" / f"shard_{index:04d}"
        remote_shard = f"{REMOTE_RUN_ROOT}/shards/shard_{index:04d}"
        receipt_path = local_shard / "receipt.json"
        ledger_path = local_shard / "terminals.jsonl.gz"
        _atomic_volume_download(f"{remote_shard}/receipt.json", receipt_path)
        _atomic_volume_download(f"{remote_shard}/terminals.jsonl.gz", ledger_path)
        receipt = _json(receipt_path)
        if (
            receipt.get("result_sha256") != result["result_sha256"]
            or _sha256_file(ledger_path) != result["ledger_sha256"]
        ):
            raise RuntimeError(f"downloaded shard {index} failed content verification")
        downloaded.append(index)

    print(
        json.dumps(
            {
                "status": "modal_shards_downloaded",
                "executed": len(results),
                "downloaded": len(downloaded),
                "first": min(downloaded) if downloaded else None,
                "last": max(downloaded) if downloaded else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
