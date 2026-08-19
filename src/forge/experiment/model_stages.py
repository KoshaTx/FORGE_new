"""Stage adapters for the authorized Phase 1 model pipeline.

Adapters live at the orchestration edge: they translate a verified ``RunContext`` into a domain
API call, but contain no chemistry or scientific policy of their own.
"""

from __future__ import annotations

import gc
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_file
from forge.core.io import write_json
from forge.experiment.errors import StageError
from forge.experiment.registry import stage
from forge.experiment.stage import ProducedArtifact, RunContext, StageResult


def _require_config_inputs(
    context: RunContext,
    config: dict[str, Any],
    *,
    labels: set[str] | None = None,
) -> None:
    configured = config.get("inputs")
    if not isinstance(configured, dict):
        raise StageError(f"stage {context.stage.stage_id} config has no input mapping")
    expected = set(context.inputs) if labels is None else labels
    if set(configured) != expected:
        raise StageError(
            f"experiment inputs and stage config differ for {context.stage.stage_id}: "
            f"experiment={sorted(expected)}, config={sorted(configured)}"
        )
    for label in sorted(expected):
        pin = context.stage.inputs[label]
        record = configured[label]
        if not isinstance(record, dict) or record != pin.to_mapping():
            raise StageError(f"experiment pin differs from stage config for {label!r}")


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _deterministic_tar(paths: list[Path], target: Path, *, base: Path) -> None:
    """Archive checkpoint shards without filesystem timestamps or ownership."""

    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(paths, key=lambda value: value.relative_to(base).as_posix()):
            name = path.relative_to(base).as_posix()
            info = tarfile.TarInfo(name=name)
            info.size = path.stat().st_size
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with path.open("rb") as handle:
                archive.addfile(info, handle)


def _resume_training_work(context: RunContext, work: Path) -> bool:
    """Resume an authenticated checkpoint, or restart unauthenticated scratch safely."""

    if not context.resume:
        return False
    if (work / "checkpoint_latest.pt").is_file():
        return True
    # A process can die before its first atomic checkpoint.  Nothing in that directory is a
    # resumable state, so rebuild it deterministically under the same stage fingerprint.
    if work.exists():
        shutil.rmtree(work)
    return False


def _publish_training_outputs(
    context: RunContext,
    work: Path,
    *,
    result_schema: str,
    checkpoint_schema: str,
    progress_schema: str,
) -> tuple[ProducedArtifact, ...]:
    result = json.loads((work / "result.json").read_text())
    for key, filename in (
        ("checkpoint", "checkpoint_best.pt"),
        ("checkpoint_latest", "checkpoint_latest.pt"),
    ):
        record = result.get(key)
        if not isinstance(record, dict) or record.get("sha256") != str(
            sha256_file(work / filename)
        ):
            raise StageError(f"training result does not authenticate {filename}")
        record["path"] = filename
    snapshots = sorted(work.glob("checkpoint_step_*.pt"))
    snapshot_records = []
    for path in snapshots:
        snapshot_records.append(
            {
                "member": path.name,
                "sha256": str(sha256_file(path)),
                "step": int(path.stem.rsplit("_", 1)[1]),
            }
        )
    result["checkpoint_snapshots"] = snapshot_records
    result["execution"] = {
        "backend": context.backend,
        "profile": context.profile,
        "resumed_partial_stage": context.resume,
    }
    write_json(context.output_path("result.json"), result)
    _copy(work / "checkpoint_best.pt", context.output_path("checkpoint_best.pt"))
    _copy(work / "checkpoint_latest.pt", context.output_path("checkpoint_latest.pt"))
    _copy(work / "progress.json", context.output_path("progress.json"))
    _deterministic_tar(snapshots, context.output_path("checkpoint_snapshots.tar"), base=work)
    return (
        ProducedArtifact("result", "result.json", result_schema),
        ProducedArtifact("checkpoint", "checkpoint_best.pt", checkpoint_schema),
        ProducedArtifact("checkpoint_latest", "checkpoint_latest.pt", checkpoint_schema),
        ProducedArtifact("progress", "progress.json", progress_schema),
        ProducedArtifact(
            "checkpoint_snapshots",
            "checkpoint_snapshots.tar",
            "forge.checkpoint_archive.v1",
            rows=len(snapshot_records),
        ),
    )


@stage("corpus.phase1.freeze.v1")
def freeze_phase1_corpus(context: RunContext) -> StageResult:
    """Build the exact Phase 1 corpus contract inside an isolated stage directory."""

    from forge.corpus import freeze_phase1_data_contract

    config = context.config()
    _require_config_inputs(context, config)
    paths = {
        "ugi_assignments": context.output_path("ugi_l1_assignments.csv.gz"),
        "ugi_provenance": context.output_path("ugi_l1_constitutional_provenance.csv.gz"),
        "manifest": context.output_path("manifest.json"),
        "result": context.output_path("result.json"),
    }
    result = freeze_phase1_data_contract(
        context.config_path,
        context.repo,
        output_paths=paths,
        output_display_root=context.output_dir,
    )
    summary = result["summary"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "assignments",
                "ugi_l1_assignments.csv.gz",
                "phase1_ugi_l1_assignments.v1",
                rows=int(summary["ugi_l1_products"]),
            ),
            ProducedArtifact(
                "provenance",
                "ugi_l1_constitutional_provenance.csv.gz",
                "phase1_ugi_l1_constitutional_provenance.v1",
                rows=int(summary["ugi_source_rows"]),
            ),
            ProducedArtifact("manifest", "manifest.json", "phase1_product_l1_split_manifest.v3"),
            ProducedArtifact("result", "result.json", "phase1_product_l1_data_result.v3"),
        ),
        metrics={
            "r0_rows": int(summary["r0_rows"]),
            "r1_rows": int(summary["r1_rows"]),
            "ugi_l1_products": int(summary["ugi_l1_products"]),
        },
        summary={
            "guidance_enabled": False,
            "r1_sampling_weight": "realism_weight",
            "status": result["status"],
        },
    )


@stage("corpus.ugi.training-cache-verify.v1")
def verify_ugi_training_cache(context: RunContext) -> StageResult:
    """Validate the frozen tensor cache before any trainer can consume it."""

    import torch

    config = context.config()
    configured_labels = set(config.get("inputs", {}))
    _require_config_inputs(context, config, labels=configured_labels)
    cache_path = context.input("prepared_cache")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "phase1_ugi_training_cache.v1":
        raise StageError("prepared training cache schema changed")
    cached_inputs = payload.get("inputs")
    if not isinstance(cached_inputs, dict):
        raise StageError("prepared training cache has no authenticated inputs")
    for label in sorted(configured_labels):
        if cached_inputs.get(label, {}).get("sha256") != config["inputs"][label]["sha256"]:
            raise StageError(f"prepared training cache input changed for {label!r}")
    records = payload.get("joint_records_by_fold")
    if not isinstance(records, dict):
        raise StageError("prepared training cache has no fold records")
    counts = {fold: len(values) for fold, values in records.items()}
    if counts != config.get("expected_fold_counts"):
        raise StageError(f"prepared training cache fold counts changed: {counts}")
    receipt = {
        "cache": context.stage.inputs["prepared_cache"].to_mapping(),
        "fold_counts": counts,
        "inputs": {label: config["inputs"][label] for label in sorted(configured_labels)},
        "schema_version": "forge.ugi_training_cache_receipt.v1",
        "status": "verified",
    }
    del payload, records
    gc.collect()
    write_json(context.output_path("receipt.json"), receipt)
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "receipt",
                "receipt.json",
                "forge.ugi_training_cache_receipt.v1",
                rows=sum(counts.values()),
            ),
        ),
        metrics={"records": sum(counts.values())},
        summary={"cache_sha256": context.stage.inputs["prepared_cache"].sha256},
    )


@stage("generate.ugi.joint-train.v1")
def train_ugi_joint_stage(context: RunContext) -> StageResult:
    """Run the frozen all-fold joint-flow contract with deterministic resume state."""

    from forge.product.ugi_joint_sparse_training import train_ugi_joint_sparse

    context.dependency("cache", "receipt")
    config = context.config()
    _require_config_inputs(context, config)
    runtime = config.get(context.profile)
    if not isinstance(runtime, dict) or runtime.get("device") != context.resources.device:
        raise StageError("joint training config and declared device differ")
    work = context.work_dir / "joint_training"
    resume_training = _resume_training_work(context, work)
    result = train_ugi_joint_sparse(
        context.config_path,
        context.repo,
        work,
        smoke=context.profile == "smoke",
        overwrite=False,
        resume=resume_training,
    )
    artifacts = _publish_training_outputs(
        context,
        work,
        result_schema="phase1_ugi_joint_sparse_training_result.v1",
        checkpoint_schema="phase1_ugi_joint_sparse_checkpoint.v1",
        progress_schema="phase1_ugi_joint_sparse_progress.v1",
    )
    return StageResult(
        artifacts=artifacts,
        metrics={
            "completed_steps": int(result["selection"]["completed_steps"]),
            "training_records": int(result["training_partition"]["training_records"]),
        },
        summary={
            "route_guidance": False,
            "oracle_guidance": False,
            "selection_mode": result["selection"]["mode"],
        },
    )


@stage("generate.ugi.closure-train.v1")
def train_ugi_closure_stage(context: RunContext) -> StageResult:
    """Train the sparse closure scorer with resumable optimizer and RNG state."""

    from forge.product.ugi_closure_training import train_ugi_closure_scorer

    config = context.config()
    _require_config_inputs(context, config)
    runtime = config.get(context.profile)
    if not isinstance(runtime, dict) or runtime.get("device") != context.resources.device:
        raise StageError("closure training config and declared device differ")
    work = context.work_dir / "closure_training"
    resume_training = _resume_training_work(context, work)
    result = train_ugi_closure_scorer(
        context.config_path,
        context.repo,
        work,
        smoke=context.profile == "smoke",
        overwrite=False,
        resume=resume_training,
    )
    artifacts = _publish_training_outputs(
        context,
        work,
        result_schema="phase1_ugi_sparse_closure_result.v2",
        checkpoint_schema="phase1_ugi_sparse_closure_checkpoint.v2",
        progress_schema="phase1_ugi_sparse_closure_progress.v1",
    )
    return StageResult(
        artifacts=artifacts,
        metrics={
            "best_step": int(result["selection"]["best_step"]),
            "completed_steps": int(result["selection"]["completed_steps"]),
        },
        summary={"selection_metric": result["selection"]["metric"]},
    )


@stage("generate.ugi.sample-shards.v1")
def sample_ugi_shards(context: RunContext) -> StageResult:
    """Sample verified contiguous program shards and merge them without retries."""

    from forge.product.ugi_joint_end_to_end_sampling import (
        joint_sampling_result_matches_request,
        sample_ugi_joint_end_to_end,
    )

    config = context.config()
    _require_config_inputs(context, config)
    runtime = config.get("profiles", {}).get(context.profile)
    if not isinstance(runtime, dict):
        raise StageError(f"sampling config has no {context.profile!r} profile")
    if context.resources.device != "cpu":
        raise StageError("the qualified end-to-end sampler currently runs on CPU")
    program_count = int(runtime["program_count"])
    shard_size = int(runtime["shard_size"])
    if program_count < 1 or shard_size < 1:
        raise StageError("sampling program and shard counts must be positive")
    program_document = json.loads(context.input("matched_programs").read_text())
    available = program_document.get("samples")
    if not isinstance(available, list) or program_count > len(available):
        raise StageError(
            f"sampling requested {program_count} programs but only "
            f"{len(available) if isinstance(available, list) else 0} are available"
        )

    work = context.work_dir / "sampling_shards"
    work.mkdir(exist_ok=context.resume)
    terminal_mode = str(runtime["terminal_decoder_mode"])
    merged_rows: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    shard_results: list[Path] = []
    for shard_index, offset in enumerate(range(0, program_count, shard_size)):
        limit = min(shard_size, program_count - offset)
        flow_seed = context.derive_seed("flow", shard_index)
        terminal_seed = (
            context.derive_seed("terminal", shard_index)
            if terminal_mode != "argmax"
            else None
        )
        shard_dir = work / f"shard_{shard_index:05d}"
        result_path = shard_dir / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if not joint_sampling_result_matches_request(
                result,
                seed=flow_seed,
                program_offset=offset,
                program_limit=limit,
                terminal_decoder_mode=terminal_mode,
                terminal_decoder_seed=terminal_seed,
                terminal_temperature=float(runtime["terminal_temperature"]),
                checkpoint_filename=context.input("joint_checkpoint").name,
            ):
                raise StageError(f"persisted sampling shard {shard_index} changed")
        else:
            if shard_dir.exists():
                # Only result.json authenticates a completed shard.  Discard a directory left by
                # interruption before that atomic receipt and replay its fixed seed from scratch.
                shutil.rmtree(shard_dir)
            result = sample_ugi_joint_end_to_end(
                context.repo,
                shard_dir,
                joint_checkpoint_path=context.input("joint_checkpoint"),
                closure_checkpoint_path=context.input("closure_checkpoint"),
                matched_staged_result_path=context.input("matched_programs"),
                sample_steps=int(runtime["sample_steps"]),
                batch_size=int(runtime["batch_size"]),
                seed=flow_seed,
                overwrite=False,
                maximum_adjacent_branch_runs=runtime["maximum_adjacent_branch_runs"],
                qualified_reactions_path=context.input("qualified_reactions"),
                evaluate_exact_l1_terminal_admission=True,
                terminal_decoder_mode=terminal_mode,
                terminal_decoder_seed=terminal_seed,
                terminal_temperature=float(runtime["terminal_temperature"]),
                program_offset=offset,
                program_limit=limit,
                reference_comparison_mode="deferred",
                render=False,
                record_timing=False,
            )
        rows = result.get("samples")
        if not isinstance(rows, list) or len(rows) != limit:
            raise StageError(f"sampling shard {shard_index} returned the wrong row count")
        for local_index, row in enumerate(rows):
            normalized = dict(row)
            global_index = offset + local_index
            normalized["pipeline_index"] = global_index
            normalized["structure_id"] = f"pipeline_generated_{global_index:06d}"
            merged_rows.append(normalized)
        digest = str(sha256_file(result_path))
        shard_results.append(result_path)
        shard_records.append(
            {
                "flow_seed": flow_seed,
                "offset": offset,
                "rows": limit,
                "sha256": digest,
                "shard": shard_index,
                "terminal_seed": terminal_seed,
            }
        )

    exact_l1 = sum(
        bool((row.get("l1_forward_verification") or {}).get("exact_product_reconstructed"))
        for row in merged_rows
    )
    terminal_valid = sum(bool(row.get("terminal_valid")) for row in merged_rows)
    valid = sum(bool(row.get("valid")) for row in merged_rows)
    merged = {
        "inputs": {
            label: pin.to_mapping() for label, pin in sorted(context.stage.inputs.items())
        },
        "profile": context.profile,
        "samples": merged_rows,
        "schema_version": "forge.phase1_ugi_sampling_result.v1",
        "shards": shard_records,
        "statistics": {
            "exact_l1": exact_l1,
            "requested": program_count,
            "returned": len(merged_rows),
            "terminal_valid": terminal_valid,
            "valid": valid,
        },
        "status": "complete",
    }
    write_json(context.output_path("result.json"), merged)
    _deterministic_tar(shard_results, context.output_path("shards.tar"), base=work)
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.phase1_ugi_sampling_result.v1",
                rows=len(merged_rows),
            ),
            ProducedArtifact(
                "shards",
                "shards.tar",
                "forge.sampling_shard_archive.v1",
                rows=len(shard_records),
            ),
        ),
        metrics={
            "exact_l1": exact_l1,
            "terminal_valid": terminal_valid,
            "valid": valid,
        },
        summary={
            "candidate_selection": False,
            "route_calls": 0,
            "oracle_calls": 0,
            "retries_or_repairs": False,
        },
    )
__all__ = [
    "freeze_phase1_corpus",
    "sample_ugi_shards",
    "train_ugi_closure_stage",
    "train_ugi_joint_stage",
    "verify_ugi_training_cache",
]
