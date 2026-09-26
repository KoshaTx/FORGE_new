"""Product/L1 sharded sampling stage adapter."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from experiments._runtime.errors import StageError
from experiments._runtime.registry import stage
from experiments._runtime.stage import (
    ProducedArtifact,
    RunContext,
    StageResult,
    require_config_inputs,
)
from experiments.phase1.product_l1._stage_support import _deterministic_tar
from forge.core.hashing import sha256_file
from forge.core.io import write_json


@stage("generate.ugi.sample-shards.v1")
def sample_ugi_shards(context: RunContext) -> StageResult:
    """Sample verified contiguous program shards and merge them without retries."""

    from experiments.phase1.product_l1.sampling.ugi_joint_end_to_end_sampling import (
        joint_sampling_result_matches_request,
        sample_ugi_joint_end_to_end,
    )

    config = context.config()
    require_config_inputs(context, config)
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
            context.derive_seed("terminal", shard_index) if terminal_mode != "argmax" else None
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
                prepared_cache_path=(
                    context.input("prepared_cache") if "prepared_cache" in context.inputs else None
                ),
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
        "inputs": {label: pin.to_mapping() for label, pin in sorted(context.stage.inputs.items())},
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
