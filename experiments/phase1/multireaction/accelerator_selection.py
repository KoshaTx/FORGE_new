"""Adjudicate verified production-accelerator benchmark runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments._runtime import verify_run_directory
from forge.core.hashing import pin_record, resolve_pin
from forge.core.io import atomic_write, read_json_object, write_json

from .production_preflight import CONFIG_SCHEMA, RESULT_SCHEMA

SELECTION_SCHEMA = "forge.synthesis_program_production_accelerator_selection.v1"
STAGE_IMPLEMENTATION = "model.shared-synthesis-program-production-accelerator-benchmark.v1"


class AcceleratorSelectionError(ValueError):
    """The accelerator evidence cannot support a production target selection."""


def _benchmark_artifacts(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    verify_run_directory(run_dir)
    run = read_json_object(
        run_dir / "run.json",
        error=AcceleratorSelectionError,
        label="accelerator benchmark run manifest",
    )
    stages = run.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != {"benchmark"}:
        raise AcceleratorSelectionError(
            "accelerator benchmark run must contain one benchmark stage"
        )
    stage_path = run_dir / "stages" / "benchmark" / "manifest.json"
    stage = read_json_object(
        stage_path,
        error=AcceleratorSelectionError,
        label="accelerator benchmark stage manifest",
    )
    if stage.get("implementation") != STAGE_IMPLEMENTATION:
        raise AcceleratorSelectionError("accelerator benchmark used the wrong stage implementation")
    artifacts = stage.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"result"}:
        raise AcceleratorSelectionError("accelerator benchmark result artifact is missing")
    result_path = run_dir / "stages" / "benchmark" / str(artifacts["result"]["path"])
    read_json_object(
        result_path,
        error=AcceleratorSelectionError,
        label="accelerator benchmark result",
    )
    return run, stage, result_path


def _selection_rows(
    results: Sequence[Mapping[str, Any]],
    *,
    primary_metric: str = "projected_training_gpu_cost_usd_all_replicates",
    tie_breaker_metric: str = "projected_training_hours_per_replicate",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.get("schema_version") != RESULT_SCHEMA:
            raise AcceleratorSelectionError("unsupported accelerator benchmark result")
        gates = result.get("gates")
        if (
            result.get("status") != "pass"
            or not isinstance(gates, Mapping)
            or not gates
            or not all(gates.values())
        ):
            gpu = result.get("device", {}).get("declared_gpu_type", "unknown")
            raise AcceleratorSelectionError(f"accelerator benchmark did not pass: {gpu}")
        device = result.get("device")
        projection = result.get("projection")
        if not isinstance(device, Mapping) or not isinstance(projection, Mapping):
            raise AcceleratorSelectionError("accelerator benchmark result is incomplete")
        if projection.get("scope") != "training_only_excludes_evaluation_and_startup":
            raise AcceleratorSelectionError("accelerator projection scope changed")
        rows.append(
            {
                "gpu_type": str(device["declared_gpu_type"]),
                "physical_device": str(device["name"]),
                "peak_reserved_bytes": int(result["peak_reserved_bytes"]),
                "total_memory_bytes": int(device["total_memory_bytes"]),
                "projected_training_hours_per_replicate": float(projection["hours_per_replicate"]),
                "projected_training_gpu_cost_usd_all_replicates": float(
                    projection["gpu_cost_usd_all_replicates"]
                ),
            }
        )
    gpu_types = [row["gpu_type"] for row in rows]
    if len(gpu_types) != len(set(gpu_types)):
        raise AcceleratorSelectionError("accelerator benchmark targets are duplicated")
    allowed_metrics = {
        "projected_training_gpu_cost_usd_all_replicates",
        "projected_training_hours_per_replicate",
    }
    if (
        primary_metric not in allowed_metrics
        or tie_breaker_metric not in allowed_metrics
        or primary_metric == tie_breaker_metric
    ):
        raise AcceleratorSelectionError("accelerator ranking metrics are invalid")
    return sorted(
        rows,
        key=lambda row: (row[primary_metric], row[tie_breaker_metric], row["gpu_type"]),
    )


def adjudicate_accelerator_benchmarks(
    run_dirs: Sequence[Path], repo: Path, output_path: Path
) -> dict[str, Any]:
    """Select the lowest projected-cost accelerator after every frozen gate passes."""

    if len(run_dirs) < 2:
        raise AcceleratorSelectionError("accelerator selection requires at least two runs")
    try:
        output_path.resolve().relative_to(repo.resolve())
    except ValueError as error:
        raise AcceleratorSelectionError(
            "accelerator selection output must stay inside the repository"
        ) from error
    loaded = [_benchmark_artifacts(path.resolve()) for path in run_dirs]
    runs = [item[0] for item in loaded]
    stages = [item[1] for item in loaded]
    result_paths = [item[2] for item in loaded]
    results = [
        read_json_object(
            path,
            error=AcceleratorSelectionError,
            label="accelerator benchmark result",
        )
        for path in result_paths
    ]

    source_hashes = {run.get("source_sha256") for run in runs}
    config_pins = {str(stage.get("config")) for stage in stages}
    external_inputs = {str(stage.get("external_inputs")) for stage in stages}
    if len(source_hashes) != 1:
        raise AcceleratorSelectionError("accelerator runs used different executable source")
    if len(config_pins) != 1 or len(external_inputs) != 1:
        raise AcceleratorSelectionError("accelerator runs used different pinned inputs")
    if any(result.get("runtime") != results[0].get("runtime") for result in results[1:]):
        raise AcceleratorSelectionError("accelerator runs used different training runtimes")
    if any(result.get("model") != results[0].get("model") for result in results[1:]):
        raise AcceleratorSelectionError("accelerator runs used different models")
    if any(result.get("math_mode") != results[0].get("math_mode") for result in results[1:]):
        raise AcceleratorSelectionError("accelerator runs used different CUDA math modes")
    if any(
        result.get("design_sha256") != results[0].get("design_sha256") for result in results[1:]
    ):
        raise AcceleratorSelectionError("accelerator runs used different scientific designs")
    if any(result.get("cache") != results[0].get("cache") for result in results[1:]):
        raise AcceleratorSelectionError("accelerator runs used different production caches")

    config_record = results[0].get("config")
    if not isinstance(config_record, Mapping):
        raise AcceleratorSelectionError("accelerator benchmark config pin is missing")
    config_path = resolve_pin(
        {"path": config_record.get("path"), "sha256": config_record.get("sha256")},
        repo,
        label="accelerator benchmark config",
    )
    config = read_json_object(
        config_path,
        error=AcceleratorSelectionError,
        label="accelerator benchmark config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise AcceleratorSelectionError("accelerator selection config changed")
    benchmark = config.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise AcceleratorSelectionError("accelerator benchmark policy is missing")
    adjudication = benchmark.get("adjudication")
    if not isinstance(adjudication, Mapping):
        raise AcceleratorSelectionError("accelerator adjudication policy is missing")
    required_targets = adjudication.get("required_targets")
    if not isinstance(required_targets, list) or any(
        not isinstance(target, str) for target in required_targets
    ):
        raise AcceleratorSelectionError("accelerator target list is invalid")

    primary_metric = str(
        adjudication.get("primary_metric", "projected_training_gpu_cost_usd_all_replicates")
    )
    tie_breaker_metric = str(
        adjudication.get("tie_breaker_metric", "projected_training_hours_per_replicate")
    )
    ranking = _selection_rows(
        results,
        primary_metric=primary_metric,
        tie_breaker_metric=tie_breaker_metric,
    )
    if {row["gpu_type"] for row in ranking} != set(required_targets):
        raise AcceleratorSelectionError("accelerator benchmark set is incomplete")
    selected = ranking[0]
    evidence = []
    for run_dir, run, stage, result_path in zip(run_dirs, runs, stages, result_paths, strict=True):
        gpu_type = str(
            read_json_object(
                result_path,
                error=AcceleratorSelectionError,
                label="accelerator benchmark result",
            )["device"]["declared_gpu_type"]
        )
        evidence_slug = gpu_type.lower().replace("-", "_").replace("!", "_exact")
        if not evidence_slug or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in evidence_slug
        ):
            raise AcceleratorSelectionError(f"accelerator target has unsafe identity: {gpu_type!r}")
        evidence_dir = output_path.parent / "evidence" / evidence_slug
        snapshot_paths = {
            "run_manifest": evidence_dir / "run.json",
            "stage_manifest": evidence_dir / "stage_manifest.json",
            "result": evidence_dir / "result.json",
        }
        sources = {
            "run_manifest": run_dir.resolve() / "run.json",
            "stage_manifest": run_dir.resolve() / "stages" / "benchmark" / "manifest.json",
            "result": result_path,
        }
        for label, destination in snapshot_paths.items():
            atomic_write(destination, sources[label].read_bytes())
        evidence.append(
            {
                "gpu_type": gpu_type,
                "run_id": run["run_id"],
                "run_manifest": pin_record(snapshot_paths["run_manifest"], repo),
                "stage_manifest": pin_record(snapshot_paths["stage_manifest"], repo),
                "result": pin_record(snapshot_paths["result"], repo),
                "stage_fingerprint": stage["fingerprint"],
            }
        )

    output = {
        "schema_version": SELECTION_SCHEMA,
        "status": "selected_execution_target",
        "selected_gpu_type": selected["gpu_type"],
        "selection_objective": adjudication["primary_order"],
        "tie_breaker": adjudication["tie_breaker"],
        "selection_metric": primary_metric,
        "tie_breaker_metric": tie_breaker_metric,
        "ranking": ranking,
        "benchmark_config": pin_record(config_path, repo),
        "source_sha256": next(iter(source_hashes)),
        "design_sha256": results[0]["design_sha256"],
        "cache_sha256": results[0]["cache"]["sha256"],
        "math_mode": results[0]["math_mode"],
        "evidence": sorted(evidence, key=lambda item: item["gpu_type"]),
        "production_requirement": adjudication["production_requirement"],
        "calls": {"route": 0, "oracle": 0},
        "candidate_selection": False,
        "nonclaims": [
            "This is an execution-cost selection, not model-quality evidence.",
            "Projected cost covers training GPU time only.",
        ],
    }
    write_json(output_path, output)
    return output


__all__ = [
    "SELECTION_SCHEMA",
    "AcceleratorSelectionError",
    "_selection_rows",
    "adjudicate_accelerator_benchmarks",
]
