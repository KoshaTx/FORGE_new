"""Aggregate three fresh Ugi tree-Transformer production seeds."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments._runtime import verify_run_directory
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, write_json

CONFIG_SCHEMA = "forge.ugi_tree_transformer_production_aggregate_config.v1"
RESULT_SCHEMA = "forge.ugi_tree_transformer_production_aggregate.v1"
EVALUATION_SCHEMA = "forge.ugi_tree_transformer_production_evaluation.v1"
EXPERIMENT_ID = "phase1-ugi-tree-relational-production-h100-v1"
STAGE_ID = "evaluate"


class UgiTreeTransformerProductionAggregateError(ValueError):
    """The three-seed production evidence or frozen reporting contract changed."""


def _artifact_path(run_dir: Path, manifest: Mapping[str, Any], label: str) -> Path:
    artifacts = manifest.get("artifacts")
    record = artifacts.get(label) if isinstance(artifacts, Mapping) else None
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise UgiTreeTransformerProductionAggregateError(
            f"production run has no {label!r} artifact"
        )
    path = run_dir / "stages" / STAGE_ID / str(record["path"])
    if sha256_file(path) != record.get("sha256"):
        raise UgiTreeTransformerProductionAggregateError(
            f"production artifact digest changed: {label}"
        )
    return path


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != 3:
        raise UgiTreeTransformerProductionAggregateError(
            "production summary requires three training seeds"
        )
    return {
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "seed_values": list(values),
    }


def aggregate_metric_mappings(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Aggregate declared numeric metrics in seed order."""

    result: dict[str, dict[str, Any]] = {}
    for key in keys:
        try:
            values = [float(row[key]) for row in rows]
        except (KeyError, TypeError, ValueError) as error:
            raise UgiTreeTransformerProductionAggregateError(
                f"production metric {key!r} is missing or nonnumeric"
            ) from error
        result[key] = _summary(values)
    return result


def aggregate_tree_transformer_production(
    run_dirs: Sequence[Path],
    config_path: Path,
    repo: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Verify and aggregate the three fixed-seed post-selection production runs."""

    config = read_json_object(
        config_path,
        error=UgiTreeTransformerProductionAggregateError,
        label="tree-Transformer production aggregate config",
    )
    required = {
        "schema_version",
        "status",
        "experiment_id",
        "expected_training_seeds",
        "selected_model",
        "inputs",
        "reported_metrics",
        "gates",
        "policy",
        "nonclaims",
    }
    if config.get("schema_version") != CONFIG_SCHEMA or set(config) != required:
        raise UgiTreeTransformerProductionAggregateError("aggregate config fields changed")
    if (
        config.get("status") != "frozen_before_fresh_production_seeds"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("selected_model")
        != {
            "arm_id": "tree_relations_and_routing",
            "checkpoint_step": 2700,
            "candidate_id": "tree_relations_and_routing:step_2700",
        }
    ):
        raise UgiTreeTransformerProductionAggregateError(
            "aggregate selected-model contract changed"
        )
    seeds = config.get("expected_training_seeds")
    if seeds != [20260905, 20260906, 20260907] or len(run_dirs) != 3:
        raise UgiTreeTransformerProductionAggregateError(
            "aggregate requires the three frozen training seeds"
        )
    policy = config.get("policy")
    if policy != {
        "architecture_or_checkpoint_reselection": False,
        "heldout_rows_descriptive_only": True,
        "negative_result_completes_experiment": True,
        "training_seed_is_unit_of_replication": True,
        "molecule_rows_are_not_independent_replicates": True,
        "reductive_amination_substructure_rate_reported": False,
        "route_calls": 0,
        "oracle_calls": 0,
        "candidate_selection": False,
    }:
        raise UgiTreeTransformerProductionAggregateError("aggregate policy changed")

    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "program_draw",
        "v0_reference_assessment",
    }:
        raise UgiTreeTransformerProductionAggregateError("aggregate inputs changed")
    program_draw = resolve_pin(inputs["program_draw"], repo, label="program_draw")
    v0_path = resolve_pin(
        inputs["v0_reference_assessment"], repo, label="v0_reference_assessment"
    )
    v0 = read_json_object(
        v0_path,
        error=UgiTreeTransformerProductionAggregateError,
        label="frozen v0 production assessment",
    )
    v0_metrics = v0.get("methods", {}).get("forge_v0_production", {}).get("metrics")
    if not isinstance(v0_metrics, Mapping):
        raise UgiTreeTransformerProductionAggregateError(
            "frozen v0 production metrics are unavailable"
        )

    evidence: list[tuple[int, dict[str, Any], Path, dict[str, Any]]] = []
    source_hashes: set[str] = set()
    replicates: set[int] = set()
    for raw_dir in run_dirs:
        run_dir = raw_dir.resolve()
        verify_run_directory(run_dir)
        run = read_json_object(
            run_dir / "run.json",
            error=UgiTreeTransformerProductionAggregateError,
            label="production run",
        )
        manifest = read_json_object(
            run_dir / "stages" / STAGE_ID / "manifest.json",
            error=UgiTreeTransformerProductionAggregateError,
            label="production evaluation manifest",
        )
        if (
            run.get("status") != "complete"
            or run.get("profile") != "full"
            or run.get("experiment_id") != EXPERIMENT_ID
            or run.get("stages", {}).get(STAGE_ID, {}).get("status") != "complete"
            or manifest.get("backend") != "modal"
        ):
            raise UgiTreeTransformerProductionAggregateError(
                "production evidence is not one completed full-profile Modal run"
            )
        replicate = int(run.get("replicate", -1))
        if replicate in replicates or replicate not in {0, 1, 2}:
            raise UgiTreeTransformerProductionAggregateError(
                "production replicate set changed"
            )
        replicates.add(replicate)
        source_hashes.add(str(run.get("source_sha256", "")))
        result_path = _artifact_path(run_dir, manifest, "result")
        result = read_json_object(
            result_path,
            error=UgiTreeTransformerProductionAggregateError,
            label="production evaluation result",
        )
        seed = int(result.get("training_seed", -1))
        if (
            result.get("schema_version") != EVALUATION_SCHEMA
            or result.get("status") != "complete"
            or result.get("profile") != "full"
            or result.get("arm_id") != "tree_relations_and_routing"
            or result.get("checkpoint_step") != 2700
            or result.get("programs") != 3072
            or result.get("heldout_selects_architecture_checkpoint_or_threshold") is not False
            or result.get("candidate_selection") is not False
            or result.get("route_calls") != 0
            or result.get("oracle_calls") != 0
            or result.get("inputs", {}).get("program_draw", {}).get("sha256")
            != inputs["program_draw"]["sha256"]
        ):
            raise UgiTreeTransformerProductionAggregateError(
                "production evaluation result violates the frozen contract"
            )
        evidence.append((seed, result, result_path, run))
    evidence.sort(key=lambda item: item[0])
    if [item[0] for item in evidence] != seeds or len(source_hashes) != 1:
        raise UgiTreeTransformerProductionAggregateError(
            "production seeds or executable source hashes differ"
        )

    reported = config.get("reported_metrics")
    if not isinstance(reported, list) or not reported or len(set(reported)) != len(reported):
        raise UgiTreeTransformerProductionAggregateError("reported metric list changed")
    metric_rows = [item[1]["metrics"] for item in evidence]
    metrics = aggregate_metric_mappings(metric_rows, [str(key) for key in reported])

    stratum_keys = set(evidence[0][1].get("strata", {}))
    if not stratum_keys or any(set(item[1].get("strata", {})) != stratum_keys for item in evidence):
        raise UgiTreeTransformerProductionAggregateError(
            "production descriptive strata differ across seeds"
        )
    stratum_metrics = (
        "valid_fraction_per_attempt",
        "exact_l1_yield_per_attempt",
        "open_ended_exact_l1_yield_per_attempt",
        "held_component_exact_l1_products_per_1000_attempts",
    )
    strata = {
        label: aggregate_metric_mappings(
            [item[1]["strata"][label] for item in evidence], stratum_metrics
        )
        for label in sorted(stratum_keys)
    }

    gates = config.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != {
        "per_seed_minimums",
        "mean_minimums",
        "minimum_mean_effective_component_count_ratio_to_v0",
    }:
        raise UgiTreeTransformerProductionAggregateError("production gate fields changed")
    per_seed = gates["per_seed_minimums"]
    mean_minimums = gates["mean_minimums"]
    if not isinstance(per_seed, Mapping) or not isinstance(mean_minimums, Mapping):
        raise UgiTreeTransformerProductionAggregateError("production gates are malformed")
    checks: dict[str, bool] = {}
    for key, threshold in per_seed.items():
        checks[f"every_seed_{key}"] = all(
            float(row[key]) >= float(threshold) for row in metric_rows
        )
    for key, threshold in mean_minimums.items():
        checks[f"mean_{key}"] = float(metrics[key]["mean"]) >= float(threshold)
    ratio = float(metrics["effective_component_count"]["mean"]) / float(
        v0_metrics["effective_component_count"]
    )
    checks["mean_effective_component_count_ratio_to_v0"] = ratio >= float(
        gates["minimum_mean_effective_component_count_ratio_to_v0"]
    )
    decision = (
        "qualified_for_ugi_production_reporting"
        if all(checks.values())
        else "negative_full_ugi_evaluation"
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "decision": decision,
        "selected_model": config["selected_model"],
        "training_seeds": seeds,
        "source_sha256": next(iter(source_hashes)),
        "attempts_per_seed": 3072,
        "metrics": metrics,
        "strata": strata,
        "v0_fixed_reference": {
            "metrics": {str(key): v0_metrics[key] for key in reported},
            "training_seed_replication": False,
            "descriptive_only": True,
        },
        "mean_effective_component_count_ratio_to_v0": ratio,
        "gates": {"checks": checks, "status": "pass" if all(checks.values()) else "fail"},
        "inputs": {
            "config": pin_record(config_path, repo),
            "program_draw": pin_record(program_draw, repo),
            "v0_reference_assessment": pin_record(v0_path, repo),
            "runs": {
                str(seed): {
                    "run": artifact_record(run_dir / "run.json"),
                    "evaluation": artifact_record(result_path),
                }
                for (seed, _, result_path, _), run_dir in zip(
                    evidence, [item[2].parents[3] for item in evidence], strict=True
                )
            },
        },
        "candidate_selection": False,
        "heldout_rows_descriptive_only": True,
        "nonclaims": list(config["nonclaims"]),
    }
    if output_path.exists():
        raise UgiTreeTransformerProductionAggregateError(
            f"aggregate output already exists: {output_path}"
        )
    write_json(output_path, result)
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiTreeTransformerProductionAggregateError",
    "aggregate_metric_mappings",
    "aggregate_tree_transformer_production",
]
