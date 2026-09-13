"""Check passive zero-strength observation and authenticate the frozen route source."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments._runtime.historical import HistoricalPinArchiveError, resolve_pinned_input
from experiments.phase1.multireaction.ugi_realism_diagnostic_suite import implementation_snapshot
from experiments.phase1.multireaction.ugi_sampling_trace import (
    _compare_historical,
    _complete_endpoint,
    _prepare,
    _read,
    _validate,
)
from experiments.phase1.synthesis_guidance.guidance.current_sampler_zero import (
    CurrentSamplerZeroObserver,
)
from experiments.phase1.synthesis_guidance.sources.ugi3_cumulative_production_source import (
    load_cumulative_production_ugi3_source,
)
from experiments.phase1.synthesis_guidance.sources.ugi3_source_qualified_cumulative_inputs import (
    source_qualified_cumulative_paths,
)
from forge.assembly import Ugi3AssemblyAdapter
from forge.core.hashing import PinError, pin_record, resolve_pin, sha256_file
from forge.core.io import write_json
from forge.model.common_ugi_benchmark import CommonUgiAttempt
from forge.model.synthesis_program_sampling import sample_synthesis_program_products
from forge.model.ugi_sampling_trace import SamplingTrace
from forge.model.ugi_train_only_assessment import (
    assess_ugi_train_only_attempts,
    load_ugi_training_identities,
)

SCHEMA = "forge.current_sampler_guidance_preflight.v1"
RUNTIME = {"program_count": 128, "sample_steps": 32, "checkpoints": [8, 16, 24, 28, 31]}
POLICY = {
    "guidance_strength": 0.0,
    "training_calls": 0,
    "candidate_route_assessment_calls": 0,
    "oracle_calls": 0,
    "remote_compute": False,
    "heldout_structure_access": False,
    "candidate_selection": False,
    "gate_changes": False,
}


def _error_chain(error: Exception) -> list[dict[str, str]]:
    result = []
    while error is not None:
        result.append({"type": type(error).__name__, "message": str(error)})
        error = error.__cause__
    return result


def inspect_route_source(
    repo: Path, source_config: Path, historical_config: Path
) -> dict[str, Any]:
    """Keep unavailable evidence distinct from a candidate's unresolved route."""
    config, historical = _read(source_config), _read(historical_config)
    inputs = {}
    for name, pin in config["inputs"].items():
        try:
            resolved = resolve_pinned_input(repo, pin["path"], pin["sha256"])
            inputs[name] = {
                "expected": pin,
                "status": "authenticated",
                "resolved": pin_record(resolved, repo),
            }
        except (PinError, HistoricalPinArchiveError) as error:
            inputs[name] = {"expected": pin, "status": "unavailable", "errors": _error_chain(error)}
    paths = source_qualified_cumulative_paths(
        repo, repo / "results/phase1/ugi3_source_qualified_cumulative_inputs_v1"
    )
    inventory = {
        name: {"path": str(path.relative_to(repo)), "exists": path.is_file()}
        for name, path in vars(paths).items()
    }
    record = {
        "assessment_as_of_utc": historical["assessment_as_of_utc"],
        "expected_cumulative_source_inputs_sha256": historical["cumulative_source_inputs_sha256"],
        "pinned_inputs": inputs,
        "loader_paths": inventory,
        "source_loader_calls": 1,
        "candidate_assessment_calls": 0,
        "current_availability_claim": False,
    }
    try:
        _, metadata = load_cumulative_production_ugi3_source(
            repo_root=repo, assessment_as_of_utc=record["assessment_as_of_utc"], paths=paths
        )
    except Exception as error:
        return {**record, "status": "source_authentication_failed", "errors": _error_chain(error)}
    return {
        **record,
        "status": "loader_returned_requires_separate_qualification",
        "metadata": metadata,
    }


def equivalence_checks(control: dict, observed: dict, saved: dict, count: int) -> dict[str, bool]:
    return {
        "rows_equal": len(control["rows"]) == count and control["rows"] == observed["rows"],
        "sampling_summaries_equal": control["sampling_summary"] == observed["sampling_summary"],
        "terminal_and_rng_state_equal": control["endpoint_state"] == observed["endpoint_state"],
        "complete_endpoint_capture": all(
            _complete_endpoint(run["endpoint_state"], count) for run in (control, observed)
        ),
        "trace_capture_complete": all(
            run["trace_summary"]["status"] == "complete" for run in (control, observed)
        ),
        "historical_prefix_equal": _compare_historical(control["rows"], saved)["equal"],
        "model_calls_equal": control["model_forward_calls"]
        == observed["model_forward_calls"]
        == 34,
    }


def run_preflight(repo: Path, config_path: Path, output: Path) -> dict[str, Any]:
    repo, config_path, output = repo.resolve(), config_path.resolve(), output.resolve()
    if output.exists() or output == repo or not output.is_relative_to(repo):
        raise ValueError("use a fresh output directory within the repository")
    config = _read(config_path)
    if (
        config.get("schema_version") != SCHEMA
        or config.get("runtime") != RUNTIME
        or config.get("policy") != POLICY
    ):
        raise ValueError("preflight schema, runtime or zero-only policy differs")
    config_pin = pin_record(config_path, repo)
    inputs = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
    trace_config = _read(inputs["trace_config"])
    trace_inputs = {
        name: resolve_pin(pin, repo, label=name) for name, pin in trace_config["inputs"].items()
    }
    comparison = _read(trace_inputs["comparison_config"])
    native = _validate(trace_config, comparison)
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in comparison["inputs"].items()}
    if (
        trace_config["runtime"]["program_count"] != RUNTIME["program_count"]
        or native["sample_steps"] != RUNTIME["sample_steps"]
    ):
        raise ValueError("historical sampler no longer matches the bounded preflight")
    sources = implementation_snapshot(repo)
    output.mkdir(parents=True)
    write_json(output / "source_snapshot.json", sources)
    started = time.perf_counter()
    try:
        source = inspect_route_source(
            repo, inputs["source_config"], inputs["historical_guidance_config"]
        )
        write_json(output / "route_source_preflight.json", source)
        prepared = _prepare(trace_config, comparison, paths, native)
        runs = {}
        for arm in ("control", "zero_observed"):
            write_json(output / "progress.json", {"status": "sampling", "arm": arm})
            print(f"sampling {arm}: one original batch of 128", flush=True)
            trace = SamplingTrace(
                selected_attempts=(0,), max_events=20000, max_candidates=50000, decisions=False
            )
            observer = CurrentSamplerZeroObserver(
                prepared["args"][0], steps=32, checkpoints=tuple(RUNTIME["checkpoints"])
            )
            model_calls = []
            handle = prepared["args"][0].register_forward_pre_hook(lambda *_: model_calls.append(1))
            arm_start = time.perf_counter()
            try:
                with trace:
                    if arm == "zero_observed":
                        with observer:
                            rows, summary = sample_synthesis_program_products(
                                *prepared["args"],
                                **prepared["common"],
                                **prepared["arms"]["baseline"],
                            )
                        observer.require_complete()
                    else:
                        rows, summary = sample_synthesis_program_products(
                            *prepared["args"], **prepared["common"], **prepared["arms"]["baseline"]
                        )
            finally:
                handle.remove()
            record = {
                "rows": rows,
                "sampling_summary": summary,
                "endpoint_state": trace.endpoint_state,
                "trace_summary": trace.summary,
                "model_forward_calls": len(model_calls),
                "observations": observer.forwards,
                "zero_decisions": observer.decisions,
                "wall_seconds": time.perf_counter() - arm_start,
            }
            write_json(output / f"{arm}.json", record)
            runs[arm] = record
        saved = _read(trace_inputs["historical_baseline_sampling"])
        checks = equivalence_checks(runs["control"], runs["zero_observed"], saved, 128)
        write_json(output / "equivalence.json", checks)
        if not all(checks.values()):
            raise ValueError(f"zero observation or historical replay failed: {checks}")
        adapter = Ugi3AssemblyAdapter.from_registry(
            paths["qualified_reactions"],
            expected_sha256=str(sha256_file(paths["qualified_reactions"])),
        )
        identities = load_ugi_training_identities(
            paths["ugi_assignments"], adapter, str(sha256_file(paths["ugi_assignments"]))
        )
        attempts = [
            CommonUgiAttempt(
                method_id="current_sampler_zero_preflight",
                seed=native["flow_seed"],
                attempt_index=row["sample_index"],
                status="generated" if row["valid"] else "invalid",
                product_smiles=row["canonical_smiles"] if row["valid"] else None,
                method_visible_component_ids=(),
                generator_calls=1,
                reaction_calls=0,
                route_calls=0,
                oracle_calls=0,
                wall_seconds=0.0,
            )
            for row in runs["control"]["rows"]
        ]
        assessed, assessment = assess_ugi_train_only_attempts(
            attempts, adapter=adapter, training_identities=identities
        )
        write_json(output / "assessed_attempts.json", assessed)
        write_json(output / "assessment.json", assessment)
        if implementation_snapshot(repo) != sources:
            raise ValueError("implementation changed during preflight")
        for name, pin in {
            "config": config_pin,
            **config["inputs"],
            **trace_config["inputs"],
            **comparison["inputs"],
        }.items():
            resolve_pin(
                {key: pin[key] for key in ("path", "sha256")}, repo, label=f"post-run {name}"
            )
        result = {
            "schema_version": SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": (
                "zero_observation_verified_route_source_blocked"
                if source["status"] == "source_authentication_failed"
                else "zero_observation_verified_further_qualification_required"
            ),
            "config": config_pin,
            "inputs": config["inputs"],
            "trace_inputs": trace_config["inputs"],
            "comparison_inputs": comparison["inputs"],
            "checkpoint_member": prepared["checkpoint_member"],
            "runtime": {**native, **RUNTIME},
            "policy": POLICY,
            "equivalence": checks,
            "assessment": assessment,
            "route_source_status": source["status"],
            "passive_route_contrast_executed": False,
            "trajectory_resumption_qualified": False,
            "nonzero_guidance_qualified": False,
            "calls": {
                "sampling_batches": 2,
                "attempted_products": 256,
                "model_forwards": 68,
                "candidate_route_assessments": 0,
                "source_loader_preflights": 1,
                "training": 0,
                "oracle": 0,
            },
            "wall_seconds": time.perf_counter() - started,
            "artifacts": {
                p.name: pin_record(p, repo)
                for p in sorted(output.glob("*.json"))
                if p.name != "progress.json"
            },
            "nonclaims": [
                "Identity observation does not qualify restartable trajectories or nonzero guidance.",
                "Synthetic zero-controller values are not route scores; no route contrast was measured.",
                "Missing source evidence is not evidence of an unsuccessful synthesis or zero utility.",
                "The two runs repeat the same 128 attempts; they are not 256 independent candidates.",
                "No realism, route-yield, biological or current procurement improvement is established.",
                "Identity preserves every row-derived validity, novelty and diversity statistic in this batch only.",
            ],
        }
        write_json(output / "result.json", result)
        write_json(output / "progress.json", {"status": result["status"]})
        return result
    except Exception as error:
        write_json(
            output / "failure.json",
            {"status": "failed", "config": config_pin, "errors": _error_chain(error)},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run_preflight(args.repo_root, args.config, args.output_dir)["status"])


if __name__ == "__main__":
    main()
