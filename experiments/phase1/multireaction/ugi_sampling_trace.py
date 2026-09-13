"""Authenticate and observe one unchanged Ugi sampling batch, with train-only assessment."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiments.phase1.multireaction.production_evaluation import _conditioning_contract
from experiments.phase1.multireaction.reaction_specialization import (
    _load_base_package,
    _set_determinism,
)
from experiments.phase1.multireaction.ugi_realism_diagnostic_suite import implementation_snapshot
from experiments.phase1.product_l1.evaluation.ugi_all_role_semantic_program_comparison import (
    _load_draw,
    _runtime,
    _semantic_support_audit,
)
from experiments.phase1.product_l1.evaluation.ugi_chemistry_specialist_comparison import (
    _topology_policy,
)
from experiments.phase1.product_l1.evaluation.ugi_v0_current_program_comparison import (
    project_program_for_mixed_transformer,
)
from forge.assembly import Ugi3AssemblyAdapter
from forge.core.hashing import pin_record, resolve_pin, sha256_file, sha256_json
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.common_ugi_benchmark import CommonUgiAttempt
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.reaction_core_saturation import ReactionCoreSaturationPolicy
from forge.model.synthesis_program_layout import SynthesisProgramLayoutPrior
from forge.model.synthesis_program_sampling import sample_synthesis_program_products
from forge.model.synthesis_program_training import build_synthesis_program_flow
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_mog_semantic_guidance import UgiMogSemanticGuidancePolicy
from forge.model.ugi_role_chemistry_prior import UgiRoleChemistryPrior

CONFIG_SCHEMA = "forge.ugi_sampling_trace_config.v1"
RESULT_SCHEMA = "forge.ugi_sampling_trace_result.v1"
ARMS = ("baseline", "treatment")


class UgiSamplingTraceError(ValueError):
    """A sampling observation cannot be authenticated or was not passive."""


def _read(path: Path) -> dict[str, Any]:
    return read_json_object(path, error=UgiSamplingTraceError)


def _digest(value: Any) -> str:
    return str(sha256_json(value))


def _authenticate_record(record: dict[str, Any], repo: Path, *, label: str) -> Path:
    return resolve_pin({key: record[key] for key in ("path", "sha256")}, repo, label=label)


def _validate(config: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    if (
        set(config)
        != {"schema_version", "inputs", "checkpoint", "runtime", "policy", "scientific_question"}
        or config["schema_version"] != CONFIG_SCHEMA
    ):
        raise UgiSamplingTraceError("trace config schema/fields differ")
    if config["policy"] != {
        "training_calls": 0,
        "route_or_oracle_calls": 0,
        "heldout_or_calibration_structure_access": False,
        "candidate_selection": False,
        "repairs_or_retries": False,
        "promotion_gate_changes": False,
    }:
        raise UgiSamplingTraceError("trace policy differs")
    native = dict(_runtime(comparison, profile="h100_preflight", device="cpu"))
    runtime = config["runtime"]
    if set(runtime) != {
        "program_count",
        "cpu_threads",
        "selected_attempts",
        "max_events",
        "max_candidates",
    }:
        raise UgiSamplingTraceError("trace runtime fields differ")
    count = runtime["program_count"]
    selected = runtime["selected_attempts"]
    if (
        type(count) is not int
        or count != native["batch_size"]
        or not 1 <= count <= native["program_count"]
        or type(runtime["cpu_threads"]) is not int
        or not 1 <= runtime["cpu_threads"] <= 4
        or not isinstance(selected, list)
        or not selected
        or len(set(selected)) != len(selected)
        or any(type(index) is not int or not 0 <= index < count for index in selected)
        or any(
            type(runtime[key]) is not int or runtime[key] < 1
            for key in ("max_events", "max_candidates")
        )
    ):
        raise UgiSamplingTraceError(
            "trace must retain one complete original batch and valid bounds"
        )
    if (
        comparison.get("baseline_semantic_guidance") is not None
        or comparison.get("treatment_semantic_target_source") is not None
    ):
        raise UgiSamplingTraceError(
            "trace supports the frozen amine baseline and original paired targets"
        )
    if any(
        config["checkpoint"].get(key) != comparison["base_checkpoint"][key]
        for key in ("arm_id", "step")
    ):
        raise UgiSamplingTraceError("checkpoint identity differs from comparison")
    return native


def _historical_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare only actual saved fields; never confuse layout exactness with exact L1."""
    return [
        {
            "pipeline_index": row["sample_index"],
            "smiles": row["canonical_smiles"] if row["valid"] else None,
            "valid": row["valid"],
            "constraint_abstention_reason": row.get("constraint_abstention_reason"),
            "sampled_topology": row.get("sampled_topology"),
        }
        for row in rows
    ]


def _compare_historical(rows: list[dict[str, Any]], saved: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        raise UgiSamplingTraceError("historical comparison requires attempted rows")
    observed = _historical_projection(rows)
    if len(saved.get("samples", [])) < len(rows):
        raise UgiSamplingTraceError("historical sample ledger is too short")
    expected = [{key: row[key] for key in observed[0]} for row in saved["samples"][: len(rows)]]
    differences = [
        index
        for index, (left, right) in enumerate(zip(observed, expected, strict=True))
        if left != right
    ]
    return {
        "compared_fields": list(observed[0]),
        "attempts": len(rows),
        "equal": not differences,
        "different_attempts": differences,
        "observed_sha256": _digest(observed),
        "historical_sha256": _digest(expected),
    }


def _complete_endpoint(state: dict[str, Any], count: int) -> bool:
    returns = state.get("sampler_returns", [])
    if len(returns) != 1:
        return False
    captured = returns[0]
    tensors = captured.get("terminal_states", {})
    generators = captured.get("generator_states", {})
    return (
        captured.get("terminal_state_scope") == "final_batch"
        and captured.get("batch_offset") == 0
        and captured.get("total_attempts") == count
        and len(captured.get("batch_record_ids", [])) == count
        and bool(tensors)
        and all(isinstance(value, list) and len(value) == count for value in tensors.values())
        and set(generators)
        == {
            "generator",
            "topology_generator",
            "topology_conditioned_chemistry_generator",
            "terminal_generator",
        }
        and generators["generator"] is not None
        and generators["topology_generator"] is not None
    )


def _complete_decisions(
    events: list[dict[str, Any]], selected: list[int], rows: list[dict[str, Any]]
) -> bool:
    """Require decisions as well as independently captured sampler-return outcomes."""
    outcomes = [event for event in events if event["kind"] == "attempt_outcome"]
    if len(outcomes) != len(selected) or {
        event["context"]["attempt_index"] for event in outcomes
    } != set(selected):
        return False
    for index in selected:
        local = [event for event in events if event["context"]["attempt_index"] == index]
        if not rows[index]["valid"]:
            if not rows[index].get("constraint_abstention_reason"):
                return False
            continue
        selectors = {event["selector"] for event in local if event["kind"] == "selector_outcome"}
        topology = {event["selector"] for event in local if event["kind"] == "categorical_choice"}
        chemistry = {event["selector"] for event in local if event["kind"] == "terminal_assignment"}
        if (
            not {"decode_ugi_exact_topology", "_strict_terminal_record"}.issubset(selectors)
            or not {
                "_sample_amine_semantic_topology",
                "_sample_constructive_ester_offspring",
            }.issubset(topology)
            or not {"_select_ugi_amine_atom_states", "_ugi_ester_motif_constraints"}.issubset(
                chemistry
            )
        ):
            return False
    return True


def _prepare(
    config: dict[str, Any],
    comparison: dict[str, Any],
    paths: dict[str, Path],
    native: dict[str, Any],
) -> dict[str, Any]:
    """Reuse production loaders and count-only TRAIN layout projection once per replay."""
    count = config["runtime"]["program_count"]
    programs, baseline_targets, treatment_targets = _load_draw(
        paths["all_role_semantic_program_draw"], count=count
    )
    local = LocalChemistrySupport.from_mapping(_read(paths["role_morphology_policy"]))
    topology = _topology_policy(paths)
    registry = paths["qualified_reactions"]
    core = ReactionCoreSaturationPolicy.from_qualified_registry(
        registry, reaction_id="ugi_3cr_agile", expected_sha256=sha256_file(registry)
    )
    ester = UgiEsterChemotypePolicy.from_qualified_registry(
        registry,
        training_assignments_path=paths["ugi_assignments"],
        reaction_id="ugi_3cr_agile",
        expected_sha256=sha256_file(registry),
        expected_training_assignments_sha256=sha256_file(paths["ugi_assignments"]),
    )
    support = _semantic_support_audit(
        programs,
        treatment_targets,
        topology_policy=topology,
        ester_policy=ester,
        local_chemistry_support=local,
    )
    if not support["all_pairs_supported"]:
        raise UgiSamplingTraceError("frozen request has unsupported semantic targets")
    guidance = UgiMogSemanticGuidancePolicy.from_mapping(comparison["semantic_guidance"])
    if guidance.uses_joint_realism or not guidance.uses_local_reference:
        raise UgiSamplingTraceError("trace reference binding differs from frozen audited treatment")
    prior = UgiRoleChemistryPrior.from_training_data(
        assignments_path=paths["ugi_assignments"],
        semantic_atoms_path=paths["semantic_atoms"],
        semantic_bonds_path=paths["semantic_bonds"],
        reaction_id="ugi_3cr_agile",
        expected_assignments_sha256=str(sha256_file(paths["ugi_assignments"])),
        expected_semantic_atoms_sha256=str(sha256_file(paths["semantic_atoms"])),
        expected_semantic_bonds_sha256=str(sha256_file(paths["semantic_bonds"])),
        maximum_bond_depth_bucket=guidance.local_chemistry_bond_maximum_depth_bucket,
    )
    guidance = guidance.bind_local_chemistry(prior)
    device = torch.device("cpu")
    _set_determinism(native["flow_seed"], config["runtime"]["cpu_threads"], device)
    checkpoint = config["checkpoint"]
    package, member = _load_base_package(
        archive_path=paths["base_checkpoint_archive"],
        training=_read(paths["base_training_result"]),
        arm_id=checkpoint["arm_id"],
        step=checkpoint["step"],
        expected_member_sha256=checkpoint["member_sha256"],
        expected_model_state_sha256=checkpoint["model_state_sha256"],
        expected_design_sha256=str(sha256_file(paths["base_design"])),
        expected_cache_sha256=str(sha256_file(paths["production_cache"])),
        device=device,
    )
    cache = SynthesisProgramProductionCache(paths["production_cache"])
    try:
        layout_prior = SynthesisProgramLayoutPrior(cache, ugi_ester_chemotype_policy=ester)
        records = tuple(
            project_program_for_mixed_transformer(layout_prior, program, sample_index=index)
            for index, program in enumerate(programs)
        )
        model = build_synthesis_program_flow(
            vocabulary=cache.vocabulary,
            node_classes=len(cache.atom_vocabulary),
            model_config=package["model_config"],
            device=device,
        )
        model.load_state_dict(package["model_state"], strict=True)
        model.eval()
        conditioning, state_mapping = _conditioning_contract(package, cache)
        atoms = cache.atom_vocabulary
    finally:
        cache.close()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise UgiSamplingTraceError("checkpoint precision differs from float32")
    common = {
        "samples_per_program": 1,
        "sample_steps": native["sample_steps"],
        "batch_size": native["batch_size"],
        "seed": native["flow_seed"],
        "device": "cpu",
        "conditioning_mode": conditioning,
        "program_state_mapping": state_mapping,
        "terminal_temperature": native["terminal_temperature"],
        "local_chemistry_support": local,
        "ugi_topology_policy": topology,
        "reaction_core_saturation_policy": core,
        "ugi_ester_chemotype_policy": ester,
    }
    arms = {
        "baseline": {
            "terminal_decode_policy": native["baseline_terminal_decode_policy"],
            "terminal_seed": None,
            "ugi_amine_semantic_targets": baseline_targets,
        },
        "treatment": {
            "terminal_decode_policy": native["treatment_terminal_decode_policy"],
            "terminal_seed": native["treatment_terminal_decoder_seed"],
            "ugi_all_role_semantic_targets": treatment_targets,
            "ugi_mog_semantic_guidance_policy": guidance,
        },
    }
    return {
        "args": (
            model,
            records,
            atoms,
            package["node_marginal"].cpu().numpy(),
            package["bond_marginal"].cpu().numpy(),
        ),
        "common": common,
        "arms": arms,
        "support": support,
        "checkpoint_member": member,
    }


def run_ugi_sampling_trace(repo: Path, config_path: Path, output: Path) -> dict[str, Any]:
    from forge.model.ugi_sampling_trace import SamplingTrace
    from forge.model.ugi_train_only_assessment import (
        assess_ugi_train_only_attempts,
        load_ugi_training_identities,
    )

    repo, config_path, output = repo.resolve(), config_path.resolve(), output.resolve()
    if not output.is_relative_to(repo) or output == repo or output.exists():
        raise UgiSamplingTraceError("use a fresh output directory within the repository")
    config = _read(config_path)
    inputs = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
    comparison = _read(inputs["comparison_config"])
    native = _validate(config, comparison)
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in comparison["inputs"].items()}
    sources = implementation_snapshot(repo)
    source_digest = _digest(sources)
    output.mkdir(parents=True)
    write_json(output / "source_snapshot.json", sources)
    progress = {
        "schema_version": RESULT_SCHEMA,
        "config": pin_record(config_path, repo),
        "source_snapshot_sha256": source_digest,
        "status": "preparing",
    }
    write_json(output / "progress.json", progress)
    started = time.perf_counter()
    try:
        prepared = _prepare(config, comparison, paths, native)
        preparation_seconds = time.perf_counter() - started
        adapter = Ugi3AssemblyAdapter.from_registry(
            paths["qualified_reactions"],
            expected_sha256=str(sha256_file(paths["qualified_reactions"])),
        )
        identities = load_ugi_training_identities(
            paths["ugi_assignments"], adapter, str(sha256_file(paths["ugi_assignments"]))
        )
        results = {}
        for arm in ARMS:
            saved = _read(inputs[f"historical_{arm}_sampling"])
            arm_runs = {}
            for decisions in (False, True):
                mode = "observed" if decisions else "control"
                label = f"{arm}_{mode}"
                print(f"sampling {label}", flush=True)
                write_json(
                    output / "progress.json", {**progress, "status": "sampling", "stage": label}
                )
                local_start = time.perf_counter()
                trace = SamplingTrace(
                    selected_attempts=tuple(config["runtime"]["selected_attempts"]),
                    max_events=config["runtime"]["max_events"],
                    max_candidates=config["runtime"]["max_candidates"],
                    decisions=decisions,
                )
                try:
                    with trace:
                        rows, summary = sample_synthesis_program_products(
                            *prepared["args"], **prepared["common"], **prepared["arms"][arm]
                        )
                finally:
                    write_json(output / label / "trace_summary.json", trace.summary)
                    write_json(output / label / "events.json", trace.events)
                record = {
                    "rows": rows,
                    "sampling_summary": summary,
                    "endpoint_state": trace.endpoint_state,
                    "endpoint_digest": trace.endpoint_digest,
                    "trace_summary": trace.summary,
                    "wall_seconds": time.perf_counter() - local_start,
                }
                write_json(output / label / "sampling.json", record)
                arm_runs[mode] = record
            control, observed = arm_runs["control"], arm_runs["observed"]
            checks = {
                "complete_decision_capture": _complete_decisions(
                    trace.events, config["runtime"]["selected_attempts"], observed["rows"]
                ),
                "trace_status_complete": all(
                    value["trace_summary"]["status"] == "complete" for value in arm_runs.values()
                ),
                "rows_equal": control["rows"] == observed["rows"],
                "sampling_summaries_equal": control["sampling_summary"]
                == observed["sampling_summary"],
                "terminal_and_rng_state_equal": control["endpoint_state"]
                == observed["endpoint_state"],
                "complete_endpoint_capture": _complete_endpoint(
                    control["endpoint_state"], config["runtime"]["program_count"]
                ),
                "complete_selected_attempt_capture": observed["trace_summary"][
                    "observed_attempt_indices"
                ]
                == sorted(config["runtime"]["selected_attempts"]),
            }
            historical = _compare_historical(control["rows"], saved)
            checks["historical_prefix_equal"] = historical["equal"]
            write_json(
                output / arm / "equivalence.json", {"checks": checks, "historical": historical}
            )
            if not all(checks.values()):
                raise UgiSamplingTraceError(
                    f"{arm}: observation/replay equivalence failed: {checks}"
                )
            attempts = [
                CommonUgiAttempt(
                    method_id=f"ugi_trace_{arm}",
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
                for row in control["rows"]
            ]
            assessed, assessment = assess_ugi_train_only_attempts(
                attempts, adapter=adapter, training_identities=identities
            )
            write_json(output / arm / "assessed_attempts.json", assessed)
            write_json(output / arm / "assessment.json", assessment)
            results[arm] = {
                "equivalence": checks,
                "historical_comparison": historical,
                "assessment": assessment,
                "trace_summary": observed["trace_summary"],
                "sampling_seconds": {key: value["wall_seconds"] for key, value in arm_runs.items()},
            }
        if implementation_snapshot(repo) != sources:
            raise UgiSamplingTraceError("implementation changed during trace replay")
        _authenticate_record(progress["config"], repo, label="post-run trace config")
        for name, pin in config["inputs"].items():
            resolve_pin(pin, repo, label=f"post-run {name}")
        for name, pin in comparison["inputs"].items():
            resolve_pin(pin, repo, label=f"post-run {name}")
        result = {
            **progress,
            "status": "complete_diagnostic_no_improvement_claim",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {name: pin_record(path, repo) for name, path in inputs.items()},
            "comparison_inputs": {name: pin_record(path, repo) for name, path in paths.items()},
            "source_snapshot": pin_record(output / "source_snapshot.json", repo),
            "runtime": {**native, **config["runtime"]},
            "policy": config["policy"],
            "checkpoint_member": prepared["checkpoint_member"],
            "semantic_support_audit": prepared["support"],
            "arms": results,
            "preparation_seconds": preparation_seconds,
            "wall_seconds": time.perf_counter() - started,
            "artifacts": {
                p.relative_to(output).as_posix(): pin_record(p, repo)
                for p in sorted(output.rglob("*.json"))
                if p.name != "progress.json"
            },
            "nonclaims": [
                "Passive replay is not a new model or realism improvement.",
                "Observed candidate scores are not calibrated whole-molecule probabilities.",
                "Shared seed labels do not prove per-choice arm coupling.",
                "Historical-prefix equality covers the explicitly listed saved fields.",
                "Endpoint-only control records terminal/RNG state after the unchanged computation; it does not wrap choices.",
                "All novelty comparisons use the complete TRAIN identity reference, with failed attempts retained.",
                "Per-attempt timing is unmeasured and encoded as zero; aggregate sampling wall time is reported separately.",
            ],
        }
        write_json(output / "result.json", result)
        write_json(output / "progress.json", {**progress, "status": result["status"]})
        return result
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                **progress,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run_ugi_sampling_trace(args.repo_root, args.config, args.output_dir)["status"])


if __name__ == "__main__":
    main()
