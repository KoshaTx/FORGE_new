"""Run the fixed, oracle TRAIN committed-atom bond-response sensitivity probe."""

from __future__ import annotations

import argparse
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.phase1.multireaction.reaction_specialization import (
    _load_base_package,
    _set_determinism,
)
from experiments.phase1.multireaction.ugi_realism_diagnostic_suite import implementation_snapshot
from experiments.phase1.multireaction.ugi_realism_model_attribution import (
    _tensor_digest,
    _training_objective_audit,
)
from forge.core.hashing import pin_record, resolve_pin, sha256_file, sha256_json
from forge.core.io import iter_csv, read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.reaction_program_flow import noise_synthesis_program_batch
from forge.model.synthesis_program_training import (
    _synthesis_program_predict,
    build_synthesis_program_flow,
    collate_synthesis_program_training_batch,
)
from forge.model.ugi_committed_bond_probe import (
    UgiCommittedBondProbeError,
    oracle_ester_commitment_masks,
    paired_committed_states,
    select_component_covering_panel,
    summarize_bond_response,
)
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_measured_joint_program_prior import (
    program_admitted_by_ester_policy,
    program_from_layout_record,
)

CONFIG_SCHEMA = "forge.ugi_committed_bond_probe_config.v1"
RESULT_SCHEMA = "forge.ugi_committed_bond_probe.v1"


def _validate(config: dict[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "scientific_question",
            "inputs",
            "sources",
            "base_checkpoint",
            "target_program",
            "roles",
            "seed",
            "records",
            "flow_times",
            "cpu_threads",
            "device",
            "precision",
            "repeat_atol",
            "policy",
            "nonclaims",
        }
        or config["schema_version"] != CONFIG_SCHEMA
    ):
        raise UgiCommittedBondProbeError("probe config schema/fields changed")
    if (
        type(config["records"]) is not int
        or config["records"] != 16
        or config["flow_times"] != [0.5, 0.9]
        or config["device"] != "cpu"
        or config["precision"] != "float32"
        or type(config["cpu_threads"]) is not int
        or not 1 <= config["cpu_threads"] <= 4
        or type(config["seed"]) is not int
        or config["seed"] < 0
        or config["repeat_atol"] != 1e-6
        or len(config["roles"]) != 3
        or len(set(config["roles"])) != 3
    ):
        raise UgiCommittedBondProbeError("fixed CPU probe geometry changed")
    if config["policy"] != {
        "training_calls": 0,
        "generation_calls": 0,
        "route_or_oracle_calls": 0,
        "heldout_or_calibration_structure_access": False,
        "candidate_selection": False,
        "promotion_gate_changes": False,
        "source_adjudicated_measured_train_only": True,
        "existing_program_support_required": True,
        "all_admitted_components_required": True,
        "morphology_quantile": 0.25,
    }:
        raise UgiCommittedBondProbeError("probe policy changed")
    if (
        set(config["inputs"])
        != {
            "base_checkpoint_archive",
            "base_training_result",
            "base_training_config",
            "base_design",
            "production_cache",
            "ugi_assignments",
            "qualified_reactions",
            "program_draw",
            "protocol",
        }
        or not config["sources"]
    ):
        raise UgiCommittedBondProbeError("probe input/source contract changed")


def _admitted_panel(
    cache: Any, paths: dict[str, Path], config: dict[str, Any], policy: Any
) -> tuple[list[Any], dict[str, Any]]:
    # Read fold/source metadata first; no cache.record or molecular field is touched for non-TRAIN.
    measured = {}
    masked = 0
    for row in iter_csv(paths["ugi_assignments"]):
        if row["primary_product_fold"] != "train":
            masked += 1
            continue
        if row["is_source_adjudicated_measured_product"] != "true":
            continue
        if row["product_id"] in measured:
            raise UgiCommittedBondProbeError("duplicate measured TRAIN product ID")
        if any(row[f"{role}_family_fold"] != "train" for role in config["roles"]):
            raise UgiCommittedBondProbeError("measured TRAIN component fold disagrees")
        measured[row["product_id"]] = row
    admitted, indices, seen = [], {}, set()
    for raw in cache.indices(program_id=config["target_program"], fold="train"):
        index = int(raw)
        identifier = cache.record_id(index)
        if identifier not in measured:
            continue
        if identifier in seen:
            raise UgiCommittedBondProbeError("duplicate measured TRAIN cache ID")
        seen.add(identifier)
        if cache.canonical_smiles(index) != measured[identifier]["canonical_product_smiles"]:
            raise UgiCommittedBondProbeError("TRAIN assignment/cache identity differs")
        program = program_from_layout_record(cache.record(index), vocabulary=cache.vocabulary)
        if program_admitted_by_ester_policy(program, policy):
            admitted.append(measured[identifier])
            indices[identifier] = index
    if seen != set(measured):
        raise UgiCommittedBondProbeError("cache lacks measured TRAIN products")
    selected, selection = select_component_covering_panel(
        admitted, roles=config["roles"], records=config["records"], seed=config["seed"]
    )
    selected_indices = [indices[row["product_id"]] for row in selected]
    selection.update(
        {
            "nontrain_rows_masked_before_structure_access": masked,
            "measured_train_products": len(measured),
            "admitted_products": len(admitted),
            "selected_cache_indices": selected_indices,
            "selected_cache_indices_sha256": str(sha256_json(selected_indices)),
        }
    )
    return cache.records(selected_indices), selection


def run_ugi_committed_bond_probe(
    repo_root: Path, config_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Authenticate inputs, score one fixed panel, and retain any failure receipt."""

    repo = repo_root.resolve()
    config_path = (repo / config_path).resolve()
    output = (repo / output_dir).resolve()
    if not output.is_relative_to(repo) or output.exists():
        raise UgiCommittedBondProbeError(
            "probe requires a fresh output directory inside repository"
        )
    started = time.perf_counter()
    output.mkdir(parents=True)
    config_pin = pin_record(config_path, repo)
    try:
        config = read_json_object(
            config_path, error=UgiCommittedBondProbeError, label="probe config"
        )
        _validate(config)
        paths = {key: resolve_pin(pin, repo, label=key) for key, pin in config["inputs"].items()}
        for key, pin in config["sources"].items():
            resolve_pin(pin, repo, label=key)
        sources = implementation_snapshot(repo)
        write_json(output / "source_snapshot.json", sources)
        draw = read_json_object(
            paths["program_draw"], error=UgiCommittedBondProbeError, label="draw"
        )
        if draw["reaction_id"] != config["target_program"]:
            raise UgiCommittedBondProbeError("program draw reaction differs")
        for name in ("production_cache", "ugi_assignments", "qualified_reactions"):
            if draw["inputs"][name]["sha256"] != config["inputs"][name]["sha256"]:
                raise UgiCommittedBondProbeError(f"program draw input differs: {name}")
        policy = UgiEsterChemotypePolicy.from_qualified_registry(
            paths["qualified_reactions"],
            training_assignments_path=paths["ugi_assignments"],
            expected_sha256=config["inputs"]["qualified_reactions"]["sha256"],
            expected_training_assignments_sha256=config["inputs"]["ugi_assignments"]["sha256"],
            reaction_id=config["target_program"],
            morphology_quantile=config["policy"]["morphology_quantile"],
        )
        training = read_json_object(
            paths["base_training_result"], error=UgiCommittedBondProbeError, label="training"
        )
        training_config = read_json_object(
            paths["base_training_config"], error=UgiCommittedBondProbeError, label="training config"
        )
        design = read_json_object(
            paths["base_design"], error=UgiCommittedBondProbeError, label="design"
        )
        if {key: training["config"][key] for key in ("path", "sha256")} != config["inputs"][
            "base_training_config"
        ]:
            raise UgiCommittedBondProbeError("training config provenance differs")
        device = torch.device("cpu")
        _set_determinism(config["seed"], config["cpu_threads"], device)
        base = config["base_checkpoint"]
        package, member = _load_base_package(
            archive_path=paths["base_checkpoint_archive"],
            training=training,
            arm_id=base["arm_id"],
            step=base["step"],
            expected_member_sha256=base["member_sha256"],
            expected_model_state_sha256=base["model_state_sha256"],
            expected_design_sha256=str(sha256_file(paths["base_design"])),
            expected_cache_sha256=str(sha256_file(paths["production_cache"])),
            device=device,
        )
        cache = SynthesisProgramProductionCache(paths["production_cache"])
        try:
            records, selection = _admitted_panel(cache, paths, config, policy)
            clean = collate_synthesis_program_training_batch(
                records,
                maximum_closures=package["model_config"]["maximum_closures"],
                conditioning="program",
                vocabulary=cache.vocabulary,
            )
            commitments, commitment_receipt = oracle_ester_commitment_masks(
                clean,
                records,
                cache.atom_vocabulary,
                policy,
                bond_classes=package["model_config"]["bond_classes"],
            )
            # The panel and oracle masks are fixed on disk before the first prediction.
            write_json(output / "selection.json", selection)
            write_json(
                output / "commitments.json",
                {"rows": commitment_receipt, "mask_sha256": _tensor_digest(commitments)},
            )
            if selection["admitted_products"] != draw["prior_audit"]["eligible_rows"]:
                raise UgiCommittedBondProbeError("admitted census differs from frozen draw")
            objective = _training_objective_audit(
                package=package,
                training=training,
                design=design,
                training_config=training_config,
                arm_id=base["arm_id"],
                program_state=cache.vocabulary.program_to_index[config["target_program"]],
            )
            roles = {role: cache.vocabulary.role_to_index[role] for role in config["roles"]}
            model = build_synthesis_program_flow(
                vocabulary=cache.vocabulary,
                node_classes=len(cache.atom_vocabulary),
                model_config=package["model_config"],
                device=device,
            )
            model.load_state_dict(package["model_state"], strict=True)
            model.eval()
            if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
                raise UgiCommittedBondProbeError("checkpoint is not float32")
            time_results = {}
            with torch.inference_mode():
                for offset, flow_time in enumerate(config["flow_times"]):
                    seed = config["seed"] + 1000 + offset
                    t = torch.full((len(records),), flow_time, dtype=torch.float32)
                    noisy = noise_synthesis_program_batch(
                        clean,
                        torch.as_tensor(package["node_marginal"], dtype=torch.float32),
                        torch.as_tensor(package["bond_marginal"], dtype=torch.float32),
                        t,
                        torch.Generator(device="cpu").manual_seed(seed),
                    )
                    stale_input, refreshed_input = paired_committed_states(
                        clean, noisy, commitments
                    )
                    stale = _synthesis_program_predict(model, clean, stale_input, t)
                    repeat = _synthesis_program_predict(model, clean, stale_input, t)
                    refreshed = _synthesis_program_predict(model, clean, refreshed_input, t)
                    metrics = summarize_bond_response(
                        clean=clean,
                        noisy=noisy,
                        commitments=commitments,
                        stale=stale,
                        repeat=repeat,
                        refreshed=refreshed,
                        roles=roles,
                        repeat_atol=config["repeat_atol"],
                    )
                    time_results[str(flow_time)] = {
                        "seed": seed,
                        "noisy_state_sha256": _tensor_digest(noisy),
                        "stale_input_sha256": _tensor_digest(stale_input),
                        "refreshed_input_sha256": _tensor_digest(refreshed_input),
                        "changed_atom_coordinates": int(
                            (stale_input["nodes"] != refreshed_input["nodes"]).sum()
                        ),
                        "changed_committed_bond_coordinates": {
                            field: int((stale_input[field] != refreshed_input[field]).sum())
                            for field in commitments
                        },
                        "evaluated_bond_input_changes": 0,
                        **metrics,
                    }
                    # Raw TRAIN tensor/probability evidence permits coordinate-level replay audits.
                    payload = {
                        f"{name}.{field}": value.detach().cpu().numpy()
                        for name, state in (
                            ("clean", clean),
                            ("noisy", noisy),
                            ("stale_input", stale_input),
                            ("refreshed_input", refreshed_input),
                            ("stale", stale),
                            ("repeat", repeat),
                            ("refreshed", refreshed),
                        )
                        for field, value in state.items()
                        if torch.is_tensor(value)
                    }
                    np.savez_compressed(output / f"flow_{flow_time}.npz", **payload)
        finally:
            cache.close()
        if sources != implementation_snapshot(repo):
            raise UgiCommittedBondProbeError("implementation changed during probe")
        for key, pin in {**config["inputs"], **config["sources"]}.items():
            resolve_pin(pin, repo, label=key)
        if config_pin != pin_record(config_path, repo):
            raise UgiCommittedBondProbeError("config changed during probe")
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete_oracle_train_sensitivity_only",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config_pin,
            "inputs": config["inputs"],
            "sources": config["sources"],
            "implementation_snapshot_sha256": str(sha256_json(sources)),
            "base_checkpoint": {**base, "member_name": member},
            "selection": selection,
            "training_objective": objective,
            "ester_policy": policy.to_mapping(),
            "results_by_flow_time": time_results,
            "mechanism_gate_passed": all(row["gate_passed"] for row in time_results.values()),
            "policy": config["policy"],
            "neural_forward_calls": 3 * len(config["flow_times"]),
            "runtime": {
                "device": "cpu",
                "precision": "float32",
                "cpu_threads": config["cpu_threads"],
                "seed": config["seed"],
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "seconds": time.perf_counter() - started,
            },
            "nonclaims": config["nonclaims"],
            "artifacts": {
                path.name: pin_record(path, repo)
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        }
        write_json(output / "result.json", result)
        return result
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "schema_version": RESULT_SCHEMA,
                "status": "failed_no_admitted_metrics",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": config_pin,
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
    print(run_ugi_committed_bond_probe(args.repo_root, args.config, args.output_dir)["status"])


if __name__ == "__main__":
    main()
