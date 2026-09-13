"""Run a bounded measured-train checkpoint, topology and probability-law diagnostic."""

from __future__ import annotations

import argparse
import inspect
import math
import platform
import sys
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
from forge.core.hashing import pin_record, resolve_pin, sha256_file, sha256_json
from forge.core.io import iter_csv, read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.reaction_program_flow import ROLE_MORPHOLOGY_FIELDS
from forge.model.reaction_program_transformer import synthesis_program_offspring_targets
from forge.model.sparse_topology_feasibility import (
    _endpoint_candidate_mask,
    _parent_candidate_mask,
)
from forge.model.synthesis_program_training import (
    build_synthesis_program_flow,
    collate_synthesis_program_training_batch,
    synthesis_program_forward,
    synthesis_program_topology_conditioned_forward,
)
from forge.model.ugi_realism_model_attribution import (
    UgiRealismModelAttributionError,
    coordinate_summaries,
    select_measured_train_rows,
)

CONFIG_SCHEMA = "forge.ugi_realism_model_attribution_config.v1"
RESULT_SCHEMA = "forge.ugi_realism_model_attribution.v1"


def _implementation_manifest(repo: Path) -> dict[str, Any]:
    paths = {
        Path(module.__file__).resolve()
        for name, module in tuple(sys.modules.items())
        if (name.startswith("forge.") or name.startswith("experiments."))
        and getattr(module, "__file__", None)
    }
    paths.add(Path(__file__).resolve())
    return {str(path.relative_to(repo)): pin_record(path, repo) for path in sorted(paths)}


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "scientific_question",
        "target_program",
        "roles",
        "seed",
        "records",
        "batch_size",
        "flow_times",
        "device",
        "precision",
        "cpu_threads",
        "inputs",
        "sources",
        "base_checkpoint",
        "policy",
        "nonclaims",
    }
    if set(config) != required or config["schema_version"] != CONFIG_SCHEMA:
        raise UgiRealismModelAttributionError("attribution config fields/schema changed")
    if (
        config["target_program"] != "ugi_3cr_agile"
        or config["device"] != "cpu"
        or config["precision"] != "float32"
        or type(config["records"]) is not int
        or not 1 <= config["records"] <= config["batch_size"] <= 16
        or type(config["cpu_threads"]) is not int
        or not 1 <= config["cpu_threads"] <= 4
        or type(config["seed"]) is not int
        or not config["flow_times"]
        or len(config["flow_times"]) > 3
        or len(set(config["flow_times"])) != len(config["flow_times"])
        or any(not 0 < value < 1 for value in config["flow_times"])
        or len(config["roles"]) != 3
        or len(set(config["roles"])) != 3
    ):
        raise UgiRealismModelAttributionError("CPU attribution geometry/scope changed")
    if config["policy"] != {
        "candidate_selection": False,
        "training_calls": 0,
        "generation_calls": 0,
        "route_or_oracle_calls": 0,
        "heldout_or_calibration_structure_access": False,
        "biological_conditioning": False,
    }:
        raise UgiRealismModelAttributionError("attribution policy changed")
    if (
        set(config["inputs"])
        != {
            "base_checkpoint_archive",
            "base_training_result",
            "base_training_config",
            "base_design",
            "production_cache",
            "ugi_assignments",
            "decoder_comparison_config",
        }
        or not config["sources"]
    ):
        raise UgiRealismModelAttributionError("attribution input/source pins changed")


def _numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy()


def _training_objective_audit(
    *,
    package: dict[str, Any],
    training: dict[str, Any],
    design: dict[str, Any],
    training_config: dict[str, Any],
    arm_id: str,
    program_state: int,
) -> dict[str, Any]:
    """Distinguish the effective checkpoint objective from its base design snapshot."""

    objective = package["model_config"].get("semantic_objective", {})
    field = "topology_conditioned_chemistry_weight"
    weight = objective.get(field)
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (float, int))
        or not math.isfinite(weight)
    ):
        raise UgiRealismModelAttributionError("checkpoint lacks a finite topology-chemistry weight")
    declared = training_config.get("intervention", {}).get(
        "topology_conditioned_chemistry_second_pass"
    )
    if not isinstance(declared, bool) or declared != (weight > 0):
        raise UgiRealismModelAttributionError(
            "training second-pass declaration/checkpoint disagree"
        )
    loss_field = f"program_{program_state}_topology_conditioned_chemistry_ce"
    loss = training["arms"][arm_id]["final_loss"].get(loss_field)
    if declared and (
        isinstance(loss, bool)
        or not isinstance(loss, (float, int))
        or not math.isfinite(loss)
        or loss < 0
    ):
        raise UgiRealismModelAttributionError(
            "enabled training objective has no valid loss receipt"
        )
    return {
        "effective_semantic_objective": objective,
        "topology_conditioned_chemistry_weight": weight,
        "second_pass_declared_in_training_config": declared,
        "final_ugi_topology_conditioned_chemistry_ce": loss,
        "evidence_fields": {
            "checkpoint_member": f"model_config.semantic_objective.{field}",
            "training_config": "intervention.topology_conditioned_chemistry_second_pass",
            "training_result": f"arms.{arm_id}.final_loss.{loss_field}",
            "base_design": f"model.semantic_objective.{field}",
            "training_result_base_model_snapshot": f"model.semantic_objective.{field}",
        },
        "base_design_snapshot_weight": design["model"].get("semantic_objective", {}).get(field),
        "training_result_base_model_snapshot_weight": training["model"]
        .get("semantic_objective", {})
        .get(field),
        "interpretation": (
            "The authenticated checkpoint records the effective mechanism-arm objective. Base "
            "design and training-result model snapshots can omit arm overrides; absence there "
            "does not mean the second pass was absent. The final loss is historical training "
            "evidence, not a newly measured holdout or generation metric."
        ),
    }


def _tensor_digest(states: dict[str, Any]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key, value in sorted(states.items()):
        if torch.is_tensor(value):
            array = _numpy(value)
            digest.update(str((key, str(array.dtype), array.shape)).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _summarize_predictions(
    predictions: dict[str, Any],
    clean: dict[str, Any],
    noisy: dict[str, Any],
    roles: dict[str, int],
    *,
    include_topology: bool,
    guidance: dict[str, Any],
) -> dict[str, Any]:
    exterior = clean["node_mask"] & (clean["core_position_states"] == 1)
    fields: dict[str, tuple[Any, Any, Any, Any, Any]] = {}
    for field, mask in (
        ("nodes", "atom_variable_mask"),
        ("parent_bonds", "parent_bond_variable_mask"),
    ):
        fields[field] = (
            clean[field],
            noisy[field],
            exterior & clean[mask],
            clean["role_states"],
            torch.ones_like(predictions[field], dtype=torch.bool),
        )
    if include_topology:
        fields["parents"] = (
            clean["parents"],
            noisy["parents"],
            exterior & clean["parent_variable_mask"],
            clean["role_states"],
            _parent_candidate_mask(clean["node_mask"]),
        )
        if "offspring" in predictions:
            maximum_children = predictions["offspring"].shape[-1] - 1
            target, mask = synthesis_program_offspring_targets(
                clean, maximum_children=maximum_children
            )
            # Noisy topology may exceed the training offspring support; the diagnostic compares
            # observed child counts without truncation. No prediction targets are changed.
            noisy_target, _ = synthesis_program_offspring_targets(
                {**clean, "parents": noisy["parents"]}, maximum_children=clean["nodes"].shape[1]
            )
            fields["offspring"] = (
                target,
                noisy_target,
                mask,
                clean["role_states"],
                torch.ones_like(predictions["offspring"], dtype=torch.bool),
            )
    closure_roles = clean["role_states"].gather(1, clean["closure_left"])
    same_closure_role = closure_roles == clean["role_states"].gather(1, clean["closure_right"])
    exterior_closure = (
        (clean["core_position_states"].gather(1, clean["closure_left"]) == 1)
        & (clean["core_position_states"].gather(1, clean["closure_right"]) == 1)
        & same_closure_role
    )
    closure_fields = (
        ("closure_left", "closure_right", "closure_bonds")
        if include_topology
        else ("closure_bonds",)
    )
    for field in closure_fields:
        bond = field == "closure_bonds"
        variable = clean["closure_bond_variable_mask" if bond else "closure_endpoint_variable_mask"]
        if torch.any(variable & ~exterior_closure):
            raise UgiRealismModelAttributionError(
                "variable closure outside a reported exterior role"
            )
        fields[field] = (
            clean[field],
            noisy[field],
            variable & exterior_closure,
            closure_roles,
            (
                torch.ones_like(predictions[field], dtype=torch.bool)
                if bond
                else _endpoint_candidate_mask(clean["node_mask"], clean["closure_left"].shape[1])
            ),
        )
    result = {}
    for field, (target, corrupted, active, role_states, allowed) in fields.items():
        rank_law = None
        if field in {"nodes", "parent_bonds"}:
            rank_law = {
                "model_rank_weight": float(guidance["model_rank_weight"]),
                "rank_temperature": float(guidance["rank_temperature"]),
                "uniform_probability_mass": float(
                    guidance[
                        (
                            "uniform_probability_mass"
                            if field == "nodes"
                            else "local_chemistry_bond_uniform_probability_mass"
                        )
                    ]
                ),
            }
        result[field] = coordinate_summaries(
            logits=_numpy(predictions[field]),
            targets=_numpy(target),
            noisy=_numpy(corrupted),
            active_mask=_numpy(active),
            role_states=_numpy(role_states),
            roles=roles,
            allowed=_numpy(allowed),
            rank_law=rank_law,
        )
    return result


def run_ugi_realism_model_attribution(
    repo_root: Path, config_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Authenticate inputs and run exactly one small measured-train batch at fixed times."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    config = read_json_object(
        config_path, error=UgiRealismModelAttributionError, label="model attribution config"
    )
    _validate_config(config)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise UgiRealismModelAttributionError(f"output directory is not empty: {output_dir}")
    paths = {
        name: resolve_pin(pin, repo_root, label=name) for name, pin in config["inputs"].items()
    }
    source_paths = {
        name: resolve_pin(pin, repo_root, label=name) for name, pin in config["sources"].items()
    }
    config_pin = pin_record(config_path, repo_root)
    decoder = read_json_object(
        paths["decoder_comparison_config"],
        error=UgiRealismModelAttributionError,
        label="frozen decoder comparison",
    )
    for key in (
        "base_checkpoint_archive",
        "base_training_result",
        "base_design",
        "production_cache",
    ):
        if decoder["inputs"][key] != config["inputs"][key]:
            raise UgiRealismModelAttributionError(f"decoder/checkpoint provenance disagrees: {key}")
    rows, selection = select_measured_train_rows(
        iter_csv(paths["ugi_assignments"]),
        roles=config["roles"],
        records=config["records"],
        seed=config["seed"],
    )
    training = read_json_object(
        paths["base_training_result"],
        error=UgiRealismModelAttributionError,
        label="base training result",
    )
    training_config = read_json_object(
        paths["base_training_config"],
        error=UgiRealismModelAttributionError,
        label="base training configuration",
    )
    if {key: training["config"][key] for key in ("path", "sha256")} != config["inputs"][
        "base_training_config"
    ]:
        raise UgiRealismModelAttributionError("training result/configuration pin disagrees")
    design = read_json_object(
        paths["base_design"], error=UgiRealismModelAttributionError, label="base design"
    )
    device = torch.device("cpu")
    _set_determinism(config["seed"], config["cpu_threads"], device)
    started = time.perf_counter()
    base = config["base_checkpoint"]
    package, member_name = _load_base_package(
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
        train_indices = cache.indices(program_id=config["target_program"], fold="train")
        train_by_id = {cache.record_id(int(index)): int(index) for index in train_indices}
        if len(train_by_id) != len(train_indices):
            raise UgiRealismModelAttributionError("duplicate Ugi train IDs in cache")
        try:
            selected = [train_by_id[row["product_id"]] for row in rows]
        except KeyError as error:
            raise UgiRealismModelAttributionError(
                f"measured train ID absent from cache: {error}"
            ) from error
        for row, index in zip(rows, selected, strict=True):
            if cache.canonical_smiles(index) != row["canonical_product_smiles"]:
                raise UgiRealismModelAttributionError(
                    f"cache train structure disagrees: {row['product_id']}"
                )
        selection["selected_cache_row_indices"] = selected
        selection["selected_cache_row_indices_sha256"] = str(sha256_json(selected))
        records = cache.records(selected)
        clean = collate_synthesis_program_training_batch(
            records,
            maximum_closures=int(package["model_config"]["maximum_closures"]),
            conditioning="program",
            vocabulary=cache.vocabulary,
        )
        roles = {role: cache.vocabulary.role_to_index[role] for role in config["roles"]}
        training_objective = _training_objective_audit(
            package=package,
            training=training,
            design=design,
            training_config=training_config,
            arm_id=base["arm_id"],
            program_state=cache.vocabulary.program_to_index[config["target_program"]],
        )
        training_objective["provenance"] = {
            "checkpoint_archive": pin_record(paths["base_checkpoint_archive"], repo_root),
            "checkpoint_member_name": member_name,
            "checkpoint_member_sha256": base["member_sha256"],
            "training_config": pin_record(paths["base_training_config"], repo_root),
            "training_result": pin_record(paths["base_training_result"], repo_root),
            "base_design": pin_record(paths["base_design"], repo_root),
        }
        model = build_synthesis_program_flow(
            vocabulary=cache.vocabulary,
            node_classes=len(cache.atom_vocabulary),
            model_config=package["model_config"],
            device=device,
        )
        model.load_state_dict(package["model_state"], strict=True)
        model.eval()
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise UgiRealismModelAttributionError("checkpoint parameters are not float32")
        node_p0 = torch.as_tensor(package["node_marginal"], dtype=torch.float32)
        bond_p0 = torch.as_tensor(package["bond_marginal"], dtype=torch.float32)
        conditioning = {
            "model_class": type(model).__name__,
            "forward_parameters": list(inspect.signature(model.forward).parameters),
            "role_morphology_fields": list(ROLE_MORPHOLOGY_FIELDS),
            "checkpoint_model_config": package["model_config"],
            "training_objective": training_objective,
            "role_morphology_tensor_shape": list(clean["role_morphology_states"].shape),
            "biological_condition": None,
            "evidence_kind": "source_signature_and_checkpoint_config_not_causal_test",
            "interpretation": (
                "Detailed decoder head composition, ester-arm and unsaturation targets are absent "
                "from this forward interface. Noisy whole-graph chemistry still supplies chemical "
                "context; this audit does not imply independent learned coordinates."
            ),
        }
        sources_before = _implementation_manifest(repo_root)
        time_results = {}
        with torch.inference_mode():
            for time_index, flow_time in enumerate(config["flow_times"]):
                generator_seed = config["seed"] + 1000 + time_index
                t = torch.full((len(selected),), flow_time, dtype=torch.float32)
                joint, noisy = synthesis_program_forward(
                    model,
                    clean,
                    node_p0,
                    bond_p0,
                    t,
                    torch.Generator(device="cpu").manual_seed(generator_seed),
                )
                teacher = synthesis_program_topology_conditioned_forward(model, clean, noisy, t)
                for field, mask in (
                    ("nodes", "atom_variable_mask"),
                    ("parents", "parent_variable_mask"),
                    ("parent_bonds", "parent_bond_variable_mask"),
                    ("closure_left", "closure_endpoint_variable_mask"),
                    ("closure_right", "closure_endpoint_variable_mask"),
                    ("closure_bonds", "closure_bond_variable_mask"),
                ):
                    if torch.any((noisy[field] != clean[field]) & ~clean[mask]):
                        raise UgiRealismModelAttributionError(
                            f"fixed-state noising violation: {field}"
                        )
                time_results[str(flow_time)] = {
                    "generator_seed": generator_seed,
                    "noisy_state_sha256": _tensor_digest(noisy),
                    "fixed_state_corruption_count": 0,
                    "joint_noisy_state": _summarize_predictions(
                        joint,
                        clean,
                        noisy,
                        roles,
                        include_topology=True,
                        guidance=decoder["semantic_guidance"],
                    ),
                    "target_topology_noisy_chemistry": _summarize_predictions(
                        teacher,
                        clean,
                        noisy,
                        roles,
                        include_topology=False,
                        guidance=decoder["semantic_guidance"],
                    ),
                }
        sources_after = _implementation_manifest(repo_root)
        for name, pin in sources_before.items():
            if sources_after.get(name) != pin:
                raise UgiRealismModelAttributionError(f"implementation changed during run: {name}")
        for name, pin in config["sources"].items():
            resolve_pin(pin, repo_root, label=name)
        if pin_record(config_path, repo_root) != config_pin:
            raise UgiRealismModelAttributionError("config changed during run")
        support = {
            "selected_node_counts": [record.node_count for record in records],
            "maximum_heavy_atoms": package["model_config"]["maximum_heavy_atoms"],
            "atom_vocabulary_classes": len(cache.atom_vocabulary),
            "bond_classes": package["model_config"]["bond_classes"],
            "closure_slots": package["model_config"]["maximum_closures"],
        }
    finally:
        cache.close()
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "completed_diagnostic_no_promotion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_question": config["scientific_question"],
        "config": config_pin,
        "inputs": {name: pin_record(path, repo_root) for name, path in sorted(paths.items())},
        "sources": {
            name: pin_record(path, repo_root) for name, path in sorted(source_paths.items())
        },
        "implementation_manifest": sources_after,
        "implementation_manifest_sha256": str(sha256_json(sources_after)),
        "base_checkpoint": {**base, "member_name": member_name},
        "selection": selection,
        "conditioning_audit": conditioning,
        "support": support,
        "sampling_contract": {
            "seed": config["seed"],
            "device": "cpu",
            "precision": "float32",
            "cpu_threads": config["cpu_threads"],
            "batch_size": config["batch_size"],
            "records": config["records"],
            "flow_times": config["flow_times"],
            "deterministic_algorithms": True,
            "training_rows_only": True,
            "noise": "existing_source_marginal_interpolation_fixed_adapter_states",
            "paired_teacher_forcing": "same_noisy_chemistry_replace_parent_and_closure_endpoints",
            "actual_molecular_rollouts": 0,
            "neural_forward_calls": 2 * len(config["flow_times"]),
            "probability_law_probe": (
                "off_policy_categorical_training_support_equal_non_neural_scores; "
                "not_complete_feasible_terminal_arrangements_or_actual_sampler_probabilities"
            ),
            "metrics": (
                "Variable exterior coordinates only; parental candidates use the training-loss "
                "mask; roles use clean origins; calibration is descriptive train-panel evidence."
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "results_by_flow_time": time_results,
        "policy": config["policy"],
        "nonclaims": config["nonclaims"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_ugi_realism_model_attribution(args.repo_root, args.config, args.output_dir)
    print(result["status"])


if __name__ == "__main__":
    main()
