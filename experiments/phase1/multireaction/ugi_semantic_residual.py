"""Fit a bounded pair of amine residual heads on frozen TRAIN-only features."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from experiments.phase1.multireaction.reaction_specialization import (
    _load_base_package,
    _set_determinism,
)
from experiments.phase1.multireaction.ugi_realism_diagnostic_suite import implementation_snapshot
from experiments.phase1.multireaction.ugi_realism_model_attribution import _tensor_digest
from forge.core.hashing import pin_record, resolve_pin, sha256_file, sha256_json
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.reaction_program_flow import noise_synthesis_program_batch
from forge.model.synthesis_program_training import (
    _synthesis_program_predict,
    build_synthesis_program_flow,
    collate_synthesis_program_training_batch,
)
from forge.model.ugi_amine_semantic_program import UgiAmineSemanticTarget
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_semantic_placement import (
    baseline_head_placement_candidates,
    compare_candidate_universes,
)
from forge.model.ugi_semantic_residual import UgiSemanticResidualHead, apply_semantic_residual


class UgiSemanticResidualStudyError(ValueError):
    """A matched residual experiment violates its frozen evidence contract."""


def _read(path: Path) -> dict[str, Any]:
    return read_json_object(path, error=UgiSemanticResidualStudyError)


def _bucket(value: str, domain: str) -> int:
    return int(hashlib.sha256((domain + value).encode()).hexdigest(), 16) % 5


def _select_draws(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Apply fixed product subdivision, preserving original relative source weights."""
    groups: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("fit", "component_disjoint", "repeated_component")
    }
    for row in rows:
        if type(row["applicable"]) is not bool or row["component_partition"] not in {
            "fit",
            "evaluation",
        }:
            raise UgiSemanticResidualStudyError("coverage applicability/partition is malformed")
        if not row["applicable"]:
            continue
        if row["component_partition"] == "evaluation":
            name = "component_disjoint"
        elif _bucket(row["product_id"], config["partition"]["product_domain"]) == 0:
            name = "repeated_component"
        else:
            name = "fit"
        groups[name].append(row)
    if any(not rows for rows in groups.values()):
        raise UgiSemanticResidualStudyError("fixed subdivision has an empty population")
    fit_heads = {row["amine_smiles"] for row in groups["fit"]}
    disjoint_heads = {row["amine_smiles"] for row in groups["component_disjoint"]}
    if fit_heads & disjoint_heads:
        raise UgiSemanticResidualStudyError("amine identity leaked across component split")
    rng = np.random.default_rng(config["seeds"]["draws"])
    output: dict[str, Any] = {}
    for name, population in groups.items():
        weights = np.asarray([row["source_weight"] for row in population], dtype=np.float64)
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise UgiSemanticResidualStudyError("invalid source measure")
        choices = rng.choice(
            len(population), size=config["draws"][name], replace=True, p=weights / weights.sum()
        )
        selected = [population[int(index)] for index in choices]
        output[name] = {
            "population_rows": len(population),
            "source_weight_sum": float(weights.sum()),
            "population_components": len({row["amine_smiles"] for row in population}),
            "sampled_components": len({row["amine_smiles"] for row in selected}),
            "sampled_unique_products": len({row["product_id"] for row in selected}),
            "heads_absent_from_fit_population": sorted(
                {row["amine_smiles"] for row in population} - fit_heads
            ),
            "draws": selected,
        }
    sampled_fit_heads = {row["amine_smiles"] for row in output["fit"]["draws"]}
    for group in output.values():
        group["draws_with_head_seen_in_actual_fit_draws"] = sum(
            row["amine_smiles"] in sampled_fit_heads for row in group["draws"]
        )
        group["sampled_heads_absent_from_actual_fit_draws"] = sorted(
            {row["amine_smiles"] for row in group["draws"]} - sampled_fit_heads
        )
    return output


def _extract_features(
    *,
    model: Any,
    package: dict[str, Any],
    cache: Any,
    selection: dict[str, Any],
    config: dict[str, Any],
    output: Path,
) -> tuple[dict[str, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    """Extract once, pack only learnable head positions, keep every selected draw/time."""
    features, ledger = {}, []
    amine_role = cache.vocabulary.role_to_index[config["amine_role"]]
    total_calls = 0
    for group, selected in selection.items():
        rows = selected["draws"]
        records = cache.records([row["cache_index"] for row in rows])
        positions = [
            np.flatnonzero(
                (record.role_states == amine_role) & ~record.fixed_atom_mask.astype(bool)
            )
            for record in records
        ]
        if any(
            np.any(record.core_position_states[pos] != 1)
            for record, pos in zip(records, positions, strict=True)
        ):
            raise UgiSemanticResidualStudyError("variable amine mask includes non-exterior atoms")
        width = max(map(len, positions))
        if min(map(len, positions)) < 1:
            raise UgiSemanticResidualStudyError("selected product lacks variable amine atoms")
        chunks: dict[str, list[torch.Tensor]] = {}
        metadata = []
        for time_index, flow_time in enumerate(config["flow_times"]):
            for start in range(0, len(rows), config["feature_batch_size"]):
                stop = min(len(rows), start + config["feature_batch_size"])
                batch_records = records[start:stop]
                clean = collate_synthesis_program_training_batch(
                    batch_records,
                    maximum_closures=package["model_config"]["maximum_closures"],
                    conditioning="program",
                    vocabulary=cache.vocabulary,
                )
                t = torch.full((stop - start,), flow_time, dtype=torch.float32)
                seed = config["seeds"]["noise"] + total_calls
                noisy = noise_synthesis_program_batch(
                    clean,
                    torch.as_tensor(package["node_marginal"], dtype=torch.float32),
                    torch.as_tensor(package["bond_marginal"], dtype=torch.float32),
                    t,
                    torch.Generator().manual_seed(seed),
                )
                conditioned = dict(noisy)
                for field in ("parents", "closure_left", "closure_right"):
                    conditioned[field] = clean[field]
                with torch.no_grad():
                    predictions = _synthesis_program_predict(
                        model, clean, conditioned, t, return_hidden_state=True
                    )
                packed: dict[str, torch.Tensor] = {
                    "hidden": torch.zeros(
                        stop - start, width, predictions["hidden_state"].shape[-1]
                    ),
                    "logits": torch.zeros(stop - start, width, predictions["nodes"].shape[-1]),
                    "mask": torch.zeros(stop - start, width, dtype=torch.bool),
                    "target": torch.zeros(stop - start, width, dtype=torch.long),
                    "noisy": torch.zeros(stop - start, width, dtype=torch.long),
                    "positions": torch.full((stop - start, width), -1, dtype=torch.long),
                    "t": t,
                    "semantics": torch.tensor(
                        [row["semantic_target"] for row in rows[start:stop]], dtype=torch.float32
                    ),
                }
                for local, global_index in enumerate(range(start, stop)):
                    pos = torch.as_tensor(positions[global_index])
                    n = len(pos)
                    packed["mask"][local, :n] = True
                    packed["positions"][local, :n] = pos
                    for name, value in (
                        ("hidden", predictions["hidden_state"]),
                        ("logits", predictions["nodes"]),
                        ("target", clean["nodes"]),
                        ("noisy", noisy["nodes"]),
                    ):
                        packed[name][local, :n] = value[local, pos].detach()
                    metadata.append(
                        {
                            "draw_index": global_index,
                            "flow_time": flow_time,
                            "time_index": time_index,
                        }
                    )
                for key, value in packed.items():
                    chunks.setdefault(key, []).append(value)
                ledger.append(
                    {
                        "group": group,
                        "start": start,
                        "stop": stop,
                        "flow_time": flow_time,
                        "noise_seed": seed,
                        "clean_sha256": _tensor_digest(clean),
                        "noisy_sha256": _tensor_digest(noisy),
                        "conditioned_sha256": _tensor_digest(conditioned),
                        "predictions_sha256": _tensor_digest(predictions),
                    }
                )
                total_calls += 1
        features[group] = {key: torch.cat(value, dim=0) for key, value in chunks.items()}
        np.savez_compressed(
            output / f"features_{group}.npz",
            **{key: value.numpy() for key, value in features[group].items()},
        )
        write_json(output / f"features_{group}_rows.json", {"rows": metadata})
    write_json(output / "feature_batches.json", {"rows": ledger, "forward_calls": total_calls})
    return features, ledger


def _head_logits(head: Any, data: dict[str, torch.Tensor]) -> torch.Tensor:
    return apply_semantic_residual(
        {"nodes": data["logits"]},
        head,
        hidden=data["hidden"],
        flow_time=data["t"],
        amine_semantics=data["semantics"],
        variable_amine_mask=data["mask"],
    )["nodes"]


def _fit_heads(
    data: dict[str, torch.Tensor], config: dict[str, Any], output: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    n = len(data["t"])
    if n != config["optimizer_steps"] * config["training_batch_size"]:
        raise UgiSemanticResidualStudyError("fixed exposure does not cover all features once")
    order = torch.randperm(n, generator=torch.Generator().manual_seed(config["seeds"]["order"]))
    write_json(output / "exposure_order.json", {"feature_indices": order.tolist()})
    heads, receipts = {}, {}
    initial = None
    for arm in ("control", "semantic"):
        head = UgiSemanticResidualHead(
            data["hidden"].shape[-1],
            data["logits"].shape[-1],
            semantic_scales=config["semantic_scales"],
            initialization_seed=config["seeds"]["initialization"],
            use_semantics=arm == "semantic",
        )
        if initial is None:
            initial = copy.deepcopy(head.state_dict())
            torch.save(initial, output / "initial_head.pt")
        elif any(not torch.equal(initial[key], value) for key, value in head.state_dict().items()):
            raise UgiSemanticResidualStudyError("matched head initialization differs")
        with torch.no_grad():
            if not torch.equal(_head_logits(head, data), data["logits"]):
                raise UgiSemanticResidualStudyError("zero-initialized head changes baseline")
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        losses = []
        for start in range(0, n, config["training_batch_size"]):
            indices = order[start : start + config["training_batch_size"]]
            batch = {key: value[indices] for key, value in data.items()}
            logits = _head_logits(head, batch)
            coordinate_loss = functional.cross_entropy(
                logits.transpose(1, 2), batch["target"], reduction="none"
            )
            loss = ((coordinate_loss * batch["mask"]).sum(1) / batch["mask"].sum(1)).mean()
            if not torch.isfinite(loss):
                raise UgiSemanticResidualStudyError("nonfinite head training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            losses.append(float(loss.detach()))
        head.eval()
        heads[arm] = head
        receipts[arm] = {
            "losses": losses,
            "updates": len(losses),
            "examples": n,
            "parameters": head.trainable_parameter_receipt(),
            "initial_state_sha256": _tensor_digest(initial),
            "final_state_sha256": _tensor_digest(head.state_dict()),
        }
        torch.save(
            {
                "head_state": head.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "arm": arm,
            },
            output / f"head_{arm}.pt",
        )
        write_json(output / f"training_{arm}.json", receipts[arm])
    return heads, receipts


def _atom_metrics(logits: torch.Tensor, data: dict[str, torch.Tensor]) -> dict[str, Any]:
    loss = functional.cross_entropy(logits.transpose(1, 2), data["target"], reduction="none")
    correct = logits.argmax(-1) == data["target"]
    metrics = {}
    for name, mask in (
        ("all_variable", data["mask"]),
        ("corrupted", data["mask"] & (data["noisy"] != data["target"])),
        ("unchanged", data["mask"] & (data["noisy"] == data["target"])),
    ):
        count = int(mask.sum())
        metrics[name] = {
            "coordinates": count,
            "correct": int(correct[mask].sum()),
            "accuracy": float(correct[mask].float().mean()) if count else None,
            "nll": float(loss[mask].mean()) if count else None,
        }
    return metrics


def _placement_summary(ranks: list[dict[str, Any]]) -> dict[str, Any]:
    multi = [rank for rank in ranks if rank["candidate_count"] > 1]
    return {
        "attempts": len(ranks),
        "empty": sum(rank["candidate_count"] == 0 for rank in ranks),
        "singleton": sum(rank["singleton"] for rank in ranks),
        "target_absent": sum(not rank["target_present"] for rank in ranks),
        "selected_correct": sum(rank["selected_is_correct"] for rank in ranks),
        "multi_candidate_attempts": len(multi),
        "multi_candidate_selected_correct": sum(rank["selected_is_correct"] for rank in multi),
        "primary_accuracy": (
            sum(rank["selected_is_correct"] for rank in multi) / len(multi) if multi else None
        ),
        "informative_with_present_target": sum(rank["informative_ranking"] for rank in ranks),
        "wrong_score_ties": sum(
            (rank["wrong_candidates_tied_with_best_correct"] or 0) > 0 for rank in ranks
        ),
    }


def _advancement(
    evaluation: dict[str, Any],
    same_universe: bool,
    unchanged_padding: bool,
    paired_target_losses: int = 0,
) -> dict[str, bool]:
    group = evaluation["component_disjoint"]
    primary = {arm: report["placement"]["primary_accuracy"] for arm, report in group.items()}
    atom = {arm: report["atoms"]["corrupted"]["accuracy"] for arm, report in group.items()}
    available = all(value is not None for value in (*primary.values(), *atom.values()))
    return {
        "nonempty_informative_population": all(
            report["placement"]["informative_with_present_target"] > 0 for report in group.values()
        ),
        "same_candidate_symbol_universe": same_universe,
        "unmodified_padding": unchanged_padding,
        "semantic_beats_baseline_placement": available
        and primary["semantic"] > primary["baseline"],
        "semantic_beats_control_placement": available and primary["semantic"] > primary["control"],
        "no_corrupted_atom_accuracy_loss_vs_baseline": available
        and atom["semantic"] >= atom["baseline"],
        "no_corrupted_atom_accuracy_loss_vs_control": available
        and atom["semantic"] >= atom["control"],
        "no_target_presence_loss": all(
            group["semantic"]["placement"]["target_absent"]
            <= group[arm]["placement"]["target_absent"]
            for arm in ("baseline", "control")
        ),
        "no_paired_target_presence_loss_in_either_population": paired_target_losses == 0,
    }


def _evaluate_heads(
    *,
    heads: dict[str, Any],
    features: dict[str, Any],
    selection: dict[str, Any],
    cache: Any,
    paths: dict[str, Path],
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    local = LocalChemistrySupport.from_mapping(_read(paths["role_morphology_policy"]))
    ester = UgiEsterChemotypePolicy.from_qualified_registry(
        paths["qualified_reactions"],
        training_assignments_path=paths["ugi_assignments"],
        reaction_id=config["target_program"],
        expected_sha256=str(sha256_file(paths["qualified_reactions"])),
        expected_training_assignments_sha256=str(sha256_file(paths["ugi_assignments"])),
    )
    all_results = {}
    same_universe, unchanged_padding = True, True
    target_losses = {}
    reused = {}
    with (
        torch.no_grad(),
        gzip.open(output / "placement_rows.jsonl.gz", "wt", encoding="utf-8") as ledger,
    ):
        for group in ("component_disjoint", "repeated_component"):
            data, rows = features[group], selection[group]["draws"]
            logits_by_arm = {
                "baseline": data["logits"],
                **{arm: _head_logits(head, data) for arm, head in heads.items()},
            }
            for logits in logits_by_arm.values():
                unchanged_padding &= torch.equal(
                    logits[~data["mask"]], data["logits"][~data["mask"]]
                )
            records = cache.records([row["cache_index"] for row in rows])
            ranks = {arm: [] for arm in logits_by_arm}
            for index in range(len(data["t"])):
                draw_index = index % len(rows)
                row, record = rows[draw_index], records[draw_index]
                pos = data["positions"][index][data["mask"][index]].numpy()
                baseline_report = None
                for arm, logits in logits_by_arm.items():
                    # The original selector reads only its variable head logits. Other positions
                    # are ignored by that selector and the clean record supplies fixed states.
                    full = np.zeros((record.node_count, logits.shape[-1]), dtype=np.float32)
                    full[pos] = logits[index][data["mask"][index]].numpy()
                    report = baseline_head_placement_candidates(
                        record=record,
                        atom_vocabulary=cache.atom_vocabulary,
                        node_logits=full,
                        local_chemistry_support=local,
                        ester_policy=ester,
                        semantic_target=UgiAmineSemanticTarget.from_key(
                            tuple(row["semantic_target"])
                        ),
                        reference_candidates=reused.get(row["cache_index"]),
                    )
                    reused[row["cache_index"]] = report
                    if arm == "baseline":
                        baseline_report = report
                        comparison = None
                    else:
                        comparison = compare_candidate_universes(baseline_report, report)
                        same_universe &= comparison["same_legal_symbol_universe"]
                    ranks[arm].append(report["rank"])
                    ledger.write(
                        json.dumps(
                            {
                                "group": group,
                                "draw_index": draw_index,
                                "cache_index": row["cache_index"],
                                "flow_time": config["flow_times"][index // len(rows)],
                                "arm": arm,
                                "universe_comparison": comparison,
                                "report": report,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            group_result = {}
            target_losses[group] = {
                arm: sum(
                    reference["target_present"] and not semantic["target_present"]
                    for reference, semantic in zip(ranks[arm], ranks["semantic"], strict=True)
                )
                for arm in ("baseline", "control")
            }
            for arm, logits in logits_by_arm.items():
                by_time = {}
                for time_index, flow_time in enumerate(config["flow_times"]):
                    start, stop = time_index * len(rows), (time_index + 1) * len(rows)
                    subset = {key: value[start:stop] for key, value in data.items()}
                    by_time[str(flow_time)] = {
                        "atoms": _atom_metrics(logits[start:stop], subset),
                        "placement": _placement_summary(ranks[arm][start:stop]),
                    }
                group_result[arm] = {
                    "atoms": _atom_metrics(logits, data),
                    "placement": _placement_summary(ranks[arm]),
                    "by_flow_time": by_time,
                }
            all_results[group] = group_result
            write_json(output / f"evaluation_{group}.json", group_result)
    checks = _advancement(
        all_results,
        same_universe,
        unchanged_padding,
        paired_target_losses=sum(sum(values.values()) for values in target_losses.values()),
    )
    return {
        "groups": all_results,
        "paired_target_presence_losses": target_losses,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def _validate(config: dict[str, Any]) -> None:
    constants = {
        "schema_version": "forge.ugi_semantic_residual_config.v1",
        "draws": {"fit": 1024, "component_disjoint": 256, "repeated_component": 256},
        "flow_times": [0.2, 0.5, 0.8, 0.95],
        "semantic_scales": [10, 10, 4, 4],
        "feature_batch_size": 64,
        "training_batch_size": 64,
        "optimizer_steps": 64,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "cpu_threads": 2,
        "device": "cpu",
        "precision": "float32",
        "policy": {
            "backbone_training": False,
            "molecular_generation": False,
            "heldout_structure_access": False,
            "remote_compute": False,
            "gate_changes": False,
            "final_checkpoint_only": True,
        },
    }
    if any(config.get(key) != value for key, value in constants.items()):
        raise UgiSemanticResidualStudyError("fixed residual study contract changed")
    if set(config["seeds"]) != {"draws", "noise", "order", "initialization"} or any(
        type(value) is not int or value < 0 for value in config["seeds"].values()
    ):
        raise UgiSemanticResidualStudyError("invalid fixed seeds")


def _resume_saved_fit(
    manifest_path: Path,
    config: dict[str, Any],
    selection: dict[str, Any],
    repo: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate a failed run and reuse its exact final heads without new fitting."""
    manifest = _read(manifest_path)
    original_path = resolve_pin(manifest["prior_config"], repo, label="prior config")
    original = _read(original_path)
    repair_sources = {
        "forge/model/ugi_semantic_placement.py",
        "experiments/phase1/multireaction/ugi_semantic_residual.py",
    }
    if set(manifest["source_archive"]) != set(original["sources"]) or set(config["sources"]) != set(
        original["sources"]
    ):
        raise UgiSemanticResidualStudyError("recovery source inventory changed")
    if any(
        config["sources"][key] != pin
        for key, pin in original["sources"].items()
        if key not in repair_sources
    ):
        raise UgiSemanticResidualStudyError("recovery changes sources outside evaluation repair")
    for key, value in original.items():
        if key not in {"inputs", "sources"} and config[key] != value:
            raise UgiSemanticResidualStudyError(f"recovery changes original study field: {key}")
    if {
        key: value for key, value in config["inputs"].items() if key != "recovery_manifest"
    } != original["inputs"]:
        raise UgiSemanticResidualStudyError("recovery changes original study inputs")
    for name, archived in manifest["source_archive"].items():
        resolve_pin(archived, repo, label="archived source")
        if archived["sha256"] != original["sources"][name]["sha256"]:
            raise UgiSemanticResidualStudyError("recovery source archive differs from prior run")
    paths = {
        name: resolve_pin(pin, repo, label=f"recovery {name}")
        for name, pin in manifest["artifacts"].items()
    }
    if _read(paths["selection.json"]) != selection:
        raise UgiSemanticResidualStudyError("recovery changes selected examples")
    failure = _read(paths["failure.json"])
    if (
        failure["status"] != "failed_no_admitted_result"
        or "evaluation_component_disjoint.json" in paths
        or "evaluation_repeated_component.json" in paths
    ):
        raise UgiSemanticResidualStudyError("recovery is only for an incomplete evaluation")
    features, heads, training = {}, {}, {}
    for group in selection:
        with np.load(paths[f"features_{group}.npz"], allow_pickle=False) as arrays:
            features[group] = {key: torch.from_numpy(arrays[key].copy()) for key in arrays.files}
        if len(features[group]["t"]) != len(selection[group]["draws"]) * len(config["flow_times"]):
            raise UgiSemanticResidualStudyError("recovery feature exposure changed")
    for arm in ("control", "semantic"):
        data = features["fit"]
        head = UgiSemanticResidualHead(
            data["hidden"].shape[-1],
            data["logits"].shape[-1],
            semantic_scales=config["semantic_scales"],
            initialization_seed=config["seeds"]["initialization"],
            use_semantics=arm == "semantic",
        )
        saved = torch.load(paths[f"head_{arm}.pt"], map_location="cpu", weights_only=True)
        training[arm] = _read(paths[f"training_{arm}.json"])
        if (
            saved["arm"] != arm
            or _tensor_digest(saved["head_state"]) != training[arm]["final_state_sha256"]
        ):
            raise UgiSemanticResidualStudyError(
                "recovery head differs from original final checkpoint"
            )
        head.load_state_dict(saved["head_state"], strict=True)
        heads[arm] = head.eval()
    return features, heads, training, manifest


def run_study(repo_root: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    repo = repo_root.resolve()
    config_path = (repo / config_path).resolve()
    output = (repo / output_dir).resolve()
    if output.exists() or not output.is_relative_to(repo):
        raise UgiSemanticResidualStudyError("fresh output directory inside repository required")
    config = _read(config_path)
    _validate(config)
    config_pin = pin_record(config_path, repo)
    paths = {name: resolve_pin(pin, repo, label=name) for name, pin in config["inputs"].items()}
    for name, pin in config["sources"].items():
        resolve_pin(pin, repo, label=name)
    output.mkdir(parents=True)
    started = time.perf_counter()
    try:
        sources = implementation_snapshot(repo)
        write_json(output / "source_snapshot.json", sources)
        coverage = _read(paths["coverage_result"])
        if (
            coverage["artifacts"]["rows.jsonl"]["sha256"]
            != config["inputs"]["coverage_rows"]["sha256"]
        ):
            raise UgiSemanticResidualStudyError("coverage result/ledger pin differs")
        coverage_config_path = resolve_pin(coverage["config"], repo, label="coverage config")
        coverage_config = _read(coverage_config_path)
        for name, pin in coverage_config["inputs"].items():
            resolve_pin(pin, repo, label=f"coverage {name}")
        if coverage_config["product_partition_domain"] != config["partition"]["product_domain"]:
            raise UgiSemanticResidualStudyError("coverage product split changed")
        rows = [json.loads(line) for line in paths["coverage_rows"].read_text().splitlines()]
        selection = _select_draws(rows, config)
        write_json(output / "selection.json", selection)
        coverage_receipt = {
            "total_train_rows": len(rows),
            "applicable_rows": sum(row["applicable"] for row in rows),
            "excluded_rows": sum(not row["applicable"] for row in rows),
            "original_applicable_source_mass": sum(
                row["source_probability"] for row in rows if row["applicable"]
            ),
            "original_excluded_source_mass": sum(
                row["source_probability"] for row in rows if not row["applicable"]
            ),
            "exclusions_preserved_in": config["inputs"]["coverage_rows"],
        }
        write_json(output / "coverage_receipt.json", coverage_receipt)
        _set_determinism(
            config["seeds"]["initialization"], config["cpu_threads"], torch.device("cpu")
        )
        base = config["base_checkpoint"]
        package, member = _load_base_package(
            archive_path=paths["base_checkpoint_archive"],
            training=_read(paths["base_training_result"]),
            arm_id=base["arm_id"],
            step=base["step"],
            expected_member_sha256=base["member_sha256"],
            expected_model_state_sha256=base["model_state_sha256"],
            expected_design_sha256=str(sha256_file(paths["base_design"])),
            expected_cache_sha256=str(sha256_file(paths["production_cache"])),
            device=torch.device("cpu"),
        )
        cache = SynthesisProgramProductionCache(paths["production_cache"])
        try:
            train_indices = set(
                map(int, cache.indices(program_id=config["target_program"], fold="train"))
            )
            if (
                len(rows) != len(train_indices)
                or {row["cache_index"] for row in rows} != train_indices
            ):
                raise UgiSemanticResidualStudyError(
                    "coverage ledger does not cover complete Ugi TRAIN"
                )
            for row in rows:
                index = row["cache_index"]
                if (
                    cache.record_id(index) != row["product_id"]
                    or float(cache.arrays["source_weights"][index]) != row["source_weight"]
                ):
                    raise UgiSemanticResidualStudyError(
                        "coverage row identity/source weight differs"
                    )
            before = _tensor_digest(package["model_state"])
            recovered = None
            if "recovery_manifest" in paths:
                features, heads, training, recovered = _resume_saved_fit(
                    paths["recovery_manifest"], config, selection, repo
                )
                feature_ledger = []
            else:
                model = build_synthesis_program_flow(
                    vocabulary=cache.vocabulary,
                    node_classes=len(cache.atom_vocabulary),
                    model_config=package["model_config"],
                    device=torch.device("cpu"),
                )
                model.load_state_dict(package["model_state"], strict=True)
                model.eval().requires_grad_(False)
                features, feature_ledger = _extract_features(
                    model=model,
                    package=package,
                    cache=cache,
                    selection=selection,
                    config=config,
                    output=output,
                )
                heads, training = _fit_heads(features["fit"], config, output)
            results = _evaluate_heads(
                heads=heads,
                features=features,
                selection=selection,
                cache=cache,
                paths=paths,
                config=config,
                output=output,
            )
            if before != _tensor_digest(package["model_state"]) or (
                recovered is None and before != _tensor_digest(model.state_dict())
            ):
                raise UgiSemanticResidualStudyError("frozen backbone changed")
        finally:
            cache.close()
        if implementation_snapshot(repo) != sources or config_pin != pin_record(config_path, repo):
            raise UgiSemanticResidualStudyError("source/config changed during study")
        for name, pin in {**config["inputs"], **config["sources"]}.items():
            resolve_pin(pin, repo, label=name)
        result = {
            "schema_version": "forge.ugi_semantic_residual_result.v1",
            "status": "complete_development_reconstruction_only",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config_pin,
            "inputs": config["inputs"],
            "sources": config["sources"],
            "source_snapshot_sha256": str(sha256_json(sources)),
            "checkpoint": {**base, "member_name": member, "before_after_state_sha256": before},
            "selection": {
                name: {key: value for key, value in group.items() if key != "draws"}
                for name, group in selection.items()
            },
            "training": training,
            "coverage": coverage_receipt,
            "evaluation": results,
            "mechanism_gate_passed": results["gate_passed"],
            "forward_calls": len(feature_ledger),
            "additional_optimizer_updates": (
                0 if recovered is not None else 2 * config["optimizer_steps"]
            ),
            "reused_artifacts": recovered["artifacts"] if recovered is not None else {},
            "runtime": {
                "seconds": time.perf_counter() - started,
                "device": "cpu",
                "precision": "float32",
                "cpu_threads": config["cpu_threads"],
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
            "artifacts": {
                path.name: pin_record(path, repo)
                for path in sorted(output.iterdir())
                if path.is_file()
            },
            "nonclaims": config["nonclaims"],
        }
        write_json(output / "result.json", result)
        return result
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "status": "failed_no_admitted_result",
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
    print(run_study(args.repo_root, args.config, args.output_dir)["status"])


if __name__ == "__main__":
    main()
