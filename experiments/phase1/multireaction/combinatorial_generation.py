"""Retain the balanced shared checkpoint and evaluate generation across twelve libraries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem, rdBase

from experiments.phase1.multireaction.combinatorial_training_pilot import (
    _record_losses,
    _validate_contract,
)
from forge.assembly.families import load_assembly_libraries
from forge.assembly.library_generation import check_generated_program
from forge.assembly.library_programs import LibraryProgramLimits
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.library_splits import constitution_id
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model._synthesis_sampling import SAMPLING_SOURCE_FILES
from forge.model.defog_feasibility import _model_state_sha256
from forge.model.synthesis_program_sampling import (
    CHECKPOINT_SCHEMA,
    load_synthesis_program_checkpoint,
    sample_synthesis_program_products,
)
from forge.model.synthesis_program_training import (
    build_synthesis_program_flow,
    collate_synthesis_program_training_batch,
    synthesis_program_fixed_state_exact,
    synthesis_program_forward,
)
from forge.model.tensor_checkpoint import encode_tensor_state

IMPLEMENTATION_SOURCES = (
    "experiments/phase1/multireaction/combinatorial_training_pilot.py",
    "forge/assembly/library_generation.py",
    "forge/assembly/library_programs.py",
    "forge/assembly/families.py",
    *SAMPLING_SOURCE_FILES,
    "forge/model/reaction_program_flow.py",
    "forge/model/synthesis_program_training.py",
    "forge/corpus/synthesis_program_production_cache.py",
)


class CombinatorialGenerationError(ValueError):
    """A generation input or invariant violates the declared experiment."""


def _pin(path: Path, repo: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(repo)), "sha256": str(sha256_file(path))}


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _inputs(repo: Path, config: dict[str, Any]) -> dict[str, Path]:
    return {key: resolve_pin(pin, repo, label=key) for key, pin in config["inputs"].items()}


def _train(cache: SynthesisProgramProductionCache, config: dict[str, Any]) -> tuple[Any, dict]:
    """Replay the frozen pilot's training stream without evaluating calibration or heldout."""
    training, seed = config["training"], config["seed"]
    torch.manual_seed(seed)
    programs = tuple(cache.vocabulary.program_states[1:])
    measure = cache.training_measure({p: 1 / len(programs) for p in programs})
    choices = {p: cache.indices(program_id=p, fold="train") for p in programs}
    probabilities = {p: measure[i] / measure[i].sum() for p, i in choices.items()}
    nodes, bonds = cache.source_marginals(
        measure,
        node_classes=len(cache.atom_vocabulary),
        bond_classes=config["model"]["bond_classes"],
        probability_floor=training["source_probability_floor"],
    )
    model = build_synthesis_program_flow(
        vocabulary=cache.vocabulary,
        node_classes=len(cache.atom_vocabulary),
        model_config=config["model"],
        device=torch.device("cpu"),
    )
    initial_hash = _model_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    rng = np.random.default_rng(seed + 1)
    noise = torch.Generator(device="cpu").manual_seed(seed + 2)
    node_p0, bond_p0 = (torch.as_tensor(a, dtype=torch.float32) for a in (nodes, bonds))
    digest = hashlib.sha256()
    model.train()
    for step in range(1, training["steps"] + 1):
        indices = [int(rng.choice(choices[p], p=probabilities[p])) for p in programs]
        for p, i in zip(programs, indices, strict=True):
            digest.update(f"{step}\t{p}\t{cache.record_id(i)}\n".encode())
        clean = collate_synthesis_program_training_batch(
            cache.records(indices),
            maximum_closures=config["model"]["maximum_closures"],
            conditioning="program",
            vocabulary=cache.vocabulary,
        )
        times = torch.empty(len(programs), dtype=torch.float32).uniform_(
            training["minimum_flow_time"], training["maximum_flow_time"], generator=noise
        )
        optimizer.zero_grad(set_to_none=True)
        predictions, noisy = synthesis_program_forward(model, clean, node_p0, bond_p0, times, noise)
        loss = torch.stack(_record_losses(predictions, clean)).mean()
        if not bool(torch.isfinite(loss)) or not synthesis_program_fixed_state_exact(noisy, clean):
            raise CombinatorialGenerationError(f"training invariant failed at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
        if not bool(torch.isfinite(norm)):
            raise CombinatorialGenerationError(f"nonfinite gradient at step {step}")
        optimizer.step()
    return model, {
        "schema_version": CHECKPOINT_SCHEMA,
        "trusted_local_checkpoint": True,
        "run_kind": "balanced_training_pilot_replay",
        "model_config": config["model"],
        "program_vocabulary": asdict(cache.vocabulary),
        "atom_vocabulary": [asdict(atom) for atom in cache.atom_vocabulary],
        "node_marginal": nodes.tolist(),
        "bond_marginal": bonds.tolist(),
        "initial_state_sha256": initial_hash,
        "model_state_sha256": _model_state_sha256(model),
        "model_state": encode_tensor_state(model.state_dict()),
        "sampling_ledger_sha256": digest.hexdigest(),
        "training_config": config,
        "training_resumption_supported": False,
        "layout_contract": "semantic_coordinates_plus_adapter_fixed_states_only",
    }


def retain_checkpoint(repo: Path, config: dict, work: Path) -> dict:
    paths = _inputs(repo, config)
    pilot_config = json.loads(paths["pilot_config"].read_text())
    pilot = json.loads(paths["pilot_result"].read_text())
    pilot_paths = _inputs(repo, pilot_config)
    cache_result = json.loads(pilot_paths["cache_result"].read_text())
    _validate_contract(pilot_config, cache_result)
    if pilot.get("status") != "pass" or pilot.get("config") != config["inputs"]["pilot_config"]:
        raise CombinatorialGenerationError("pilot result is not authenticated and passing")
    with SynthesisProgramProductionCache(pilot_paths["cache"]) as cache:
        _, package = _train(cache, pilot_config)
    checks = {
        "initial_state_identical": package["initial_state_sha256"]
        == pilot["model"]["initial_state_sha256"],
        "final_state_identical": package["model_state_sha256"]
        == pilot["model"]["final_state_sha256"],
        "training_stream_identical": package["sampling_ledger_sha256"]
        == pilot["training"]["sampling_ledger_sha256"],
    }
    if not all(checks.values()):
        raise CombinatorialGenerationError(f"frozen training replay differs: {checks}")
    package["inputs"] = {**config["inputs"], **pilot_config["inputs"]}
    _write(work / "checkpoint.json", package)
    loaded, *_ = load_synthesis_program_checkpoint(work / "checkpoint.json", device="cpu")
    checks["checkpoint_reload_identical"] = (
        _model_state_sha256(loaded) == package["model_state_sha256"]
    )
    if not all(checks.values()):
        raise CombinatorialGenerationError("checkpoint reload differs")
    return {
        "status": "pass",
        "checks": checks,
        "model_state_sha256": package["model_state_sha256"],
        "training_examples": pilot["training"]["examples_seen"],
        "cache": pilot_config["inputs"]["cache"],
        "checkpoint_sha256": str(sha256_file(work / "checkpoint.json")),
    }


def summarize(rows: list[dict], families: list[str]) -> dict:
    result = {}
    for family in families:
        local = [r for r in rows if r["program_id"] == family]
        valid = [r for r in local if r["valid_connected"]]
        exact = [r for r in valid if r["assembly"]["status"] == "exact_computed_program"]
        unique = {r["canonical_smiles"] for r in valid}
        unique_exact = {r["canonical_smiles"] for r in exact}
        frequencies = Counter(r["canonical_smiles"] for r in valid)
        result[family] = {
            "attempts": len(local),
            "valid_connected": len(valid),
            "valid_fraction": len(valid) / len(local) if local else None,
            "exact_program": len(exact),
            "exact_program_fraction_all_attempts": len(exact) / len(local) if local else None,
            "exact_program_fraction_valid": len(exact) / len(valid) if valid else None,
            "unique_valid_products": len(unique),
            "unique_exact_products": len(unique_exact),
            "unique_fraction_valid": len(unique) / len(valid) if valid else None,
            "effective_product_count": (
                len(valid) ** 2 / sum(n * n for n in frequencies.values()) if valid else 0
            ),
            "novel_valid_attempts_vs_all_cache_train": sum(r["novel_vs_train"] for r in valid),
            "novel_exact_attempts_vs_all_cache_train": sum(r["novel_vs_train"] for r in exact),
            "unique_novel_exact_products": len(
                {r["canonical_smiles"] for r in exact if r["novel_vs_train"]}
            ),
            "exact_with_novel_component_in_every_witness": sum(
                r["all_witnesses_have_novel_component"] for r in exact
            ),
            "assembly_search_abstentions": sum(r["assembly"]["status"] == "abstain" for r in valid),
            "failure_counts": dict(
                Counter(r["failure_reason"] for r in local if r["failure_reason"])
            ),
        }
    return result


def generate(repo: Path, config: dict, work: Path) -> dict:
    paths = _inputs(repo, config)
    model, vocabulary, atoms, nodes, bonds, package = load_synthesis_program_checkpoint(
        paths["checkpoint"], device="cpu"
    )
    if package["inputs"]["cache"] != config["inputs"]["cache"]:
        raise CombinatorialGenerationError("checkpoint and sampling cache differ")
    dataset = json.loads(paths["dataset_config"].read_text())
    families = sorted(dataset["programs"])
    if list(vocabulary.program_states[1:]) != families or len(families) != 12:
        raise CombinatorialGenerationError("checkpoint does not cover the twelve declared families")
    registries = [
        (resolve_pin(dataset["inputs"][key], repo, label=key), dataset["inputs"][key]["sha256"])
        for key in dataset["registries"]
    ]
    libraries = load_assembly_libraries(registries, expected_families=families)
    train_components = {
        row["constitution_id"]
        for row in json.loads(paths["component_partitions"].read_text())
        if row["fold"] == "train"
    }
    sampling = config["sampling"]
    rng = np.random.default_rng(config["seed"])
    with SynthesisProgramProductionCache(paths["cache"]) as cache:
        if cache.vocabulary != vocabulary or tuple(cache.atom_vocabulary) != atoms:
            raise CombinatorialGenerationError("checkpoint and cache vocabularies differ")
        measure = cache.training_measure({p: 1 / len(families) for p in families})
        selected = []
        for family in families:
            indices = cache.indices(program_id=family, fold="train")
            selected.extend(
                int(i)
                for i in rng.choice(
                    indices,
                    size=sampling["attempts_per_family"],
                    p=measure[indices] / measure[indices].sum(),
                )
            )
        records = cache.records(selected)
        if any(cache.fold(i) != "train" for i in selected):
            raise CombinatorialGenerationError("nontraining layout selected")
        train_products = {cache.canonical_smiles(int(i)) for i in cache.indices(fold="train")}
        layout_ledger = [
            {
                "index": i,
                "record_id": cache.record_id(i),
                "family": r.program_id,
                "depth": r.program_depth,
                "nodes": r.node_count,
                "fold": "train",
            }
            for i, r in zip(selected, records, strict=True)
        ]
    _write(work / "layouts.json", layout_ledger)
    # The same layouts, p0, seeds and decoder are used in both arms. No learned weights are
    # selected using the generated results. Initialization is the frozen pilot initialization.
    torch.manual_seed(package["training_config"]["seed"])
    untrained = build_synthesis_program_flow(
        vocabulary=vocabulary,
        node_classes=len(atoms),
        model_config=package["model_config"],
        device=torch.device("cpu"),
    )
    if _model_state_sha256(untrained) != package["initial_state_sha256"]:
        raise CombinatorialGenerationError("untrained control differs from pilot initialization")
    summaries, all_rows, sampler_reports = {}, [], {}
    for arm, network in (("untrained", untrained), ("trained", model)):
        with rdBase.BlockLogs():
            raw_rows, report = sample_synthesis_program_products(
                network,
                records,
                atoms,
                nodes,
                bonds,
                samples_per_program=1,
                sample_steps=sampling["flow_steps"],
                batch_size=sampling["batch_size"],
                seed=sampling["flow_seed"],
                device="cpu",
                terminal_decode_policy="strict_valence_topology_argmax",
            )
        if len(raw_rows) != len(records) or report["fixed_state_failures"]:
            raise CombinatorialGenerationError("sampler dropped attempts or changed fixed states")
        rows = []
        for row, record in zip(raw_rows, records, strict=True):
            family = record.program_id
            policy = dataset["programs"][family]
            smiles = row["canonical_smiles"]
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(smiles) if smiles else None
            valid = molecule is not None and len(Chem.GetMolFrags(molecule)) == 1
            assembly = {"status": "invalid_graph", "programs": [], "expansions": 0}
            if valid:
                assembly = asdict(
                    check_generated_program(
                        libraries[family],
                        smiles,
                        depth=record.program_depth,
                        accumulator_role=policy["accumulator_role"],
                        limits=LibraryProgramLimits(policy["maximum_steps"], **dataset["limits"]),
                    )
                )
            witnesses = assembly["programs"]
            enriched = {
                **row,
                "arm": arm,
                "valid_connected": valid,
                "target_scope": policy["target_scope"],
                "requested_depth": record.program_depth,
                "assembly": assembly,
                "novel_vs_train": valid and smiles not in train_products,
                "all_witnesses_have_novel_component": bool(witnesses)
                and all(
                    any(
                        constitution_id(s) not in train_components for s in w["components"].values()
                    )
                    for w in witnesses
                ),
                "failure_reason": (
                    row["constraint_abstention_reason"] or "invalid_or_disconnected_graph"
                    if not valid
                    else (None if witnesses else assembly["status"])
                ),
            }
            rows.append(enriched)
        summaries[arm] = summarize(rows, families)
        sampler_reports[arm] = report
        all_rows.extend(rows)
    with (work / "attempts.jsonl").open("w") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    demonstration = {
        p: summaries["trained"][p]["unique_novel_exact_products"] > 0 for p in families
    }
    return {
        "status": "numerical_complete",
        "generation_demonstrated_all_families": all(demonstration.values()),
        "novel_exact_generation_demonstrated_by_family": demonstration,
        "per_arm": summaries,
        "sampler": sampler_reports,
        "family_target_scope": {p: dataset["programs"][p]["target_scope"] for p in families},
        "reference_train_product_count": len(train_products),
        "reference_train_component_count": len(train_components),
        "component_reference": "all train-assigned source blocks; not only sampled blocks",
        "layout_policy": "realism_weighted_empirical_train_semantic_layouts",
        "access": {
            "cache_arrays_include_heldout": True,
            "heldout_records_materialized": 0,
            "heldout_used_for_training_sampling_or_metrics": False,
        },
        "precision_boundary": "Exact forward replay is checked for every returned witness; source-adjudicated chemical precision is unmeasured.",
        "nonclaims": [
            "Layout-conditioned graph generation, not unconditional layout generation.",
            "Computed assembly consistency does not close L2/L3 or establish ionizability, lipid realism or synthesis success.",
            "Precursor or neutral-structure families are not complete ionizable-lipid generators.",
            "One seed and a small existing pilot checkpoint; no model promotion or quality noninferiority claim.",
        ],
    }


def run(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if not config_path.is_relative_to(repo) or not output.is_relative_to(repo) or output.exists():
        raise CombinatorialGenerationError("config/output must be local and output must be fresh")
    config = json.loads(config_path.read_text())
    stage = config.get("stage")
    if config.get("schema_version") != "forge.combinatorial_generation_config.v1" or stage not in {
        "checkpoint",
        "generation",
    }:
        raise CombinatorialGenerationError("unknown generation stage/schema")
    if config.get("remote_compute") is not False or config.get("candidate_selection") is not False:
        raise CombinatorialGenerationError("this command is a local diagnostic only")
    if type(config.get("cpu_threads")) is not int or not 1 <= config["cpu_threads"] <= 8:
        raise CombinatorialGenerationError("cpu_threads must be between one and eight")
    if stage == "generation":
        s = config["sampling"]
        if (
            set(s) != {"attempts_per_family", "flow_steps", "batch_size", "flow_seed"}
            or any(type(v) is not int or v < 1 for v in s.values())
            or s["flow_steps"] < 2
        ):
            raise CombinatorialGenerationError("invalid fixed sampling budget")
    sources = [
        Path(__file__).resolve(),
        *[repo / p for p in IMPLEMENTATION_SOURCES],
    ]
    pins = [_pin(p, repo) for p in [config_path, *sources]]
    start = time.monotonic()
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        with tempfile.TemporaryDirectory(prefix=".generation-", dir=output.parent) as temporary:
            work = Path(temporary)
            result = (retain_checkpoint if stage == "checkpoint" else generate)(repo, config, work)
            for pin in pins:
                resolve_pin(pin, repo, label="source/config")
            _inputs(repo, config)
            result.update(
                {
                    "schema_version": "forge.combinatorial_generation_result.v1",
                    "stage": stage,
                    "config": pins[0],
                    "inputs": config["inputs"],
                    "sources": pins[1:],
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": time.monotonic() - start,
                    "environment": {
                        "torch": torch.__version__,
                        "numpy": np.__version__,
                        "rdkit": rdBase.rdkitVersion,
                        "device": "cpu",
                        "deterministic": True,
                        "cpu_threads": config["cpu_threads"],
                    },
                    "artifacts": {
                        p.name: {
                            "path": str((output / p.name).relative_to(repo)),
                            "sha256": str(sha256_file(p)),
                        }
                        for p in sorted(work.iterdir())
                    },
                }
            )
            _write(work / "result.json", result)
            os.rename(work, output)
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.repo_root, args.config, args.output_dir)
    print(json.dumps({"status": result["status"], "stage": result["stage"]}))


if __name__ == "__main__":
    main()
