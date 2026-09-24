"""Evaluate the frozen whole-product flow on count-only layouts across all twelve libraries."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from rdkit import rdBase

from experiments.phase1.multireaction.combinatorial_connection_reuse import (
    verify as verify_completed,
)
from experiments.phase1.multireaction.combinatorial_generation import _pin, _write, summarize
from experiments.phase1.multireaction.combinatorial_generation_verify import (
    verify as verify_generation,
)
from experiments.phase1.multireaction.combinatorial_graph_reuse import compare
from experiments.phase1.multireaction.combinatorial_reuse_pilot import assess_rows
from forge.assembly.families import load_assembly_libraries
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.combinatorial_layout_prior import CombinatorialLayoutPrior
from forge.model.generated_program_layout import layout_signatures
from forge.model.synthesis_program_sampling import (
    load_synthesis_program_checkpoint,
    sample_synthesis_program_products,
)

SCHEMA = "forge.combinatorial_count_layout_generation.v1"
SOURCES = (
    "forge/model/combinatorial_layout_prior.py",
    "forge/model/synthesis_program_layout.py",
    "forge/model/synthesis_program_sampling.py",
    "forge/model/generated_program_layout.py",
    "experiments/phase1/multireaction/combinatorial_count_layout_generation.py",
)
POLICY = {
    "layout_distribution": "factorized_count_only_train_realism_weight",
    "source_product_layouts_used": False,
    "source_variable_graph_values_used": False,
    "training_calls": 0,
    "remote_compute": False,
    "heldout_use": False,
    "gate_changes": False,
    "all_attempts_retained": True,
    "automatic_promotion": False,
}


def calculate(repo, config_path):
    config = json.loads(config_path.read_text())
    if config["schema_version"] != SCHEMA + "_config" or config["policy"] != POLICY:
        raise ValueError("count-layout generation contract differs")
    if {p["path"] for p in config["source_pins"]} != set(SOURCES):
        raise ValueError("count-layout source set differs")
    for pin in config["source_pins"]:
        resolve_pin(pin, repo, label="count-layout frozen source")
    paths = {k: resolve_pin(p, repo, label=k) for k, p in config["inputs"].items()}
    verify_generation(repo, paths["baseline_result"])
    verify_completed(repo, paths["completed_result"])
    baseline = json.loads(paths["baseline_result"].read_text())
    completed = json.loads(paths["completed_result"].read_text())
    if any(
        config["inputs"][k] != baseline["inputs"][k]
        for k in ("cache", "checkpoint", "dataset_config", "component_partitions")
    ):
        raise ValueError("count-layout baseline/checkpoint/data mismatch")
    dataset = json.loads(paths["dataset_config"].read_text())
    libraries = load_assembly_libraries(
        [
            (resolve_pin(dataset["inputs"][k], repo, label=k), dataset["inputs"][k]["sha256"])
            for k in dataset["registries"]
        ],
        expected_families=dataset["programs"],
    )
    if len(libraries) != 12 or config["sampling"]["attempts_per_family"] != 64:
        raise ValueError("count-layout diagnostic requires 64 attempts in each of twelve families")
    threads, determinism = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        model, vocabulary, atoms, nodes, bonds, package = load_synthesis_program_checkpoint(
            paths["checkpoint"], device="cpu"
        )
        if package["model_config"]["architecture"] != "reaction_program_sparse_whole_lipid_flow":
            raise ValueError("count-layout model architecture changed")
        with SynthesisProgramProductionCache(paths["cache"]) as cache:
            prior = CombinatorialLayoutPrior(cache)
            train_products = {cache.canonical_smiles(int(i)) for i in cache.indices(fold="train")}
        records = []
        for i, family in enumerate(sorted(libraries)):
            records.extend(
                prior.sample(
                    family,
                    sample_count=config["sampling"]["attempts_per_family"],
                    seed=config["sampling"]["layout_seed"] + i,
                )
            )
        if any(
            r.graph.canonical_smiles
            or np.any(r.graph.node_states[~r.fixed_atom_mask])
            or np.any(r.graph.parents[~r.fixed_parent_bond_mask])
            for r in records
        ):
            raise ValueError("count-only layout contains a variable target or product identity")
        train_components = {
            r["constitution_id"]
            for r in json.loads(paths["component_partitions"].read_text())
            if r["fold"] == "train"
        }
        with rdBase.BlockLogs():
            raw, sampler = sample_synthesis_program_products(
                model,
                records,
                atoms,
                nodes,
                bonds,
                samples_per_program=1,
                sample_steps=config["sampling"]["flow_steps"],
                batch_size=config["sampling"]["batch_size"],
                seed=config["sampling"]["flow_seed"],
                device="cpu",
                terminal_decode_policy="strict_valence_topology_argmax",
            )
            if len(raw) != len(records) or sampler["fixed_state_failures"]:
                raise ValueError("count-only sampler accounting or fixed-state failure")
            # Synthetic layouts have no target molecule. Do not publish tensor reconstruction
            # accuracy against their zero-filled placeholder fields.
            raw = [
                {
                    k: r[k]
                    for k in (
                        "sample_index",
                        "program_id",
                        "layout_record_id",
                        "canonical_smiles",
                        "valid",
                        "constraint_abstention_reason",
                    )
                }
                for r in raw
            ]
            rows = assess_rows(
                raw, records, "graph_reuse", dataset, libraries, train_products, train_components
            )

        def originals(result, arm):
            path = resolve_pin(
                result["artifacts"]["attempts.jsonl"], repo, label="baseline attempts"
            )
            return [
                {**r, "arm": "baseline"}
                for r in map(json.loads, path.read_text().splitlines())
                if r["arm"] == arm
            ]

        comparison = compare(originals(baseline, "trained") + rows, sorted(libraries))
        comparison_completed = compare(
            originals(completed, "graph_reuse") + rows, sorted(libraries)
        )
        layout_rows = [
            {
                "family": r.program_id,
                "record_id": r.graph.structure_id,
                "depth": r.program_depth,
                "nodes": r.node_count,
                "closures": r.graph.closure_count,
                "signatures": layout_signatures(r, maximum_closures=1),
            }
            for r in records
        ]
        metrics = {
            "by_family": summarize(rows, sorted(libraries)),
            "comparison_vs_source_layouts": comparison,
            "comparison_vs_previous_completion": comparison_completed,
            "all_families_preserved_vs_source_layouts": comparison[
                "all_family_preservation_screen_passed"
            ],
            "prior_support": prior.support_report,
            "source_topology_counts_by_depth": prior.topology_support_counts,
            "neural_attempts": len(rows),
            "sampler": {
                k: sampler[k]
                for k in (
                    "fixed_state_failures",
                    "strict_constraint_abstentions",
                    "strict_constraint_abstention_reasons",
                    "terminal_decode_policy",
                    "repairs",
                )
                if k in sampler
            },
            "selection": "none; all generated attempts retained",
        }
        return json.loads(json.dumps(metrics)), json.loads(
            json.dumps({"attempts.jsonl": rows, "layouts.json": layout_rows})
        )
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(determinism)


def run(repo, config_path, output):
    repo = repo.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output).resolve()
    if not config_path.is_relative_to(repo) or not output.is_relative_to(repo) or output.exists():
        raise ValueError("config/output must be local; output must be fresh")
    config_pin, sources = _pin(config_path, repo), [_pin(repo / p, repo) for p in SOURCES]
    config = json.loads(config_path.read_text())
    start = time.monotonic()
    metrics, artifacts = calculate(repo, config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".count-layouts-", dir=output.parent) as temp:
        work = Path(temp)
        for name, value in artifacts.items():
            if name.endswith("jsonl"):
                (work / name).write_text(
                    "".join(json.dumps(r, sort_keys=True) + "\n" for r in value)
                )
            else:
                _write(work / name, value)
        for pin in [config_pin, *sources, *config["inputs"].values()]:
            resolve_pin(pin, repo, label="final count layout input")
        result = {
            "schema_version": SCHEMA,
            "status": "numerical_complete",
            "config": config_pin,
            "inputs": config["inputs"],
            "sources": sources,
            "policy": POLICY,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.monotonic() - start,
            **metrics,
            "artifacts": {
                name: {
                    "path": str((output / name).relative_to(repo)),
                    "sha256": str(sha256_file(work / name)),
                }
                for name in artifacts
            },
            "nonclaims": [
                "Count-only layouts are sampled from weighted TRAIN summary distributions without product SMILES, precursor identities, or variable graph targets.",
                "Context novelty is descriptive, not a chemical novelty or quality gate. Exact family-specific reconstruction is evaluated for every generated attempt.",
                "The layout distribution and tensor shapes change, so sharing a seed with the source-layout baseline does not establish coordinate-paired noise or a causal comparison.",
                "This bounded diagnostic does not automatically replace the existing completion pipeline. Any preservation failure prevents promotion.",
            ],
        }
        _write(work / "result.json", result)
        os.rename(work, output)
    return result


def verify(repo, path):
    repo, path = repo.resolve(), (repo / path).resolve()
    result = json.loads(path.read_text())
    if (
        result["schema_version"] != SCHEMA
        or result["status"] != "numerical_complete"
        or result["policy"] != POLICY
        or {p["path"] for p in result["sources"]} != set(SOURCES)
    ):
        raise ValueError("count-layout result contract differs")
    for pin in [
        result["config"],
        *result["inputs"].values(),
        *result["sources"],
        *result["artifacts"].values(),
    ]:
        resolve_pin(pin, repo, label="count layout verification")
    config = json.loads((repo / result["config"]["path"]).read_text())
    if config["inputs"] != result["inputs"]:
        raise ValueError("count-layout inputs substituted")
    metrics, artifacts = calculate(repo, repo / result["config"]["path"])
    if any(result[k] != v for k, v in metrics.items()) or set(artifacts) != set(
        result["artifacts"]
    ):
        raise ValueError("count-layout metrics/artifacts differ")
    for name, expected in artifacts.items():
        text = (repo / result["artifacts"][name]["path"]).read_text()
        actual = (
            [json.loads(s) for s in text.splitlines()]
            if name.endswith("jsonl")
            else json.loads(text)
        )
        if actual != expected:
            raise ValueError("count-layout ledger differs: " + name)
    return {"status": "verified", "result_sha256": str(sha256_file(path))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--config", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--verify", type=Path)
    a = p.parse_args()
    if a.verify:
        result = verify(a.repo_root, a.verify)
    elif a.config and a.output_dir:
        result = run(a.repo_root, a.config, a.output_dir)
    else:
        p.error("provide --verify or --config and --output-dir")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "by_family",
                    "all_families_preserved_vs_source_layouts",
                )
                if k in result
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
