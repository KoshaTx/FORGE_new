"""A paired terminal-core test across all twelve libraries, with unchanged neural scores."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from rdkit import rdBase

from experiments.phase1.multireaction.combinatorial_count_layout_generation import (
    SOURCES as BASE_SOURCES,
)
from experiments.phase1.multireaction.combinatorial_generation import _pin, _write
from experiments.phase1.multireaction.combinatorial_graph_reuse import compare
from experiments.phase1.multireaction.combinatorial_reuse_pilot import assess_rows
from forge.assembly.families import load_assembly_libraries
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.combinatorial_core_scaffold import (
    CoreScaffoldPrior,
    apply_core_scaffold,
    canonical_core_slots,
)
from forge.model.combinatorial_layout_prior import CombinatorialLayoutPrior
from forge.model.generated_program_layout import layout_signatures
from forge.model.paired_core_decoding import paired_terminal_core_decode
from forge.model.synthesis_program_sampling import load_synthesis_program_checkpoint

SCHEMA = "forge.combinatorial_core_decoding.v1"
SOURCES = (
    *BASE_SOURCES,
    "forge/model/combinatorial_core_scaffold.py",
    "forge/model/paired_core_decoding.py",
    "experiments/phase1/multireaction/combinatorial_core_decoding.py",
)
POLICY = {
    "core_source": "weighted_all_TRAIN_registry_replayed_semantics_core_only",
    "pairing": "one_neural_trajectory_and_identical_terminal_scores",
    "scope": "terminal_constraint_diagnostic",
    "source_product_layouts_used": False,
    "source_precursor_interiors_used": False,
    "training_calls": 0,
    "remote_compute": False,
    "heldout_use": False,
    "gate_changes": False,
    "all_attempts_retained": True,
    "automatic_promotion": False,
    "core_draws_per_layout": 1,
    "retry_unrepresentable_core": False,
}


def calculate(repo, config_path):
    config = json.loads(config_path.read_text())
    if config["schema_version"] != SCHEMA + "_config" or config["policy"] != POLICY:
        raise ValueError("paired core experiment contract differs")
    if {p["path"] for p in config["source_pins"]} != set(SOURCES):
        raise ValueError("paired core source set differs")
    for pin in config["source_pins"]:
        resolve_pin(pin, repo, label="paired core source")
    paths = {k: resolve_pin(p, repo, label=k) for k, p in config["inputs"].items()}
    previous = json.loads(paths["count_layout_result"].read_text())
    if any(
        previous["inputs"][k] != pin
        for k, pin in config["inputs"].items()
        if k != "count_layout_result"
    ):
        raise ValueError("paired core data differ from count-prior experiment")
    for p in [previous["config"], *previous["sources"], *previous["artifacts"].values()]:
        resolve_pin(p, repo, label="preceding count result")
    dataset = json.loads(paths["dataset_config"].read_text())
    libraries = load_assembly_libraries(
        [
            (resolve_pin(dataset["inputs"][k], repo, label=k), dataset["inputs"][k]["sha256"])
            for k in dataset["registries"]
        ],
        expected_families=dataset["programs"],
    )
    sampling = config["sampling"]
    if (
        len(libraries) != 12
        or sampling["attempts_per_family"] != 64
        or sampling["flow_steps"] != 32
    ):
        raise ValueError(
            "paired core diagnostic requires twelve families and its fixed neural budget"
        )
    old_threads, old_determinism = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
    )
    try:
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        model, _, atoms, nodes, bonds, _ = load_synthesis_program_checkpoint(
            paths["checkpoint"], device="cpu"
        )
        with SynthesisProgramProductionCache(paths["cache"]) as cache:
            counts, cores = CombinatorialLayoutPrior(cache), CoreScaffoldPrior(cache)
            train_products = {cache.canonical_smiles(int(i)) for i in cache.indices(fold="train")}
        train_components = {
            r["constitution_id"]
            for r in json.loads(paths["component_partitions"].read_text())
            if r["fold"] == "train"
        }
        records, treated, plans = [], [], []
        for f, family in enumerate(sorted(libraries)):
            for record in counts.sample(family, sample_count=64, seed=sampling["layout_seed"] + f):
                record = canonical_core_slots(record)
                core = cores.sample(record, seed=sampling["core_seed"] + len(records))
                scaffold, reason = apply_core_scaffold(record, core)
                plans.append(
                    {
                        "sample_index": len(records),
                        "program_id": family,
                        "layout_record_id": record.graph.structure_id,
                        "depth": record.program_depth,
                        "nodes": record.node_count,
                        "closures": record.graph.closure_count,
                        "core": asdict(core),
                        "qualification": reason or "qualified",
                        "baseline_signature": layout_signatures(record, maximum_closures=1),
                        "scaffold_signature": (
                            None
                            if scaffold is None
                            else layout_signatures(scaffold, maximum_closures=1)
                        ),
                    }
                )
                records.append(record)
                treated.append(scaffold if scaffold is not None else record)
        raw_base, raw_core, evidence = [], [], []
        with rdBase.BlockLogs():
            for offset in range(0, len(records), sampling["batch_size"]):
                end = offset + sampling["batch_size"]
                base, core, proof = paired_terminal_core_decode(
                    model,
                    records[offset:end],
                    treated[offset:end],
                    atoms,
                    nodes,
                    bonds,
                    seed=sampling["flow_seed"] + offset,
                    flow_steps=sampling["flow_steps"],
                )
                evidence.append({"offset": offset, "attempts": len(base), **proof})
                for i, (a, b) in enumerate(zip(base, core, strict=True), start=offset):
                    a["sample_index"] = b["sample_index"] = i
                    if plans[i]["qualification"] != "qualified":
                        b.update(
                            canonical_smiles=None,
                            valid=False,
                            constraint_abstention_reason=plans[i]["qualification"],
                        )
                    raw_base.append(a)
                    raw_core.append(b)
            original = assess_rows(
                raw_base, records, "baseline", dataset, libraries, train_products, train_components
            )
            revised = assess_rows(
                raw_core,
                records,
                "graph_reuse",
                dataset,
                libraries,
                train_products,
                train_components,
            )
        comparison = compare(original + revised, sorted(libraries))
        completed = json.loads(paths["completed_result"].read_text())
        completed_rows = [
            {**r, "arm": "baseline"}
            for r in map(
                json.loads,
                resolve_pin(
                    completed["artifacts"]["attempts.jsonl"], repo, label="completed attempts"
                )
                .read_text()
                .splitlines(),
            )
            if r["arm"] == "graph_reuse"
        ]
        metrics = {
            "comparison": comparison,
            "comparison_vs_existing_pipeline": compare(completed_rows + revised, sorted(libraries)),
            "core_support": cores.report(),
            "neural_attempts": len(records),
            "decoded_attempts_per_arm": len(records),
            "qualification_by_family": {
                f: dict(Counter(p["qualification"] for p in plans if p["program_id"] == f))
                for f in sorted(libraries)
            },
            "pairing": evidence,
            "promoted": False,
        }
        return json.loads(json.dumps(metrics)), json.loads(
            json.dumps(
                {
                    "attempts.jsonl": original + revised,
                    "layouts.json": plans,
                }
            )
        )
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_determinism)


def run(repo, config_path, output):
    repo = repo.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output).resolve()
    if not config_path.is_relative_to(repo) or not output.is_relative_to(repo) or output.exists():
        raise ValueError("config/output must be local and output must be fresh")
    start = time.monotonic()
    config = json.loads(config_path.read_text())
    config_pin, sources = _pin(config_path, repo), [_pin(repo / p, repo) for p in SOURCES]
    metrics, artifacts = calculate(repo, config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".paired-core-", dir=output.parent) as temp:
        work = Path(temp)
        for name, value in artifacts.items():
            if name.endswith("jsonl"):
                (work / name).write_text(
                    "".join(json.dumps(r, sort_keys=True) + "\n" for r in value)
                )
            else:
                _write(work / name, value)
        for pin in [config_pin, *sources, *config["inputs"].values()]:
            resolve_pin(pin, repo, label="final paired core input")
        result = {
            "schema_version": SCHEMA,
            "status": "numerical_complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.monotonic() - start,
            "config": config_pin,
            "inputs": config["inputs"],
            "sources": sources,
            "policy": POLICY,
            **metrics,
            "artifacts": {
                name: {
                    "path": str((output / name).relative_to(repo)),
                    "sha256": str(sha256_file(work / name)),
                }
                for name in artifacts
            },
            "limits": [
                "This tests terminal constraints using identical logits, not in-trajectory guidance or learned improvement.",
                "Core alternatives come from computed-transform-consistent TRAIN semantics, not independently qualified L2/L3 routes.",
                "Core slot normalization is shared by the paired arms. Comparisons with earlier layouts are distributional.",
                "Unsupported core serialization abstains within the original closure budget, with no redraw or fallback treatment success.",
                "All twelve libraries are reported separately; not every source library scope is a complete ionizable lipid.",
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
        or result["policy"] != POLICY
        or result["status"] != "numerical_complete"
    ):
        raise ValueError("paired core result contract differs")
    config = json.loads(resolve_pin(result["config"], repo, label="paired core config").read_text())
    if config["inputs"] != result["inputs"] or config["source_pins"] != result["sources"]:
        raise ValueError("paired core inputs or sources substituted")
    for pin in [*result["inputs"].values(), *result["sources"], *result["artifacts"].values()]:
        resolve_pin(pin, repo, label="paired core verification")
    metrics, artifacts = calculate(repo, repo / result["config"]["path"])
    if any(result[k] != value for k, value in metrics.items()) or set(artifacts) != set(
        result["artifacts"]
    ):
        raise ValueError("paired core metrics differ")
    for name, expected in artifacts.items():
        content = (repo / result["artifacts"][name]["path"]).read_text()
        actual = (
            [json.loads(line) for line in content.splitlines()]
            if name.endswith("jsonl")
            else json.loads(content)
        )
        if actual != expected:
            raise ValueError("paired core ledger differs: " + name)
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
                for k in ("status", "comparison", "qualification_by_family")
                if k in result
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
