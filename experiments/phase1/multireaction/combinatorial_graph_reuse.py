"""Test an explicit graph-copy completion stage against the frozen twelve-library generator."""

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

import numpy as np
import torch
from rdkit import rdBase

from experiments.phase1.multireaction.combinatorial_generation import _pin, _write, summarize
from experiments.phase1.multireaction.combinatorial_generation_verify import (
    verify as verify_baseline,
)
from experiments.phase1.multireaction.combinatorial_reuse_pilot import _digest, assess_rows
from forge.assembly.families import load_assembly_libraries
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.precursor_reuse import PrecursorReuseError, qualify_reuse
from forge.model.precursor_reuse_projection import TerminalTrace, complete_reuse
from forge.model.synthesis_program_sampling import (
    load_synthesis_program_checkpoint,
    sample_synthesis_program_products,
)

SCHEMA = "forge.combinatorial_graph_reuse.v1"
METRICS = (
    "valid_connected",
    "exact_program",
    "unique_valid_products",
    "effective_product_count",
    "unique_exact_products",
    "novel_valid_attempts_vs_all_cache_train",
    "unique_novel_exact_products",
    "exact_with_novel_component_in_every_witness",
)


def compare(rows: list[dict], families: list[str]) -> dict:
    arms = {
        arm: summarize([r for r in rows if r["arm"] == arm], families)
        for arm in ("baseline", "graph_reuse")
    }
    deltas = {
        f: {k: arms["graph_reuse"][f][k] - arms["baseline"][f][k] for k in METRICS}
        for f in families
    }
    depth = []
    for arm in arms:
        for family in families:
            for n in sorted({r["requested_depth"] for r in rows if r["program_id"] == family}):
                local = [
                    r
                    for r in rows
                    if r["arm"] == arm and r["program_id"] == family and r["requested_depth"] == n
                ]
                depth.append(
                    {"arm": arm, "family": family, "depth": n, **summarize(local, [family])[family]}
                )
    return {
        "per_arm": arms,
        "by_depth": depth,
        "per_family_deltas": deltas,
        "all_family_preservation_screen_passed": all(
            v >= 0 for d in deltas.values() for v in d.values()
        ),
        "total_exact_gain": sum(d["exact_program"] for d in deltas.values()),
    }


def validate(config: dict) -> None:
    if config.get("schema_version") != SCHEMA + "_config" or config.get("policy") != {
        "remote_compute": False,
        "training_calls": 0,
        "heldout_record_use": False,
        "gate_changes": False,
        "candidate_selection": False,
        "model_promotion": False,
        "selection": "minimum_graph_edits_then_smiles_then_donor",
        "novelty_used_for_selection": False,
    }:
        raise PrecursorReuseError("graph-copy experiment policy/schema changed")
    if (
        type(config.get("use_saved_layouts")) is not bool
        or type(config.get("cpu_threads")) is not int
        or not 1 <= config["cpu_threads"] <= 8
    ):
        raise PrecursorReuseError("invalid layout/CPU policy")
    for name in ("layout_seed", "flow_seed", "attempts_per_family", "flow_steps", "batch_size"):
        if type(config["sampling"].get(name)) is not int or config["sampling"][name] < 1:
            raise PrecursorReuseError("invalid positive sampling coordinate: " + name)
    if (
        config["sampling"]["attempts_per_family"] > 256
        or not 2 <= config["sampling"]["flow_steps"] <= 64
    ):
        raise PrecursorReuseError("sampling exceeds the local bounded experiment")


def run(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if not config_path.is_relative_to(repo) or not output.is_relative_to(repo) or output.exists():
        raise PrecursorReuseError("config/output must be local and output fresh")
    config = json.loads(config_path.read_text())
    validate(config)
    paths = {k: resolve_pin(p, repo, label=k) for k, p in config["inputs"].items()}
    verify_baseline(repo, paths["baseline_result"])
    baseline = json.loads(paths["baseline_result"].read_text())
    base_config = json.loads((repo / baseline["config"]["path"]).read_text())
    if any(
        config["inputs"][k] != baseline["inputs"][k]
        for k in ("cache", "checkpoint", "dataset_config", "component_partitions")
    ):
        raise PrecursorReuseError("frozen checkpoint/data inputs changed")
    dataset = json.loads(paths["dataset_config"].read_text())
    registries = [dataset["inputs"][k] for k in dataset["registries"]]
    libraries = load_assembly_libraries(
        [(resolve_pin(p, repo, label="registry"), p["sha256"]) for p in registries],
        expected_families=dataset["programs"],
    )
    families = sorted(libraries)
    source_names = sorted(
        set(
            [p["path"] for p in baseline["sources"]]
            + [
                "forge/model/precursor_reuse.py",
                "forge/model/precursor_reuse_projection.py",
                "forge/model/defog_feasibility.py",
                "forge/model/sparse_topology_feasibility.py",
                "experiments/phase1/multireaction/combinatorial_reuse_pilot.py",
                "experiments/phase1/multireaction/combinatorial_graph_reuse.py",
            ]
        )
    )
    source_pins = [_pin(repo / p, repo) for p in source_names]
    config_pin = _pin(config_path, repo)
    sampling = config["sampling"]
    start = time.monotonic()
    previous_threads, previous_determinism = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        with tempfile.TemporaryDirectory(prefix=".graph-reuse-", dir=output.parent) as temporary:
            work = Path(temporary)
            model, vocabulary, atoms, nodes, bonds, _ = load_synthesis_program_checkpoint(
                paths["checkpoint"], device="cpu"
            )
            with SynthesisProgramProductionCache(paths["cache"]) as cache, rdBase.BlockLogs():
                if config["use_saved_layouts"]:
                    layouts = json.loads(
                        (repo / baseline["artifacts"]["layouts.json"]["path"]).read_text()
                    )
                    if sampling != {"layout_seed": base_config["seed"], **base_config["sampling"]}:
                        raise PrecursorReuseError("saved-layout baseline sampling changed")
                    indices = [r["index"] for r in layouts]
                else:
                    rng = np.random.default_rng(sampling["layout_seed"])
                    measure = cache.training_measure({f: 1 / len(families) for f in families})
                    indices = []
                    for f in families:
                        choices = cache.indices(program_id=f, fold="train")
                        indices.extend(
                            int(i)
                            for i in rng.choice(
                                choices,
                                size=sampling["attempts_per_family"],
                                p=measure[choices] / measure[choices].sum(),
                            )
                        )
                    layouts = [
                        {
                            "index": i,
                            "record_id": cache.record_id(i),
                            "family": cache.program_id(i),
                            "depth": cache.record(i).program_depth,
                            "nodes": cache.record(i).node_count,
                            "fold": "train",
                        }
                        for i in indices
                    ]
                if any(cache.fold(i) != "train" for i in indices):
                    raise PrecursorReuseError("nontraining layout selected")
                records = cache.records(indices)
                plans = [
                    qualify_reuse(
                        r,
                        libraries[r.program_id],
                        dataset["programs"][r.program_id],
                        dataset["limits"],
                    )
                    for r in records
                ]
                train_products = {
                    cache.canonical_smiles(int(i)) for i in cache.indices(fold="train")
                }
            train_components = {
                r["constitution_id"]
                for r in json.loads(paths["component_partitions"].read_text())
                if r["fold"] == "train"
            }
            trace = TerminalTrace(model, records, atoms)
            with rdBase.BlockLogs():
                raw, sampler = sample_synthesis_program_products(
                    trace,
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
                if (
                    len(raw) != len(records)
                    or len(trace.terminals) != len(records)
                    or sampler["fixed_state_failures"]
                ):
                    raise PrecursorReuseError("original sampler accounting/fixed-state failure")
                originals = assess_rows(
                    raw, records, "baseline", dataset, libraries, train_products, train_components
                )
                if config["use_saved_layouts"]:
                    expected = [
                        json.loads(s)
                        for s in (repo / baseline["artifacts"]["attempts.jsonl"]["path"])
                        .read_text()
                        .splitlines()
                    ]
                    if _digest([r for r in expected if r["arm"] == "trained"]) != _digest(
                        [{**r, "arm": "trained"} for r in originals]
                    ):
                        raise PrecursorReuseError("instrumentation changed original generation")
                completions = [
                    complete_reuse(
                        record=r,
                        plan=p,
                        terminal=t,
                        original=o,
                        atoms=atoms,
                        adapter=libraries[r.program_id],
                        policy=dataset["programs"][r.program_id],
                        limits=dataset["limits"],
                    )
                    for r, p, t, o in zip(records, plans, trace.terminals, originals, strict=True)
                ]
                # Keep original tensor-reconstruction diagnostics in their own raw ledger. They
                # cannot be assigned to a graph changed by the separate completion stage.
                completed_raw = [
                    {
                        "sample_index": i,
                        "program_id": r.program_id,
                        "layout_record_id": r.graph.structure_id,
                        "canonical_smiles": c["selected_smiles"],
                        "valid": c["selected_smiles"] is not None,
                        "constraint_abstention_reason": raw[i]["constraint_abstention_reason"],
                    }
                    for i, (r, c) in enumerate(zip(records, completions, strict=True))
                ]
                completed = assess_rows(
                    completed_raw,
                    records,
                    "graph_reuse",
                    dataset,
                    libraries,
                    train_products,
                    train_components,
                )
            rows = originals + completed
            for name, values in (
                ("attempts.jsonl", rows),
                ("terminals.jsonl", trace.terminals),
                ("completions.jsonl", completions),
            ):
                with (work / name).open("w") as stream:
                    for value in values:
                        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
            _write(work / "layouts.json", layouts)
            _write(work / "reuse_plans.json", [asdict(p) for p in plans])
            for p in [config_pin, *source_pins, *config["inputs"].values(), *registries]:
                resolve_pin(p, repo, label="final graph-reuse authentication")
            result = {
                "schema_version": SCHEMA,
                "status": "numerical_complete",
                "config": config_pin,
                "inputs": config["inputs"],
                "sources": source_pins,
                "registry_inputs": registries,
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
                "sampler": sampler,
                "reuse_qualification": dict(Counter(p.status for p in plans)),
                "completion_dispositions": dict(Counter(c["disposition"] for c in completions)),
                "proposal_dispositions": dict(
                    Counter(p["status"] for c in completions for p in c["proposals"])
                ),
                "additional_assembly_checks": sum(
                    c["additional_assembly_checks"] for c in completions
                ),
                "graph_copy_proposals": sum(len(c["proposals"]) for c in completions),
                "reference_train_products": len(train_products),
                "reference_train_components": len(train_components),
                "frozen_generation_reproduced": config["use_saved_layouts"],
                "training_calls": 0,
                "extra_neural_generation_calls": 0,
                "model_promoted": False,
                **compare(rows, families),
                "nonclaims": [
                    "This is a bounded completion stage using exact program checks, not an improved learned checkpoint or a zero-cost comparison.",
                    "All original attempts, rejected proposals and fallbacks are recorded; unmodified failures remain failures.",
                    "Correspondence is source-derived TRAIN layout context; target atom/bond/pointer values are not copied.",
                    "Validity/exact preservation is partly guaranteed by original fallback; diversity and novelty remain empirical checks.",
                    "No statistical noninferiority, realism, ionizability, source-executed chemistry or L2/L3 closure is established.",
                    "No heldout records, remote compute or prospective candidate selection were used.",
                ],
            }
            result["artifacts"] = {
                p.name: {
                    "path": str((output / p.name).relative_to(repo)),
                    "sha256": str(sha256_file(p)),
                }
                for p in sorted(work.iterdir())
            }
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
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "total_exact_gain",
                    "all_family_preservation_screen_passed",
                    "completion_dispositions",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
