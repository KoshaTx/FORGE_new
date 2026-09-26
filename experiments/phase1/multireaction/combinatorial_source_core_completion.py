"""Add bounded source-core proposals to the existing admitted twelve-library pipeline."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase

from experiments.phase1.multireaction.combinatorial_connection_reuse import (
    verify as verify_baseline,
)
from experiments.phase1.multireaction.combinatorial_generation import _pin, _write
from experiments.phase1.multireaction.combinatorial_graph_reuse import compare
from experiments.phase1.multireaction.combinatorial_reuse_pilot import _digest, assess_rows
from forge.assembly.families import load_assembly_libraries
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model._synthesis_sampling import SAMPLING_SOURCE_FILES
from forge.model.precursor_reuse_admission import admit_completions
from forge.model.precursor_reuse_projection import fixed_graph_preserved, state_graph
from forge.model.source_core_completion import SourceCoreTrace
from forge.model.synthesis_program_sampling import (
    load_synthesis_program_checkpoint,
    sample_synthesis_program_products,
)

SCHEMA = "forge.combinatorial_source_core_completion.v1"
SOURCES = (
    "forge/model/source_core_completion.py",
    "forge/model/fixed_closure_decoding.py",
    "forge/model/combinatorial_core_scaffold.py",
    *SAMPLING_SOURCE_FILES,
    "forge/model/precursor_reuse_projection.py",
    "forge/model/precursor_reuse_admission.py",
    "experiments/phase1/multireaction/combinatorial_source_core_completion.py",
)
POLICY = {
    "source_context": "existing_TRAIN_layout_plus_its_computed_program_core",
    "source_precursor_interiors_copied": False,
    "neural_trajectories": "exact_replay_of_existing_layouts_and_noise",
    "core_proposals_per_attempt": 1,
    "retries": 0,
    "original_invalid_and_exact_outputs_immutable": True,
    "original_atom_and_edge_counts_preserved": True,
    "admission": "existing_train_novelty_and_product_multiplicity_rule",
    "gate_changes": False,
    "training_calls": 0,
    "remote_compute": False,
    "heldout_use": False,
    "model_promotion": False,
    "all_attempts_retained": True,
}


def calculate(repo, config_path):
    config = json.loads(config_path.read_text())
    if (
        config["schema_version"] != SCHEMA + "_config"
        or config["policy"] != POLICY
        or config["cpu_threads"] != 2
    ):
        raise ValueError("source-core completion contract differs")
    if config["source_pins"] != [_pin(repo / p, repo) for p in SOURCES]:
        raise ValueError("source-core frozen implementation differs")

    def load(pin):
        return json.loads(resolve_pin(pin, repo, label="source-core ancestry").read_text())

    baseline_path = resolve_pin(config["baseline_result"], repo, label="completed baseline")
    verify_baseline(repo, baseline_path)
    previous = json.loads(baseline_path.read_text())
    occurrence = load(previous["baseline_result"])
    admitted = load(occurrence["baseline_result"])
    parent = load(admitted["parent_result"])
    sampling = load(parent["config"])["sampling"]
    paths = {key: resolve_pin(p, repo, label=key) for key, p in parent["inputs"].items()}
    dataset = json.loads(paths["dataset_config"].read_text())
    libraries = load_assembly_libraries(
        [
            (resolve_pin(p, repo, label="family registry"), p["sha256"])
            for p in parent["registry_inputs"]
        ],
        expected_families=dataset["programs"],
    )
    if len(libraries) != 12 or sampling["flow_steps"] != 32:
        raise ValueError("source-core population or flow budget differs")

    def read(result, name):
        content = resolve_pin(result["artifacts"][name], repo, label=name).read_text()
        return (
            [json.loads(line) for line in content.splitlines()]
            if name.endswith("jsonl")
            else json.loads(content)
        )

    layouts = read(parent, "layouts.json")
    expected_terminals = read(parent, "terminals.jsonl")
    expected_raw = [r for r in read(parent, "attempts.jsonl") if r["arm"] == "baseline"]
    originals = [
        {**r, "arm": "baseline"}
        for r in read(previous, "attempts.jsonl")
        if r["arm"] == "graph_reuse"
    ]
    with SynthesisProgramProductionCache(paths["cache"]) as cache:
        if any(cache.fold(r["index"]) != "train" for r in layouts):
            raise ValueError("source-core layout is not TRAIN")
        records = cache.records([r["index"] for r in layouts])
        if any(
            r.graph.structure_id != entry["record_id"]
            for r, entry in zip(records, layouts, strict=True)
        ):
            raise ValueError("source-core layout identity differs")
        train = {cache.canonical_smiles(int(i)) for i in cache.indices(fold="train")}
    components = {
        r["constitution_id"]
        for r in json.loads(paths["component_partitions"].read_text())
        if r["fold"] == "train"
    }
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        model, _, atoms, nodes, bonds, _ = load_synthesis_program_checkpoint(
            paths["checkpoint"], device="cpu"
        )
        trace = SourceCoreTrace(model, records, atoms)
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
            replay = assess_rows(raw, records, "baseline", dataset, libraries, train, components)
            if (
                sampler["fixed_state_failures"]
                or _digest(replay) != _digest(expected_raw)
                or trace.terminals != expected_terminals
            ):
                raise ValueError(
                    "source-core replay differs from saved neural outputs or terminal states"
                )
            candidates = assess_rows(
                trace.core_rows, records, "core_proposal", dataset, libraries, train, components
            )
            proposals = []
            for source, original, candidate, old, new in zip(
                records, originals, candidates, trace.terminals, trace.core_terminals, strict=True
            ):
                if any(
                    original[k] != candidate[k]
                    for k in ("sample_index", "program_id", "layout_record_id")
                ):
                    raise ValueError("source-core proposal is assigned to another attempt")
                p = {
                    "donor": 0,
                    "smiles": candidate["canonical_smiles"],
                    "status": candidate["assembly"]["status"],
                    "atom_edits": None,
                    "bond_edits": None,
                    "edit_reference": "frozen_neural_terminal_before_previous_completions",
                }
                if (
                    original["valid_connected"]
                    and candidate["valid_connected"]
                    and old["state"] is not None
                    and new["state"] is not None
                ):
                    old_nodes, old_edges = state_graph(old["state"])
                    new_nodes, new_edges = state_graph(new["state"])
                    p["source_fixed_graph_preserved"] = fixed_graph_preserved(
                        new_nodes, new_edges, source
                    )
                    p["atom_edits"] = int(np.count_nonzero(old_nodes != new_nodes))
                    p["bond_edits"] = int(np.count_nonzero(np.triu(old_edges != new_edges, k=1)))
                    molecule = Chem.MolFromSmiles(original["canonical_smiles"])
                    p["original_atom_and_edge_counts_preserved"] = bool(
                        molecule is not None
                        and molecule.GetNumAtoms() == len(new_nodes)
                        and molecule.GetNumBonds() == np.count_nonzero(np.triu(new_edges, k=1))
                    )
                    if (
                        not p["source_fixed_graph_preserved"]
                        or not p["original_atom_and_edge_counts_preserved"]
                    ):
                        p["status"] = "graph_invariant_failure"
                elif original["valid_connected"] and p["status"] == "exact_computed_program":
                    raise ValueError("eligible source-core proposal lacks its graph correspondence")
                proposals.append({"proposals": [p]})
            decisions = admit_completions(originals, proposals, train)
            selected = [
                {
                    "sample_index": o["sample_index"],
                    "program_id": o["program_id"],
                    "layout_record_id": o["layout_record_id"],
                    "canonical_smiles": d["selected_smiles"],
                    "valid": d["selected_smiles"] is not None,
                    "constraint_abstention_reason": o["constraint_abstention_reason"],
                }
                for o, d in zip(originals, decisions, strict=True)
            ]
            completed = assess_rows(
                selected, records, "graph_reuse", dataset, libraries, train, components
            )
        comparison = compare(originals + completed, sorted(libraries))
        if not comparison["all_family_preservation_screen_passed"]:
            raise ValueError("source-core admission violated a frozen preservation invariant")
        metrics = {
            **comparison,
            "proposal_comparison": compare(
                originals + [{**r, "arm": "graph_reuse"} for r in candidates], sorted(libraries)
            ),
            "qualification": dict(Counter(reason or "qualified" for reason in trace.qualification)),
            "proposal_statuses": dict(Counter(p["proposals"][0]["status"] for p in proposals)),
            "admission_dispositions": dict(Counter(d["disposition"] for d in decisions)),
            "neural_attempts_replayed": len(records),
            "new_neural_noise_or_layout_draws": False,
            "frozen_neural_terminal_states_identical": True,
            "saved_sampling": sampling,
            "new_core_proposal_program_checks": sum(c["valid_connected"] for c in candidates),
            "promoted": False,
        }
        return metrics, {
            "attempts.jsonl": originals + completed,
            "core_proposals.jsonl": candidates,
            "proposals.jsonl": proposals,
            "admissions.jsonl": decisions,
            "core_terminals.jsonl": trace.core_terminals,
        }
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


def run(repo: Path, config_path: Path, output_dir: Path):
    repo = repo.resolve()
    config_path = (repo / config_path).resolve()
    output = (repo / output_dir).resolve()
    if not config_path.is_relative_to(repo) or not output.is_relative_to(repo) or output.exists():
        raise ValueError("config/output must be local and output fresh")
    pins = [_pin(repo / p, repo) for p in SOURCES]
    config_pin = _pin(config_path, repo)
    config = json.loads(config_path.read_text())
    start = time.monotonic()
    metrics, artifacts = calculate(repo, config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".source-core-", dir=output.parent) as temp:
        work = Path(temp)
        for name, values in artifacts.items():
            if name.endswith("jsonl"):
                (work / name).write_text(
                    "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in values)
                )
            else:
                _write(work / name, values)
        for pin in [config_pin, *pins, config["baseline_result"]]:
            resolve_pin(pin, repo, label="final source-core input")
        result = {
            "schema_version": SCHEMA,
            "status": "numerical_complete",
            "config": config_pin,
            "baseline_result": config["baseline_result"],
            "sources": pins,
            "policy": POLICY,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.monotonic() - start,
            **metrics,
            "nonclaims": [
                "This exposes the existing TRAIN layout's typed reaction core as additional context. Core atom states and core bonds are source-derived; no precursor exterior graph is copied.",
                "Neural trajectories and terminal states reproduce the saved originals exactly. Core proposals add bounded terminal decoding and full-program checking, not training or new layout/noise seeds.",
                "Edit counts refer to the frozen neural terminal before earlier completion stages. There is only one new core proposal per attempt, so edit distance does not select among alternatives.",
                "Admission uses the unchanged TRAIN-novelty and family-multiplicity rule and preserves original invalid/exact outputs. Every proposal and rejection remains visible.",
                "These are previously inspected saved TRAIN populations, not heldout or across-training evidence. No learned improvement, lipid realism or L2/L3 closure is established.",
            ],
            "artifacts": {
                p.name: {
                    "path": str((output / p.name).relative_to(repo)),
                    "sha256": str(sha256_file(p)),
                }
                for p in sorted(work.iterdir())
            },
        }
        _write(work / "result.json", result)
        os.rename(work, output)
    return result


def verify(repo: Path, result_path: Path):
    repo = repo.resolve()
    result_path = (repo / result_path).resolve()
    result = json.loads(result_path.read_text())
    if (
        result.get("schema_version") != SCHEMA
        or result.get("status") != "numerical_complete"
        or result.get("policy") != POLICY
        or {p["path"] for p in result["sources"]} != set(SOURCES)
    ):
        raise ValueError("source-core result contract differs")
    for pin in [
        result["config"],
        result["baseline_result"],
        *result["sources"],
        *result["artifacts"].values(),
    ]:
        resolve_pin(pin, repo, label="source-core verification")
    config_path = repo / result["config"]["path"]
    if json.loads(config_path.read_text())["baseline_result"] != result["baseline_result"]:
        raise ValueError("source-core baseline substituted")
    metrics, artifacts = calculate(repo, config_path)
    if any(result[key] != value for key, value in metrics.items()):
        raise ValueError("source-core summary/decision differs from recomputation")
    if set(artifacts) != set(result["artifacts"]):
        raise ValueError("source-core artifact set differs")
    for name, expected in artifacts.items():
        text = (repo / result["artifacts"][name]["path"]).read_text()
        actual = (
            [json.loads(s) for s in text.splitlines()]
            if name.endswith("jsonl")
            else json.loads(text)
        )
        if _digest(expected) != _digest(actual):
            raise ValueError("source-core ledger differs from recomputation: " + name)
    return {
        "status": "verified",
        "result_sha256": str(sha256_file(result_path)),
        "attempts_recomputed": len(artifacts["attempts.jsonl"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify(args.repo_root, args.verify)
    elif args.config and args.output_dir:
        result = run(args.repo_root, args.config, args.output_dir)
    else:
        parser.error("provide --verify or both --config and --output-dir")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "total_exact_gain",
                    "all_family_preservation_screen_passed",
                    "qualification",
                    "attempts_recomputed",
                )
                if k in result
            }
        )
    )


if __name__ == "__main__":
    main()
