"""Complete generated precursor interiors with registry-qualified reaction connections."""

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

from rdkit import rdBase

from experiments.phase1.multireaction.combinatorial_generation import _pin, _write
from experiments.phase1.multireaction.combinatorial_graph_reuse import compare
from experiments.phase1.multireaction.combinatorial_occurrence_reuse import (
    verify as verify_baseline,
)
from experiments.phase1.multireaction.combinatorial_reuse_pilot import _digest, assess_rows
from forge.assembly.families import load_assembly_libraries
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.precursor_reuse import PrecursorReuseError
from forge.model.precursor_reuse_admission import admit_completions
from forge.model.program_connection_reuse import (
    propose_program_connections,
    qualify_program_connections,
)
from forge.model.synthesis_program_sampling import load_synthesis_program_checkpoint

SCHEMA = "forge.combinatorial_connection_reuse.v1"
SOURCES = (
    "forge/model/program_connection_reuse.py",
    "experiments/phase1/multireaction/combinatorial_connection_reuse.py",
)
POLICY = {
    "target": "all_repeated_programs",
    "source_program_connections_used": True,
    "source_interior_edges_copied": False,
    "connections_must_match_registry_template_core_positions": True,
    "original_atom_and_edge_counts_preserved": True,
    "all_exact_witnesses_and_atom_mappings_must_agree": True,
    "source_atom_values_copied": False,
    "original_exact_outputs_immutable": True,
    "reuse_existing_novelty_and_multiplicity_admission": True,
    "gate_changes": False,
    "neural_generation_calls": 0,
    "training_calls": 0,
    "remote_compute": False,
    "heldout_record_use": False,
    "prospective_candidate_selection": False,
    "model_promotion": False,
}


def calculate(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != SCHEMA + "_config" or config.get("policy") != POLICY:
        raise PrecursorReuseError("connection experiment policy/schema changed")
    if config.get("maximum_canonical_matches") != 4096:
        raise PrecursorReuseError("connection canonical mapping bound changed")
    previous_path = resolve_pin(config["baseline_result"], repo, label="previous admitted pipeline")
    verify_baseline(repo, previous_path)
    previous = json.loads(previous_path.read_text())
    admitted_path = resolve_pin(previous["baseline_result"], repo, label="admitted baseline")
    admitted = json.loads(admitted_path.read_text())
    parent_path = resolve_pin(admitted["parent_result"], repo, label="parent terminal traces")
    parent = json.loads(parent_path.read_text())
    sampling = json.loads(resolve_pin(parent["config"], repo, label="saved sampling").read_text())[
        "sampling"
    ]
    dataset = json.loads((repo / parent["inputs"]["dataset_config"]["path"]).read_text())
    libraries = load_assembly_libraries(
        [(repo / p["path"], p["sha256"]) for p in parent["registry_inputs"]],
        expected_families=dataset["programs"],
    )

    def read(result, name):
        path = resolve_pin(result["artifacts"][name], repo, label=name)
        text = path.read_text()
        return (
            [json.loads(s) for s in text.splitlines()]
            if name.endswith("jsonl")
            else json.loads(text)
        )

    layouts = read(parent, "layouts.json")
    terminals = read(parent, "terminals.jsonl")
    originals = [
        {**r, "arm": "baseline"}
        for r in read(previous, "attempts.jsonl")
        if r["arm"] == "graph_reuse"
    ]
    with SynthesisProgramProductionCache(repo / parent["inputs"]["cache"]["path"]) as cache:
        if any(cache.fold(row["index"]) != "train" for row in layouts):
            raise PrecursorReuseError("non-TRAIN connection layout")
        records = cache.records([r["index"] for r in layouts])
        train = {cache.canonical_smiles(int(i)) for i in cache.indices(fold="train")}
    components = {
        r["constitution_id"]
        for r in json.loads((repo / parent["inputs"]["component_partitions"]["path"]).read_text())
        if r["fold"] == "train"
    }
    _, vocabulary, atoms, *_ = load_synthesis_program_checkpoint(
        repo / parent["inputs"]["checkpoint"]["path"], device="cpu"
    )
    plans = []
    with rdBase.BlockLogs():
        for record in records:
            plans.append(
                qualify_program_connections(
                    record,
                    libraries[record.program_id],
                    dataset["programs"][record.program_id],
                    dataset["limits"],
                    vocabulary,
                )
            )
        proposals = [
            propose_program_connections(
                record=r,
                plan=p,
                terminal=t,
                original=o,
                atoms=atoms,
                adapter=libraries[r.program_id],
                policy=dataset["programs"][r.program_id],
                limits=dataset["limits"],
            )
            for r, p, t, o in zip(records, plans, terminals, originals, strict=True)
        ]
        decisions = admit_completions(originals, proposals, train)
        raw = [
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
        completed = assess_rows(raw, records, "graph_reuse", dataset, libraries, train, components)
    rows = originals + completed
    metrics = compare(rows, sorted(libraries))
    if not metrics["all_family_preservation_screen_passed"]:
        raise PrecursorReuseError("connection completion violated a preservation invariant")
    source_rows = [r for r in read(parent, "attempts.jsonl") if r["arm"] == "baseline"]
    return {
        **metrics,
        "comparison_vs_frozen_model": compare(source_rows + completed, sorted(libraries)),
        "qualification": dict(Counter(p.status for p in plans)),
        "qualification_reasons": dict(Counter(p.reason for p in plans if p.reason)),
        "proposal_dispositions": dict(
            Counter(p["status"] for entry in proposals for p in entry["proposals"])
        ),
        "admission_dispositions": dict(Counter(d["disposition"] for d in decisions)),
        "additional_assembly_checks": sum(
            entry["additional_assembly_checks"] for entry in proposals
        ),
        "connection_completion_proposals": sum(len(entry["proposals"]) for entry in proposals),
        "reference_train_products": len(train),
        "reference_train_components": len(components),
        "saved_generation_sampling": sampling,
        "new_random_sampling": False,
    }, {
        "attempts.jsonl": rows,
        "plans.json": [asdict(p) for p in plans],
        "proposals.jsonl": proposals,
        "admissions.jsonl": decisions,
    }


def run(repo: Path, config_path: Path, output_dir: Path):
    repo = repo.resolve()
    config_path = (repo / config_path).resolve()
    output = (repo / output_dir).resolve()
    if not config_path.is_relative_to(repo) or not output.is_relative_to(repo) or output.exists():
        raise PrecursorReuseError("config/output must be local and output fresh")
    pins = [_pin(repo / p, repo) for p in SOURCES]
    config_pin = _pin(config_path, repo)
    config = json.loads(config_path.read_text())
    start = time.monotonic()
    metrics, artifacts = calculate(repo, config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".connection-reuse-", dir=output.parent) as temp:
        work = Path(temp)
        for name, values in artifacts.items():
            if name.endswith("jsonl"):
                (work / name).write_text(
                    "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in values)
                )
            else:
                _write(work / name, values)
        for pin in [config_pin, *pins, config["baseline_result"]]:
            resolve_pin(pin, repo, label="final connection input")
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
                "Exact occurrence mapping and source-derived L1 connections are additional TRAIN layout context; arbitrary new-layout generation is not established.",
                "Only qualified cross-occurrence connection edges come from source context. Atom states and internal precursor edges come from the original generated donor; accumulator interiors remain generated.",
                "The program connections must match registry product-template core positions and form a tree over the accumulator and co-reactant occurrences. Ambiguous source correspondences abstain.",
                "This is extra bounded constrained completion, not a learned checkpoint improvement or a matched-total-budget comparison. No lipid realism or L2/L3 closure is established.",
                "Novelty and diversity admission uses explicit TRAIN membership and family multiplicities. All original failures and rejected proposals remain in the ledger.",
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
        raise PrecursorReuseError("connection result contract differs")
    for pin in [
        result["config"],
        result["baseline_result"],
        *result["sources"],
        *result["artifacts"].values(),
    ]:
        resolve_pin(pin, repo, label="connection verification")
    config_path = repo / result["config"]["path"]
    if json.loads(config_path.read_text())["baseline_result"] != result["baseline_result"]:
        raise PrecursorReuseError("connection baseline substituted")
    metrics, artifacts = calculate(repo, config_path)
    if any(result[key] != value for key, value in metrics.items()):
        raise PrecursorReuseError("connection summary/decision differs from recomputation")
    if set(artifacts) != set(result["artifacts"]):
        raise PrecursorReuseError("connection artifact set differs")
    for name, expected in artifacts.items():
        text = (repo / result["artifacts"][name]["path"]).read_text()
        actual = (
            [json.loads(s) for s in text.splitlines()]
            if name.endswith("jsonl")
            else json.loads(text)
        )
        if _digest(expected) != _digest(actual):
            raise PrecursorReuseError("connection ledger differs from recomputation: " + name)
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
