"""Extend the authenticated completion pipeline to exact, joined precursor occurrences."""

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
from experiments.phase1.multireaction.combinatorial_reuse_admission import verify as verify_baseline
from experiments.phase1.multireaction.combinatorial_reuse_pilot import _digest, assess_rows
from forge.assembly.families import load_assembly_libraries
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.precursor_occurrence_reuse import (
    OccurrenceReusePlan,
    propose_occurrence_reuse,
    qualify_occurrence_reuse,
)
from forge.model.precursor_reuse import PrecursorReuseError
from forge.model.precursor_reuse_admission import admit_completions
from forge.model.synthesis_program_sampling import load_synthesis_program_checkpoint

SCHEMA = "forge.combinatorial_occurrence_reuse.v1"
SOURCES = (
    "forge/assembly/library_occurrences.py",
    "forge/model/precursor_occurrence_reuse.py",
    "experiments/phase1/multireaction/combinatorial_occurrence_reuse.py",
)
POLICY = {
    "target_previous_status": "occurrence_partition_mismatch",
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
        raise PrecursorReuseError("occurrence experiment policy/schema changed")
    if config.get("maximum_canonical_matches") != 4096:
        raise PrecursorReuseError("occurrence canonical mapping bound changed")
    previous_path = resolve_pin(config["baseline_result"], repo, label="previous admitted pipeline")
    verify_baseline(repo, previous_path)
    previous = json.loads(previous_path.read_text())
    parent_path = resolve_pin(previous["parent_result"], repo, label="parent terminal traces")
    parent = json.loads(parent_path.read_text())
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
    previous_plans = read(parent, "reuse_plans.json")
    originals = [
        {**r, "arm": "baseline"}
        for r in read(previous, "attempts.jsonl")
        if r["arm"] == "graph_reuse"
    ]
    with SynthesisProgramProductionCache(repo / parent["inputs"]["cache"]["path"]) as cache:
        if any(cache.fold(row["index"]) != "train" for row in layouts):
            raise PrecursorReuseError("non-TRAIN occurrence layout")
        records = cache.records([r["index"] for r in layouts])
        train = {cache.canonical_smiles(int(i)) for i in cache.indices(fold="train")}
    components = {
        r["constitution_id"]
        for r in json.loads((repo / parent["inputs"]["component_partitions"]["path"]).read_text())
        if r["fold"] == "train"
    }
    _, _, atoms, *_ = load_synthesis_program_checkpoint(
        repo / parent["inputs"]["checkpoint"]["path"], device="cpu"
    )
    plans = []
    with rdBase.BlockLogs():
        for record, old in zip(records, previous_plans, strict=True):
            if old["status"] == POLICY["target_previous_status"]:
                plan = qualify_occurrence_reuse(
                    record,
                    libraries[record.program_id],
                    dataset["programs"][record.program_id],
                    dataset["limits"],
                    config["maximum_canonical_matches"],
                )
            else:
                plan = OccurrenceReusePlan(
                    record.graph.structure_id, (), "previous_" + old["status"]
                )
            plans.append(plan)
        proposals = [
            propose_occurrence_reuse(
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
        raise PrecursorReuseError("occurrence completion violated a preservation invariant")
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
        "graph_copy_proposals": sum(len(entry["proposals"]) for entry in proposals),
        "reference_train_products": len(train),
        "reference_train_components": len(components),
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
    with tempfile.TemporaryDirectory(prefix=".occurrence-reuse-", dir=output.parent) as temp:
        work = Path(temp)
        for name, values in artifacts.items():
            if name.endswith("jsonl"):
                (work / name).write_text(
                    "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in values)
                )
            else:
                _write(work / name, values)
        for pin in [config_pin, *pins, config["baseline_result"]]:
            resolve_pin(pin, repo, label="final occurrence input")
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
                "Exact occurrence provenance is additional source-derived TRAIN layout context, not arbitrary new-layout generation.",
                "The original role-component cache and historical results remain unchanged. New sidecars identify reaction occurrences that share a connected role block.",
                "This is extra bounded constrained completion; no learned checkpoint improvement, total-budget matching, lipid realism or L2/L3 closure is claimed.",
                "Novelty and product-count preservation still use explicit TRAIN identity and current-family multiplicities; original failures remain failures when no admissible proposal exists.",
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
        raise PrecursorReuseError("occurrence result contract differs")
    for pin in [
        result["config"],
        result["baseline_result"],
        *result["sources"],
        *result["artifacts"].values(),
    ]:
        resolve_pin(pin, repo, label="occurrence verification")
    config_path = repo / result["config"]["path"]
    if json.loads(config_path.read_text())["baseline_result"] != result["baseline_result"]:
        raise PrecursorReuseError("occurrence baseline substituted")
    metrics, artifacts = calculate(repo, config_path)
    if any(result[key] != value for key, value in metrics.items()):
        raise PrecursorReuseError("occurrence summary/decision differs from recomputation")
    if set(artifacts) != set(result["artifacts"]):
        raise PrecursorReuseError("occurrence artifact set differs")
    for name, expected in artifacts.items():
        text = (repo / result["artifacts"][name]["path"]).read_text()
        actual = (
            [json.loads(s) for s in text.splitlines()]
            if name.endswith("jsonl")
            else json.loads(text)
        )
        if _digest(expected) != _digest(actual):
            raise PrecursorReuseError("occurrence ledger differs from recomputation: " + name)
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
