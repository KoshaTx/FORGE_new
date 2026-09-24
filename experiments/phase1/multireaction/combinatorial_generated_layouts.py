"""Audit exact generated program layouts against all TRAIN sampling contexts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rdkit import rdBase

from experiments.phase1.multireaction.combinatorial_connection_reuse import verify as verify_parent
from experiments.phase1.multireaction.combinatorial_generation import _pin, _write
from forge.assembly.families import load_assembly_libraries
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.generated_program_layout import compile_generated_layouts, layout_signatures
from forge.synthesis.engine.qualified_forward import load_qualified_forward_reaction

SCHEMA = "forge.combinatorial_generated_layouts.v1"
SOURCES = (
    "forge/model/generated_program_layout.py",
    "experiments/phase1/multireaction/combinatorial_generated_layouts.py",
    "forge/model/reaction_program_flow.py",
    "forge/model/synthesis_program_graph.py",
    "forge/assembly/library_semantics.py",
    "forge/potency/annotations.py",
)
POLICY = {
    "all_original_attempts_retained": True,
    "all_exact_program_alternatives_retained": True,
    "within_witness_semantic_ambiguity": "abstain",
    "reference_fold": "train",
    "new_random_sampling": False,
    "neural_generation_calls": 0,
    "training_calls": 0,
    "remote_compute": False,
    "gate_changes": False,
}


def calculate(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != SCHEMA + "_config" or config.get("policy") != POLICY:
        raise ValueError("generated-layout audit contract differs")
    paths = {k: resolve_pin(p, repo, label=k) for k, p in config["inputs"].items()}
    parent = json.loads(paths["parent_result"].read_text())
    verify_parent(repo, paths["parent_result"])
    semantic = json.loads(paths["semantic_config"].read_text())
    dataset = json.loads(paths["dataset_config"].read_text())
    checkpoint = json.loads(paths["checkpoint"].read_text())
    if checkpoint["model_config"]["architecture"] != "reaction_program_sparse_whole_lipid_flow":
        raise ValueError("context signature applies only to the pinned sparse flow")
    if checkpoint["inputs"]["cache"] != config["inputs"]["cache"]:
        raise ValueError("checkpoint/cache mismatch")
    registries = [
        (resolve_pin(semantic["inputs"][k], repo, label=k), semantic["inputs"][k]["sha256"])
        for k in semantic["registries"]
    ]
    libraries = load_assembly_libraries(registries, expected_families=dataset["programs"])
    ugi = load_qualified_forward_reaction(
        resolve_pin(semantic["inputs"]["ugi_registry"], repo, label="Ugi semantic registry"),
        resolve_pin(semantic["inputs"]["ugi_variant"], repo, label="Ugi variant"),
        reaction_id="ugi_3cr_agile",
    )
    source = resolve_pin(parent["artifacts"]["attempts.jsonl"], repo, label="parent attempts")
    originals = [json.loads(line) for line in source.read_text().splitlines()]
    originals = [r for r in originals if r["arm"] == "graph_reuse"]
    references = {
        f: {k: set() for k in ("sampler_context", "sparse_flow_context")} for f in libraries
    }
    rows, layouts, compiled = [], [], {}
    with SynthesisProgramProductionCache(paths["cache"]) as cache:
        if cache.metadata["inputs"]["semantic_config"] != config["inputs"]["semantic_config"]:
            raise ValueError("compiler semantic policy differs from the training cache")
        if semantic["support_bounds"] != {
            k: checkpoint["model_config"][k] for k in ("maximum_heavy_atoms", "maximum_closures")
        }:
            raise ValueError("compiler graph support differs from the trained model")
        for index in cache.indices(fold="train"):
            record = cache.record(int(index))
            for kind, signature in layout_signatures(record, maximum_closures=1).items():
                references[record.program_id][kind].add(signature)
        train_products = {cache.canonical_smiles(int(i)) for i in cache.indices(fold="train")}
        with rdBase.BlockLogs():
            for original in originals:
                f, smiles, depth = (
                    original["program_id"],
                    original["canonical_smiles"],
                    original["requested_depth"],
                )
                key = f, smiles, depth
                if key not in compiled:
                    compiled[key] = compile_generated_layouts(
                        smiles=smiles,
                        depth=depth,
                        adapter=libraries[f],
                        program_policy=dataset["programs"][f],
                        semantic_policy=semantic["programs"][f],
                        limits=dataset["limits"],
                        support=semantic["support_bounds"],
                        vocabulary=cache.vocabulary,
                        atoms=cache.atom_vocabulary,
                        ugi_reaction=ugi,
                    )
                result = compiled[key]
                row = {
                    "sample_index": original["sample_index"],
                    "family": f,
                    "canonical_smiles": smiles,
                    "depth": depth,
                    "status": result.status,
                    "reason": result.reason,
                    "witness_count": result.witness_count,
                    "layout_indices": [],
                }
                for record, witness in zip(result.records, result.witnesses, strict=True):
                    signatures = layout_signatures(record, maximum_closures=1)
                    row["layout_indices"].append(len(layouts))
                    layouts.append(
                        {
                            "sample_index": original["sample_index"],
                            "family": f,
                            "canonical_smiles": smiles,
                            "depth": depth,
                            "witness": witness,
                            "signatures": signatures,
                            "product_novel_vs_train": smiles not in train_products,
                            "novel_vs_train": {
                                k: v not in references[f][k] for k, v in signatures.items()
                            },
                        }
                    )
                rows.append(row)
        train_count = len(cache.indices(fold="train"))
    summary = {}
    for family in libraries:
        local = [r for r in rows if r["family"] == family]
        choices = [r for r in layouts if r["family"] == family]
        summary[family] = {
            "original_attempts": len(local),
            "status_counts": dict(Counter(r["status"] for r in local)),
            "qualified_program_layouts": len(choices),
            "unique_novel_product_layouts": len(
                {
                    (r["canonical_smiles"], r["signatures"]["sparse_flow_context"])
                    for r in choices
                    if r["product_novel_vs_train"]
                }
            ),
            "unique_novel_sparse_flow_contexts": len(
                {
                    r["signatures"]["sparse_flow_context"]
                    for r in choices
                    if r["novel_vs_train"]["sparse_flow_context"]
                }
            ),
            "unique_novel_sampler_contexts": len(
                {
                    r["signatures"]["sampler_context"]
                    for r in choices
                    if r["novel_vs_train"]["sampler_context"]
                }
            ),
            "train_reference_unique_contexts": {k: len(v) for k, v in references[family].items()},
        }
    return {
        "by_family": summary,
        "train_reference_records": train_count,
        "all_twelve_families_have_novel_sparse_flow_context": len(summary) == 12
        and all(r["unique_novel_sparse_flow_contexts"] > 0 for r in summary.values()),
    }, {
        "attempts.jsonl": rows,
        "layouts.json": layouts,
        "train_layout_signatures.json": {
            f: {k: sorted(v) for k, v in values.items()} for f, values in references.items()
        },
    }


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
    with tempfile.TemporaryDirectory(prefix=".generated-layouts-", dir=output.parent) as temp:
        work = Path(temp)
        for name, value in artifacts.items():
            if name.endswith("jsonl"):
                (work / name).write_text(
                    "".join(json.dumps(r, sort_keys=True) + "\n" for r in value)
                )
            else:
                _write(work / name, value)
        for pin in [config_pin, *sources, *config["inputs"].values()]:
            resolve_pin(pin, repo, label="final generated layout input")
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
                "Layouts are compiled from earlier generated graphs, not invented independently of all prior TRAIN layout ancestry.",
                "Novel context means absent from all TRAIN cache contexts under the specified signature; morphology ignored by the sparse flow cannot establish novelty alone.",
                "Different exact reaction programs remain alternative hypotheses, not identified experimental routes. Each witness separately passes the unchanged atom-semantics gate.",
                "This qualification audit does not itself demonstrate generation on the new contexts or improve any molecule.",
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
        raise ValueError("generated-layout result contract differs")
    for pin in [
        result["config"],
        *result["inputs"].values(),
        *result["sources"],
        *result["artifacts"].values(),
    ]:
        resolve_pin(pin, repo, label="generated layout verification")
    config = json.loads((repo / result["config"]["path"]).read_text())
    if config["inputs"] != result["inputs"]:
        raise ValueError("generated-layout inputs substituted")
    metrics, artifacts = calculate(repo, repo / result["config"]["path"])
    if any(result[k] != v for k, v in metrics.items()) or set(artifacts) != set(
        result["artifacts"]
    ):
        raise ValueError("generated-layout metrics/artifacts differ")
    for name, expected in artifacts.items():
        text = (repo / result["artifacts"][name]["path"]).read_text()
        actual = (
            [json.loads(s) for s in text.splitlines()]
            if name.endswith("jsonl")
            else json.loads(text)
        )
        if actual != expected:
            raise ValueError("generated-layout ledger differs: " + name)
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
                    "all_twelve_families_have_novel_sparse_flow_context",
                )
                if k in result
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
