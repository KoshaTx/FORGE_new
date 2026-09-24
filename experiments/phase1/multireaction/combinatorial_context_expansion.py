"""Propose new program contexts using generated precursors and frozen registry transforms."""

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
from rdkit import rdBase

from experiments.phase1.multireaction.combinatorial_generated_layouts import (
    verify as verify_layouts,
)
from experiments.phase1.multireaction.combinatorial_generation import _pin, _write
from forge.assembly.families import load_assembly_libraries
from forge.assembly.library_programs import LibraryProgramLimits
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.generated_program_expansion import expand_generated_program
from forge.model.generated_program_layout import compile_generated_layouts, layout_signatures
from forge.synthesis.engine.qualified_forward import load_qualified_forward_reaction

SCHEMA = "forge.combinatorial_context_expansion.v1"
SOURCES = (
    "forge/model/generated_program_expansion.py",
    "experiments/phase1/multireaction/combinatorial_context_expansion.py",
)
POLICY = {
    "component_pool": "unique_precursors_of_qualified_generated_programs",
    "component_draws": "uniform_by_role_with_replacement",
    "depth_draws": "uniform_allowed_depths",
    "neural_generation_calls": 0,
    "training_calls": 0,
    "remote_compute": False,
    "heldout_use": False,
    "gate_changes": False,
    "all_trials_and_paths_retained": True,
    "proposal_products_are_neural_outputs": False,
}


def calculate(repo, config_path):
    config = json.loads(config_path.read_text())
    if config["schema_version"] != SCHEMA + "_config" or config["policy"] != POLICY:
        raise ValueError("context-expansion contract differs")
    if {p["path"] for p in config["source_pins"]} != set(SOURCES):
        raise ValueError("context-expansion source set differs")
    if type(config["trials_per_family"]) is not int or not 1 <= config["trials_per_family"] <= 256:
        raise ValueError("context expansion exceeds the bounded pilot")
    for pin in config["source_pins"]:
        resolve_pin(pin, repo, label="frozen context expansion source")
    parent_path = resolve_pin(config["layout_audit"], repo, label="generated layout audit")
    verify_layouts(repo, parent_path)
    parent = json.loads(parent_path.read_text())
    semantic = json.loads(
        resolve_pin(parent["inputs"]["semantic_config"], repo, label="semantic config").read_text()
    )
    dataset = json.loads(
        resolve_pin(parent["inputs"]["dataset_config"], repo, label="dataset config").read_text()
    )
    refs = json.loads(
        resolve_pin(
            parent["artifacts"]["train_layout_signatures.json"], repo, label="TRAIN contexts"
        ).read_text()
    )
    refs = {f: set(r["sparse_flow_context"]) for f, r in refs.items()}
    generated = json.loads(
        resolve_pin(
            parent["artifacts"]["layouts.json"], repo, label="generated witnesses"
        ).read_text()
    )
    libraries = load_assembly_libraries(
        [
            (resolve_pin(semantic["inputs"][k], repo, label=k), semantic["inputs"][k]["sha256"])
            for k in semantic["registries"]
        ],
        expected_families=dataset["programs"],
    )
    ugi = load_qualified_forward_reaction(
        resolve_pin(semantic["inputs"]["ugi_registry"], repo, label="Ugi registry"),
        resolve_pin(semantic["inputs"]["ugi_variant"], repo, label="Ugi variant"),
        reaction_id="ugi_3cr_agile",
    )
    pool = {
        f: {
            r: sorted({x["witness"]["components"][r] for x in generated if x["family"] == f})
            for r in adapter.roles
        }
        for f, adapter in libraries.items()
    }
    rng = np.random.default_rng(config["seed"])
    trials, layouts, summary, compiled = [], [], {}, {}
    with (
        SynthesisProgramProductionCache(
            resolve_pin(parent["inputs"]["cache"], repo, label="vocabulary cache")
        ) as cache,
        rdBase.BlockLogs(),
    ):
        for family, adapter in sorted(libraries.items()):
            status, novel = Counter(), set()
            for trial in range(config["trials_per_family"]):
                components = {
                    r: values[int(rng.integers(len(values)))] for r, values in pool[family].items()
                }
                depth = int(rng.choice(semantic["programs"][family]["allowed_depths"]))
                p = dataset["programs"][family]
                expansion = expand_generated_program(
                    adapter,
                    components,
                    depth=depth,
                    accumulator_role=p["accumulator_role"],
                    limits=LibraryProgramLimits(p["maximum_steps"], **dataset["limits"]),
                )
                row = {
                    "family": family,
                    "trial": trial,
                    "depth": depth,
                    "components": components,
                    "expansion": asdict(expansion),
                    "products": [],
                }
                for smiles in sorted({path[-1] for path in expansion.paths}):
                    key = family, depth, smiles
                    if key not in compiled:
                        compiled[key] = compile_generated_layouts(
                            smiles=smiles,
                            depth=depth,
                            adapter=adapter,
                            program_policy=p,
                            semantic_policy=semantic["programs"][family],
                            limits=dataset["limits"],
                            support=semantic["support_bounds"],
                            vocabulary=cache.vocabulary,
                            atoms=cache.atom_vocabulary,
                            ugi_reaction=ugi,
                        )
                    result = compiled[key]
                    status[result.status] += 1
                    entry = {
                        "canonical_smiles": smiles,
                        "status": result.status,
                        "reason": result.reason,
                        "layout_indices": [],
                    }
                    for record, witness in zip(result.records, result.witnesses, strict=True):
                        signatures = layout_signatures(record, maximum_closures=1)
                        is_novel = signatures["sparse_flow_context"] not in refs[family]
                        if is_novel:
                            novel.add(signatures["sparse_flow_context"])
                        entry["layout_indices"].append(len(layouts))
                        layouts.append(
                            {
                                "family": family,
                                "trial": trial,
                                "canonical_smiles": smiles,
                                "depth": depth,
                                "witness": witness,
                                "signatures": signatures,
                                "novel_sparse_flow_context": is_novel,
                            }
                        )
                    row["products"].append(entry)
                trials.append(row)
            summary[family] = {
                "trials": config["trials_per_family"],
                "pool_sizes_by_role": {r: len(v) for r, v in pool[family].items()},
                "product_compilation_status_counts": dict(status),
                "unique_novel_sparse_flow_contexts": len(novel),
            }
    return {
        "by_family": summary,
        "all_twelve_have_new_contexts": len(summary) == 12
        and all(r["unique_novel_sparse_flow_contexts"] for r in summary.values()),
        "forward_expansions": sum(r["expansion"]["expansions"] for r in trials),
    }, {"trials.jsonl": trials, "layouts.json": layouts, "component_pool.json": pool}


def run(repo, config_path, output):
    repo = repo.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output).resolve()
    if output.exists() or not output.is_relative_to(repo) or not config_path.is_relative_to(repo):
        raise ValueError("output must be fresh and paths local")
    config = json.loads(config_path.read_text())
    pin = _pin(config_path, repo)
    start = time.monotonic()
    metrics, artifacts = calculate(repo, config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".context-expansion-", dir=output.parent) as temp:
        work = Path(temp)
        for name, values in artifacts.items():
            if name.endswith("jsonl"):
                (work / name).write_text(
                    "".join(json.dumps(r, sort_keys=True) + "\n" for r in values)
                )
            else:
                _write(work / name, values)
        for value in [pin, config["layout_audit"], *config["source_pins"]]:
            resolve_pin(value, repo, label="final expansion pin")
        result = {
            "schema_version": SCHEMA,
            "status": "numerical_complete",
            "config": pin,
            "sources": config["source_pins"],
            "layout_audit": config["layout_audit"],
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
                "Registry-enumerated proposal products are not neural outputs or evidence of improved model quality.",
                "All proposals use generated precursor pools from an earlier TRAIN-seeded run. No independent holdout, unconditional layout generation or experimental synthesis claim is established.",
                "Novelty is an exact context-membership diagnostic; ignored metadata cannot establish a new sparse-flow context.",
            ],
        }
        _write(work / "result.json", result)
        os.rename(work, output)
    return result


def verify(repo, path):
    repo = repo.resolve()
    result = json.loads((repo / path).read_text())
    if (
        result["schema_version"] != SCHEMA
        or result["status"] != "numerical_complete"
        or result["policy"] != POLICY
    ):
        raise ValueError("context-expansion result contract differs")
    for pin in [
        result["config"],
        result["layout_audit"],
        *result["sources"],
        *result["artifacts"].values(),
    ]:
        resolve_pin(pin, repo, label="context-expansion verification")
    config_path = repo / result["config"]["path"]
    config = json.loads(config_path.read_text())
    if (
        config["layout_audit"] != result["layout_audit"]
        or config["source_pins"] != result["sources"]
    ):
        raise ValueError("context-expansion source inputs substituted")
    metrics, artifacts = calculate(repo, config_path)
    if any(result[k] != v for k, v in metrics.items()) or set(artifacts) != set(
        result["artifacts"]
    ):
        raise ValueError("context-expansion metrics/artifacts differ")
    for name, expected in artifacts.items():
        text = (repo / result["artifacts"][name]["path"]).read_text()
        actual = (
            [json.loads(line) for line in text.splitlines()]
            if name.endswith("jsonl")
            else json.loads(text)
        )
        if actual != json.loads(json.dumps(expected)):
            raise ValueError("context-expansion ledger differs: " + name)
    return {"status": "verified", "result_sha256": str(sha256_file(repo / path))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify(Path("."), args.verify)
    elif args.config and args.output_dir:
        result = run(Path("."), args.config, args.output_dir)
    else:
        parser.error("provide --verify or --config and --output-dir")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "all_twelve_have_new_contexts",
                    "by_family",
                    "forward_expansions",
                )
                if k in result
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
