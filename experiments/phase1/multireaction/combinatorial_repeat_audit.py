"""Audit saved repeated-reaction failures without new generation or gate changes."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rdkit import rdBase

from experiments.phase1.multireaction.combinatorial_generation_verify import (
    verify as verify_generation,
)
from forge.assembly.families import LibraryAssemblyError, load_assembly_libraries
from forge.assembly.library_generation import check_generated_program
from forge.assembly.library_programs import LibraryProgramLimits
from forge.assembly.library_repeat_audit import audit_repeated_program, classify_repeat_attempt
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache

SCHEMA = "forge.combinatorial_repeat_audit.v1"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["family"], row["requested_depth"])].append(row)
    summaries = []
    for (arm, family, depth), local in sorted(groups.items()):
        counts = Counter(row["classification"] for row in local)
        exact = sum(row["original_status"] == "exact_computed_program" for row in local)
        full_diagnostic = sum(
            bool(row["diagnostic"] and row["diagnostic"]["requested_depth_programs"])
            for row in local
        )
        summaries.append(
            {
                "arm": arm,
                "family": family,
                "requested_depth": depth,
                "attempts": len(local),
                "valid_connected": sum(row["original_status"] != "invalid_graph" for row in local),
                "original_exact": exact,
                "original_exact_fraction_all_attempts": exact / len(local),
                "diagnostic_full_depth_attempts": full_diagnostic,
                "diagnostic_fraction_all_attempts": full_diagnostic / len(local),
                "classifications": dict(sorted(counts.items())),
            }
        )
    return summaries


def run(repo_root: Path, config_path: Path, output_dir: Path) -> dict:
    repo = repo_root.resolve()
    config_path, output = (repo / config_path).resolve(), (repo / output_dir).resolve()
    if not config_path.is_relative_to(repo) or not output.is_relative_to(repo) or output.exists():
        raise LibraryAssemblyError("audit config/output must be local and output must be fresh")
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != "forge.combinatorial_repeat_audit_config.v1" or config.get(
        "policy"
    ) != {
        "training_calls": 0,
        "generation_calls": 0,
        "gate_changes": False,
        "remote_compute": False,
        "heldout_record_use": False,
        "candidate_selection": False,
    }:
        raise LibraryAssemblyError("audit policy/schema changed")
    controls_per_stratum = config.get("train_controls_per_family_depth")
    if type(controls_per_stratum) is not int or not 1 <= controls_per_stratum <= 4:
        raise LibraryAssemblyError("train_controls_per_family_depth must be one to four")
    paths = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    source_names = [
        "experiments/phase1/multireaction/combinatorial_repeat_audit.py",
        "experiments/phase1/multireaction/combinatorial_generation_verify.py",
        "forge/assembly/library_repeat_audit.py",
        "forge/assembly/library_generation.py",
        "forge/assembly/families.py",
        "forge/assembly/library_programs.py",
        "forge/assembly/program.py",
        "forge/assembly/registry.py",
    ]
    source_pins = [{"path": name, "sha256": str(sha256_file(repo / name))} for name in source_names]
    config_pin = {
        "path": str(config_path.relative_to(repo)),
        "sha256": str(sha256_file(config_path)),
    }
    start = time.monotonic()
    verified = verify_generation(repo, paths["generation_result"])
    generation = json.loads(paths["generation_result"].read_text())
    if (
        generation["artifacts"]["attempts.jsonl"] != config["inputs"]["attempts"]
        or generation["inputs"]["dataset_config"] != config["inputs"]["dataset_config"]
        or generation["inputs"]["cache"] != config["inputs"]["cache"]
    ):
        raise LibraryAssemblyError("audit inputs differ from the authenticated generation result")
    dataset = json.loads(paths["dataset_config"].read_text())
    registries = [dataset["inputs"][key] for key in dataset["registries"]]
    libraries = load_assembly_libraries(
        [(resolve_pin(pin, repo, label="registry"), pin["sha256"]) for pin in registries],
        expected_families=dataset["programs"],
    )
    families = sorted(
        p for p, policy in dataset["programs"].items() if policy["accumulator_role"] is not None
    )
    if len(families) != config["expected_repeated_families"]:
        raise LibraryAssemblyError("declared repeated-family population changed")
    rows = [json.loads(line) for line in paths["attempts"].read_text().splitlines()]
    audited, controls = [], []

    def check(family: str, smiles: str, depth: int) -> tuple[dict, dict]:
        policy = dataset["programs"][family]
        limits = LibraryProgramLimits(policy["maximum_steps"], **dataset["limits"])
        strict = asdict(
            check_generated_program(
                libraries[family],
                smiles,
                depth=depth,
                accumulator_role=policy["accumulator_role"],
                limits=limits,
            )
        )
        diagnostic = audit_repeated_program(
            libraries[family],
            smiles,
            depth=depth,
            accumulator_role=policy["accumulator_role"],
            limits=limits,
        )
        return strict, diagnostic

    for row in rows:
        family = row["program_id"]
        if family not in families:
            continue
        diagnostic = None
        if row["valid_connected"]:
            strict, diagnostic = check(family, row["canonical_smiles"], row["requested_depth"])
            if json.loads(json.dumps(strict)) != row["assembly"]:
                raise LibraryAssemblyError("frozen identical-repeat reconstruction changed")
        classification = classify_repeat_attempt(row["assembly"]["status"], diagnostic)
        audited.append(
            {
                "arm": row["arm"],
                "sample_index": row["sample_index"],
                "family": family,
                "requested_depth": row["requested_depth"],
                "canonical_smiles": row["canonical_smiles"],
                "original_status": row["assembly"]["status"],
                "classification": classification,
                "diagnostic": diagnostic,
            }
        )
    # Small positive controls from every represented training-family/depth stratum, never selected
    # by diagnostic success. The cache loads packed arrays; only TRAIN records are materialized.
    with SynthesisProgramProductionCache(paths["cache"]) as cache:
        for family in families:
            indices = cache.indices(program_id=family, fold="train")
            depths = cache.arrays["program_depths"][indices]
            for depth in sorted(set(int(d) for d in depths)):
                chosen = indices[depths == depth][:controls_per_stratum]
                for index in chosen:
                    record = cache.record(int(index))
                    strict, diagnostic = check(family, record.graph.canonical_smiles, depth)
                    classification = classify_repeat_attempt(strict["status"], diagnostic)
                    controls.append(
                        {
                            "record_id": cache.record_id(int(index)),
                            "fold": cache.fold(int(index)),
                            "family": family,
                            "requested_depth": depth,
                            "original_status": strict["status"],
                            "diagnostic_status": diagnostic["status"],
                            "classification": classification,
                        }
                    )
    generation_config = json.loads((repo / generation["config"]["path"]).read_text())
    expected = 2 * len(families) * generation_config["sampling"]["attempts_per_family"]
    if len(audited) != expected:
        raise LibraryAssemblyError("audit dropped attempts")
    summary = summarize(audited)
    counts = {
        arm: dict(sorted(Counter(r["classification"] for r in audited if r["arm"] == arm).items()))
        for arm in ("untrained", "trained")
    }
    gates = {
        "all_saved_attempts_in_repeated_families_audited": len(audited) == expected,
        "original_results_reproduced_without_changes": True,
        "all_training_controls_recovered": all(
            r["classification"] == "original_exact_program" for r in controls
        ),
        "all_diagnostics_search_complete": all(
            r["classification"] != "diagnostic_search_abstained" for r in audited
        ),
    }
    for pin in [config_pin, *source_pins, *registries, *config["inputs"].values()]:
        resolve_pin(pin, repo, label="audit input/source unchanged")
    result = {
        "schema_version": SCHEMA,
        "status": "complete" if all(gates.values()) else "complete_with_limits",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.monotonic() - start,
        "config": config_pin,
        "inputs": config["inputs"],
        "registry_pins": registries,
        "sources": source_pins,
        "environment": {"numpy": np.__version__, "rdkit": rdBase.rdkitVersion, "device": "cpu"},
        "prior_generation_verification": verified,
        "audited_attempts": len(audited),
        "families": families,
        "by_arm_family_depth": summary,
        "classifications_by_arm": counts,
        "train_controls": controls,
        "gates": gates,
        "policy": config["policy"],
        "random_sampling_used": False,
        "inherited_sampling_seed": generation_config["sampling"]["flow_seed"],
        "controls_selection": "first cache indices per TRAIN family/depth, before diagnostic results",
        "search_limits": dataset["limits"],
        "precision": "Every returned full or shorter program strictly forward-replays; source-adjudicated chemical precision remains unmeasured.",
        "nonclaims": [
            "Mixed-reactant diagnostic matches are outside the original identical-repeat contract and remain unaccepted.",
            "Failure to reconstruct through this bounded registry does not prove chemical impossibility.",
            "Shorter witnesses show partial registry consistency, not proof that the model omitted a particular bond.",
            "Differences across depths use different sampled layouts and are descriptive, not causal depth effects.",
            "No new training, molecular sampling, holdout evaluation, experimental synthesis or model promotion occurred.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".repeat-audit-", dir=output.parent) as temporary:
        work = Path(temporary)
        with (work / "attempt_audit.jsonl").open("w") as stream:
            for row in audited:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        result["artifacts"] = {
            "attempt_audit.jsonl": {
                "path": str((output / "attempt_audit.jsonl").relative_to(repo)),
                "sha256": str(sha256_file(work / "attempt_audit.jsonl")),
            }
        }
        _write(work / "result.json", result)
        os.rename(work, output)
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
                for k in ("status", "audited_attempts", "classifications_by_arm", "gates")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
