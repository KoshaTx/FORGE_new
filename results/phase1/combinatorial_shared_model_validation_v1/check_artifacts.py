"""Read-only integrity checks for the twelve-library shared-model artifacts."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache


class ArtifactCheckError(ValueError):
    pass


def _verify_artifact(record: dict[str, Any], repo: Path, label: str) -> Path:
    if set(record) not in ({"path", "sha256"}, {"path", "sha256", "bytes"}):
        raise ArtifactCheckError(f"malformed artifact record: {label}")
    path = (repo / record["path"]).resolve()
    if not path.is_relative_to(repo) or not path.is_file():
        raise ArtifactCheckError(f"artifact is missing or external: {label}")
    if str(sha256_file(path)) != record["sha256"]:
        raise ArtifactCheckError(f"artifact hash changed: {label}")
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise ArtifactCheckError(f"artifact size changed: {label}")
    return path


def _verify_result(path: Path, repo: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if result.get("status") != "pass" or not all(result.get("gates", {}).values()):
        raise ArtifactCheckError(f"result is not passing: {path}")
    resolve_pin(result["config"], repo, label=f"{path.name}/config")
    for label, pin in sorted(result["inputs"].items()):
        resolve_pin(pin, repo, label=f"{path.name}/{label}")
    for source, digest in sorted(result["sources"].items()):
        if str(sha256_file(repo / source)) != digest:
            raise ArtifactCheckError(f"source hash changed: {source}")
    for label, record in sorted(result.get("artifacts", {}).items()):
        _verify_artifact(record, repo, f"{path.name}/{label}")
    return result


def check(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    semantic_path = repo / "results/phase1/combinatorial_program_semantics_v3/result.json"
    cache_path = repo / "results/phase1/combinatorial_program_cache_v2/result.json"
    smoke_path = repo / "results/phase1/combinatorial_training_smoke_v2/result.json"
    semantic = _verify_result(semantic_path, repo)
    cache_result = _verify_result(cache_path, repo)
    smoke = _verify_result(smoke_path, repo)
    semantic_records = _verify_artifact(
        semantic["artifacts"]["semantic_records.jsonl.gz"], repo, "semantic records"
    )
    semantic_failures = _verify_artifact(
        semantic["artifacts"]["semantic_failures.jsonl.gz"], repo, "semantic failures"
    )
    with gzip.open(semantic_records, "rt") as handle:
        semantic_count = sum(1 for _ in handle)
    with gzip.open(semantic_failures, "rt") as handle:
        abstention_count = sum(1 for _ in handle)
    packed_path = _verify_artifact(cache_result["artifacts"]["cache.npz"], repo, "cache")
    with SynthesisProgramProductionCache(packed_path) as packed:
        programs = tuple(packed.vocabulary.program_states[1:])
        measure = packed.training_measure({program: 1.0 / len(programs) for program in programs})
        family_mass = {
            program: float(measure[packed.indices(program_id=program, fold="train")].sum())
            for program in programs
        }
        if (
            len(packed) != semantic_count
            or packed.fold_counts() != semantic["semantic_fold_counts"]
            or any(not np.isclose(value, 1.0 / len(programs)) for value in family_mass.values())
        ):
            raise ArtifactCheckError("packed cache content differs from semantic qualification")
    if (
        semantic_count != semantic["representation"]["records"]
        or abstention_count != semantic["representation"]["abstentions"]
        or cache_result["records"] != semantic_count
        or len(smoke["selected_records"]) != len(programs)
        or smoke["optimizer_steps_executed"] != 2
        or smoke["generator_sampling_calls"] != 0
    ):
        raise ArtifactCheckError("cross-artifact counts changed")
    return {
        "status": "pass",
        "semantic_records": semantic_count,
        "semantic_abstentions": abstention_count,
        "cache_records": cache_result["records"],
        "programs": list(programs),
        "equal_family_training_mass": family_mass,
        "smoke_optimizer_steps": smoke["optimizer_steps_executed"],
        "generator_sampling_calls": smoke["generator_sampling_calls"],
        "inputs": {
            "semantic_result": {
                "path": str(semantic_path.relative_to(repo)),
                "sha256": str(sha256_file(semantic_path)),
            },
            "cache_result": {
                "path": str(cache_path.relative_to(repo)),
                "sha256": str(sha256_file(cache_path)),
            },
            "smoke_result": {
                "path": str(smoke_path.relative_to(repo)),
                "sha256": str(sha256_file(smoke_path)),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.repo_root)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
