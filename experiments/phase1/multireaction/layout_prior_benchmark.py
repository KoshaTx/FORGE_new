"""Reproduce the factorized-layout sampling benchmark on the pinned production cache."""

from __future__ import annotations

import argparse
import platform
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import pin_record
from forge.core.io import write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.synthesis_program_layout import (
    SynthesisProgramLayoutPrior,
    _compile_program_distribution,
    _RecordSummary,
    _summarize_program,
    _WeightedSupport,
)

SCHEMA_VERSION = "forge.factorized_layout_sampling_benchmark.v1"
UGI_PROGRAM = "ugi_3cr_agile"


def _legacy_choice(
    values: Sequence[Any],
    weights: Sequence[float],
    rng: np.random.Generator,
) -> Any:
    return _WeightedSupport.build(values, weights).sample(rng)


def _legacy_sample_fields(
    summaries: Sequence[_RecordSummary],
    rng: np.random.Generator,
) -> tuple[Any, ...]:
    """Execute the pre-repair repeated-scan algorithm for a measured reference."""

    weights = [value.weight for value in summaries]
    depth = int(_legacy_choice([value.depth for value in summaries], weights, rng))
    closure_count = int(_legacy_choice([value.closure_count for value in summaries], weights, rng))
    depth_rows = [value for value in summaries if value.depth == depth]
    depth_weights = [value.weight for value in depth_rows]
    multiplicity_pattern = _legacy_choice(
        [tuple(sorted(Counter(role for role, _ in value.blocks).items())) for value in depth_rows],
        depth_weights,
        rng,
    )
    blocks: list[tuple[int, int]] = []
    for role_state, multiplicity in multiplicity_pattern:
        sizes: list[int] = []
        size_weights: list[float] = []
        for row in depth_rows:
            local = [size for role, size in row.blocks if role == role_state]
            for size in local:
                sizes.append(size)
                size_weights.append(row.weight / len(local))
        size = int(_legacy_choice(sizes, size_weights, rng))
        blocks.extend((int(role_state), size) for _ in range(int(multiplicity)))
    core_pattern = _legacy_choice(
        [value.core_pattern for value in depth_rows],
        depth_weights,
        rng,
    )
    fixed_signature = _legacy_choice(
        [value.fixed_signature for value in depth_rows],
        depth_weights,
        rng,
    )
    return depth, closure_count, blocks, core_pattern, fixed_signature


def benchmark_layout_prior(
    cache_path: Path,
    output_path: Path,
    repo: Path,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Time old repeated scans against the compiled categorical implementation."""

    if draws < 1 or seed < 0:
        raise ValueError("benchmark draws and seed must be nonnegative, with at least one draw")
    cache = SynthesisProgramProductionCache(cache_path)
    try:
        summaries, _ = _summarize_program(cache, UGI_PROGRAM)
    finally:
        cache.close()

    legacy_rng = np.random.default_rng(seed)
    started = time.perf_counter()
    for _ in range(draws):
        _legacy_sample_fields(summaries, legacy_rng)
    legacy_seconds = time.perf_counter() - started

    started = time.perf_counter()
    compiled_distribution = _compile_program_distribution(summaries)
    compile_seconds = time.perf_counter() - started
    prior = object.__new__(SynthesisProgramLayoutPrior)
    prior._distributions = {UGI_PROGRAM: compiled_distribution}
    compiled_rng = np.random.default_rng(seed)
    started = time.perf_counter()
    for _ in range(draws):
        prior._sample_fields(UGI_PROGRAM, compiled_rng)
    compiled_seconds = time.perf_counter() - started

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "program_id": UGI_PROGRAM,
        "draws": draws,
        "seed": seed,
        "training_summaries": len(summaries),
        "legacy_repeated_scan_seconds": legacy_seconds,
        "one_time_compilation_seconds": compile_seconds,
        "compiled_support_seconds": compiled_seconds,
        "speedup": legacy_seconds / compiled_seconds,
        "input": pin_record(cache_path, repo),
        "implementation": pin_record(Path(__file__), repo),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "interpretation": (
            "The reference reproduces the removed per-draw scans over all Ugi training summaries. "
            "The compiled path samples immutable weighted supports with the same number of draws."
        ),
    }
    write_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("results/phase1/shared_synthesis_program_production_cache_v1/cache.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase1/factorized_layout_sampling_benchmark_v1/result.json"),
    )
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3561515994897245985)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    benchmark_layout_prior(
        (repo / args.cache).resolve(),
        (repo / args.output).resolve(),
        repo,
        draws=args.draws,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "benchmark_layout_prior"]
