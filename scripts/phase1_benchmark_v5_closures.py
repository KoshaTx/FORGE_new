#!/usr/bin/env python3
"""Benchmark and audit the exact V5 sparse closure-completion contract."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

from forge.design.flow.defog_feasibility import sha256_file
from forge.design.flow.v5_closure_feasibility import exact_closure_completion

REPO = Path(__file__).resolve().parents[1]
IMPLEMENTATION = REPO / "src/forge/product/v5_closure_feasibility.py"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _timed_maximum_case(repeats: int) -> dict[str, Any]:
    node_count = 282
    tree = tuple((node, node + 1) for node in range(node_count - 1))
    capacities = (2,) * node_count
    durations = []
    peak_bytes = []
    completions = []
    for _ in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()
        result = exact_closure_completion(
            node_count=node_count,
            tree_edges=tree,
            residual_capacities=capacities,
            remaining_closures=12,
        )
        durations.append(time.perf_counter() - start)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_bytes.append(peak)
        completions.append(result.completion)
        if not result.feasible or len(result.completion) != 12:
            raise RuntimeError("maximum supported closure problem did not complete")
    if len(set(completions)) != 1:
        raise RuntimeError("maximum supported closure completion is not deterministic")
    return {
        "node_count": node_count,
        "cycle_rank": 12,
        "repeats": repeats,
        "median_wall_seconds": statistics.median(durations),
        "maximum_wall_seconds": max(durations),
        "maximum_tracemalloc_peak_bytes": max(peak_bytes),
        "completion": [list(edge) for edge in completions[0]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/v5_closure_feasibility_benchmark.json",
    )
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("repeats must be positive")

    maximum = _timed_maximum_case(args.repeats)
    greedy_trap = exact_closure_completion(
        node_count=4,
        tree_edges=((0, 1), (0, 2), (0, 3)),
        residual_capacities=(0, 1, 1, 2),
        remaining_closures=2,
    )
    infeasible = exact_closure_completion(
        node_count=4,
        tree_edges=((0, 1), (1, 2), (2, 3)),
        residual_capacities=(1, 1, 1, 0),
        remaining_closures=2,
    )
    passed = bool(
        greedy_trap.feasible
        and not greedy_trap.greedy_succeeded
        and greedy_trap.exact_states_visited > 0
        and not infeasible.feasible
        and maximum["maximum_wall_seconds"] < 5.0
        and maximum["maximum_tracemalloc_peak_bytes"] < 64 * 1024 * 1024
    )
    result = {
        "schema_version": "phase1_v5_closure_feasibility_benchmark.v1",
        "status": "pass" if passed else "fail",
        "scope": (
            "exact topology-and-coarse-capacity closure completion; "
            "exact atom valence, bond order, aromaticity, and synthesis are out of scope"
        ),
        "implementation": {
            "path": str(IMPLEMENTATION.resolve()),
            "sha256": sha256_file(IMPLEMENTATION),
        },
        "maximum_supported_case": maximum,
        "adversarial_greedy_trap": {
            "feasible": greedy_trap.feasible,
            "greedy_succeeded": greedy_trap.greedy_succeeded,
            "exact_states_visited": greedy_trap.exact_states_visited,
            "completion": [list(edge) for edge in greedy_trap.completion],
        },
        "globally_infeasible_case": {
            "feasible": infeasible.feasible,
            "exact_states_visited": infeasible.exact_states_visited,
        },
        "gates": {
            "maximum_supported_case_wall_seconds_lt": 5.0,
            "maximum_supported_case_peak_bytes_lt": 64 * 1024 * 1024,
            "greedy_trap_requires_exact_fallback": True,
            "infeasible_case_must_be_rejected": True,
        },
    }
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
