"""Reproducible CPU benchmark and output-identity digest for the Ugi assessment suite.

The production comparison assesses 3,072 attempts per program per seed.  Assessment is CPU work
and its verdicts are evidence, so a performance change is only admissible when every emitted row
and metric is byte-identical.  This module builds one deterministic, content-addressed attempt
ledger from already hash-pinned inputs, runs the complete assessor suite over it, and returns both
a per-phase timing record and a SHA-256 digest of every emitted row and metric.

The ledger is derived, never sampled from a model: it draws real Ugi corpus products, real observed
R0 lipids, disconnected and unparseable products, and native invalid/failed attempts so that the
valid, abstained, ambiguous, disconnected and failed branches are all exercised.  Nothing here
selects a candidate, calls a planner, or touches a paid accelerator.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from forge.assembly import Ugi3AssemblyAdapter
from forge.core.hashing import sha256_file, sha256_json
from forge.core.io import iter_csv, read_json_object
from forge.model.common_lipid_realism import (
    RealismPolicy,
    assess_lipid_realism,
    build_realism_reference,
)
from forge.model.common_ugi_benchmark import (
    CommonUgiAttempt,
    assess_common_ugi_attempts,
    load_attempt_ledger,
    write_attempt_ledger,
)
from forge.synthesis.assessment.common_route_evidence import (
    assess_common_route_evidence,
    load_frozen_component_evidence,
)

BENCHMARK_SCHEMA = "forge.ugi_assessment_benchmark.v1"
METHOD_ID = "assessment_benchmark"
ATTEMPTS_PER_SEED = 3072

UGI_PROTOCOL = Path("configs/multireaction/common_ugi_baseline_protocol_v1.json")
UGI_ASSESSMENT_CONFIG = Path("configs/multireaction/common_ugi_assessment_v1.json")
REALISM_CONFIG = Path("configs/multireaction/common_lipid_realism_v1.json")
R0_CONSTITUTIONAL = Path("results/m0_03/r0_constitutional.csv.gz")

# Composition of one benchmark ledger.  The counts are frozen so that two runs of this module on
# the same repository produce the same input bytes.
UGI_PRODUCTS = 2000
OBSERVED_LIPIDS = 800
DISCONNECTED_PRODUCTS = 64
UNPARSEABLE_PRODUCTS = 32
NATIVE_INVALID = 88
NATIVE_FAILED = 88

UNPARSEABLE_SMILES = (
    "C1CCCC",
    "C(((C",
    "CCN(C",
    "[Xx]CC",
)


class AssessmentBenchmarkError(ValueError):
    """The benchmark cannot be built from the pinned inputs."""


def _rank(seed: int, population: str, key: str) -> str:
    return hashlib.sha256(f"{seed}|{population}|{key}".encode()).hexdigest()


def _select(
    rows: Sequence[tuple[str, str]], *, seed: int, population: str, limit: int
) -> list[str]:
    """Rank rows by a seeded digest and take a fixed prefix, exactly like the frozen samplers."""

    ordered = sorted(rows, key=lambda row: (_rank(seed, population, row[0]), row[0]))
    if len(ordered) < limit:
        raise AssessmentBenchmarkError(
            f"{population} has {len(ordered)} rows, fewer than the required {limit}"
        )
    return [row[1] for row in ordered[:limit]]


def _corpus_rows(assignments: Path, roles: Sequence[str]) -> list[dict[str, str]]:
    return [
        {
            "product_id": row["product_id"],
            "canonical_product_smiles": row["canonical_product_smiles"],
            **{role: row[f"{role}_smiles"] for role in roles},
        }
        for row in iter_csv(assignments)
    ]


def _observed_rows(r0_path: Path) -> list[tuple[str, str]]:
    return [
        (row["r0_structure_id"], row["canonical_constitutional_smiles"])
        for row in iter_csv(r0_path)
        if row.get("r0_pretraining_eligible") == "True"
    ]


def build_benchmark_ledger(
    *,
    assignments: Path,
    r0_path: Path,
    roles: Sequence[str],
    seed: int,
) -> tuple[CommonUgiAttempt, ...]:
    """Build one deterministic attempt ledger that exercises every assessment branch."""

    from rdkit import Chem, rdBase

    corpus = _corpus_rows(assignments, roles)
    by_id = {row["product_id"]: row for row in corpus}
    product_ids = _select(
        [(row["product_id"], row["product_id"]) for row in corpus],
        seed=seed,
        population="ugi_products",
        limit=UGI_PRODUCTS + 2 * DISCONNECTED_PRODUCTS,
    )
    kept = product_ids[:UGI_PRODUCTS]
    joined_left = product_ids[UGI_PRODUCTS : UGI_PRODUCTS + DISCONNECTED_PRODUCTS]
    joined_right = product_ids[UGI_PRODUCTS + DISCONNECTED_PRODUCTS :]
    observed = _select(
        _observed_rows(r0_path),
        seed=seed,
        population="observed_lipids",
        limit=OBSERVED_LIPIDS,
    )

    records: list[tuple[str, str | None, tuple[str, ...]]] = []
    with rdBase.BlockLogs():
        for product_id in kept:
            row = by_id[product_id]
            visible: tuple[str, ...] = ()
            # Half of the corpus attempts declare their own precursors as method-visible so both
            # the open-ended and the closed branch of the open-endedness metric are exercised.
            if int(_rank(seed, "visibility", product_id)[:2], 16) < 128:
                canonical = []
                for role in roles:
                    molecule = Chem.MolFromSmiles(row[role])
                    if molecule is None:
                        raise AssessmentBenchmarkError(f"corpus component is invalid: {product_id}")
                    canonical.append(
                        f"{role}:"
                        + Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
                    )
                visible = tuple(sorted(canonical))
            records.append(("generated", row["canonical_product_smiles"], visible))
    for smiles in observed:
        records.append(("generated", smiles, ()))
    for left, right in zip(joined_left, joined_right, strict=True):
        records.append(
            (
                "generated",
                f"{by_id[left]['canonical_product_smiles']}."
                f"{by_id[right]['canonical_product_smiles']}",
                (),
            )
        )
    for index in range(UNPARSEABLE_PRODUCTS):
        records.append(("generated", UNPARSEABLE_SMILES[index % len(UNPARSEABLE_SMILES)], ()))
    records.extend(("invalid", None, ()) for _ in range(NATIVE_INVALID))
    records.extend(("failed", None, ()) for _ in range(NATIVE_FAILED))
    if len(records) != ATTEMPTS_PER_SEED:
        raise AssessmentBenchmarkError(
            f"benchmark ledger composition changed: {len(records)} != {ATTEMPTS_PER_SEED}"
        )
    # Interleave the branches deterministically: the position of each record is decided by a
    # seeded digest of its build index, so the ordering depends on nothing but the seed.
    ordered = [
        record
        for _, record in sorted(
            enumerate(records),
            key=lambda item: (_rank(seed, "attempt_order", str(item[0])), item[0]),
        )
    ]
    attempts = tuple(
        CommonUgiAttempt(
            method_id=METHOD_ID,
            seed=seed,
            attempt_index=index,
            status=status,  # type: ignore[arg-type]
            product_smiles=smiles,
            method_visible_component_ids=visible,
            generator_calls=1,
            reaction_calls=0,
            route_calls=0,
            oracle_calls=0,
            wall_seconds=0.0,
        )
        for index, (status, smiles, visible) in enumerate(ordered)
    )
    return attempts


def materialize_ledger(
    repo: Path,
    output: Path,
    *,
    seed: int,
    roles: Sequence[str],
) -> Path:
    """Write one benchmark ledger to disk so the benchmark input itself is hash-pinnable."""

    if output.is_file():
        return output
    assignments, _ = _protocol_inputs(repo)
    attempts = build_benchmark_ledger(
        assignments=assignments,
        r0_path=repo / R0_CONSTITUTIONAL,
        roles=roles,
        seed=seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_attempt_ledger(output, attempts)
    return output


def _pin(record: object, repo: Path, *, label: str) -> Path:
    if not isinstance(record, Mapping) or "path" not in record:
        raise AssessmentBenchmarkError(f"{label} pin is malformed")
    path = repo / str(record["path"])
    if not path.is_file():
        raise AssessmentBenchmarkError(f"{label} input is missing: {path}")
    expected = record.get("sha256")
    if isinstance(expected, str) and str(sha256_file(path)) != expected:
        raise AssessmentBenchmarkError(f"{label} input changed on disk: {path}")
    return path


def _protocol_inputs(repo: Path) -> tuple[Path, Path]:
    protocol = read_json_object(
        repo / UGI_PROTOCOL, error=AssessmentBenchmarkError, label="common Ugi protocol"
    )
    inputs = protocol["inputs"]
    return (
        _pin(inputs["ugi_assignments"], repo, label="ugi_assignments"),
        _pin(inputs["qualified_ugi_reactions"], repo, label="qualified_ugi_reactions"),
    )


@contextmanager
def _phase(timings: dict[str, float], name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - start)


@contextmanager
def _cpu(timings: dict[str, float], name: str) -> Iterator[None]:
    """Record process CPU seconds alongside wall time so a loaded host stays interpretable."""

    start = time.process_time()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + (time.process_time() - start)


def run_assessment_suite(
    repo: Path,
    attempts_path: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    """Run the complete CPU assessor suite and digest every emitted row and metric."""

    timings: dict[str, float] = {}
    cpu: dict[str, float] = {}
    assignments, registry = _protocol_inputs(repo)
    assessment_config = read_json_object(
        repo / UGI_ASSESSMENT_CONFIG, error=AssessmentBenchmarkError, label="ugi assessment config"
    )
    route_ledger = _pin(
        assessment_config["inputs"]["component_route_ledger"], repo, label="component_route_ledger"
    )
    realism_config = read_json_object(
        repo / REALISM_CONFIG, error=AssessmentBenchmarkError, label="lipid realism config"
    )
    policy = RealismPolicy.from_mapping(realism_config["policy"])
    r0_path = _pin(realism_config["inputs"]["r0_constitutional"], repo, label="r0_constitutional")
    splits_path = _pin(
        realism_config["inputs"]["r0_fold_assignments"], repo, label="r0_fold_assignments"
    )

    total_start = time.perf_counter()
    cpu_start = time.process_time()
    with _phase(timings, "load_ledger"), _cpu(cpu, "load_ledger"):
        attempts = load_attempt_ledger(
            attempts_path,
            expected_method=METHOD_ID,
            expected_seed=seed,
            expected_attempts=ATTEMPTS_PER_SEED,
        )
    with _phase(timings, "adapter"), _cpu(cpu, "adapter"):
        adapter = Ugi3AssemblyAdapter.from_registry(registry)
    with _phase(timings, "common_ugi_assessment"), _cpu(cpu, "common_ugi_assessment"):
        assessed, common = assess_common_ugi_attempts(
            attempts, adapter=adapter, assignments_path=assignments
        )
    with _phase(timings, "route_evidence"), _cpu(cpu, "route_evidence"):
        evidence = load_frozen_component_evidence(route_ledger)
        route_rows, route = assess_common_route_evidence(assessed, component_evidence=evidence)
    with _phase(timings, "realism_reference"), _cpu(cpu, "realism_reference"):
        reference = build_realism_reference(r0_path, splits_path, policy)
    with _phase(timings, "lipid_realism_assessment"), _cpu(cpu, "lipid_realism_assessment"):
        realism_rows, realism = assess_lipid_realism(attempts, reference, policy)
    timings["total"] = time.perf_counter() - total_start
    cpu["total"] = time.process_time() - cpu_start

    return {
        "cpu_seconds": {name: round(value, 6) for name, value in sorted(cpu.items())},
        "timings": {name: round(value, 6) for name, value in sorted(timings.items())},
        "digest": {
            "common_assessed_attempts": str(sha256_json(assessed)),
            "common_assessment": str(sha256_json(common)),
            "route_assessed_attempts": str(sha256_json(route_rows)),
            "route_evidence_assessment": str(sha256_json(route)),
            "realism_assessed_attempts": str(sha256_json(realism_rows)),
            "realism_assessment": str(sha256_json(realism)),
        },
        "payload": {
            "common_assessed_attempts": assessed,
            "common_assessment": common,
            "route_assessed_attempts": route_rows,
            "route_evidence_assessment": route,
            "realism_assessed_attempts": realism_rows,
            "realism_assessment": realism,
        },
    }


def run_benchmark(
    repo: Path,
    *,
    seeds: Sequence[int],
    ledger_dir: Path,
    keep_payload: bool = False,
) -> dict[str, Any]:
    """Assess one ledger per seed in a single process, mirroring the production run shape."""

    if not seeds:
        raise AssessmentBenchmarkError("at least one seed is required")
    adapter_roles = Ugi3AssemblyAdapter.from_registry(_protocol_inputs(repo)[1]).roles
    ledgers = [
        materialize_ledger(
            repo,
            ledger_dir / f"benchmark_attempts_seed{seed}.jsonl.gz",
            seed=seed,
            roles=adapter_roles,
        )
        for seed in seeds
    ]
    runs: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    for seed, ledger in zip(seeds, ledgers, strict=True):
        run = run_assessment_suite(repo, ledger, seed=seed)
        payload = run.pop("payload")
        if keep_payload:
            run["payload"] = payload
        run["seed"] = seed
        run["attempts_sha256"] = str(sha256_file(ledger))
        runs.append(run)
    wall = time.perf_counter() - wall_start
    return {
        "schema_version": BENCHMARK_SCHEMA,
        "attempts_per_seed": ATTEMPTS_PER_SEED,
        "seeds": list(seeds),
        "cpu_seconds_total": round(time.process_time() - cpu_start, 6),
        "wall_seconds_total": round(wall, 6),
        "wall_seconds_cold_first_seed": runs[0]["timings"]["total"],
        "wall_seconds_warm_mean": (
            round(sum(run["timings"]["total"] for run in runs[1:]) / (len(runs) - 1), 6)
            if len(runs) > 1
            else None
        ),
        "runs": runs,
        "combined_digest": str(sha256_json([run["digest"] for run in runs])),
    }


__all__ = [
    "ATTEMPTS_PER_SEED",
    "AssessmentBenchmarkError",
    "METHOD_ID",
    "build_benchmark_ledger",
    "materialize_ledger",
    "run_assessment_suite",
    "run_benchmark",
]
