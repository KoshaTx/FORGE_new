"""Restartable bounded parallel replay through unchanged, qualified anionic iPhos executor."""

import gzip
import hashlib
import json
import multiprocessing
import os
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus import compose_lipid_current_replay as program
from forge.corpus import compose_lipid_family_replay as family_replay
from forge.corpus.compose_lipid_full_partition import PartitionPreparationCorpus
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORKERS, BATCH_SIZE = 4, 256
_STRUCTURES, _EXECUTORS = None, None


def initialize(structures, executors):
    global _STRUCTURES, _EXECUTORS
    _STRUCTURES, _EXECUTORS = structures, executors


def evaluate(items):
    answer = []
    for item in items:
        prepared = item["preparation"]
        result = family_replay.replay_record(item, _STRUCTURES, _EXECUTORS[prepared["family"]])
        answer.append(
            {
                **prepared,
                "replay": result,
                "training_admitted": False,
                "experimental_execution_admitted": False,
            }
        )
    return answer


def digest(value):
    return hashlib.sha256(compact(value).encode()).hexdigest()


def chunks(reader, families, carried):
    for family in families:
        iterator = (
            item
            for item in reader.iter_preparation_records(family=family)
            if item["preparation"]["target_id"] not in carried
        )
        number = 0
        while items := list(islice(iterator, BATCH_SIZE)):
            yield family, number, items
            number += 1


def write_shard(destination, records, family, number, items_hash, request_hash):
    if len({r["target_id"] for r in records}) != len(records):
        raise ValueError("Parallel replay duplicated a target")
    counts = Counter(r["replay"]["disposition"] for r in records)
    with tempfile.TemporaryDirectory(prefix=".shard-", dir=destination.parent) as tmp:
        stage = Path(tmp)
        ledger = stage / "replay.jsonl.gz"
        with (
            ledger.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0, compresslevel=1) as stream,
        ):
            for row in records:
                stream.write((compact(row) + "\n").encode())
        artifact = pin(ROOT, ledger)
        artifact["path"] = str((destination / ledger.name).relative_to(ROOT))
        result = {
            "schema_version": "forge.compose_lipid_incremental_replay_shard.v1",
            "family": family,
            "number": number,
            "rows": len(records),
            "counts": dict(counts),
            "items_sha256": items_hash,
            "request_sha256": request_hash,
            "artifact": artifact,
            "target_ids_sha256": digest([r["target_id"] for r in records]),
            "training_admitted": False,
        }
        dump(stage / "result.json", result)
        os.rename(stage, destination)
    return result


def main():
    started = time.monotonic()
    output = HERE / "anionic"
    if (output / "result.json").exists():
        raise ValueError("Completed published replay is frozen")
    output.mkdir(exist_ok=True)
    config_path = ROOT / "configs/multireaction/compose_lipid_supplied_iphos_deprotonated_v1.json"
    partition_path = ROOT / "results/phase1/compose_lipid_full_partition_v1/audit-v2/result.json"
    evidence_path = (
        ROOT / "results/phase1/compose_lipid_training_readiness_v7/population-audit.json"
    )
    inputs = {
        name: pin(ROOT, path)
        for name, path in {
            "program_config": config_path,
            "partition": partition_path,
            "previous_evidence": evidence_path,
        }.items()
    }
    implementations = (
        set(family_replay.IMPLEMENTATION)
        | set(program.IMPLEMENTATION)
        | {
            "forge/corpus/compose_lipid_full_partition.py",
            str(Path(__file__).resolve().relative_to(ROOT)),
        }
    )
    implementation = {name: pin(ROOT, ROOT / name) for name in sorted(implementations)}
    request = {
        "schema_version": "forge.compose_lipid_incremental_replay_request.v1",
        "inputs": inputs,
        "implementation": implementation,
        "policy": {
            "seed": 0,
            "device": "cpu",
            "workers": WORKERS,
            "batch_size": BATCH_SIZE,
            "maximum_pending_batches": WORKERS,
            "multiprocessing": "spawn",
            "population": "current_partition_eligible_only",
            "source_roles_quantities_and_ids_modified": False,
            "search_bounds": "unchanged_qualified_program_bounds",
            "record_cap": None,
            "size_filter": None,
            "training_admitted": False,
            "training_calls": 0,
        },
    }
    request_path = output / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError("Cannot resume replay with changed inputs, policy or implementation")
    else:
        dump(request_path, request)
    request_hash = pin(ROOT, request_path)["sha256"]
    print("Authenticating frozen programs and positive/ambiguity controls once", flush=True)
    cfg, _, adapters, programs, bounds, controls = program.load_contract(ROOT, config_path)
    executors = {
        f: {
            "kind": "repeated",
            "adapter": adapters[f],
            "program": programs[f],
            "binding": binding,
            "bounds": bounds,
        }
        for f, binding in cfg["families"].items()
    }
    reader = PartitionPreparationCorpus(ROOT, partition_path)
    catalogue = resolve_pin(reader.result["inputs"]["catalogue"], ROOT, label="complete precursors")
    structures = {r["component_id"]: r["constitution"] for r in rows(catalogue)}
    previous = json.loads(evidence_path.read_text())
    if previous["inputs"]["partition_receipt"] != inputs["partition"]:
        raise ValueError("Carried evidence uses another partition")
    evidence = resolve_pin(previous["artifact"], ROOT, label="previous exact evidence")
    with sqlite3.connect(evidence.as_uri() + "?mode=ro", uri=True) as db:
        carried = {r[0] for r in db.execute("SELECT target_id FROM exact")}
    initialize(structures, executors)
    families = sorted(executors)
    samples = []
    for family in families:
        largest = None
        for n, item in enumerate(reader.iter_preparation_records(family=family)):
            if n < 8:
                samples.append(item)
            if largest is None or item["source"]["heavy_atoms"] > largest["source"]["heavy_atoms"]:
                largest = item
        if largest is not None and largest not in samples:
            samples.append(largest)
    if not samples:
        raise ValueError("Parallel numerical-equivalence check has no eligible samples")
    serial_start = time.monotonic()
    serial = evaluate(samples)
    serial_seconds = time.monotonic() - serial_start
    with ProcessPoolExecutor(
        max_workers=WORKERS,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(structures, executors),
    ) as pool:
        parallel_start = time.monotonic()
        parallel = list(pool.map(evaluate, [[item] for item in samples]))
        parallel_seconds = time.monotonic() - parallel_start
        parallel = [row for batch in parallel for row in batch]
        if parallel != serial:
            raise ValueError("Parallel replay changed a computed chemical result")
        dump(
            output / "preflight.json",
            {
                "request": pin(ROOT, request_path),
                "controls": controls,
                "sample_targets": [r["target_id"] for r in serial],
                "rows": len(serial),
                "serial_result_sha256": digest(serial),
                "parallel_result_sha256": digest(parallel),
                "serial_seconds": serial_seconds,
                "parallel_seconds_including_worker_startup": parallel_seconds,
                "all_eligible_families_sampled": families,
                "largest_eligible_molecule_per_family_included": True,
                "equivalent": True,
                "training_admitted": False,
            },
        )
        print(
            f"Serial/parallel equivalence passed on {len(samples)} records; starting incremental replay",
            flush=True,
        )
        pending = deque()
        completed = []
        counts = defaultdict(Counter)

        def finish():
            family, number, items_hash, targets_hash, destination, future = pending.popleft()
            records = future.result()
            if digest([r["target_id"] for r in records]) != targets_hash:
                raise ValueError("Parallel replay changed input target order or population")
            report = write_shard(destination, records, family, number, items_hash, request_hash)
            completed.append(pin(ROOT, destination / "result.json"))
            counts[family].update(report["counts"])
            counts[family]["rows"] += report["rows"]
            dump(
                output / "progress.json",
                {
                    "request": pin(ROOT, request_path),
                    "completed_shards": len(completed),
                    "by_family": dict(counts),
                    "elapsed_seconds": time.monotonic() - started,
                    "complete": False,
                    "training_admitted": False,
                },
            )
            print(
                json.dumps(
                    {
                        "family": family,
                        "completed_rows": counts[family]["rows"],
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )

        for family, number, items in chunks(reader, families, carried):
            destination = output / f"{family}-{number:05d}"
            items_hash = digest(items)
            targets_hash = digest([i["preparation"]["target_id"] for i in items])
            if destination.exists():
                report = json.loads((destination / "result.json").read_text())
                if (
                    report["request_sha256"] != request_hash
                    or report["items_sha256"] != items_hash
                    or report["target_ids_sha256"] != targets_hash
                    or report["rows"] != len(items)
                ):
                    raise ValueError("Completed replay shard does not match restart inputs")
                resolve_pin(report["artifact"], ROOT, label="completed shard")
                completed.append(pin(ROOT, destination / "result.json"))
                counts[family].update(report["counts"])
                counts[family]["rows"] += report["rows"]
            else:
                pending.append(
                    (
                        family,
                        number,
                        items_hash,
                        targets_hash,
                        destination,
                        pool.submit(evaluate, items),
                    )
                )
                if len(pending) == WORKERS:
                    finish()
        while pending:
            finish()
    for family in families:
        expected = previous["summary"]["by_family"][family]["eligible_pending_chemistry"]
        if counts[family]["rows"] != expected:
            raise ValueError("Incremental replay did not cover every eligible pending row")
    for name, value in {**inputs, **implementation}.items():
        resolve_pin(value, ROOT, label=name)
    dump(
        output / "result.json",
        {
            "schema_version": "forge.compose_lipid_incremental_replay.v1",
            "request": pin(ROOT, request_path),
            "preflight": pin(ROOT, output / "preflight.json"),
            "shards": sorted(completed, key=lambda p: p["path"]),
            "summary": dict(counts),
            "elapsed_seconds": time.monotonic() - started,
            "complete": True,
            "training_admitted": False,
            "training_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
