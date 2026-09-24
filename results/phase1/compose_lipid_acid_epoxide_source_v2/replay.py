"""Restartable source-qualified acid/epoxide replay on every current pending eligible row."""

import json
import multiprocessing
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_evidence_index import EvidencePreparationCorpus
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_staged_replay import IMPLEMENTATION, load_contract, replay_record
from forge.corpus.compose_lipid_supplement import rows
from results.phase1.compose_lipid_full_replay_v1.replay_michael import digest, write_shard

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FAMILY = "acid_epoxide_diester_multistep"
STRUCTURES, EXECUTOR = None, None


def initialize(structures):
    global STRUCTURES, EXECUTOR
    STRUCTURES = structures
    EXECUTOR = load_contract(ROOT, HERE / "replay-config.json")[2][FAMILY]


def evaluate(batch):
    return [
        {
            **item["preparation"],
            "program_id": EXECUTOR["program"].specification["program_id"],
            "replay": replay_record(item, STRUCTURES, EXECUTOR),
            "training_admitted": False,
            "experimental_execution_admitted": False,
        }
        for item in batch
    ]


def main():
    started = time.monotonic()
    output = HERE / "replay"
    output.mkdir(exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("Replay already complete; verify its receipt instead")
    config_path = HERE / "replay-config.json"
    config, _, _, controls = load_contract(ROOT, config_path)
    index_path = ROOT / "results/phase1/compose_lipid_aema_source_v1/reconciled/result.json"
    print("Authenticating existing evidence index and frozen source population", flush=True)
    reader = EvidencePreparationCorpus(ROOT, index_path)
    partition = json.loads(
        (ROOT / reader.result["inputs"]["partition_receipt"]["path"]).read_text()
    )
    catalogue = resolve_pin(partition["inputs"]["catalogue"], ROOT, label="precursor catalogue")
    structures = {row["component_id"]: row["constitution"] for row in rows(catalogue)}
    request = {
        "schema_version": "forge.compose_lipid_acid_epoxide_request.v1",
        "seed": 0,
        "inputs": {
            "config": pin(ROOT, config_path),
            "evidence_index": pin(ROOT, index_path),
            "catalogue": partition["inputs"]["catalogue"],
            **config["inputs"],
        },
        "implementation": {
            name: pin(ROOT, ROOT / name)
            for name in sorted(
                {
                    *IMPLEMENTATION,
                    "forge/corpus/compose_lipid_evidence_index.py",
                    "results/phase1/compose_lipid_full_replay_v1/replay_michael.py",
                    str(Path(__file__).resolve().relative_to(ROOT)),
                }
            )
        },
        "runtime": {
            "workers": 4,
            "batch_size": 256,
            "maximum_pending_batches": 4,
            "device": "cpu",
            "multiprocessing": "spawn",
        },
        "policy": {
            "population": "current_full_partition_eligible_pending_only",
            "record_cap": None,
            "size_filter": None,
            "program_choice": "source_declared_family_contract",
            "training_admitted": False,
            "training_calls": 0,
        },
    }
    request_path = output / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError("Restart inputs or implementation changed")
    dump(request_path, request)
    request_pin = pin(ROOT, request_path)
    initialize(structures)
    sample = list(islice(reader.iter_preparation_records(family=FAMILY, exact=False), 8))
    tick = time.monotonic()
    serial = evaluate(sample)
    serial_seconds = time.monotonic() - tick
    counts, shards = Counter(), []
    pending = deque()
    with ProcessPoolExecutor(
        max_workers=4,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(structures,),
    ) as pool:
        tick = time.monotonic()
        parallel = [row for batch in pool.map(evaluate, [[v] for v in sample]) for row in batch]
        parallel_seconds = time.monotonic() - tick
        if serial != parallel:
            raise ValueError("Parallel execution changed source-program evidence")
        dump(
            output / "preflight.json",
            {
                "request": request_pin,
                "source_controls": controls,
                "serial_parallel_equivalent": True,
                "sample_targets": [r["target_id"] for r in serial],
                "serial_sha256": digest(serial),
                "parallel_sha256": digest(parallel),
                "serial_seconds": serial_seconds,
                "parallel_cold_seconds": parallel_seconds,
                "training_admitted": False,
            },
        )
        print("Source and serial/parallel controls passed", flush=True)

        def finish():
            number, destination, item_hash, target_hash, future = pending.popleft()
            result = future.result()
            if digest([r["target_id"] for r in result]) != target_hash:
                raise ValueError("Worker changed source record order")
            shard = write_shard(
                destination, result, FAMILY, number, item_hash, request_pin["sha256"]
            )
            shards.append(pin(ROOT, destination / "result.json"))
            counts.update(shard["counts"])
            counts["rows"] += shard["rows"]
            dump(
                output / "progress.json",
                {
                    "request": request_pin,
                    "counts": dict(counts),
                    "completed_shards": len(shards),
                    "seconds": time.monotonic() - started,
                    "complete": False,
                    "training_admitted": False,
                },
            )
            if len(shards) % 20 == 0:
                print(
                    json.dumps({"counts": counts, "seconds": time.monotonic() - started}),
                    flush=True,
                )

        iterator = reader.iter_preparation_records(family=FAMILY, exact=False)
        number = 0
        while batch := list(islice(iterator, 256)):
            destination = output / f"{FAMILY}-{number:05d}"
            item_hash = digest(batch)
            target_hash = digest([v["preparation"]["target_id"] for v in batch])
            if destination.exists():
                shard = json.loads((destination / "result.json").read_text())
                if (
                    shard["request_sha256"] != request_pin["sha256"]
                    or shard["items_sha256"] != item_hash
                    or shard["target_ids_sha256"] != target_hash
                    or shard["rows"] != len(batch)
                ):
                    raise ValueError("Completed shard differs from current request")
                resolve_pin(shard["artifact"], ROOT, label="completed shard")
                shards.append(pin(ROOT, destination / "result.json"))
                counts.update(shard["counts"])
                counts["rows"] += shard["rows"]
            else:
                pending.append(
                    (number, destination, item_hash, target_hash, pool.submit(evaluate, batch))
                )
                if len(pending) == 4:
                    finish()
            number += 1
        while pending:
            finish()
    if counts["rows"] != reader.result["by_family"][FAMILY]["pending"]:
        raise ValueError("Replay omitted pending eligible acid/epoxide rows")
    for name, value in {**request["inputs"], **request["implementation"]}.items():
        resolve_pin(value, ROOT, label=name)
    dump(
        output / "result.json",
        {
            "schema_version": "forge.compose_lipid_acid_epoxide_replay.v1",
            "seed": 0,
            "request": request_pin,
            "preflight": pin(ROOT, output / "preflight.json"),
            "shards": sorted(shards, key=lambda v: v["path"]),
            "counts": dict(counts),
            "seconds": time.monotonic() - started,
            "complete": True,
            "training_admitted": False,
            "experimental_execution_admitted": False,
            "training_calls": 0,
        },
    )
    print(
        json.dumps({"complete": True, "counts": counts, "seconds": time.monotonic() - started}),
        flush=True,
    )


if __name__ == "__main__":
    main()
