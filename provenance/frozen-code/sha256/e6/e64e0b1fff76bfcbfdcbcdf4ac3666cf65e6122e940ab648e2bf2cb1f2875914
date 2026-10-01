"""Run existing qualified programs on every pending eligible recipe, in restartable shards."""

import json
import multiprocessing
import sqlite3
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus import compose_lipid_current_replay as repeated
from forge.corpus import compose_lipid_family_replay as family_engine
from forge.corpus import compose_lipid_fixed_profiles as profiles
from forge.corpus import compose_lipid_fixed_replay as fixed
from forge.corpus import compose_lipid_grouped_replay as grouped
from forge.corpus import compose_lipid_staged_replay as staged
from forge.corpus.compose_lipid_full_partition import PartitionPreparationCorpus
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import rows
from results.phase1.compose_lipid_full_replay_v1.replay_michael import digest, write_shard

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ENGINES = {
    "family": family_engine,
    "repeated": repeated,
    "profiles": profiles,
    "fixed": fixed,
    "staged": staged,
    "grouped": grouped,
}
_STRUCTURES, _EXECUTORS = None, None


def initialize(structures, executors):
    global _STRUCTURES, _EXECUTORS
    _STRUCTURES, _EXECUTORS = structures, executors


def evaluate(items):
    answer = []
    for item in items:
        p = item["preparation"]
        specification = _EXECUTORS[p["family"]]
        executor = specification["executor"]
        kind = specification["loader"]
        if kind == "profiles":
            executor = profiles.select_executor(item, executor)
        if specification.get("role_mappings"):
            roles = [v[0] for v in p["component_instances"]]
            matches = [
                m for m in specification["role_mappings"] if sorted(m.values()) == sorted(roles)
            ]
            if len(matches) > 1:
                raise ValueError("Ambiguous previously qualified source role namespace")
            if matches:
                executor = {**executor, "mapping": matches[0]}
        if kind in {"staged", "grouped"}:
            result = ENGINES[kind].replay_record(item, _STRUCTURES, executor)
        else:
            result = family_engine.replay_record(item, _STRUCTURES, executor)
        answer.append(
            {
                **p,
                "replay": result,
                "program_contract": specification["contract"],
                "training_admitted": False,
                "experimental_execution_admitted": False,
            }
        )
    return answer


def load_programs(config, inputs):
    executors, controls = {}, {}
    for name, spec in config["contracts"].items():
        path = resolve_pin(spec["config"], ROOT, label=name)
        loader = spec["loader"]
        print(f"Authenticating {name} source contracts and controls", flush=True)
        if loader == "family":
            cfg = json.loads(path.read_text())
            if (
                cfg["schema_version"] != family_engine.CONFIG_SCHEMA
                or cfg["policy"] != family_engine.POLICY
            ):
                raise ValueError("Frozen family contract changed")
            paths = {k: resolve_pin(v, ROOT, label=k) for k, v in cfg["inputs"].items()}
            found, checked = family_engine.load_executors(ROOT, paths)
            if sorted(found) != cfg["qualified_executor_families"]:
                raise ValueError("Frozen qualified family scope changed")
        elif loader == "repeated":
            cfg, _, adapters, programs, bounds, checked = repeated.load_contract(ROOT, path)
            found = {
                f: {
                    "kind": "repeated",
                    "adapter": adapters[f],
                    "binding": binding,
                    "program": programs[f],
                    "bounds": bounds,
                }
                for f, binding in cfg["families"].items()
            }
        else:
            _, _, found, checked = ENGINES[loader].load_contract(ROOT, path)
        controls[name] = checked
        for family in spec["families"]:
            if family in executors or family not in found:
                raise ValueError("Missing or duplicated qualified family executor")
            executors[family] = {"loader": loader, "contract": name, "executor": found[family]}
    # This is the already qualified role namespace, matched solely to source roles.
    # No precursor, site, quantity or namespace is inferred from a target product.
    binding = json.loads(inputs["ugi3_role_binding"].read_text())
    for group in ("implementation", "inputs", "artifacts"):
        for name, value in binding[group].items():
            resolve_pin(value, ROOT, label=name)
    witness = json.loads(
        resolve_pin(binding["artifacts"]["binding.json"], ROOT, label="role namespace").read_text()
    )
    executor = executors["aldehyde_ugi3"]["executor"]
    alternative = witness["registry_to_supplied_role"]
    if set(alternative) != set(executor["mapping"]) or len(set(alternative.values())) != len(
        alternative
    ):
        raise ValueError("Qualified role namespace changed chemical role coverage")
    mappings = [executor["mapping"]]
    if alternative not in mappings:
        mappings.append(alternative)
    executors["aldehyde_ugi3"]["role_mappings"] = mappings
    return executors, controls


def main():
    started = time.monotonic()
    config_path = HERE / "config.json"
    config = json.loads(config_path.read_text())
    runtime = config["runtime"]
    output = HERE / "programs"
    if (output / "result.json").exists():
        raise ValueError("Completed published replay is frozen")
    output.mkdir(exist_ok=True)
    inputs = {k: resolve_pin(v, ROOT, label=k) for k, v in config["inputs"].items()}
    implementation = set().union(*(engine.IMPLEMENTATION for engine in ENGINES.values())) | {
        "forge/corpus/compose_lipid_full_partition.py",
        "results/phase1/compose_lipid_full_replay_v1/replay_michael.py",
        str(Path(__file__).resolve().relative_to(ROOT)),
    }
    request = {
        "schema_version": "forge.compose_lipid_parallel_program_request.v1",
        "config": pin(ROOT, config_path),
        "inputs": config["inputs"],
        "policy": config["policy"],
        "runtime": runtime,
        "implementation": {n: pin(ROOT, ROOT / n) for n in sorted(implementation)},
    }
    request_path = output / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError("Restart inputs or implementation changed")
    dump(request_path, request)
    request_pin = pin(ROOT, request_path)
    executors, controls = load_programs(config, inputs)
    reader = PartitionPreparationCorpus(ROOT, inputs["partition"])
    catalogue = resolve_pin(reader.result["inputs"]["catalogue"], ROOT, label="complete precursors")
    structures = {r["component_id"]: r["constitution"] for r in rows(catalogue)}
    previous = json.loads(inputs["previous_evidence"].read_text())
    original = json.loads(inputs["original_evidence"].read_text())
    if (
        previous["inputs"]["previous"] != config["inputs"]["original_evidence"]
        or original["inputs"]["partition_receipt"] != config["inputs"]["partition"]
    ):
        raise ValueError("Carried exact evidence uses another source partition")
    carried = set()
    for document, query in (
        (original, "SELECT target_id FROM exact"),
        (previous, "SELECT target_id FROM new_evidence WHERE exact=1"),
    ):
        ledger = resolve_pin(document["artifact"], ROOT, label="carried exact evidence")
        with sqlite3.connect(ledger.as_uri() + "?mode=ro", uri=True) as db:
            carried.update(r[0] for r in db.execute(query))
    initialize(structures, executors)
    samples = []
    for family in sorted(executors):
        largest = None
        for n, item in enumerate(reader.iter_preparation_records(family=family)):
            if n < 8:
                samples.append(item)
            if largest is None or item["source"]["heavy_atoms"] > largest["source"]["heavy_atoms"]:
                largest = item
        if largest is not None and largest not in samples:
            samples.append(largest)
    if not samples:
        raise ValueError("No eligible records for numerical equivalence")
    mark = time.monotonic()
    serial = evaluate(samples)
    serial_seconds = time.monotonic() - mark
    counts = defaultdict(Counter)
    completed = []
    with ProcessPoolExecutor(
        max_workers=runtime["workers"],
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(structures, executors),
    ) as pool:
        mark = time.monotonic()
        parallel = [r for batch in pool.map(evaluate, [[v] for v in samples]) for r in batch]
        cold_seconds = time.monotonic() - mark
        mark = time.monotonic()
        warmed = [r for batch in pool.map(evaluate, [[v] for v in samples]) for r in batch]
        warm_seconds = time.monotonic() - mark
        if serial != parallel or serial != warmed:
            raise ValueError("Parallel execution changed computed program results")
        preflight = {
            "request": request_pin,
            "controls": controls,
            "equivalent": True,
            "rows": len(samples),
            "sample_targets": [r["target_id"] for r in serial],
            "serial_sha256": digest(serial),
            "parallel_sha256": digest(parallel),
            "serial_seconds": serial_seconds,
            "parallel_cold_seconds": cold_seconds,
            "parallel_warm_seconds": warm_seconds,
            "largest_per_family_included": True,
            "training_admitted": False,
        }
        dump(output / "preflight.json", preflight)
        print(
            f"Serial/parallel equivalence passed for {len(samples)} records in {len(executors)} families",
            flush=True,
        )
        pending = deque()

        def finish():
            family, number, destination, item_hash, target_hash, future = pending.popleft()
            records = future.result()
            if digest([r["target_id"] for r in records]) != target_hash:
                raise ValueError("Parallel replay changed source target order")
            report = write_shard(
                destination, records, family, number, item_hash, request_pin["sha256"]
            )
            completed.append(pin(ROOT, destination / "result.json"))
            counts[family].update(report["counts"])
            counts[family]["rows"] += report["rows"]
            dump(
                output / "progress.json",
                {
                    "request": request_pin,
                    "summary": dict(counts),
                    "completed_shards": len(completed),
                    "elapsed_seconds": time.monotonic() - started,
                    "complete": False,
                    "training_admitted": False,
                },
            )
            print(
                json.dumps(
                    {
                        "family": family,
                        "rows": counts[family]["rows"],
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )

        for family in sorted(executors):
            iterator = (
                v
                for v in reader.iter_preparation_records(family=family)
                if v["preparation"]["target_id"] not in carried
            )
            number = 0
            while items := list(islice(iterator, runtime["batch_size"])):
                destination = output / f"{family}-{number:05d}"
                item_hash = digest(items)
                target_hash = digest([v["preparation"]["target_id"] for v in items])
                if destination.exists():
                    report = json.loads((destination / "result.json").read_text())
                    if (
                        report["request_sha256"] != request_pin["sha256"]
                        or report["items_sha256"] != item_hash
                        or report["target_ids_sha256"] != target_hash
                        or report["rows"] != len(items)
                    ):
                        raise ValueError("Completed shard differs from restart inputs")
                    resolve_pin(report["artifact"], ROOT, label="completed replay shard")
                    completed.append(pin(ROOT, destination / "result.json"))
                    counts[family].update(report["counts"])
                    counts[family]["rows"] += report["rows"]
                else:
                    pending.append(
                        (
                            family,
                            number,
                            destination,
                            item_hash,
                            target_hash,
                            pool.submit(evaluate, items),
                        )
                    )
                    if len(pending) == runtime["maximum_pending_batches"]:
                        finish()
                number += 1
        while pending:
            finish()
    for family in executors:
        if (
            counts[family]["rows"]
            != previous["summary"]["by_family"][family]["eligible_pending_chemistry"]
        ):
            raise ValueError("Family replay lost eligible pending source records")
    for name, value in {
        **request["inputs"],
        **request["implementation"],
        "configuration": request["config"],
    }.items():
        resolve_pin(value, ROOT, label=name)
    result = {
        "schema_version": "forge.compose_lipid_parallel_program_replay.v1",
        "request": request_pin,
        "preflight": pin(ROOT, output / "preflight.json"),
        "shards": sorted(completed, key=lambda v: v["path"]),
        "summary": dict(counts),
        "elapsed_seconds": time.monotonic() - started,
        "complete": True,
        "training_admitted": False,
        "training_calls": 0,
    }
    dump(output / "result.json", result)


if __name__ == "__main__":
    main()
