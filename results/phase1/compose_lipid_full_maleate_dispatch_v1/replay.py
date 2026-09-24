"""Select frozen maleate executors solely by the original source mechanism label."""

import json
import sqlite3
import time
from collections import Counter
from itertools import islice
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus import compose_lipid_current_replay as programs
from forge.corpus import compose_lipid_family_replay as engine
from forge.corpus.compose_lipid_full_partition import PartitionPreparationCorpus
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import rows
from results.phase1.compose_lipid_full_replay_v1.replay_michael import digest, write_shard

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FAMILY = "maleate_addition"


def select_mechanism(metadata):
    mechanism = metadata.get("mechanism")
    if mechanism not in {"amine", "thiol"}:
        raise ValueError("Maleate source must explicitly declare amine or thiol mechanism")
    return mechanism


def main():
    started = time.monotonic()
    output = HERE / "replay"
    if output.exists():
        raise ValueError("Maleate replay output must be fresh")
    paths = {
        "partition": ROOT / "results/phase1/compose_lipid_full_partition_v1/audit-v2/result.json",
        "previous": ROOT
        / "results/phase1/compose_lipid_training_readiness_v8/combined-replay-audit.json",
        "base_evidence": ROOT
        / "results/phase1/compose_lipid_training_readiness_v7/population-audit.json",
        "expanded_evidence": ROOT
        / "results/phase1/compose_lipid_training_readiness_v8/incremental-replay-audit.json",
        "amine": ROOT / "configs/multireaction/compose_lipid_supplied_maleate_ester_v1.json",
        "thiol": ROOT / "configs/multireaction/compose_lipid_supplied_thiol_a3_v1.json",
    }
    implementation = (
        set(programs.IMPLEMENTATION)
        | set(engine.IMPLEMENTATION)
        | {
            "forge/corpus/compose_lipid_full_partition.py",
            "results/phase1/compose_lipid_full_replay_v1/replay_michael.py",
            str(Path(__file__).resolve().relative_to(ROOT)),
        }
    )
    request = {
        "schema_version": "forge.compose_lipid_source_mechanism_replay_request.v1",
        "inputs": {k: pin(ROOT, p) for k, p in paths.items()},
        "implementation": {k: pin(ROOT, ROOT / k) for k in sorted(implementation)},
        "policy": {
            "seed": 0,
            "device": "cpu",
            "workers": 1,
            "batch_size": 256,
            "population": "all_current_eligible_maleate_records",
            "program_selection": "original_primary_metadata.mechanism_only",
            "target_based_program_selection": False,
            "unknown_or_missing_mechanism": "error",
            "source_roles_quantities_and_ids_modified": False,
            "search_bounds": "unchanged_qualified_program_bounds",
            "record_cap": None,
            "size_filter": None,
            "training_admitted": False,
            "training_calls": 0,
        },
    }
    executors, controls = {}, {}
    for mechanism in ("amine", "thiol"):
        cfg, _, adapters, specifications, bounds, checked = programs.load_contract(
            ROOT, paths[mechanism]
        )
        executors[mechanism] = {
            "kind": "repeated",
            "adapter": adapters[FAMILY],
            "binding": cfg["families"][FAMILY],
            "program": specifications[FAMILY],
            "bounds": bounds,
        }
        controls[mechanism] = checked
    # Selection accepts metadata only; it has no product or product-derived input.
    for invalid in ({}, {"mechanism": "unknown"}, {"mechanism": None}):
        try:
            select_mechanism(invalid)
        except ValueError:
            pass
        else:
            raise ValueError("Invalid source mechanism silently selected a program")
    if any(select_mechanism({"mechanism": name}) != name for name in executors):
        raise ValueError("Source mechanism selection changed")
    prior = set()
    partition = json.loads(paths["partition"].read_text())
    partition_database = resolve_pin(partition["artifact"], ROOT, label="partition database")
    for key, table, predicate in (
        ("base_evidence", "exact", "family=?"),
        ("expanded_evidence", "new_evidence", "family=? AND exact=1"),
    ):
        receipt = json.loads(paths[key].read_text())
        database = resolve_pin(receipt["artifact"], ROOT, label=key)
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
            db.execute(
                "ATTACH DATABASE ? AS population", (partition_database.as_uri() + "?mode=ro",)
            )
            selected = {
                r[0]
                for r in db.execute(
                    f"SELECT target_id FROM {table} WHERE {predicate} "
                    "AND target_id IN (SELECT target_id FROM population.eligible)",
                    (FAMILY,),
                )
            }
        if prior.intersection(selected):
            raise ValueError("Carried maleate evidence overlaps")
        prior.update(selected)
    previous = json.loads(paths["previous"].read_text())["summary"]["by_family"][FAMILY]
    if len(prior) != previous["exact_computed_reconstructions"]:
        raise ValueError("Carried exact maleate evidence does not reproduce")
    reader = PartitionPreparationCorpus(ROOT, paths["partition"])
    catalogue = resolve_pin(reader.result["inputs"]["catalogue"], ROOT, label="catalogue")
    structures = {r["component_id"]: r["constitution"] for r in rows(catalogue)}
    output.mkdir(parents=True)
    request_path = output / "request.json"
    dump(request_path, request)
    request_pin = pin(ROOT, request_path)
    dump(output / "preflight.json", {"request": request_pin, "controls": controls, "pass": True})
    iterator = reader.iter_preparation_records(family=FAMILY)
    counts, mechanisms, comparison = Counter(), Counter(), Counter()
    shards, seen_prior = [], set()
    number = 0
    while batch := list(islice(iterator, 256)):
        records = []
        for item in batch:
            prepared = item["preparation"]
            mechanism = select_mechanism(item["source"]["primary_metadata"])
            replay = engine.replay_record(item, structures, executors[mechanism])
            target = prepared["target_id"]
            exact = replay["computed_consistency_pass"]
            if target in prior:
                seen_prior.add(target)
                comparison["prior_exact_still_exact" if exact else "prior_exact_regressed"] += 1
            elif exact:
                comparison["new_exact"] += 1
            counts[replay["disposition"]] += 1
            mechanisms[mechanism] += 1
            records.append({**prepared, "source_mechanism": mechanism, "replay": replay})
        destination = output / f"{FAMILY}-{number:05d}"
        write_shard(destination, records, FAMILY, number, digest(batch), request_pin["sha256"])
        shards.append(pin(ROOT, destination / "result.json"))
        number += 1
        print(
            json.dumps({"rows": sum(counts.values()), "seconds": time.monotonic() - started}),
            flush=True,
        )
    if (
        sum(counts.values()) != previous["eligible_preparation_rows"]
        or seen_prior != prior
        or comparison["prior_exact_regressed"]
    ):
        raise ValueError(
            "Maleate replay omitted eligible records or regressed prior exact evidence"
        )
    dump(
        output / "result.json",
        {
            "schema_version": "forge.compose_lipid_source_mechanism_replay.v1",
            "request": request_pin,
            "preflight": pin(ROOT, output / "preflight.json"),
            "shards": shards,
            "summary": {FAMILY: {**counts, "rows": sum(counts.values())}},
            "source_mechanisms": mechanisms,
            "comparison": comparison,
            "elapsed_seconds": time.monotonic() - started,
            "complete": True,
            "training_admitted": False,
            "training_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
