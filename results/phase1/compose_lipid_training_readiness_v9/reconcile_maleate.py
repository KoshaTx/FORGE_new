"""Independently reconcile every mechanism-selected maleate replay and carried positive."""

import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from itertools import islice
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_full_partition import PartitionPreparationCorpus
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import rows
from results.phase1.compose_lipid_full_replay_v1.replay_michael import digest

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FAMILY = "maleate_addition"


def main():
    result_path = (
        ROOT / "results/phase1/compose_lipid_full_maleate_dispatch_v1/incremental/result.json"
    )
    result = json.loads(result_path.read_text())
    if result["complete"] is not True or result["training_admitted"] is not False:
        raise ValueError("Maleate result is incomplete or claims training admission")
    request = json.loads(resolve_pin(result["request"], ROOT, label="request").read_text())
    paths = {k: resolve_pin(v, ROOT, label=k) for k, v in request["inputs"].items()}
    for name, value in request["implementation"].items():
        resolve_pin(value, ROOT, label=name)
    preflight = json.loads(resolve_pin(result["preflight"], ROOT, label="controls").read_text())
    if preflight["pass"] is not True or preflight["request"] != result["request"]:
        raise ValueError("Maleate controls are absent or substituted")
    previous = json.loads(paths["previous"].read_text())
    prior = set()
    partition = json.loads(paths["partition"].read_text())
    partition_database = resolve_pin(partition["artifact"], ROOT, label="partition database")
    for key, table, condition in (
        ("base_evidence", "exact", "family=?"),
        ("expanded_evidence", "new_evidence", "family=? AND exact=1"),
    ):
        receipt = json.loads(paths[key].read_text())
        database = resolve_pin(receipt["artifact"], ROOT, label=key)
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
            db.execute(
                "ATTACH DATABASE ? AS population", (partition_database.as_uri() + "?mode=ro",)
            )
            for (target,) in db.execute(
                f"SELECT target_id FROM {table} WHERE {condition} "
                "AND target_id IN (SELECT target_id FROM population.eligible)",
                (FAMILY,),
            ):
                if target in prior:
                    raise ValueError("Carried evidence duplicates a maleate target")
                prior.add(target)
    if len(prior) != previous["summary"]["by_family"][FAMILY]["exact_computed_reconstructions"]:
        raise ValueError("Carried positive count differs from current checkpoint")
    reader = PartitionPreparationCorpus(ROOT, paths["partition"])
    iterator = (
        item
        for item in reader.iter_preparation_records(family=FAMILY)
        if item["preparation"]["target_id"] not in prior
    )
    output = HERE / "maleate-evidence.sqlite"
    if output.exists():
        raise ValueError("Completed evidence cannot be overwritten")
    counts, mechanisms, comparison = Counter(), Counter(), Counter()
    comparison["carried_exact_reconstructions"] = len(prior)
    with tempfile.TemporaryDirectory(prefix=".evidence-", dir=HERE) as tmp:
        stage = Path(tmp) / output.name
        db = sqlite3.connect(stage)
        db.execute(
            "CREATE TABLE new_evidence(target_id TEXT PRIMARY KEY,family TEXT,exact INTEGER,"
            "disposition TEXT,shard TEXT,source_line INTEGER)"
        )
        for number, shard_pin in enumerate(result["shards"]):
            shard = json.loads(resolve_pin(shard_pin, ROOT, label="shard").read_text())
            expected = list(islice(iterator, 256))
            if (
                not expected
                or shard["number"] != number
                or shard["family"] != FAMILY
                or shard["request_sha256"] != result["request"]["sha256"]
                or shard["items_sha256"] != digest(expected)
                or shard["training_admitted"] is not False
            ):
                raise ValueError("Maleate shard changed its request, input population or order")
            ledger = list(rows(resolve_pin(shard["artifact"], ROOT, label="ledger")))
            if len(ledger) != len(expected) or len(ledger) != shard["rows"]:
                raise ValueError("Maleate shard omitted or added a source record")
            local, identifiers = Counter(), []
            for line, (row, source) in enumerate(zip(ledger, expected), 1):
                prepared = source["preparation"]
                if any(row[k] != v for k, v in prepared.items()):
                    raise ValueError(
                        "Replay changed source membership, identity, roles or quantities"
                    )
                mechanism = source["source"]["primary_metadata"].get("mechanism")
                if mechanism not in {"amine", "thiol"} or row["source_mechanism"] != mechanism:
                    raise ValueError("Replay substituted its source-declared mechanism")
                replay = row["replay"]
                exact = replay["computed_consistency_pass"]
                if type(exact) is not bool or exact != (
                    replay["disposition"] == "exact_computed_reconstruction"
                ):
                    raise ValueError("Exactness and disposition disagree")
                if exact and (
                    not replay.get("checks")
                    or any(v is not True for v in replay["checks"].values())
                    or replay.get("bound_reasons")
                    or len(replay["forward_layers"][-1]) != 1
                    or hashlib.sha256(replay["forward_layers"][-1][0].encode()).hexdigest()
                    != prepared["constitution_id"]
                    or replay["verified_target_constitution_id"] != prepared["constitution_id"]
                ):
                    raise ValueError("An exact replay lacks full checks or a unique target product")
                target = prepared["target_id"]
                if target in prior:
                    raise ValueError("Incremental replay duplicates carried exact evidence")
                if exact:
                    comparison["new_exact"] += 1
                local[replay["disposition"]] += 1
                mechanisms[mechanism] += 1
                identifiers.append(target)
                db.execute(
                    "INSERT INTO new_evidence VALUES (?,?,?,?,?,?)",
                    (target, FAMILY, int(exact), replay["disposition"], shard_pin["path"], line),
                )
            if dict(local) != shard["counts"] or digest(identifiers) != shard["target_ids_sha256"]:
                raise ValueError("Shard counts or target digest do not reproduce")
            counts.update(local)
        if next(iterator, None) is not None:
            raise ValueError("Replay omitted eligible maleate rows")
        expected_summary = {**counts, "rows": sum(counts.values())}
        if (
            {FAMILY: expected_summary} != result["summary"]
            or mechanisms != result["source_mechanisms"]
            or comparison != result["comparison"]
            or sum(counts.values())
            != previous["summary"]["by_family"][FAMILY]["eligible_pending_chemistry"]
        ):
            raise ValueError("Maleate summary or carried-positive comparison does not reproduce")
        db.commit()
        db.close()
        os.rename(stage, output)
    current = {}
    for family, values in previous["summary"]["by_family"].items():
        added = comparison["new_exact"] if family == FAMILY else 0
        current[family] = {
            **values,
            "new_exact_computed_reconstructions": added,
            "exact_computed_reconstructions": values["exact_computed_reconstructions"] + added,
            "eligible_pending_chemistry": values["eligible_pending_chemistry"] - added,
        }
    dump(
        HERE / "combined-replay-audit.json",
        {
            "schema_version": "forge.compose_lipid_incremental_evidence_checkpoint.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {"previous": pin(ROOT, paths["previous"]), "replay": pin(ROOT, result_path)},
            "artifact": pin(ROOT, output),
            "comparison": comparison,
            "summary": {
                "by_family": current,
                "totals": dict(sum((Counter(v) for v in current.values()), Counter())),
            },
            "training_admitted": False,
            "training_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
