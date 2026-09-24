"""Admit completed, pinned replay evidence into the current chemistry work ledger."""

import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    original_path = HERE / "population-audit.json"
    replay_path = ROOT / "results/phase1/compose_lipid_full_replay_v1/michael/result.json"
    original = json.loads(original_path.read_text())
    replay = json.loads(replay_path.read_text())
    if replay["complete"] is not True or replay["training_admitted"] is not False:
        raise ValueError("Incremental replay is incomplete or claims admission")
    request_path = resolve_pin(replay["request"], ROOT, label="replay request")
    request = json.loads(request_path.read_text())
    if request["inputs"]["previous_evidence"] != pin(ROOT, original_path):
        raise ValueError("Replay starts from another chemistry checkpoint")
    for name, value in {**request["inputs"], **request["implementation"]}.items():
        resolve_pin(value, ROOT, label=name)
    preflight = json.loads(
        resolve_pin(replay["preflight"], ROOT, label="parallel equivalence").read_text()
    )
    if preflight["equivalent"] is not True or preflight["request"] != replay["request"]:
        raise ValueError("Parallel qualification is absent or substituted")
    partition_path = resolve_pin(request["inputs"]["partition"], ROOT, label="partition receipt")
    partition = json.loads(partition_path.read_text())
    database = resolve_pin(partition["artifact"], ROOT, label="partition ledger")
    joins = resolve_pin(partition["inputs"]["joins"], ROOT, label="source constructions")
    old_evidence = resolve_pin(original["artifact"], ROOT, label="carried evidence")
    output = HERE / "incremental-evidence.sqlite"
    if output.exists():
        raise ValueError("Incremental evidence checkpoint already exists")
    family_counts = defaultdict(Counter)
    with tempfile.TemporaryDirectory(prefix=".evidence-", dir=HERE) as tmp:
        stage = Path(tmp) / output.name
        db = sqlite3.connect(stage.as_uri(), uri=True)
        db.execute("ATTACH DATABASE ? AS population", (database.as_uri() + "?mode=ro",))
        db.execute("ATTACH DATABASE ? AS source", (joins.as_uri() + "?mode=ro",))
        db.execute("ATTACH DATABASE ? AS prior", (old_evidence.as_uri() + "?mode=ro",))
        db.execute(
            "CREATE TABLE new_evidence(target_id TEXT PRIMARY KEY,family TEXT,exact INTEGER,disposition TEXT,shard TEXT,source_line INTEGER)"
        )
        for shard_pin in replay["shards"]:
            shard_path = resolve_pin(shard_pin, ROOT, label="completed shard receipt")
            shard = json.loads(shard_path.read_text())
            if (
                shard["request_sha256"] != replay["request"]["sha256"]
                or shard["training_admitted"] is not False
            ):
                raise ValueError("Replay shard belongs to another request")
            ledger = resolve_pin(shard["artifact"], ROOT, label="completed replay ledger")
            counts = Counter()
            batch = []
            for line, row in enumerate(rows(ledger), 1):
                target = row["target_id"]
                bound = db.execute(
                    "SELECT e.family,e.constitution_id,c.instances FROM population.eligible e JOIN source.constructions c USING(target_id) WHERE e.target_id=?",
                    (target,),
                ).fetchone()
                if (
                    bound is None
                    or row["family"] != shard["family"]
                    or bound[0] != row["family"]
                    or bound[1] != row["constitution_id"]
                    or json.loads(bound[2]) != row["component_instances"]
                ):
                    raise ValueError("Replay changed the eligible target or complete source recipe")
                if (
                    row["training_admitted"] is not False
                    or row["eligible_for_program_preparation"] is not True
                ):
                    raise ValueError("Replay admission or preparation flags differ")
                result = row["replay"]
                exact = result["computed_consistency_pass"]
                if type(exact) is not bool or exact != (
                    result["disposition"] == "exact_computed_reconstruction"
                ):
                    raise ValueError("Replay exactness and disposition disagree")
                if exact and (
                    not result.get("checks")
                    or any(v is not True for v in result["checks"].values())
                    or result["verified_target_constitution_id"] != row["constitution_id"]
                ):
                    raise ValueError("Claimed exact result lacks complete passing checks")
                counts[result["disposition"]] += 1
                batch.append(
                    (
                        target,
                        row["family"],
                        int(exact),
                        result["disposition"],
                        shard_pin["path"],
                        line,
                    )
                )
            if sum(counts.values()) != shard["rows"] or dict(counts) != shard["counts"]:
                raise ValueError("Shard summary does not reproduce")
            db.executemany("INSERT INTO new_evidence VALUES (?,?,?,?,?,?)", batch)
            family_counts[shard["family"]].update(counts)
            family_counts[shard["family"]]["rows"] += shard["rows"]
        if dict(family_counts) != replay["summary"]:
            raise ValueError("Replay aggregate counts do not reproduce")
        if db.execute(
            "SELECT count(*) FROM new_evidence JOIN prior.exact USING(target_id)"
        ).fetchone()[0]:
            raise ValueError("Incremental replay duplicates carried exact evidence")
        for family, counts in family_counts.items():
            if (
                counts["rows"]
                != original["summary"]["by_family"][family]["eligible_pending_chemistry"]
            ):
                raise ValueError("Completed family replay omitted eligible pending targets")
        db.commit()
        db.close()
        os.rename(stage, output)
    current = {}
    for family, counts in original["summary"]["by_family"].items():
        new = family_counts[family].get("exact_computed_reconstruction", 0)
        current[family] = {
            **counts,
            "new_exact_computed_reconstructions": new,
            "exact_computed_reconstructions": counts["exact_computed_reconstructions_carried"]
            + new,
            "eligible_pending_chemistry": counts["eligible_pending_chemistry"] - new,
        }
    dump(
        HERE / "incremental-replay-audit.json",
        {
            "schema_version": "forge.compose_lipid_incremental_evidence_checkpoint.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {"previous": pin(ROOT, original_path), "replay": pin(ROOT, replay_path)},
            "artifact": pin(ROOT, output),
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
