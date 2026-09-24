"""Independently reconcile all persisted representation rows to eligible source inputs."""

import json
import sqlite3
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import rows
from results.phase1.compose_lipid_full_replay_v1.replay_michael import digest

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    result_path = ROOT / "results/phase1/compose_lipid_full_representation_v1/audit/result.json"
    result = json.loads(result_path.read_text())
    request_path = resolve_pin(result["request"], ROOT, label="representation request")
    request = json.loads(request_path.read_text())
    for name, value in {**request["inputs"], **request["implementation"]}.items():
        resolve_pin(value, ROOT, label=name)
    if (
        result["pass"] is not True
        or result["failure_count"]
        or result["training_admitted"] is not False
    ):
        raise ValueError("Full eligible representation did not pass")
    if (
        result["model_smoke"]["finite"] is not True
        or result["model_smoke"]["training_steps"] != 0
        or result["joint_synthesis_program_representation_qualified"] is not False
    ):
        raise ValueError("Representation scope changed")
    partition = json.loads(
        resolve_pin(request["inputs"]["partition_receipt"], ROOT, label="partition").read_text()
    )
    if (
        request["inputs"]["partition"] != partition["artifact"]
        or request["inputs"]["source_corpus"] != partition["inputs"]["corpus"]
    ):
        raise ValueError("Representation used different source/partition artifacts")
    database = resolve_pin(request["inputs"]["partition"], ROOT, label="partition database")
    corpus = resolve_pin(request["inputs"]["source_corpus"], ROOT, label="source corpus")
    counts = defaultdict(Counter)
    states = Counter()
    maximum = {"atoms": 0, "closures": 0}
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        db.execute("ATTACH DATABASE ? AS source", (corpus.as_uri() + "?mode=ro",))
        db.execute("PRAGMA cache_size=-32768")
        cursor = iter(
            db.execute(
                "SELECT e.target_id,e.family,e.heavy_atoms,t.constitution FROM eligible e JOIN source.targets t USING(target_id) ORDER BY e.family,e.target_id"
            )
        )
        for shard_pin in sorted(result["shards"], key=lambda v: v["path"]):
            shard = json.loads(
                resolve_pin(shard_pin, ROOT, label="representation shard").read_text()
            )
            expected = list(islice(cursor, shard["rows"]))
            if (
                shard["request_sha256"] != result["request"]["sha256"]
                or len(expected) != shard["rows"]
                or digest(expected) != shard["items_sha256"]
            ):
                raise ValueError(
                    "Representation shard does not match complete eligible source inputs"
                )
            actual = list(
                rows(resolve_pin(shard["artifact"], ROOT, label="representation records"))
            )
            if len(actual) != len(expected):
                raise ValueError("Representation shard lost records")
            for row, source in zip(actual, expected, strict=True):
                if (row["target_id"], row["family"], row["heavy_atoms"]) != source[:3]:
                    raise ValueError("Representation altered target identity, family or size")
                if (
                    row["pass_checks"] is not True
                    or row["training_admitted"] is not False
                    or not row["checks"]
                    or any(v is not True for v in row["checks"].values())
                ):
                    raise ValueError("Representation record lacks complete passing checks")
                if sum(s["atoms"] for s in row["atom_states"]) != row["heavy_atoms"]:
                    raise ValueError("Representation state inventory changes source size")
                count = counts[row["family"]]
                count.update({"rows": 1, "passed": 1, "above_96_atoms": row["heavy_atoms"] > 96})
                count.update(row["checks"].keys())
                maximum["atoms"] = max(maximum["atoms"], row["heavy_atoms"])
                maximum["closures"] = max(maximum["closures"], row["closures"])
                for state in row["atom_states"]:
                    key = tuple(
                        (k, state[k])
                        for k in ("symbol", "formal_charge", "aromatic", "explicit_hydrogens")
                    )
                    states[key] += state["atoms"]
        if next(cursor, None) is not None:
            raise ValueError("Representation did not cover the complete eligible population")
    for family in result["by_family"]:
        counts.setdefault(family, Counter())
    atom_counts = [{**dict(state), "eligible_atoms": n} for state, n in sorted(states.items())]
    totals = dict(sum(counts.values(), Counter()))
    if (
        dict(counts) != result["by_family"]
        or totals != result["totals"]
        or maximum != result["maximum_observed"]
        or atom_counts != result["atom_counts"]
    ):
        raise ValueError("Persisted representation summary does not reproduce")
    dump(
        HERE / "representation-audit.json",
        {
            "schema_version": "forge.compose_lipid_full_representation_reconciliation.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {
                "representation_result": pin(ROOT, result_path),
                "request": result["request"],
            },
            "verified_eligible_rows": totals["rows"],
            "maximum_observed": maximum,
            "model_smoke_support_atoms": result["model_smoke"]["output_shapes"]["nodes"][1],
            "pass": True,
            "training_admitted": False,
            "training_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
