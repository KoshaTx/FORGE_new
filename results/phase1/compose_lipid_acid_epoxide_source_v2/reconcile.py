"""Authenticate every acid/epoxide shard and append its evidence to the existing index."""

import copy
import json
import os
import sqlite3
import tempfile
import time
from collections import Counter
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_evidence_index import append_evidence, census
from forge.corpus.compose_lipid_grouped_readiness import validate_replay_row
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_staged_replay import load_contract
from forge.corpus.compose_lipid_supplement import rows
from results.phase1.compose_lipid_full_replay_v1.replay_michael import digest

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FAMILY = "acid_epoxide_diester_multistep"


def main():
    started = time.monotonic()
    output = HERE / "reconciled"
    if output.exists():
        raise FileExistsError(output)
    replay_path = HERE / "replay/result.json"
    replay = json.loads(replay_path.read_text())
    if (
        replay.get("complete") is not True
        or replay["training_admitted"] is not False
        or replay["training_calls"] != 0
    ):
        raise ValueError("Acid/epoxide replay is incomplete or claims training admission")
    request = json.loads(resolve_pin(replay["request"], ROOT, label="replay request").read_text())
    for name, value in {**request["inputs"], **request["implementation"]}.items():
        resolve_pin(value, ROOT, label=name)
    cfg, _, executors, controls = load_contract(ROOT, ROOT / request["inputs"]["config"]["path"])
    preflight = json.loads(resolve_pin(replay["preflight"], ROOT, label="preflight").read_text())
    if (
        preflight["request"] != replay["request"]
        or preflight["serial_parallel_equivalent"] is not True
        or preflight["source_controls"] != json.loads(json.dumps(controls))
    ):
        raise ValueError("Source or serial/parallel controls differ")
    old_path = ROOT / request["inputs"]["evidence_index"]["path"]
    old = json.loads(old_path.read_text())
    for name, value in {**old["inputs"], **old["partition_inputs"]}.items():
        resolve_pin(value, ROOT, label=name)
    resolve_pin(old["implementation"], ROOT, label="previous index implementation")
    old_database = resolve_pin(old["artifact"], ROOT, label="previous evidence index")
    previous_path = resolve_pin(old["evidence_receipts"][-1], ROOT, label="previous checkpoint")
    previous = json.loads(previous_path.read_text())
    if old["by_family"][FAMILY]["exact"] != 0:
        raise ValueError("This first acid/epoxide audit must not replace existing positives")
    counts, failed, exact_sizes = Counter(), Counter(), Counter()
    with tempfile.TemporaryDirectory(prefix=".reconcile-", dir=HERE) as tmp:
        stage = Path(tmp)
        audit_db = stage / "acid-epoxide-evidence.sqlite"
        db = sqlite3.connect(audit_db.as_uri(), uri=True)
        for name, value in old["partition_inputs"].items():
            db.execute(
                f"ATTACH DATABASE ? AS {name}", ((ROOT / value["path"]).as_uri() + "?mode=ro",)
            )
        db.execute(
            "CREATE TABLE new_evidence(target_id TEXT PRIMARY KEY,family TEXT,exact INTEGER,disposition TEXT,shard TEXT,source_line INTEGER)"
        )
        for number, shard_pin in enumerate(replay["shards"]):
            shard = json.loads(resolve_pin(shard_pin, ROOT, label="replay shard").read_text())
            if (
                shard["request_sha256"] != replay["request"]["sha256"]
                or shard["number"] != number
                or shard["family"] != FAMILY
                or shard["training_admitted"] is not False
            ):
                raise ValueError("Shard request, order, family or admission differs")
            local, targets = Counter(), []
            for line, row in enumerate(
                rows(resolve_pin(shard["artifact"], ROOT, label="shard ledger")), 1
            ):
                target = row["target_id"]
                bound = db.execute(
                    "SELECT p.family,p.constitution_id,p.heavy_atoms,c.instances,c.basis FROM partition.eligible p JOIN joins.constructions c USING(target_id) WHERE p.target_id=? AND p.family=c.family",
                    (target,),
                ).fetchone()
                if bound is None or bound[0] != FAMILY:
                    raise ValueError("Replay contains protected or unknown source identity")
                prepared = {
                    "target_id": target,
                    "family": FAMILY,
                    "constitution_id": bound[1],
                    "component_instances": json.loads(bound[3]),
                    "construction_basis": bound[4],
                    "eligible_for_program_preparation": True,
                    "old_split": "train",
                    "corrected_split": "train",
                    "exclusion_reasons": [],
                    "pending_reasons": [],
                }
                exact = validate_replay_row(row, prepared)
                result = row["replay"]
                if row["program_id"] != executors[FAMILY]["program"].specification["program_id"]:
                    raise ValueError("Source program identity was substituted")
                if exact:
                    expected = cfg["families"][FAMILY]["source_role_occurrences"]
                    if {r: n for r, _, n in prepared["component_instances"]} != {
                        r: len(v) for r, v in expected.items()
                    }:
                        raise ValueError("Exact replay changed source role multiplicities")
                    exact_sizes[bound[2]] += 1
                failed.update(k for k, v in result.get("checks", {}).items() if not v)
                local[result["disposition"]] += 1
                targets.append(target)
                db.execute(
                    "INSERT INTO new_evidence VALUES (?,?,?,?,?,?)",
                    (target, FAMILY, int(exact), result["disposition"], shard_pin["path"], line),
                )
            if (
                dict(local) != shard["counts"]
                or len(targets) != shard["rows"]
                or digest(targets) != shard["target_ids_sha256"]
            ):
                raise ValueError("Shard counts or source target digest differ")
            counts.update(local)
        db.commit()
        db.close()
        if {**counts, "rows": sum(counts.values())} != replay["counts"] or sum(
            counts.values()
        ) != old["by_family"][FAMILY]["pending"]:
            raise ValueError("Replay omitted or added eligible source records")
        added = counts["exact_computed_reconstruction"]
        current = copy.deepcopy(previous["summary"]["by_family"])
        for family, values in current.items():
            delta = added if family == FAMILY else 0
            values["new_exact_computed_reconstructions"] = delta
            values["exact_computed_reconstructions"] += delta
            values["eligible_pending_chemistry"] -= delta
        audit_pin = pin(ROOT, audit_db)
        audit_pin["path"] = str((output / audit_db.name).relative_to(ROOT))
        audit = {
            "schema_version": "forge.compose_lipid_incremental_evidence_checkpoint.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {
                "previous": pin(ROOT, previous_path),
                "replay": pin(ROOT, replay_path),
                "previous_index": pin(ROOT, old_path),
                "evidence_index_code": pin(
                    ROOT, ROOT / "forge/corpus/compose_lipid_evidence_index.py"
                ),
                "validation_code": pin(
                    ROOT, ROOT / "forge/corpus/compose_lipid_grouped_readiness.py"
                ),
            },
            "artifact": audit_pin,
            "counts": dict(counts),
            "failed_checks": dict(failed),
            "summary": {
                "by_family": current,
                "totals": dict(sum((Counter(v) for v in current.values()), Counter())),
            },
            "maximum_exact_heavy_atoms": max(exact_sizes, default=0),
            "exact_above_96_atoms": sum(n for atoms, n in exact_sizes.items() if atoms > 96),
            "training_admitted": False,
            "training_calls": 0,
        }
        audit_path = stage / "combined-replay-audit.json"
        dump(audit_path, audit)
        receipt = pin(ROOT, audit_path)
        receipt["path"] = str((output / audit_path.name).relative_to(ROOT))
        index_path = stage / "evidence.sqlite"
        with (
            sqlite3.connect(old_database.as_uri() + "?mode=ro", uri=True) as source,
            sqlite3.connect(index_path.as_uri(), uri=True) as index,
        ):
            source.backup(index)
            if census(index) != old["by_family"]:
                raise ValueError("Previous index census differs")
            population = ROOT / old["partition_inputs"]["partition"]["path"]
            index.execute("ATTACH DATABASE ? AS population", (population.as_uri() + "?mode=ro",))
            index.execute("ATTACH DATABASE ? AS incoming", (audit_db.as_uri() + "?mode=ro",))
            imported = append_evidence(index, receipt=receipt["sha256"], historical=False)
            family_counts = census(index)
            expected = {
                f: {
                    "eligible": v["eligible_preparation_rows"],
                    "exact": v["exact_computed_reconstructions"],
                    "pending": v["eligible_pending_chemistry"],
                }
                for f, v in current.items()
                if v["eligible_preparation_rows"]
            }
            if family_counts != expected or index.execute("PRAGMA quick_check").fetchone() != (
                "ok",
            ):
                raise ValueError("Updated index failed reconciliation")
        index_pin = pin(ROOT, index_path)
        index_pin["path"] = str((output / index_path.name).relative_to(ROOT))
        result = {
            **old,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {
                "previous_index": pin(ROOT, old_path),
                "incremental_checkpoint": receipt,
                "partition_receipt": old["inputs"]["partition_receipt"],
            },
            "artifact": index_pin,
            "evidence_receipts": [*old["evidence_receipts"], receipt],
            "by_family": family_counts,
            "imports": {**old["imports"], receipt["sha256"]: imported},
            "summary": {
                k: sum(v[k] for v in family_counts.values())
                for k in ("eligible", "exact", "pending")
            },
            "elapsed_seconds": time.monotonic() - started,
            "update_method": "verified_database_backup_then_append_only_new_evidence",
        }
        dump(stage / "result.json", result)
        os.rename(stage, output)
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "acid_epoxide": family_counts[FAMILY],
                "seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
