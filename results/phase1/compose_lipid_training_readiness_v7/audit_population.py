"""Reconcile frozen computed evidence with the expanded protected partition."""

import hashlib
import json
import os
import sqlite3
import tempfile
import time
from collections import Counter
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_full_partition import PartitionPreparationCorpus
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    started = time.monotonic()
    partition_path = ROOT / "results/phase1/compose_lipid_full_partition_v1/audit-v2/result.json"
    previous_path = ROOT / "results/phase1/compose_lipid_readiness_b5_v2/result.json"
    previous = json.loads(previous_path.read_text())
    if previous["training_admitted"] is not False:
        raise ValueError("Prior evidence has inconsistent admission scope")
    for name, value in previous["implementation"].items():
        resolve_pin(value, ROOT, label=name)
    reader = PartitionPreparationCorpus(ROOT, partition_path)
    artifact = resolve_pin(previous["artifact"], ROOT, label="frozen evidence ledger")
    pins = {
        "partition_receipt": pin(ROOT, partition_path),
        "partition": reader.result["artifact"],
        "prior_chemistry_receipt": pin(ROOT, previous_path),
        "prior_chemistry_ledger": previous["artifact"],
    }
    output = HERE / "evidence.sqlite"
    if output.exists():
        raise ValueError("Frozen population audit already exists")
    with tempfile.TemporaryDirectory(prefix=".evidence-", dir=HERE) as tmp:
        stage = Path(tmp) / output.name
        db = sqlite3.connect(stage.as_uri(), uri=True)
        db.execute("ATTACH DATABASE ? AS population", (reader.partition.as_uri() + "?mode=ro",))
        db.execute(
            "CREATE TABLE exact(target_id TEXT PRIMARY KEY,family TEXT NOT NULL,source_line INT NOT NULL)"
        )
        count = 0
        batch = []
        for count, row in enumerate(rows(artifact), 1):
            if (
                type(row["passes_completed_recipe_checks"]) is not bool
                or row["training_admitted"] is not False
            ):
                raise ValueError("Invalid prior chemistry evidence row")
            if row["passes_completed_recipe_checks"]:
                batch.append((row["target_id"], row["family"], count))
            if len(batch) >= 2048:
                db.executemany("INSERT INTO exact VALUES (?,?,?)", batch)
                batch.clear()
        db.executemany("INSERT INTO exact VALUES (?,?,?)", batch)
        if count != previous["summary"]["source_records"]:
            raise ValueError("Prior evidence population changed")
        exact_count = db.execute("SELECT count(*) FROM exact").fetchone()[0]
        if exact_count != previous["summary"]["exact_reconstructions"]:
            raise ValueError("Prior exact evidence total changed")
        if db.execute(
            "SELECT count(*) FROM exact e LEFT JOIN population.partition p USING(target_id) WHERE p.target_id IS NULL OR p.family!=e.family"
        ).fetchone()[0]:
            raise ValueError("Prior evidence does not join the current population")
        carried = {}
        for family, disposition, n in db.execute(
            "SELECT e.family,p.disposition,count(*) FROM exact e JOIN population.partition p USING(target_id) GROUP BY 1,2"
        ):
            carried.setdefault(family, {})[disposition] = n
        eligible = Counter()
        for family in sorted(reader.result["summary"]["by_family"]):
            for item in reader.iter_preparation_records(family=family):
                source, prepared = item["source"], item["preparation"]
                if (
                    hashlib.sha256(source["constitution"].encode()).hexdigest()
                    != prepared["constitution_id"]
                ):
                    raise ValueError("Reader product identity changed")
                instances = prepared["component_instances"]
                if not instances or any(
                    len(v) != 3 or type(v[2]) is not int or v[2] <= 0 for v in instances
                ):
                    raise ValueError("Reader changed complete source component instances")
                eligible[family] += 1
            print(json.dumps({"family": family, "reader_verified": eligible[family]}), flush=True)
        if (
            sum(eligible.values())
            != reader.result["summary"]["totals"]["eligible_for_program_preparation"]
        ):
            raise ValueError("Reader lost eligible source records")
        db.commit()
        db.close()
        os.rename(stage, output)
    by_family = {}
    for family, counts in reader.result["summary"]["by_family"].items():
        remaining = counts.get("eligible_for_program_preparation", 0)
        evidence = carried.get(family, {})
        kept = evidence.get("eligible_for_program_preparation", 0)
        by_family[family] = {
            "source_records": counts["rows"],
            "eligible_preparation_rows": remaining,
            "protected_records": counts.get("protected", 0),
            "exact_computed_reconstructions_carried": kept,
            "historical_exact_now_protected": evidence.get("protected", 0),
            "eligible_pending_chemistry": remaining - kept,
        }
    totals = dict(sum((Counter(v) for v in by_family.values()), Counter()))
    dump(
        HERE / "population-audit.json",
        {
            "schema_version": "forge.compose_lipid_current_partition_evidence.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": pins,
            "artifact": pin(ROOT, output),
            "summary": {"by_family": by_family, "totals": totals},
            "reader_verified_rows": sum(eligible.values()),
            "chemistry_recomputed": False,
            "training_admitted": False,
            "training_calls": 0,
            "elapsed_seconds": time.monotonic() - started,
        },
    )


if __name__ == "__main__":
    main()
