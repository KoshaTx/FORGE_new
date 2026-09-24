"""Join authenticated replay evidence to the current protected preparation population.

This index is a preparation interface, not a training cache. Negative attempts remain
auditable when a later source-qualified program succeeds. No weights or atom origins
are inferred from a successful product reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_full_partition import PartitionPreparationCorpus
from forge.corpus.compose_lipid_source_view import dump, pin

CONFIG_SCHEMA = "forge.compose_lipid_evidence_index_config.v1"
RESULT_SCHEMA = "forge.compose_lipid_evidence_index.v1"
BASE_SCHEMA = "forge.compose_lipid_current_partition_evidence.v1"
INCREMENT_SCHEMA = "forge.compose_lipid_incremental_evidence_checkpoint.v1"


def create_index(db: sqlite3.Connection) -> None:
    """Copy only eligible identities from the attached, authenticated population."""
    db.executescript(
        "CREATE TABLE products(target_id TEXT PRIMARY KEY,family TEXT NOT NULL,"
        "constitution_id TEXT NOT NULL,heavy_atoms INTEGER NOT NULL);"
        "INSERT INTO products SELECT target_id,family,constitution_id,heavy_atoms "
        "FROM population.eligible ORDER BY target_id;"
        "CREATE INDEX products_family ON products(family,target_id);"
        "CREATE TABLE evidence(target_id TEXT NOT NULL,receipt TEXT NOT NULL,"
        "exact INTEGER NOT NULL CHECK(exact IN (0,1)),disposition TEXT NOT NULL,"
        "shard TEXT,source_line INTEGER NOT NULL CHECK(source_line>0),"
        "PRIMARY KEY(target_id,receipt));"
        "CREATE UNIQUE INDEX unique_exact ON evidence(target_id) WHERE exact=1;"
        "CREATE VIEW exact AS SELECT p.*,e.receipt,e.shard,e.source_line FROM products p "
        "JOIN evidence e USING(target_id) WHERE e.exact=1;"
        "CREATE VIEW pending AS SELECT p.* FROM products p WHERE NOT EXISTS "
        "(SELECT 1 FROM evidence e WHERE e.target_id=p.target_id AND e.exact=1);"
    )


def append_evidence(db: sqlite3.Connection, *, receipt: str, historical: bool) -> dict[str, int]:
    """Import one attached audit, retaining failures and rejecting repeated positives."""
    if historical:
        query = (
            "SELECT target_id,family,1 AS exact,'exact_computed_reconstruction' AS disposition,"
            "NULL AS shard,source_line FROM incoming.exact"
        )
    else:
        query = (
            "SELECT target_id,family,exact,disposition,shard,source_line FROM incoming.new_evidence"
        )
    # Validate even excluded historical rows; an unknown target must not disappear in a join.
    invalid = db.execute(
        f"SELECT e.target_id FROM ({query}) e LEFT JOIN population.partition p USING(target_id) "
        "WHERE p.target_id IS NULL OR p.family IS NOT e.family OR e.exact IS NULL "
        "OR e.exact NOT IN (0,1) OR typeof(e.exact)!='integer' "
        "OR e.disposition IS NULL OR e.disposition='' "
        "OR e.exact != (e.disposition='exact_computed_reconstruction') "
        "OR typeof(e.source_line)!='integer' OR e.source_line<1 LIMIT 1"
    ).fetchone()
    if invalid:
        raise ComposeLipidError(f"Invalid evidence identity, family, status or line: {invalid[0]}")
    if (
        not historical
        and db.execute(
            f"SELECT 1 FROM ({query}) e LEFT JOIN products p USING(target_id) "
            "WHERE p.target_id IS NULL OR e.shard IS NULL OR e.shard='' LIMIT 1"
        ).fetchone()
    ):
        raise ComposeLipidError("Incremental evidence is protected, unresolved or lacks a shard")
    total = db.execute(f"SELECT count(*) FROM ({query})").fetchone()[0]
    try:
        db.execute(
            "INSERT INTO evidence SELECT e.target_id,?,e.exact,e.disposition,e.shard,e.source_line "
            f"FROM ({query}) e JOIN products p USING(target_id) ORDER BY e.target_id",
            (receipt,),
        )
    except sqlite3.IntegrityError as error:
        raise ComposeLipidError("Duplicate exact evidence or duplicate audit target") from error
    counts = dict(
        db.execute("SELECT exact,count(*) FROM evidence WHERE receipt=? GROUP BY exact", (receipt,))
    )
    included = sum(counts.values())
    return {
        "source_rows": total,
        "eligible_attempts": included,
        "exact": counts.get(1, 0),
        "nonexact": counts.get(0, 0),
        "historical_excluded": total - included,
    }


def census(db: sqlite3.Connection) -> dict[str, dict[str, int]]:
    return {
        family: {"eligible": eligible, "exact": exact, "pending": eligible - exact}
        for family, eligible, exact in db.execute(
            "SELECT p.family,count(*),count(e.target_id) FROM products p "
            "LEFT JOIN evidence e ON p.target_id=e.target_id AND e.exact=1 GROUP BY p.family"
        )
    }


def build_evidence_index(repo: Path, config_path: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA or not config.get("evidence"):
        raise ComposeLipidError("Malformed evidence index configuration")
    if output.exists():
        raise ComposeLipidError(f"Evidence index output already exists: {output}")
    partition_path = resolve_pin(config["partition"], repo, label="partition receipt")
    reader = PartitionPreparationCorpus(repo, partition_path)
    audits = []
    audit_ids = set()
    for value in config["evidence"]:
        path = resolve_pin(value, repo, label="evidence receipt")
        audit = json.loads(path.read_text())
        if (
            audit.get("schema_version") not in {BASE_SCHEMA, INCREMENT_SCHEMA}
            or audit.get("training_admitted") is not False
            or audit.get("training_calls") != 0
            or value["sha256"] in audit_ids
        ):
            raise ComposeLipidError("Unqualified or repeated evidence audit")
        audit_ids.add(value["sha256"])
        resolve_pin(audit["implementation"], repo, label="evidence implementation")
        for name, input_pin in audit["inputs"].items():
            resolve_pin(input_pin, repo, label=name)
        database = resolve_pin(audit["artifact"], repo, label="evidence database")
        audits.append((value, audit, database))
    expected_path = resolve_pin(config["checkpoint"], repo, label="combined checkpoint")
    if config["checkpoint"] != config["evidence"][-1]:
        raise ComposeLipidError("Combined checkpoint must be the last evidence audit")
    expected = json.loads(expected_path.read_text())["summary"]["by_family"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".evidence-index-", dir=output.parent) as temporary:
        stage = Path(temporary)
        database = stage / "evidence.sqlite"
        with closing(sqlite3.connect(database.as_uri(), uri=True)) as db:
            db.execute("PRAGMA cache_size=-32768")
            db.execute("ATTACH DATABASE ? AS population", (reader.partition.as_uri() + "?mode=ro",))
            create_index(db)
            imported = {}
            for value, audit, incoming in audits:
                db.execute("ATTACH DATABASE ? AS incoming", (incoming.as_uri() + "?mode=ro",))
                imported[value["sha256"]] = append_evidence(
                    db, receipt=value["sha256"], historical=audit["schema_version"] == BASE_SCHEMA
                )
                db.commit()
                db.execute("DETACH DATABASE incoming")
            families = census(db)
            expected_families = {
                family: {
                    "eligible": row["eligible_preparation_rows"],
                    "exact": row["exact_computed_reconstructions"],
                    "pending": row["eligible_pending_chemistry"],
                }
                for family, row in expected.items()
                if row["eligible_preparation_rows"]
            }
            if families != expected_families:
                raise ComposeLipidError(
                    "Unified evidence census differs from the combined checkpoint"
                )
            if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ComposeLipidError("Evidence database integrity check failed")
            unique = db.execute("SELECT count(DISTINCT constitution_id) FROM products").fetchone()[
                0
            ]
            duplicate_identities = sum(v["eligible"] for v in families.values()) - unique
        result = {
            "schema_version": RESULT_SCHEMA,
            "seed": 0,
            "inputs": {"config": pin(repo, config_path), "partition_receipt": config["partition"]},
            "implementation": pin(repo, Path(__file__)),
            "partition_inputs": {
                "partition": reader.result["artifact"],
                "corpus": reader.result["inputs"]["corpus"],
                "joins": reader.result["inputs"]["joins"],
            },
            "evidence_receipts": config["evidence"],
            "artifact": {
                **pin(repo, database),
                "path": str((output / database.name).relative_to(repo)),
            },
            "by_family": families,
            "imports": imported,
            "summary": {
                k: sum(row[k] for row in families.values())
                for k in ("eligible", "exact", "pending")
            },
            "duplicate_eligible_constitutions": duplicate_identities,
            "training_admitted": False,
            "training_calls": 0,
            "sampling_weights_fitted": False,
            "joint_program_representation_qualified": False,
            "elapsed_seconds": time.monotonic() - started,
        }
        dump(stage / "result.json", result)
        stage.rename(output)
    return result


class EvidencePreparationCorpus:
    """Stream exact or pending preparation rows without parsing protected molecules."""

    def __init__(self, repo: Path, result_path: Path):
        self.result = json.loads(result_path.read_text())
        if (
            self.result.get("schema_version") != RESULT_SCHEMA
            or self.result.get("training_admitted") is not False
            or self.result.get("sampling_weights_fitted") is not False
        ):
            raise ComposeLipidError("Evidence preparation contract is unqualified")
        resolve_pin(self.result["implementation"], repo, label="index implementation")
        for name, value in self.result["inputs"].items():
            resolve_pin(value, repo, label=name)
        self.paths = {
            name: resolve_pin(value, repo, label=name)
            for name, value in self.result["partition_inputs"].items()
        }
        self.database = resolve_pin(self.result["artifact"], repo, label="evidence index")
        self.receipts = {}
        for value in self.result["evidence_receipts"]:
            resolve_pin(value, repo, label="evidence receipt")
            self.receipts[value["sha256"]] = value

    def iter_preparation_records(self, *, family: str, exact: bool = True):
        if family not in self.result["by_family"] or type(exact) is not bool:
            raise ComposeLipidError("Unknown eligible family or non-boolean evidence selector")
        table = "exact" if exact else "pending"
        with closing(sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True)) as db:
            for name, path in self.paths.items():
                db.execute(f"ATTACH DATABASE ? AS {name}", (path.as_uri() + "?mode=ro",))
            db.execute("PRAGMA cache_size=-32768")
            evidence_columns = "e.receipt,e.shard,e.source_line" if exact else "NULL,NULL,NULL"
            query = db.execute(
                f"SELECT e.target_id,e.family,e.constitution_id,e.heavy_atoms,t.payload,"
                f"c.instances,c.basis,c.source_line,t.family,c.family,{evidence_columns} "
                f"FROM {table} e JOIN partition.eligible p USING(target_id) "
                "JOIN corpus.targets t USING(target_id) JOIN joins.constructions c USING(target_id) "
                "WHERE e.family=? AND e.family=p.family AND e.constitution_id=p.constitution_id "
                "AND e.heavy_atoms=p.heavy_atoms ORDER BY e.target_id",
                (family,),
            )
            count = 0
            for (
                target,
                actual_family,
                identity,
                atoms,
                payload,
                instances,
                basis,
                line,
                tf,
                cf,
                receipt,
                shard,
                evidence_line,
            ) in query:
                source = json.loads(payload)
                components = json.loads(instances)
                if (
                    actual_family != tf
                    or actual_family != cf
                    or source["target_id"] != target
                    or source["heavy_atoms"] != atoms
                    or hashlib.sha256(source["constitution"].encode()).hexdigest() != identity
                    or not components
                    or any(len(v) != 3 or type(v[2]) is not int or v[2] <= 0 for v in components)
                ):
                    raise ComposeLipidError("Preparation identity, family, size or recipe changed")
                count += 1
                yield {
                    "source": source,
                    "preparation": {
                        "target_id": target,
                        "family": family,
                        "constitution_id": identity,
                        "component_instances": components,
                        "construction_basis": basis,
                        "construction_source_line": line,
                        "old_projection": "train",
                        "corrected_projection": "train",
                        "eligible_for_program_preparation": True,
                        "training_admitted": False,
                    },
                    "evidence": {
                        "exact_computed_reconstruction": exact,
                        "receipt": self.receipts[receipt] if exact else None,
                        "shard": shard,
                        "source_line": evidence_line,
                    },
                }
            if count != self.result["by_family"][family][table]:
                raise ComposeLipidError(
                    "Preparation reader census differs from its evidence receipt"
                )

    def iter_training_records(self):
        raise ComposeLipidError(
            "Exact replay does not qualify joint representation or training admission"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_evidence_index(args.repo.resolve(), args.config.resolve(), args.output.resolve())
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
