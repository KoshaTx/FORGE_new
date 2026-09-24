"""Extend authenticated frozen split rules without selecting new holdout groups.

This partition admits only program preparation. It preserves prior exclusions and
closes product identity overlap after propagating complete held precursors globally.
Molecular consumers must query the eligible view before decoding source payloads.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import resolve_pin

RESULT_SCHEMA = "forge.compose_lipid_full_partition.v1"
POLICY = {
    "seed": 0,
    "source_assignments_changed": False,
    "new_holdout_groups_selected": False,
    "preparation_requires": "both_frozen_projections_train_and_no_exclusions_or_uncertainty",
    "protected_product_overlap": "constitutional_identity_global_closure",
    "unresolved_product_overlap": "constitutional_identity_global_closure",
    "protected_components": "prior_union_corrected_union_resolved_historical_global_ids",
    "record_cap": None,
    "size_filter": None,
    "training_admitted": False,
    "training_calls": 0,
}


def merge_partition(db: sqlite3.Connection) -> dict:
    """Merge authenticated staging tables; never parse or decompose held products.

    Requires attached `features` and `source` databases, and main tables
    `corrected` and `blocked_components`. The caller authenticates their bytes.
    """
    expected = db.execute("SELECT count(*) FROM features.features").fetchone()[0]
    if not expected or db.execute("SELECT count(*) FROM corrected").fetchone()[0] != expected:
        raise ComposeLipidError("Full partition input populations differ")
    invalid = db.execute(
        "SELECT count(*) FROM corrected c LEFT JOIN features.features f USING(target_id) "
        "LEFT JOIN source.constructions s USING(target_id) WHERE f.target_id IS NULL "
        "OR s.target_id IS NULL OR c.family!=f.family OR c.family!=s.family "
        "OR c.fold NOT IN ('train','calibration','test','reference') "
        "OR f.old_projection NOT IN ('train','calibration','test','reference') "
        "OR c.prior_disposition NOT IN ('eligible_for_program_preparation','protected',"
        "'unresolved_partition_or_study') OR c.study_unresolved NOT IN (0,1) "
        "OR c.study_unresolved!=f.source_study_unresolved"
    ).fetchone()[0]
    if invalid:
        raise ComposeLipidError(f"Full partition has {invalid} invalid or inconsistent joins")
    db.executescript(
        "CREATE TABLE component_matches(target_id TEXT PRIMARY KEY,component_ids TEXT NOT NULL);"
        "INSERT INTO component_matches SELECT c.target_id,json_group_array(DISTINCT b.id) "
        "FROM source.constructions c,json_each(c.instances) j JOIN blocked_components b "
        "ON b.id=json_extract(j.value,'$[1]') GROUP BY c.target_id;"
        "CREATE TABLE partition (target_id TEXT PRIMARY KEY,family TEXT NOT NULL,"
        "constitution_id TEXT,heavy_atoms INTEGER NOT NULL,old_projection TEXT NOT NULL,"
        "corrected_projection TEXT NOT NULL,prior_disposition TEXT NOT NULL,"
        "study_unresolved INTEGER NOT NULL,identity_error TEXT,"
        "protected_component_ids TEXT NOT NULL,prior_protected INTEGER NOT NULL,"
        "direct_protected INTEGER NOT NULL,canonical_protected INTEGER NOT NULL DEFAULT 0,"
        "canonical_unresolved INTEGER NOT NULL DEFAULT 0,disposition TEXT);"
        "INSERT INTO partition (target_id,family,constitution_id,heavy_atoms,old_projection,"
        "corrected_projection,prior_disposition,study_unresolved,identity_error,"
        "protected_component_ids,prior_protected,direct_protected) "
        "SELECT f.target_id,f.family,f.constitution_id,f.heavy_atoms,f.old_projection,c.fold,"
        "c.prior_disposition,c.study_unresolved,nullif(f.identity_error,''),coalesce(m.component_ids,'[]'),"
        "c.prior_disposition='protected',"
        "(c.prior_disposition='protected' OR f.old_projection!='train' OR c.fold!='train' "
        "OR m.target_id IS NOT NULL) FROM features.features f JOIN corrected c USING(target_id) "
        "LEFT JOIN component_matches m USING(target_id);"
        "CREATE INDEX partition_identity ON partition(constitution_id);"
        "CREATE TABLE protected_identities(identity TEXT PRIMARY KEY);"
        "INSERT INTO protected_identities SELECT identity FROM features.protected_identities;"
        "INSERT OR IGNORE INTO protected_identities SELECT DISTINCT constitution_id FROM partition "
        "WHERE direct_protected=1 AND constitution_id IS NOT NULL;"
        "CREATE TABLE unresolved_identities(identity TEXT PRIMARY KEY);"
        "INSERT INTO unresolved_identities SELECT DISTINCT constitution_id FROM partition "
        "WHERE (study_unresolved=1 OR identity_error IS NOT NULL) AND constitution_id IS NOT NULL;"
        "UPDATE partition SET canonical_protected=1 WHERE constitution_id IN "
        "(SELECT identity FROM protected_identities);"
        "UPDATE partition SET canonical_unresolved=1 WHERE constitution_id IN "
        "(SELECT identity FROM unresolved_identities);"
        "UPDATE partition SET disposition=CASE "
        "WHEN direct_protected=1 OR canonical_protected=1 THEN 'protected' "
        "WHEN canonical_unresolved=1 OR study_unresolved=1 OR identity_error IS NOT NULL "
        "OR constitution_id IS NULL THEN 'unresolved_partition_or_study' "
        "ELSE 'eligible_for_program_preparation' END;"
        "CREATE INDEX partition_family_disposition ON partition(family,disposition,target_id);"
        "CREATE VIEW eligible AS SELECT * FROM partition "
        "WHERE disposition='eligible_for_program_preparation';"
    )
    invariant_failures = db.execute(
        "SELECT count(*) FROM eligible WHERE old_projection!='train' OR corrected_projection!='train' "
        "OR prior_protected!=0 OR direct_protected!=0 OR canonical_protected!=0 "
        "OR canonical_unresolved!=0 OR study_unresolved!=0 OR identity_error IS NOT NULL "
        "OR constitution_id IS NULL OR protected_component_ids!='[]'"
    ).fetchone()[0]
    if invariant_failures:
        raise ComposeLipidError("Full partition released a protected or unresolved record")
    by_family = {}
    for family, disposition, count in db.execute(
        "SELECT family,disposition,count(*) FROM partition GROUP BY family,disposition"
    ):
        by_family.setdefault(family, {})[disposition] = count
    totals = Counter()
    for counts in by_family.values():
        totals.update(counts)
        counts["rows"] = sum(counts.values())
    totals["rows"] = sum(totals.values())
    if totals["rows"] != expected:
        raise ComposeLipidError("Full partition lost source records")
    transitions = [
        {"previous": old, "current": new, "rows": count}
        for old, new, count in db.execute(
            "SELECT prior_disposition,disposition,count(*) FROM partition "
            "GROUP BY prior_disposition,disposition ORDER BY 1,2"
        )
    ]
    return {
        "totals": dict(totals),
        "by_family": by_family,
        "transitions": transitions,
        "invariant_failures": invariant_failures,
        "rows_with_protected_components": db.execute(
            "SELECT count(*) FROM component_matches"
        ).fetchone()[0],
        "newly_protected_via_identity_closure": db.execute(
            "SELECT count(*) FROM partition WHERE direct_protected=0 AND canonical_protected=1"
        ).fetchone()[0],
        "eligible_unique_constitutions": db.execute(
            "SELECT count(DISTINCT constitution_id) FROM eligible"
        ).fetchone()[0],
        "maximum_heavy_atoms": db.execute("SELECT max(heavy_atoms) FROM partition").fetchone()[0],
        "above_96_atoms": db.execute(
            "SELECT count(*) FROM partition WHERE heavy_atoms>96"
        ).fetchone()[0],
        "eligible_maximum_heavy_atoms": db.execute(
            "SELECT max(heavy_atoms) FROM eligible"
        ).fetchone()[0],
        "eligible_above_96_atoms": db.execute(
            "SELECT count(*) FROM eligible WHERE heavy_atoms>96"
        ).fetchone()[0],
    }


class PartitionPreparationCorpus:
    """Stream only partition-qualified records while preserving complete source roles."""

    def __init__(self, repo_root: Path, result_path: Path):
        self.result = json.loads(result_path.read_text())
        if (
            self.result.get("schema_version") != RESULT_SCHEMA
            or self.result.get("policy") != POLICY
            or self.result.get("eligible_view_qualified") is not True
            or self.result["summary"]["invariant_failures"] != 0
        ):
            raise ComposeLipidError("Full partition preparation contract is unqualified")
        for name, value in self.result["implementation"].items():
            resolve_pin(value, repo_root, label=name)
        paths = {
            name: resolve_pin(value, repo_root, label=name)
            for name, value in self.result["inputs"].items()
        }
        self.partition = resolve_pin(self.result["artifact"], repo_root, label="partition")
        self.corpus, self.joins = paths["corpus"], paths["joins"]

    def iter_preparation_records(self, *, family: str):
        if family not in self.result["summary"]["by_family"]:
            raise ComposeLipidError(f"Unknown partition family: {family}")
        with closing(sqlite3.connect(self.partition.as_uri() + "?mode=ro", uri=True)) as db:
            db.execute("ATTACH DATABASE ? AS original", (self.corpus.as_uri() + "?mode=ro",))
            db.execute("ATTACH DATABASE ? AS source", (self.joins.as_uri() + "?mode=ro",))
            db.execute("PRAGMA cache_size=-32768")
            query = db.execute(
                "SELECT e.target_id,e.family,e.constitution_id,e.heavy_atoms,t.payload,"
                "c.instances,c.basis,c.source_line,t.family,c.family FROM eligible e "
                "JOIN original.targets t USING(target_id) JOIN source.constructions c USING(target_id) "
                "WHERE e.family=? ORDER BY e.target_id",
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
            ) in query:
                if actual_family != tf or actual_family != cf:
                    raise ComposeLipidError("Eligible source family join differs")
                source = json.loads(payload)
                if source["target_id"] != target or source["heavy_atoms"] != atoms:
                    raise ComposeLipidError("Eligible source identity/size join differs")
                count += 1
                yield {
                    "source": source,
                    "preparation": {
                        "target_id": target,
                        "family": actual_family,
                        "constitution_id": identity,
                        "component_instances": json.loads(instances),
                        "construction_basis": basis,
                        "construction_source_line": line,
                        "old_projection": "train",
                        "corrected_projection": "train",
                        "eligible_for_program_preparation": True,
                        "training_admitted": False,
                    },
                }
            expected = self.result["summary"]["by_family"][family].get(
                "eligible_for_program_preparation", 0
            )
            if count != expected:
                raise ComposeLipidError("Eligible source population differs from partition receipt")

    def iter_training_records(self):
        raise ComposeLipidError("Partition preparation has not qualified training admission")
