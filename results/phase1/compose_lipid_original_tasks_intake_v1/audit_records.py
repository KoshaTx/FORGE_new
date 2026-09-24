"""Stream source records for schema, row-count and ID-to-supplied-SMILES checks only."""

from __future__ import annotations

import gzip
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from verify_restore import CACHE, OUT, PREVIOUS, ROOT, pin


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Non-object record: {path}")
            yield row


def main():
    package = json.loads((OUT / "package_check.json").read_text())
    if package["status"] != "both_checksum_manifests_verified":
        raise ValueError("Verify source package before inspecting records")
    expected_manifest = package["inputs"]["new_manifest"]
    if pin(ROOT / expected_manifest["path"]) != expected_manifest:
        raise ValueError("Verified package manifest changed")
    manifest = json.loads((CACHE / "MANIFEST.json").read_text())
    for item in manifest["files"]:
        if pin(CACHE / item["path"])["sha256"] != item["sha256"]:
            raise ValueError(f"Verified package file changed: {item['path']}")
    task_files = [r for r in manifest["files"] if r["path"].startswith("original_generator_tasks/")]
    catalog = (
        PREVIOUS
        / "original_generator_tasks/family_03_aldehyde_ugi4_enumeration_v8/precursors.jsonl"
    )
    prior_expected = dict(
        (line.split(maxsplit=1)[1], line.split(maxsplit=1)[0])
        for line in (PREVIOUS / "SHA256SUMS").read_text().splitlines()
    )
    if pin(catalog)["sha256"] != prior_expected[str(catalog.relative_to(PREVIOUS))]:
        raise ValueError("Ugi-4 precursor catalogue bytes changed")
    components = {}
    for row in rows(catalog):
        identity, role, structure = row["id"], row["role"], row["smiles"]
        if not all(isinstance(v, str) and v for v in (identity, role, structure)):
            raise ValueError("Incomplete Ugi-4 catalogue row")
        if identity in components and components[identity] != (role, structure):
            raise ValueError(f"Conflicting catalogue identity: {identity}")
        components[identity] = (role, structure)
    expected_rows = {}
    for filename in ("family_18_staar.receipt.json", "family_21_acid_epoxide_diester.receipt.json"):
        receipt = json.loads((CACHE / "provenance" / filename).read_text())
        for shard in receipt["task_shards"]:
            expected_rows[shard["path"]] = shard["tasks"]
    summaries = []
    family3_ids = set()
    family3_parent_ids = set()
    unresolved = Counter()
    with tempfile.TemporaryDirectory(prefix=".record-audit-", dir=OUT) as temporary:
        db = sqlite3.connect(Path(temporary) / "ids.sqlite")
        db.execute("CREATE TABLE identities(family TEXT, id TEXT, PRIMARY KEY(family,id))")
        try:
            for item in task_files:
                path = CACHE / item["path"]
                if pin(path)["sha256"] != item["sha256"]:
                    raise ValueError(f"Changed task file: {path}")
                family = path.parts[path.parts.index("original_generator_tasks") + 1]
                count = 0
                flags = Counter()
                lanes = Counter()
                complete_structures = 0
                fully_resolved = 0
                fully_parent_resolved = 0
                batch = []
                for row in rows(path):
                    count += 1
                    if row.get("training_admissible") is not False:
                        raise ValueError(f"Unexpected source training-admission flag: {path}")
                    flags[str(row.get("training_admissible", "absent"))] += 1
                    lanes[str(row.get("design_lane", row.get("disposition", "absent")))] += 1
                    if "family_03_" in family:
                        ids = row["precursor_ids"]
                        parents = row["source_parent_precursor_ids"]
                        family3_ids.update(ids)
                        family3_parent_ids.update(parents)
                        missing = [v for v in ids if v not in components]
                        missing_parents = [v for v in parents if v not in components]
                        unresolved.update(missing + missing_parents)
                        fully_resolved += bool(ids) and not missing
                        fully_parent_resolved += bool(parents) and not missing_parents
                        identity = row["tuple_id"]
                    elif "family_05_" in family:
                        complete = row["complete_subcomponents"]
                        if row["precursor_ids"] != [v["subcomponent_id"] for v in complete]:
                            raise ValueError(
                                f"Ugi-3 precursor IDs disagree with components: {path}"
                            )
                        if any(
                            not isinstance(v.get("constitutional_smiles"), str)
                            or not v["constitutional_smiles"]
                            for v in complete
                        ):
                            raise ValueError(f"Missing Ugi-3 precursor structure: {path}")
                        complete_structures += bool(complete)
                        # Context ledgers can contain more than one context for a product;
                        # no new uniqueness assumption is imposed on their source IDs.
                        identity = None
                    else:
                        roles = (
                            ("head", "thiolactone", "acrylate")
                            if "family_18_" in family
                            else ("head", "epoxide", "hydrophobic_acid")
                        )
                        if any(
                            not isinstance(row.get(role + suffix), str) or not row[role + suffix]
                            for role in roles
                            for suffix in ("_id", "_smiles")
                        ):
                            raise ValueError(f"Missing task component ID/SMILES: {path}")
                        complete_structures += 1
                        identity = row["task_id"]
                    if identity is not None:
                        batch.append((family, identity))
                    if len(batch) >= 2048:
                        db.executemany("INSERT INTO identities VALUES (?,?)", batch)
                        batch.clear()
                db.executemany("INSERT INTO identities VALUES (?,?)", batch)
                if item["source"] in expected_rows and count != expected_rows[item["source"]]:
                    raise ValueError(f"Task count disagrees with family receipt: {path}")
                summaries.append(
                    {
                        "input": pin(path),
                        "family_directory": family,
                        "rows": count,
                        "source_training_admissible_values": dict(flags),
                        "lanes": dict(lanes),
                        "rows_with_complete_supplied_component_smiles": complete_structures,
                        "rows_with_all_precursor_ids_resolved_in_ugi4_catalog": fully_resolved,
                        "rows_with_all_parent_ids_resolved_in_ugi4_catalog": fully_parent_resolved,
                        "receipt_task_count_checked": item["source"] in expected_rows,
                    }
                )
        finally:
            db.close()
    family_counts = Counter()
    for summary in summaries:
        family_counts[summary["family_directory"]] += summary["rows"]
    result = {
        "schema_version": "forge.compose_lipid_original_tasks_record_audit.v1",
        "seed": 0,
        "implementation": pin(Path(__file__)),
        "inputs": {
            "package": pin(OUT / "package_check.json"),
            "catalog": pin(catalog),
            "manifest": pin(CACHE / "MANIFEST.json"),
            "helper": pin(OUT / "verify_restore.py"),
        },
        "family_rows": dict(family_counts),
        "files": summaries,
        "ugi4_catalog": {
            "catalog_unique_ids": len(components),
            "distinct_precursor_ids_used": len(family3_ids),
            "distinct_parent_ids_used": len(family3_parent_ids),
            "unresolved_id_occurrences": dict(sorted(unresolved.items())),
            "all_referenced_ids_resolve": not unresolved,
            "resolution_meaning": "exact source ID to supplied SMILES, not chemistry qualification",
        },
        "unique_family3_tuple_and_family18_21_task_ids": True,
        "task_receipt_counts_verified": len(expected_rows),
        "molecular_graphs_parsed": 0,
        "reactions_executed": 0,
        "training_admitted": False,
        "training_calls": 0,
        "scope": "Source metadata and supplied structures only; upstream chemical claims not promoted.",
    }
    (OUT / "record_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"family_rows": result["family_rows"], "ugi4_catalog": result["ugi4_catalog"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
