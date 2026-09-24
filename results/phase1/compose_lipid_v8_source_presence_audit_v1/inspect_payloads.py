"""Inventory JSON field paths without returning or parsing evaluation product structures."""

import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from forge.core.hashing import resolve_pin, sha256_file  # noqa: E402


def pin(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": str(sha256_file(path))}


def main():
    config_path = ROOT / "configs/multireaction/compose_lipid_v8_universe_v1.json"
    config = json.loads(config_path.read_text())
    imported_path = resolve_pin(config["inputs"]["import_result"], ROOT, label="import receipt")
    imported = json.loads(imported_path.read_text())
    database = resolve_pin(imported["artifacts"]["corpus.sqlite"], ROOT, label="corpus index")
    queries = {}
    inventory = {}
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-8192")
        for table in ("targets", "accepted", "assignments"):
            # Only fixed internal table names are interpolated. Molecular values stay in SQLite.
            query = (
                f"SELECT j.fullkey,j.type,count(*) FROM {table} r, json_tree(r.payload) j "
                "WHERE j.key IS NOT NULL GROUP BY j.fullkey,j.type ORDER BY j.fullkey,j.type"
            )
            queries[table] = query
            inventory[table] = [
                {"field": field, "json_type": kind, "occurrences": count}
                for field, kind, count in db.execute(query)
            ]
            print(f"Finished {table} field inventory", flush=True)
    rows = {
        table: sum(row["occurrences"] for row in fields if row["field"] == '$."target_id"')
        for table, fields in inventory.items()
    }
    assert rows == {
        "targets": imported["summary"]["universe_rows"],
        "accepted": imported["summary"]["accepted_rows"],
        "assignments": imported["summary"]["assignment_rows"],
    }
    result = {
        "schema_version": "forge.compose_lipid_payload_field_inventory.v1",
        "seed": 0,
        "random_sampling_used": False,
        "inputs": {
            "config": pin(config_path),
            "import": pin(imported_path),
            "database": pin(database),
            "implementation": pin(Path(__file__)),
        },
        "queries": queries,
        "rows_accounted_for": rows,
        "field_inventory": inventory,
        "product_graphs_returned_to_python": False,
        "product_graphs_parsed": False,
        "training_calls": 0,
        "scope": "JSON field names and types only; field names alone do not prove chemical semantics",
    }
    temp = OUTPUT / ".payload-fields.partial.json"
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temp.replace(OUTPUT / "payload-fields.json")


if __name__ == "__main__":
    main()
