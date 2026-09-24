"""Reproduce a TRAIN-only, directly contradictory source-label candidate intersection."""

import gzip
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULT = REPO / "results/phase1/compose_lipid_v8_components_v1/result.json"
EXPECTED = "55ff93818a30a3d6e2ddfda4145617ac9127c36de8a3a395b9beaaf909cedfa8"


def pin(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return {
        "path": path.relative_to(REPO).as_posix(),
        "sha256": digest.hexdigest(),
    }


def main():
    assert pin(RESULT)["sha256"] == EXPECTED
    result = json.loads(RESULT.read_text())
    source = result["artifacts"]["candidate_constraints.jsonl.gz"]
    path = REPO / source["path"]
    assert pin(path) == source
    grouped = defaultdict(list)
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["family"] != "aryl_reductive_amination" or not row["candidates"]:
                continue
            for role, label in row["labels"].items():
                grouped[(role, label)].append(row)
    for (role, label), rows in sorted(grouped.items()):
        possible = None
        witness = []
        for row in rows:
            values = {c["components"][role] for c in row["candidates"]}
            narrowed = values if possible is None else possible & values
            if narrowed != possible:
                witness.append(
                    {
                        "target_id": row["target_id"],
                        "candidate_component_constitutions": sorted(values),
                    }
                )
            possible = narrowed
            if not possible:
                break
        if possible == set():
            break
    else:
        raise RuntimeError("No direct contradiction found")
    imported = REPO / "results/phase1/compose_lipid_v8_import_v1/result.json"
    receipt = json.loads(imported.read_text())
    db_pin = receipt["artifacts"]["corpus.sqlite"]
    db_path = REPO / db_pin["path"]
    assert pin(db_path) == db_pin
    with sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True) as db:
        for row in witness:
            payload = db.execute(
                "SELECT t.payload FROM assignments a JOIN targets t USING(target_id) "
                "WHERE a.forge_split='train' AND a.target_id=?",
                (row["target_id"],),
            ).fetchone()
            assert payload is not None
            row["source_train_record"] = json.loads(payload[0])
    output = {
        "schema_version": "forge.compose_lipid_component_conflict_witness.v1",
        "inputs": {
            "component_result": pin(RESULT),
            "constraints": source,
            "import_result": pin(imported),
            "database": db_pin,
            "script": pin(Path(__file__)),
        },
        "family": "aryl_reductive_amination",
        "role": role,
        "source_label": label,
        "witnesses": witness,
        "candidate_intersection": [],
        "evidence_basis": "computed_related_transform_constraint_conflict",
        "interpretation": "One-event inverse candidates cannot represent a common source-labelled head across these TRAIN rows. This does not distinguish an incomplete related program from a metadata problem.",
        "training_admitted": False,
        "heldout_graphs_parsed": False,
    }
    Path(__file__).with_name("conflict-witness.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
