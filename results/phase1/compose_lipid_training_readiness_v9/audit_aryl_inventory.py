"""Persist source-recipe inventory conflicts without inferring replacement quantities."""

import gzip
import json
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FAMILY = "aryl_reductive_amination"


def main():
    result_path = ROOT / "results/phase1/compose_lipid_full_program_replay_v1/programs/result.json"
    partition_path = ROOT / "results/phase1/compose_lipid_full_partition_v1/audit-v2/result.json"
    result, partition = (json.loads(p.read_text()) for p in (result_path, partition_path))
    database = resolve_pin(partition["artifact"], ROOT, label="partition")
    source = resolve_pin(partition["inputs"]["corpus"], ROOT, label="source")
    db = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    db.execute("ATTACH DATABASE ? AS source", (source.as_uri() + "?mode=ro",))
    output = HERE / "aryl-inventory-conflicts.jsonl.gz"
    if output.exists():
        raise ValueError("Published inventory conflicts are frozen")
    counts, heads, signatures = Counter(), Counter(), Counter()
    shards = []
    with tempfile.TemporaryDirectory(prefix=".evidence-", dir=HERE) as tmp:
        stage = Path(tmp) / output.name
        with (
            stage.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream,
        ):
            for shard_pin in result["shards"]:
                if FAMILY not in shard_pin["path"]:
                    continue
                shard = json.loads(resolve_pin(shard_pin, ROOT, label="shard").read_text())
                if (
                    shard["family"] != FAMILY
                    or shard["request_sha256"] != result["request"]["sha256"]
                ):
                    raise ValueError("Inventory audit encountered a substituted replay")
                shards.append(shard_pin)
                for row in rows(resolve_pin(shard["artifact"], ROOT, label="replay")):
                    replay = row["replay"]
                    counts[replay["disposition"]] += 1
                    if (
                        replay.get("checks", {}).get("full_element_hydrogen_charge_balance")
                        is not False
                    ):
                        continue
                    matched = db.execute(
                        "SELECT e.family,e.constitution_id,t.payload FROM eligible e "
                        "JOIN source.targets t USING(target_id) WHERE e.target_id=?",
                        (row["target_id"],),
                    ).fetchone()
                    if matched is None or matched[:2] != (FAMILY, row["constitution_id"]):
                        raise ValueError(
                            "Inventory conflict is outside the current eligible partition"
                        )
                    original = json.loads(matched[2])
                    balance = replay["balance"]
                    left, right = balance["reactants"], balance["product_and_byproducts"]
                    delta = {
                        k: right.get(k, 0) - left.get(k, 0) for k in left.keys() | right.keys()
                    }
                    delta = {k: v for k, v in sorted(delta.items()) if v}
                    if not delta:
                        raise ValueError(
                            "Claimed inventory conflict has no nonzero inventory difference"
                        )
                    head_code = original["primary_metadata"].get("head_code")
                    heads[str(head_code)] += 1
                    signatures[",".join(delta)] += 1
                    issue = {
                        "target_id": row["target_id"],
                        "constitution_id": row["constitution_id"],
                        "family": FAMILY,
                        "source_metadata": original["primary_metadata"],
                        "component_instances": row["component_instances"],
                        "balance": balance,
                        "product_and_net_byproducts_minus_reactants": delta,
                        "replay_shard": shard_pin,
                        "proposed_component_or_quantity_correction": None,
                        "training_admitted": False,
                    }
                    stream.write((compact(issue) + "\n").encode())
        if dict(counts) != {k: v for k, v in result["summary"][FAMILY].items() if k != "rows"}:
            raise ValueError("Inventory audit did not cover the complete aryl replay")
        os.rename(stage, output)
    db.close()
    dump(
        HERE / "aryl-inventory-audit.json",
        {
            "schema_version": "forge.compose_lipid_source_inventory_conflicts.v1",
            "seed": 0,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {"replay": pin(ROOT, result_path), "partition": pin(ROOT, partition_path)},
            "shards": shards,
            "artifact": pin(ROOT, output),
            "family": FAMILY,
            "replay_dispositions": counts,
            "inventory_conflicts": sum(heads.values()),
            "by_source_head_code": heads,
            "nonzero_inventory_elements": signatures,
            "conclusion": "These complete supplied recipes fail the qualified program's atom, "
            "hydrogen and charge inventory check. The issue packet preserves the source IDs and "
            "quantities for original-generator reconciliation. No replacement quantity is inferred "
            "from the target, and no record is promoted into training.",
            "training_admitted": False,
            "training_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
