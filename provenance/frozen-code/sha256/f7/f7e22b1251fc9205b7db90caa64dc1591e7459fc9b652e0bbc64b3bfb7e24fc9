"""Build a complete, reproducible preparation partition from both frozen rulesets."""

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_full_partition import POLICY, RESULT_SCHEMA, merge_partition
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    start = time.monotonic()
    output = HERE / "audit"
    if output.exists():
        raise ComposeLipidError("Frozen full partition output already exists")
    receipt_paths = {
        "features_receipt": ROOT
        / "results/phase1/compose_lipid_full_partition_features_v1/audit/result.json",
        "corrected_receipt": ROOT
        / "results/phase1/compose_lipid_full_split_projection_v1/audit/result.json",
        "historical_component_receipt": ROOT
        / "results/phase1/compose_lipid_split_source_recovery_v2/historical-global-component-exclusions.json",
        "component_policy": ROOT
        / "results/phase1/compose_lipid_supplement_intake_v1/component_exclusion_policy.json",
    }
    receipts = {name: json.loads(path.read_text()) for name, path in receipt_paths.items()}
    features, corrected = receipts["features_receipt"], receipts["corrected_receipt"]
    historical, components = receipts["historical_component_receipt"], receipts["component_policy"]
    if features["identity_errors"] or features["historical_identity_errors"]:
        raise ComposeLipidError("Full identity audit has unresolved parsing failures")
    if historical["summary"]["ambiguous_complete_bindings"] or historical["training_admitted"]:
        raise ComposeLipidError("Historical component binding is unqualified")
    if any(features["inputs"][name] != corrected["inputs"][name] for name in ("corpus", "joins")):
        raise ComposeLipidError("Partition projections use different source corpora")
    if (
        features["inputs"]["joins"]
        != json.loads(
            resolve_pin(historical["inputs"]["intake"], ROOT, label="intake").read_text()
        )["artifacts"]["joins.sqlite"]
    ):
        raise ComposeLipidError("Held precursor bindings use a different source corpus")
    for receipt in receipts.values():
        implementation = receipt["implementation"]
        if "path" in implementation:
            implementation = {"implementation": implementation}
        for name, value in implementation.items():
            resolve_pin(value, ROOT, label=name)
    inputs = {
        **{name: pin(ROOT, path) for name, path in receipt_paths.items()},
        "features": features["artifact"],
        "corrected_ledger": corrected["artifact"],
        "corpus": features["inputs"]["corpus"],
        "joins": features["inputs"]["joins"],
        "catalogue": historical["inputs"]["catalogue"],
    }
    paths = {name: resolve_pin(value, ROOT, label=name) for name, value in inputs.items()}
    implementation = {
        name: pin(ROOT, ROOT / name)
        for name in (
            "forge/corpus/compose_lipid_full_partition.py",
            "forge/corpus/compose_lipid_source_view.py",
            "forge/corpus/compose_lipid_supplement.py",
            "forge/core/hashing.py",
            str(Path(__file__).resolve().relative_to(ROOT)),
        )
    }
    blocked = set(historical["global_protected_components"])
    blocked.update(item["component_id"] for item in components["blocked_components"])
    with tempfile.TemporaryDirectory(prefix=".partition-", dir=HERE) as temporary:
        stage = Path(temporary)
        artifact = stage / "partition.sqlite"
        db = sqlite3.connect(artifact.as_uri(), uri=True)
        try:
            db.execute("PRAGMA cache_size=-65536")
            db.execute("PRAGMA temp_store=FILE")
            db.execute("ATTACH DATABASE ? AS features", (paths["features"].as_uri() + "?mode=ro",))
            db.execute("ATTACH DATABASE ? AS source", (paths["joins"].as_uri() + "?mode=ro",))
            db.executescript(
                "CREATE TABLE corrected(target_id TEXT PRIMARY KEY,family TEXT NOT NULL,"
                "fold TEXT NOT NULL,prior_disposition TEXT NOT NULL,study_unresolved INTEGER NOT NULL);"
                "CREATE TABLE blocked_components(id TEXT PRIMARY KEY);"
            )
            db.executemany(
                "INSERT INTO blocked_components VALUES (?)", ((v,) for v in sorted(blocked))
            )
            batch, count = [], 0
            for row in rows(paths["corrected_ledger"]):
                if (
                    row["training_admitted"] is not False
                    or row["corrected_projection"]["training_admitted"] is not False
                ):
                    raise ComposeLipidError("Projection unexpectedly claims training admission")
                if type(row["source_study_identity_unresolved"]) is not bool:
                    raise ComposeLipidError("Projection study resolution flag is invalid")
                batch.append(
                    (
                        row["target_id"],
                        row["family"],
                        row["corrected_projection"]["split"],
                        row["disposition"],
                        int(row["source_study_identity_unresolved"]),
                    )
                )
                if len(batch) == 4096:
                    db.executemany("INSERT INTO corrected VALUES (?,?,?,?,?)", batch)
                    count += len(batch)
                    batch.clear()
                if count and count % 102400 == 0 and not batch:
                    db.commit()
                    print(
                        json.dumps({"loaded": count, "seconds": time.monotonic() - start}),
                        flush=True,
                    )
            db.executemany("INSERT INTO corrected VALUES (?,?,?,?,?)", batch)
            db.commit()
            print("Merging frozen protections and closing product identity overlaps", flush=True)
            summary = merge_partition(db)
            db.commit()
            if {f: v["rows"] for f, v in summary["by_family"].items()} != features["families"]:
                raise ComposeLipidError("Full partition changed family populations")
            if summary["maximum_heavy_atoms"] != features["maximum_heavy_atoms"]:
                raise ComposeLipidError("Full partition changed molecular support")
            if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ComposeLipidError("Partition database integrity check failed")
        finally:
            db.close()
        for name, value in {**inputs, **implementation}.items():
            resolve_pin(value, ROOT, label=name)
        artifact_pin = pin(ROOT, artifact)
        artifact_pin["path"] = str((output / artifact.name).relative_to(ROOT))
        dump(
            stage / "result.json",
            {
                "schema_version": RESULT_SCHEMA,
                "seed": 0,
                "implementation": implementation,
                "inputs": inputs,
                "artifact": artifact_pin,
                "policy": POLICY,
                "summary": summary,
                "eligible_view_qualified": True,
                "all_source_rows_dispositioned": True,
                "all_nonprotected_rows_resolved": summary["totals"].get(
                    "unresolved_partition_or_study", 0
                )
                == 0,
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        os.rename(stage, output)
        print(json.dumps(summary["totals"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
