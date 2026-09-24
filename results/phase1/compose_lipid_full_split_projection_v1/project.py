"""Audit every supplied construction against the frozen corrected holdout groups.

Run from the repository root with .venv/bin/python <this file>. This is a
non-learning leakage audit; it does not extract components or admit training.
"""

import gzip
import json
import os
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import closing
from itertools import zip_longest
from pathlib import Path

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_partition_projection import project_preparation_record
from forge.corpus.compose_lipid_partition_signatures import (
    FrozenGroupProjection,
    SourceSplitSignatures,
)
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import compact, rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    start = time.monotonic()
    output = HERE / "audit"
    if output.exists():
        raise ComposeLipidError("Frozen projection output already exists")
    replay_path = (
        ROOT / "results/phase1/compose_lipid_split_source_recovery_v2/frozen-group-replay.json"
    )
    replay = json.loads(replay_path.read_text())
    if replay["pass"] is not True or replay["failures"] or replay["training_admitted"]:
        raise ComposeLipidError("Frozen selected-population replay did not pass")
    for name, value in replay["implementation"].items():
        resolve_pin(value, ROOT, label=name)
    paths = {k: resolve_pin(v, ROOT, label=k) for k, v in replay["inputs"].items()}
    intake = json.loads(paths["intake"].read_text())
    signatures = SourceSplitSignatures.from_registry(
        ROOT, paths["registry"], expected_sha256=replay["inputs"]["registry"]["sha256"]
    )
    projection = FrozenGroupProjection.from_selected_groups(
        json.loads(paths["selected_groups"].read_text())
    )
    preparation_path = ROOT / "results/phase1/compose_lipid_full_preparation_v1/result.json"
    preparation = json.loads(preparation_path.read_text())
    for name, value in preparation["implementation"].items():
        resolve_pin(value, ROOT, label=name)
    ledger = resolve_pin(preparation["artifact"], ROOT, label="full preparation ledger")
    implementation = {
        name: pin(ROOT, ROOT / name)
        for name in (
            "forge/corpus/compose_lipid_partition_signatures.py",
            "forge/corpus/compose_lipid_partition_projection.py",
            "forge/corpus/compose_lipid_supplement.py",
            "forge/corpus/compose_lipid_source_view.py",
            "forge/core/hashing.py",
            str(Path(__file__).resolve().relative_to(ROOT)),
        )
    }
    inputs = {
        **replay["inputs"],
        "selected_population_replay": pin(ROOT, replay_path),
        "full_preparation": pin(ROOT, preparation_path),
        "preparation_ledger": preparation["artifact"],
    }
    totals, families, panels, crossings = Counter(), defaultdict(Counter), Counter(), Counter()
    maximum = 0
    with tempfile.TemporaryDirectory(prefix=".projection-", dir=HERE) as temporary:
        stage = Path(temporary)
        artifact = stage / "projection.jsonl.gz"
        with (
            closing(sqlite3.connect(paths["joins"].as_uri() + "?mode=ro", uri=True)) as db,
            artifact.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, filename="", mode="wb", compresslevel=1, mtime=0) as stream,
        ):
            db.execute("PRAGMA cache_size=-32768")
            db.execute("PRAGMA temp_store=FILE")
            db.execute("ATTACH DATABASE ? AS original", (paths["corpus"].as_uri() + "?mode=ro",))
            query = db.execute(
                "SELECT t.target_id,t.family,t.payload,c.instances,c.family,m.pmids,"
                "a.split,a.panels,a.combination,a.morphology FROM original.targets t "
                "LEFT JOIN constructions c USING(target_id) "
                "LEFT JOIN components m USING(target_id) LEFT JOIN assignments a USING(target_id) "
                "ORDER BY t.family,t.target_id"
            )
            for previous, values in zip_longest(rows(ledger), query):
                if previous is None or values is None:
                    raise ComposeLipidError("Full projection input populations differ")
                (
                    target,
                    family,
                    payload,
                    instances,
                    c_family,
                    pmids,
                    split,
                    expected_panels,
                    combination,
                    morphology,
                ) = values
                if (
                    target != previous["target_id"]
                    or family != previous["family"]
                    or c_family != family
                    or json.loads(instances) != previous["component_instances"]
                ):
                    raise ComposeLipidError("Full projection target/family/component join differs")
                source = json.loads(payload)
                actual = project_preparation_record(
                    source,
                    previous,
                    signatures=signatures,
                    projection=projection,
                    source_pmids=json.loads(pmids) if pmids is not None else [],
                )
                if split is not None:
                    if (
                        actual["corrected_projection"]["split"] != split
                        or actual["corrected_projection"]["test_panels"]
                        != json.loads(expected_panels)
                        or actual["combination_signature"] != combination
                        or actual["morphology_group_signature"] != morphology
                    ):
                        raise ComposeLipidError(
                            "Full projection changed a frozen selected assignment"
                        )
                    totals["selected_population_reproduced"] += 1
                projected = actual["corrected_projection"]
                totals.update(
                    {
                        "rows": 1,
                        "newly_protected": actual["newly_protected"],
                        "source_declared_above_96_atoms": source["heavy_atoms"] > 96,
                        "source_study_identity_unresolved": actual[
                            "source_study_identity_unresolved"
                        ],
                    }
                )
                maximum = max(maximum, source["heavy_atoms"])
                families[family]["rows"] += 1
                families[family][actual["disposition"]] += 1
                families[family]["newly_protected"] += actual["newly_protected"]
                panels.update(projected["test_panels"])
                crossings[(previous["disposition"], projected["split"], actual["disposition"])] += 1
                stream.write((compact(actual) + "\n").encode())
                if totals["rows"] % 100000 == 0:
                    print(
                        json.dumps(
                            {
                                "rows": totals["rows"],
                                "newly_protected": totals["newly_protected"],
                                "elapsed_seconds": time.monotonic() - start,
                            }
                        ),
                        flush=True,
                    )
        if {f: c["rows"] for f, c in families.items()} != intake["constructions"]["family_counts"]:
            raise ComposeLipidError("Full projection changed source family coverage")
        if totals["selected_population_reproduced"] != replay["counts"]["all_checks_pass"]:
            raise ComposeLipidError("Full projection lost original selected records")
        for name, value in {**implementation, **inputs}.items():
            resolve_pin(value, ROOT, label=name)
        artifact_pin = pin(ROOT, artifact)
        artifact_pin["path"] = str((output / artifact.name).relative_to(ROOT))
        dump(
            stage / "result.json",
            {
                "schema_version": "forge.compose_lipid_full_split_projection.v1",
                "seed": 0,
                "implementation": implementation,
                "inputs": inputs,
                "artifact": artifact_pin,
                "summary": {
                    "totals": dict(totals),
                    "by_family": dict(families),
                    "maximum_source_declared_atoms": maximum,
                    "test_panel_counts_overlap": dict(panels),
                    "disposition_crossings": [
                        {
                            "prior_disposition": old,
                            "corrected_projection": new,
                            "disposition": final,
                            "rows": count,
                        }
                        for (old, new, final), count in sorted(crossings.items())
                    ],
                },
                "elapsed_seconds": time.monotonic() - start,
                "policy": {
                    "purpose": "Non-learning full-universe corrected split protection audit",
                    "group_selection_repeated": False,
                    "source_assignments_changed": False,
                    "prior_protections_removed": False,
                    "new_rows_released_for_preparation": 0,
                    "reaction_decomposition_performed": False,
                    "size_filter": None,
                    "record_cap": None,
                    "training_calls": 0,
                    "training_admitted": False,
                    "full_universe_partition_qualified": False,
                    "remaining_partition_gates": [
                        "historical_group_extension",
                        "constitutional_overlap",
                        "complete_source_study_identity",
                    ],
                },
            },
        )
        os.rename(stage, output)


if __name__ == "__main__":
    main()
