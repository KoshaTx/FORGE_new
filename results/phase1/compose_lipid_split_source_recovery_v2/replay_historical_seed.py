"""Check an explicit Python hash seed against all original frozen split assignments."""

import json
import math
import os
import runpy
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_partition_signatures import SourceSplitSignatures
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    start = time.monotonic()
    seed = int(sys.argv[1])
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise ValueError("Python process hash seed differs from the requested diagnostic")
    output = HERE / f"historical-seed-{seed}.json"
    if output.exists():
        raise ValueError("Historical hash-seed diagnostic already exists")
    cache_path = HERE / "historical-selected-groups-v2/result.json"
    cache = json.loads(cache_path.read_text())
    if cache["precursor_combination_failures"] or cache["failed_fields"] != {"split": 26}:
        raise ValueError("Unexpected parent discrepancy")
    records_path = resolve_pin(
        cache["artifacts"]["records.jsonl.gz"], ROOT, label="authenticated original input records"
    )
    helper_path = resolve_pin(cache["implementation"], ROOT, label="source function loader")
    helper = runpy.run_path(str(helper_path))["functions"]
    config_path = resolve_pin(
        cache["inputs"]["original_config"], ROOT, label="original split config"
    )
    policy_path = resolve_pin(cache["inputs"]["historical_policy"], ROOT, label="historical policy")
    policy = json.loads(policy_path.read_text())
    module = resolve_pin(
        policy["source_assets"]["src/compose_lipid/data/generator_splits.py"]["file"],
        ROOT,
        label="source split algorithm",
    )
    registry = resolve_pin(cache["inputs"]["registry"], ROOT, label="digest registry")
    signatures = SourceSplitSignatures.from_registry(
        ROOT, registry, expected_sha256=sha256_file(registry)
    )
    corpus = resolve_pin(cache["inputs"]["corpus"], ROOT, label="original assignments")
    joins = resolve_pin(cache["inputs"]["joins"], ROOT, label="source-study identities")
    namespace = {
        "digest": signatures.digest,
        "math": math,
        "Counter": Counter,
        "defaultdict": defaultdict,
    }
    import ast

    names = {n.name for n in ast.parse(module.read_text()).body if isinstance(n, ast.FunctionDef)}
    helper(module, names, namespace)
    records = list(rows(records_path))
    with closing(sqlite3.connect(joins.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("ATTACH DATABASE ? AS original", (corpus.as_uri() + "?mode=ro",))
        studies = {
            target: sorted({"pmid:" + str(p).removeprefix("pmid:") for p in json.loads(pmids)})
            for target, pmids in db.execute(
                "SELECT t.target_id,m.pmids FROM original.targets t JOIN components m USING(target_id) WHERE t.source_anchor=1"
            )
        }
        expected = {
            target: json.loads(payload)
            for target, payload in db.execute("SELECT target_id,payload FROM original.assignments")
        }
    config = json.loads(config_path.read_text())
    result = namespace["make_generator_splits"](
        records,
        studies,
        **{
            k: config[k]
            for k in ("seed", "test_fraction", "calibration_fraction", "minimum_training_class")
        },
    )
    assignments = result.pop("assignments")
    failures, changes = [], Counter()
    for row in assignments:
        frozen = expected[row["target_id"]]
        different = [k for k in row if row[k] != frozen[k]]
        changes.update(different)
        if different:
            failures.append(row["target_id"])
    if len(assignments) != len(expected) or {r["target_id"] for r in assignments} != set(expected):
        raise ValueError("Original selected target population differs")
    saved_groups = None
    if not failures:
        group_path = HERE / f"historical-selected-groups-seed-{seed}.json"
        dump(group_path, result)
        saved_groups = pin(ROOT, group_path)
    dump(
        output,
        {
            "schema_version": "forge.compose_lipid_historical_hash_seed_replay.v1",
            "seed": config["seed"],
            "python_hash_seed": seed,
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {
                "parent": pin(ROOT, cache_path),
                "original_records": pin(ROOT, records_path),
                "loader": pin(ROOT, helper_path),
                "source_algorithm": pin(ROOT, module),
                "config": pin(ROOT, config_path),
                "registry": pin(ROOT, registry),
                "corpus": pin(ROOT, corpus),
                "joins": pin(ROOT, joins),
            },
            "rows": len(assignments),
            "assignment_failures": len(failures),
            "failed_fields": dict(changes),
            "pass": not failures,
            "selected_groups": saved_groups,
            "upstream_pure_functions_executed": True,
            "upstream_source_modified": False,
            "source_assignments_changed": False,
            "full_universe_group_selection_performed": False,
            "training_admitted": False,
            "elapsed_seconds": time.monotonic() - start,
        },
    )


if __name__ == "__main__":
    main()
