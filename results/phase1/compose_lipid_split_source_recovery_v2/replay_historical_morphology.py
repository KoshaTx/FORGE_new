"""Compare historical morphology keys with every frozen original assignment."""

import json
import sqlite3
import time
from collections import Counter, defaultdict
from contextlib import closing
from itertools import zip_longest
from pathlib import Path

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_historical_morphology import HistoricalMorphology
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    start = time.monotonic()
    replay_path = HERE / "frozen-group-replay.json"
    replay = json.loads(replay_path.read_text())
    if not replay["pass"] or replay["failures"]:
        raise ComposeLipidError("Corrected source descriptor replay must pass first")
    for k, value in replay["implementation"].items():
        resolve_pin(value, ROOT, label=k)
    paths = {k: resolve_pin(v, ROOT, label=k) for k, v in replay["inputs"].items()}
    policy = ROOT / "data/vendor/compose_lipid_historical_morphology_policy_v1.json"
    historical = HistoricalMorphology.from_registry(
        ROOT, policy, expected_sha256=sha256_file(policy)
    )
    implementation = {
        name: pin(ROOT, ROOT / name)
        for name in (
            "forge/corpus/compose_lipid_historical_morphology.py",
            "forge/corpus/compose_lipid_partition_signatures.py",
            str(Path(__file__).resolve().relative_to(ROOT)),
        )
    }
    totals, failures, groups, families = (
        Counter(),
        Counter(),
        defaultdict(Counter),
        defaultdict(Counter),
    )
    examples, test_groups = [], set()
    with closing(sqlite3.connect(paths["joins"].as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("ATTACH DATABASE ? AS original", (paths["corpus"].as_uri() + "?mode=ro",))
        query = db.execute(
            "SELECT t.target_id,t.family,t.payload,m.instances,a.payload FROM original.assignments a "
            "JOIN original.targets t USING(target_id) JOIN components m USING(target_id) ORDER BY t.target_id"
        )
        for supplied, values in zip_longest(rows(paths["groups"]), query):
            if supplied is None or values is None:
                raise ComposeLipidError("Historical morphology populations differ")
            target, family, raw, instances, assignment = values
            if target != supplied["target_id"] or family != supplied["family"]:
                raise ComposeLipidError("Historical morphology source-group join differs")
            source, expected = json.loads(raw), json.loads(assignment)
            actual = historical.describe(
                family=family,
                metadata=source["primary_metadata"],
                source_anchor=source["source_anchor"],
                instances=json.loads(instances),
                corrected_description=supplied,
            )
            bad = [key for key, value in actual.items() if value != expected[key]]
            totals["rows"] += 1
            totals["all_keys_exact"] += not bad
            totals["source_anchors"] += source["source_anchor"]
            families[family]["rows"] += 1
            families[family]["all_keys_exact"] += not bad
            failures.update(bad)
            group = (family, expected["structural_group_signature"])
            groups[group][expected["split"]] += 1
            if "unseen_regional_topology" in expected["test_panels"]:
                test_groups.add(group)
            if bad and len(examples) < 20:
                examples.append(
                    {
                        "target_id": target,
                        "family": family,
                        "source_anchor": source["source_anchor"],
                        "failed_keys": bad,
                        "actual": actual,
                        "expected": {k: expected[k] for k in actual},
                    }
                )
    if totals["rows"] != replay["counts"]["all_checks_pass"]:
        raise ComposeLipidError("Historical morphology lost frozen selected rows")
    for group in test_groups:
        if set(groups[group]) != {"test"}:
            raise ComposeLipidError("Original test topology group crosses old splits")
    for name, value in implementation.items():
        resolve_pin(value, ROOT, label=name)
    dump(
        HERE / "historical-morphology-replay.json",
        {
            "schema_version": "forge.compose_lipid_historical_morphology_replay.v1",
            "seed": 0,
            "implementation": implementation,
            "inputs": {
                **replay["inputs"],
                "corrected_replay": pin(ROOT, replay_path),
                "historical_policy": pin(ROOT, policy),
            },
            "counts": dict(totals),
            "by_family": dict(families),
            "failures": dict(failures),
            "examples": examples,
            "selected_test_structural_groups": [list(g) for g in sorted(test_groups)],
            "wholly_calibration_structural_groups": [
                list(g) for g, folds in sorted(groups.items()) if set(folds) == {"calibration"}
            ],
            "calibration_group_interpretation": "Observed wholly-calibration groups; original selected-calibration structural versus combination labels are not identified by this audit.",
            "pass": not failures,
            "elapsed_seconds": time.monotonic() - start,
            "source_assignments_changed": False,
            "component_inventories_changed": False,
            "training_admitted": False,
            "full_universe_partition_qualified": False,
        },
    )
    if failures:
        raise ComposeLipidError(
            "Historical morphology keys do not reproduce original source assignments"
        )


if __name__ == "__main__":
    main()
