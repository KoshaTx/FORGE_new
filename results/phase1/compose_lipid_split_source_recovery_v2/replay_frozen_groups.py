"""Reproduce every supplied split descriptor and assignment without selecting new groups."""

import json
import sqlite3
import time
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_partition_signatures import (
    FrozenGroupProjection,
    SourceSplitSignatures,
)
from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_supplement import rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    start = time.monotonic()
    intake_path = ROOT / "results/phase1/compose_lipid_supplement_intake_v1/result.json"
    intake = json.loads(intake_path.read_text())
    names = ("corpus", "groups", "assignments", "selected_groups", "components")
    paths = {k: resolve_pin(intake["inputs"][k], ROOT, label=k) for k in names}
    joins = resolve_pin(intake["artifacts"]["joins.sqlite"], ROOT, label="component joins")
    registry = ROOT / "data/vendor/compose_lipid_frozen_split_policy_v1.json"
    signatures = SourceSplitSignatures.from_registry(
        ROOT, registry, expected_sha256=sha256_file(registry)
    )
    projection = FrozenGroupProjection.from_selected_groups(
        json.loads(paths["selected_groups"].read_text())
    )
    implementation = {
        name: pin(ROOT, ROOT / name)
        for name in (
            "forge/corpus/compose_lipid_partition_signatures.py",
            str(Path(__file__).resolve().relative_to(ROOT)),
        )
    }
    counts, failures, examples = Counter(), Counter(), []
    with sqlite3.connect(joins.as_uri() + "?mode=ro", uri=True) as db:
        db.execute("ATTACH DATABASE ? AS original", (paths["corpus"].as_uri() + "?mode=ro",))
        query = db.execute(
            "SELECT a.target_id,a.family,a.split,a.panels,a.combination,a.morphology,"
            "c.instances,c.ids,c.pmids,t.constitution,t.source_anchor,t.payload "
            "FROM assignments a JOIN components c USING(target_id) JOIN original.targets t USING(target_id) "
            "ORDER BY a.target_id"
        )
        previous_id = None
        for expected, values in zip_longest(rows(paths["groups"]), query):
            if expected is None or values is None:
                raise ComposeLipidError("Split group and assignment populations differ")
            (
                target,
                family,
                split,
                panels,
                combination,
                morphology,
                raw_instances,
                raw_ids,
                raw_pmids,
                smiles,
                anchor,
                raw,
            ) = values
            if (
                target != expected["target_id"]
                or family != expected["family"]
                or (previous_id and target <= previous_id)
            ):
                raise ComposeLipidError("Split group target/family order differs")
            previous_id = target
            source = json.loads(raw)
            instances = json.loads(raw_instances)
            description = signatures.describe(
                family=family,
                smiles=smiles,
                metadata=source["primary_metadata"],
                source_anchor=bool(anchor),
                instances=instances,
            )
            expected_fields = {
                k: expected[k]
                for k in (
                    "core_scaffold_signature",
                    "regional_morphology_signature",
                    "regional_profile",
                    "morphology_context",
                )
            }
            expected_fields.update(
                combination_signature=combination, morphology_group_signature=morphology
            )
            checks = {k: description[k] == v for k, v in expected_fields.items()}
            projected = projection.project(
                family=family,
                component_ids=json.loads(raw_ids),
                source_pmids=json.loads(raw_pmids),
                morphology=description["morphology_group_signature"],
                combination=description["combination_signature"],
            )
            checks.update(
                frozen_split=projected["split"] == split,
                frozen_test_panels=projected["test_panels"] == json.loads(panels),
                no_training_admission=projected["training_admitted"] is False,
            )
            bad = [k for k, v in checks.items() if not v]
            failures.update(bad)
            counts["rows"] += 1
            counts["all_checks_pass"] += not bad
            counts["original_" + split] += 1
            if bad and len(examples) < 20:
                examples.append(
                    {
                        "target_id": target,
                        "failed_checks": bad,
                        "actual": description,
                        "expected": expected_fields,
                        "projection": projected,
                        "original_split": split,
                        "original_test_panels": json.loads(panels),
                    }
                )
            if counts["rows"] % 10000 == 0:
                print(json.dumps({"rows": counts["rows"], "failures": dict(failures)}), flush=True)
    expected_count = sum(intake["corrected_split"]["split_counts"].values())
    if counts["rows"] != expected_count:
        raise ComposeLipidError("Frozen split descriptor replay lost selected records")
    for k, value in implementation.items():
        resolve_pin(value, ROOT, label=k)
    dump(
        HERE / "frozen-group-replay.json",
        {
            "schema_version": "forge.compose_lipid_frozen_group_replay.v1",
            "seed": 0,
            "implementation": implementation,
            "inputs": {
                "intake": pin(ROOT, intake_path),
                "registry": pin(ROOT, registry),
                "joins": pin(ROOT, joins),
                **{k: intake["inputs"][k] for k in names},
            },
            "counts": dict(counts),
            "failures": dict(failures),
            "examples": examples,
            "pass": not failures,
            "elapsed_seconds": time.monotonic() - start,
            "purpose": "Non-learning partition and leakage audit only",
            "upstream_code_executed": False,
            "group_selection_repeated": False,
            "source_assignments_changed": False,
            "reaction_decomposition_performed": False,
            "full_universe_partition_qualified": False,
            "training_admitted": False,
            "training_calls": 0,
        },
    )
    if failures:
        raise ComposeLipidError(
            "Frozen split descriptor or assignment replay disagrees with source"
        )


if __name__ == "__main__":
    main()
