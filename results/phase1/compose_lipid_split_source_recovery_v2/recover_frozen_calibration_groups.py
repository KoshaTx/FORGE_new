"""Recover calibration groups from frozen assignments and original candidate groups.

Original calibration iterates a Python set. Candidate groups are deterministic;
the accepted subset is recovered from the saved assignments instead of replacing
them with the outcome of a new hash seed. A fully calibration candidate cannot
have failed the original quota or required-training-component checks: both would
contradict that final membership. Remaining whole calibration combinations are
then the original second calibration stage.
"""

import json
import math
import runpy
import sqlite3
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
    cache_path = HERE / "historical-selected-groups-v2/result.json"
    cache = json.loads(cache_path.read_text())
    if cache["precursor_combination_failures"] or cache["failed_fields"] != {"split": 26}:
        raise ValueError("The historical split discrepancy is not confined to calibration")
    changes_path = resolve_pin(
        cache["artifacts"]["assignment-differences.json"], ROOT, label="original replay differences"
    )
    differences = json.loads(changes_path.read_text())
    if any(
        r["changed_fields"] != ["split"]
        or {r["expected"]["split"], r["replayed"]["split"]} != {"train", "calibration"}
        for r in differences
    ):
        raise ValueError("Historical test-prefix replay differs")
    records_path = resolve_pin(
        cache["artifacts"]["records.jsonl.gz"], ROOT, label="original split records"
    )
    original_groups_path = resolve_pin(
        cache["artifacts"]["selected_groups.json"], ROOT, label="replayed test groups"
    )
    config_path = resolve_pin(
        cache["inputs"]["original_config"], ROOT, label="original split config"
    )
    registry = resolve_pin(cache["inputs"]["registry"], ROOT, label="split digest policy")
    corpus = resolve_pin(cache["inputs"]["corpus"], ROOT, label="immutable source assignments")
    joins = resolve_pin(cache["inputs"]["joins"], ROOT, label="immutable source studies")
    policy_path = resolve_pin(cache["inputs"]["historical_policy"], ROOT, label="historical policy")
    policy = json.loads(policy_path.read_text())
    module = resolve_pin(
        policy["source_assets"]["src/compose_lipid/data/generator_splits.py"]["file"],
        ROOT,
        label="original split functions",
    )
    loader = resolve_pin(cache["implementation"], ROOT, label="reviewed source function loader")
    signatures = SourceSplitSignatures.from_registry(
        ROOT, registry, expected_sha256=sha256_file(registry)
    )
    namespace = {
        "digest": signatures.digest,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "math": math,
    }
    runpy.run_path(str(loader))["functions"](
        module,
        {
            "stable_order",
            "_bounded_group_selection",
            "combination_signature",
            "structural_group_signature",
        },
        namespace,
    )
    records = list(rows(records_path))
    with closing(sqlite3.connect(joins.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("ATTACH DATABASE ? AS original", (corpus.as_uri() + "?mode=ro",))
        expected = {
            target: json.loads(payload)
            for target, payload in db.execute("SELECT target_id,payload FROM original.assignments")
        }
        studies = {
            target: {"pmid:" + str(p).removeprefix("pmid:") for p in json.loads(pmids)}
            for target, pmids in db.execute(
                "SELECT t.target_id,m.pmids FROM original.targets t JOIN components m USING(target_id) WHERE t.source_anchor=1"
            )
        }
    config = json.loads(config_path.read_text())
    groups = json.loads(original_groups_path.read_text())
    formal = set(groups["formal_evaluation_families"])
    family_targets, structures, combinations = defaultdict(set), defaultdict(set), defaultdict(set)
    by_id = {}
    for row in records:
        target, family = row["target_id"], row["family"]
        family_targets[family].add(target)
        structures[(family, namespace["structural_group_signature"](row))].add(target)
        combinations[(family, namespace["combination_signature"](row))].add(target)
        by_id[target] = row
    if set(by_id) != set(expected) or len(records) != config["expected_targets"]:
        raise ValueError("Frozen original population differs")
    test = {target for target, row in expected.items() if row["split"] == "test"}
    calibration = {target for target, row in expected.items() if row["split"] == "calibration"}
    selected_structures, candidate_count = set(), 0
    for family in sorted(formal):
        quota = max(1, round(len(family_targets[family]) * config["calibration_fraction"]))
        candidates = {
            signature: targets
            for (f, signature), targets in structures.items()
            if f == family and not targets & test
        }
        chosen, _ = namespace["_bounded_group_selection"](
            candidates,
            quota,
            max(1, math.floor(quota * 0.5)),
            config["seed"] + ":calibration_structure",
        )
        candidate_count += len(chosen)
        selected_structures.update(
            (family, signature) for signature in chosen if candidates[signature] <= calibration
        )
    structural_calibration = set().union(*(structures[key] for key in selected_structures))
    remaining = calibration - structural_calibration
    selected_combinations = {
        key for key, targets in combinations.items() if targets and targets <= remaining
    }
    combination_calibration = set().union(*(combinations[key] for key in selected_combinations))
    if structural_calibration | combination_calibration != calibration:
        raise ValueError("Frozen calibration does not close as original whole candidate groups")
    # Original test-combination members must retain every precursor in final TRAIN.
    train = {target for target, row in expected.items() if row["split"] == "train"}
    train_precursors = {p for target in train for p in by_id[target]["precursor_ids"]}
    for key in groups["selected_combination_groups"]:
        if any(
            set(by_id[target]["precursor_ids"]) - train_precursors
            for target in combinations[tuple(key)]
        ):
            raise ValueError("Frozen calibration violates required training precursor support")
    groups["selected_calibration_structural_groups"] = [
        list(k) for k in sorted(selected_structures)
    ]
    groups["selected_calibration_combination_groups"] = [
        list(k) for k in sorted(selected_combinations)
    ]
    old_components = set(groups["selected_precursors"])
    old_studies = {tuple(k) for k in groups["selected_source_studies"]}
    old_structures = {tuple(k) for k in groups["selected_structural_groups"]}
    old_combinations = {tuple(k) for k in groups["selected_combination_groups"]}
    for row in records:
        target, family = row["target_id"], row["family"]
        morphology = namespace["structural_group_signature"](row)
        combination = namespace["combination_signature"](row)
        tests = {
            "unseen_precursor_identity": bool(set(row["precursor_ids"]) & old_components),
            "source_study_transfer": any(
                (family, p) in old_studies for p in studies.get(target, set())
            ),
            "unseen_regional_topology": (family, morphology) in old_structures,
            "unseen_exact_combination": (family, combination) in old_combinations,
        }
        panels = sorted(k for k, value in tests.items() if value)
        projected = (
            "reference"
            if family not in formal
            else (
                "test"
                if panels
                else (
                    "calibration"
                    if (family, morphology) in selected_structures
                    or (family, combination) in selected_combinations
                    else "train"
                )
            )
        )
        expected_row = expected[target]
        if (
            projected != expected_row["split"]
            or (panels if projected == "test" else []) != expected_row["test_panels"]
        ):
            raise ValueError("Recovered historical groups changed an original assignment or panel")
    group_path = HERE / "historical-frozen-selected-groups.json"
    dump(group_path, groups)
    dump(
        HERE / "historical-frozen-group-recovery.json",
        {
            "schema_version": "forge.compose_lipid_frozen_historical_group_recovery.v1",
            "seed": config["seed"],
            "implementation": pin(ROOT, Path(__file__).resolve()),
            "inputs": {
                "original_replay": pin(ROOT, cache_path),
                "replay_differences": pin(ROOT, changes_path),
                "records": pin(ROOT, records_path),
                "original_test_groups": pin(ROOT, original_groups_path),
                "config": pin(ROOT, config_path),
                "corpus": pin(ROOT, corpus),
                "joins": pin(ROOT, joins),
                "digest_registry": pin(ROOT, registry),
                "historical_policy": pin(ROOT, policy_path),
                "source_algorithm": pin(ROOT, module),
                "loader": pin(ROOT, loader),
            },
            "artifact": pin(ROOT, group_path),
            "pass": True,
            "source_rows_reproduced": len(records),
            "assignment_failures": 0,
            "panel_failures": 0,
            "calibration_candidate_structures": candidate_count,
            "recovered_calibration_structures": len(selected_structures),
            "recovered_calibration_combinations": len(selected_combinations),
            "recovery_basis": "Original deterministic candidate groups and complete saved calibration membership; no substitution of hash-seed-dependent reassignment.",
            "original_python_hash_seed_known": False,
            "source_assignments_changed": False,
            "full_universe_group_selection_performed": False,
            "training_admitted": False,
            "elapsed_seconds": time.monotonic() - start,
        },
    )


if __name__ == "__main__":
    main()
