"""Independently audit the structure-resolved v8.1 generator split."""

from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from compose_lipid.data.assets import sha256_file
from compose_lipid.data.source_inventory import publish_json


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = (
    ROOT / "artifacts/corpus_build_v2/post_instruction_generator_splits_v8_1"
)
OLD_SPLIT_DIR = ROOT / "artifacts/corpus_build_v2/post_instruction_generator_splits_v8"
COMPONENT_DIR = (
    ROOT / "artifacts/corpus_build_v2/post_instruction_component_manifest_v8_1"
)
OUTPUT = ROOT / "audits/post_instruction_generator_splits_v8_1_independent.json"


def rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        yield from map(json.loads, handle)


def _check_receipt(directory: Path) -> dict:
    receipt = json.loads((directory / "receipt.json").read_text())
    if receipt.get("complete") is not True:
        raise ValueError("split receipt is incomplete")
    for name, expected in receipt["inputs"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError("changed bound input: " + name)
    for name, expected in receipt["producers"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError("changed producer: " + name)
    for name, expected in receipt["outputs"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError("changed split output: " + name)
    return receipt


def build() -> dict:
    receipt = _check_receipt(SPLIT_DIR)
    assignments = {
        row["target_id"]: row for row in rows(SPLIT_DIR / "assignments.jsonl.gz")
    }
    groups = {
        row["target_id"]: row for row in rows(SPLIT_DIR / "split_groups.jsonl.gz")
    }
    selected = json.loads((SPLIT_DIR / "selected_groups.json").read_text())
    summary = json.loads((SPLIT_DIR / "summary.json").read_text())
    if len(assignments) != 200_000 or set(assignments) != set(groups):
        raise ValueError("split membership is incomplete or inconsistent")

    split_counts = Counter(row["split"] for row in assignments.values())
    panel_counts = Counter(
        panel for row in assignments.values() for panel in row["test_panels"]
    )
    if dict(sorted(split_counts.items())) != summary["split_counts"]:
        raise ValueError("summary split counts differ from assignments")
    if dict(sorted(panel_counts.items())) != summary["panel_counts"]:
        raise ValueError("summary panel counts differ from assignments")
    if any(
        row["split"] == "test" and not row["test_panels"]
        for row in assignments.values()
    ):
        raise ValueError("formal test target lacks a named panel")
    if any(
        row["split"] != "test" and row["test_panels"]
        for row in assignments.values()
    ):
        raise ValueError("non-test target has a test panel")

    train_ids = {target_id for target_id, row in assignments.items() if row["split"] == "train"}
    test_ids = {target_id for target_id, row in assignments.items() if row["split"] == "test"}
    calibration_ids = {
        target_id
        for target_id, row in assignments.items()
        if row["split"] == "calibration"
    }
    reference_ids = {
        target_id
        for target_id, row in assignments.items()
        if row["split"] == "reference"
    }
    if any(
        left & right
        for index, left in enumerate(
            (train_ids, test_ids, calibration_ids, reference_ids)
        )
        for right in (train_ids, test_ids, calibration_ids, reference_ids)[index + 1 :]
    ):
        raise ValueError("split partitions overlap")

    train_components = set()
    train_family_components = set()
    train_combinations = set()
    train_studies = set()
    train_morphologies = set()
    for target_id in train_ids:
        group = groups[target_id]
        family = group["family"]
        train_components.update(group["component_ids"])
        train_family_components.update(
            (family, component_id) for component_id in group["component_ids"]
        )
        train_combinations.add((family, assignments[target_id]["combination_signature"]))
        train_studies.update(
            "pmid:" + str(value).removeprefix("pmid:")
            for value in group.get("source_pmids", [])
        )
        train_morphologies.add(
            (family, assignments[target_id]["morphology_group_signature"])
        )

    selected_components = set(selected["selected_components"])
    if selected_components & train_components:
        raise ValueError("selected unseen component constitution leaks into train")
    selected_studies = set(selected["selected_source_studies"])
    if selected_studies & train_studies:
        raise ValueError("globally held source study leaks into train")

    panel_failures = Counter()
    overlapping_panels = Counter()
    for target_id in test_ids:
        assignment = assignments[target_id]
        group = groups[target_id]
        family = group["family"]
        panels = set(assignment["test_panels"])
        if len(panels) > 1:
            overlapping_panels["multi_panel_targets"] += 1
        if "unseen_component_structure" in panels and not (
            set(group["component_ids"]) - train_components
        ):
            panel_failures["component_panel_without_unseen_structure"] += 1
        if "unseen_exact_component_combination" in panels:
            combination = (family, assignment["combination_signature"])
            if combination in train_combinations:
                panel_failures["combination_seen_in_train"] += 1
            if {
                (family, component_id) for component_id in group["component_ids"]
            } - train_family_components:
                panel_failures["combination_component_unseen_in_family_train"] += 1
        if "unseen_regional_morphology" in panels and (
            family,
            assignment["morphology_group_signature"],
        ) in train_morphologies:
            panel_failures["morphology_seen_in_train"] += 1
        if "source_study_transfer_global" in panels:
            normalized = {
                "pmid:" + str(value).removeprefix("pmid:")
                for value in group.get("source_pmids", [])
            }
            if not (normalized & selected_studies):
                panel_failures["study_panel_without_selected_study"] += 1
    if panel_failures:
        raise ValueError("panel semantics failed: " + repr(dict(panel_failures)))

    selected_morphologies = {
        tuple(value) for value in selected["selected_morphology_groups"]
    }
    selected_combinations = {
        tuple(value) for value in selected["selected_combination_groups"]
    }
    for target_id, assignment in assignments.items():
        family = assignment["family"]
        if (
            family,
            assignment["morphology_group_signature"],
        ) in selected_morphologies and target_id not in test_ids:
            raise ValueError("selected morphology group extends outside test")
        if (
            family,
            assignment["combination_signature"],
        ) in selected_combinations and target_id not in test_ids:
            raise ValueError("selected combination group extends outside test")

    reference_families = {assignments[target_id]["family"] for target_id in reference_ids}
    if reference_families != set(summary["reference_only_families"]):
        raise ValueError("reference-only family assignment changed")
    if any(
        assignments[target_id]["family"] in reference_families
        for target_id in set(assignments) - reference_ids
    ):
        raise ValueError("reference family leaks into formal partitions")

    # Quantify why v8 is superseded: its ID-level precursor panel contains rows
    # whose complete canonical component structures are all present in training.
    old_assignments = {
        row["target_id"]: row
        for row in rows(OLD_SPLIT_DIR / "assignments.jsonl.gz")
    }
    component_manifest = {
        row["target_id"]: row
        for row in rows(COMPONENT_DIR / "components.jsonl.gz")
    }
    if set(old_assignments) != set(component_manifest):
        raise ValueError("old split/component audit membership differs")
    old_train_components = {
        component_id
        for target_id, assignment in old_assignments.items()
        if assignment["split"] == "train"
        for component_id in component_manifest[target_id]["component_ids"]
    }
    old_panel_ids = {
        target_id
        for target_id, assignment in old_assignments.items()
        if "unseen_precursor_identity" in assignment["test_panels"]
    }
    old_false_unseen = sum(
        not (set(component_manifest[target_id]["component_ids"]) - old_train_components)
        for target_id in old_panel_ids
    )

    audit = {
        "schema": "post_instruction_generator_splits_v8_1_independent_audit",
        "audited_receipt_sha256": sha256_file(SPLIT_DIR / "receipt.json"),
        "audited_assignment_digest": receipt["assignment_digest"],
        "targets": len(assignments),
        "split_counts": dict(sorted(split_counts.items())),
        "panel_counts": dict(sorted(panel_counts.items())),
        "formal_families": len(summary["formal_evaluation_families"]),
        "reference_families": sorted(reference_families),
        "selected_component_structures": len(selected_components),
        "selected_source_studies": sorted(selected_studies),
        "selected_morphology_groups": len(selected_morphologies),
        "selected_combination_groups": len(selected_combinations),
        "multi_panel_test_targets": overlapping_panels["multi_panel_targets"],
        "component_structure_leaks": len(selected_components & train_components),
        "global_source_study_leaks": len(selected_studies & train_studies),
        "panel_semantic_failures": dict(panel_failures),
        "test_targets_without_named_panel": 0,
        "morphology_label_scope": "coarse regional morphology; not exact graph topology",
        "old_v8_unseen_precursor_panel_targets": len(old_panel_ids),
        "old_v8_panel_targets_without_structurally_unseen_component": old_false_unseen,
        "old_v8_superseded_for_model_evaluation": True,
        "biology_used": False,
        "beae_outcomes_used": False,
        "passed": True,
    }
    publish_json(OUTPUT, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


if __name__ == "__main__":
    build()
