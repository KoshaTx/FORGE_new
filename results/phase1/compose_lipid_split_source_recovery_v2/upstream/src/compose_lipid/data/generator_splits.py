"""Deterministic multi-panel splits for a combinatorial reaction corpus."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from compose_lipid.data.training_corpus import digest


def stable_order(label, *parts):
    return digest([label, *parts])


def combination_signature(row):
    instances = sorted(
        (item["role"], item["precursor_id"]) for item in row["precursor_instances"]
    )
    return digest(["precursor_role_multiset_v1", row["family"], instances])


def structural_group_signature(row):
    return digest(
        [
            "reaction_core_regional_topology_v1",
            row["family"],
            row["core_scaffold_signature"],
            row["regional_topology_signature"],
        ]
    )


def _add_panel(targets, name, test, panels):
    for target_id in targets:
        test.add(target_id)
        panels[target_id].add(name)


def _bounded_group_selection(groups, target, maximum_group, label):
    selected = set()
    count = 0
    candidates = sorted(groups.items(), key=lambda item: stable_order(label, item[0]))
    for key, target_ids in candidates:
        target_ids = set(target_ids)
        if not target_ids or len(target_ids) > maximum_group:
            continue
        if count + len(target_ids) <= target:
            selected.add(key)
            count += len(target_ids)
    if not selected and target:
        fitting = [
            (key, set(values))
            for key, values in candidates
            if values and len(values) <= maximum_group
        ]
        if fitting:
            key, values = min(fitting, key=lambda item: (len(item[1]), item[0]))
            selected.add(key)
            count = len(values)
    return selected, count


def make_generator_splits(
    records,
    source_groups=None,
    *,
    seed="broad_lipid_generator_split_v1",
    test_fraction=0.10,
    calibration_fraction=0.10,
    minimum_training_class=32,
):
    records = list(records)
    by_id = {row["target_id"]: row for row in records}
    if len(by_id) != len(records) or not records:
        raise ValueError("unique nonempty split records required")
    if not 0 < test_fraction < 0.5 or not 0 < calibration_fraction < 0.5:
        raise ValueError("bounded positive split fractions required")
    source_groups = source_groups or {}
    families = defaultdict(set)
    precursor_targets = defaultdict(set)
    topology_targets = defaultdict(set)
    combination_targets = defaultdict(set)
    combinations = {}
    for row in records:
        target_id = row["target_id"]
        family = row["family"]
        families[family].add(target_id)
        for precursor_id in row["precursor_ids"]:
            precursor_targets[precursor_id].add(target_id)
        topology_targets[(family, structural_group_signature(row))].add(target_id)
        combination = combination_signature(row)
        combinations[target_id] = combination
        combination_targets[(family, combination)].add(target_id)

    formal = {
        family
        for family, target_ids in families.items()
        if len(target_ids) >= minimum_training_class
    }
    test_quotas = {
        family: max(1, round(len(families[family]) * test_fraction))
        for family in formal
    }
    calibration_quotas = {
        family: max(1, round(len(families[family]) * calibration_fraction))
        for family in formal
    }
    test = set()
    calibration = set()
    test_family_counts = Counter()
    calibration_family_counts = Counter()
    panels = defaultdict(set)
    selected_studies = set()
    selected_precursors = set()
    selected_structures = set()
    selected_combinations = set()

    def add_test(target_ids, panel):
        new = set(target_ids) - test
        test_family_counts.update(by_id[target_id]["family"] for target_id in new)
        _add_panel(target_ids, panel, test, panels)

    def add_calibration(target_ids):
        new = set(target_ids) - calibration
        calibration_family_counts.update(
            by_id[target_id]["family"] for target_id in new
        )
        calibration.update(new)

    studies_by_family = defaultdict(lambda: defaultdict(set))
    for target_id, studies in source_groups.items():
        if target_id not in by_id:
            raise ValueError("source study refers to an unknown target")
        for study in studies:
            studies_by_family[by_id[target_id]["family"]][study].add(target_id)
    for family in sorted(formal):
        groups = studies_by_family[family]
        if len(groups) < 2:
            continue
        target = max(1, math.floor(test_quotas[family] * 0.15))
        maximum = max(1, math.floor(test_quotas[family] * 0.5))
        chosen, _ = _bounded_group_selection(groups, target, maximum, seed + ":study")
        for study in chosen:
            selected_studies.add((family, study))
            add_test(groups[study], "source_study_transfer")

    family_precursor_degree = defaultdict(Counter)
    for family, target_ids in families.items():
        for target_id in target_ids:
            family_precursor_degree[family].update(by_id[target_id]["precursor_ids"])
    for family in sorted(formal):
        panel_target = max(1, math.floor(test_quotas[family] * 0.30))
        added = 0
        candidates = sorted(
            family_precursor_degree[family],
            key=lambda precursor_id: stable_order(
                seed + ":precursor", family, precursor_id
            ),
        )
        for precursor_id in candidates:
            if precursor_id in selected_precursors:
                continue
            affected = precursor_targets[precursor_id]
            family_degree = family_precursor_degree[family][precursor_id]
            if family_degree < 2 or len(affected) > max(2, panel_target):
                continue
            new = affected - test
            if not new:
                continue
            by_affected_family = Counter(
                by_id[target_id]["family"] for target_id in new
            )
            if any(
                other not in formal
                or test_family_counts[other] + count > test_quotas[other]
                for other, count in by_affected_family.items()
            ):
                continue
            if added + len(new & families[family]) > panel_target:
                continue
            selected_precursors.add(precursor_id)
            add_test(affected, "unseen_precursor_identity")
            added += len(new & families[family])
            if added >= panel_target:
                break
        if test_family_counts[family] > test_quotas[family]:
            raise ValueError("source panel already exceeds family test quota")

    for family in sorted(formal):
        panel_target = max(1, math.floor(test_quotas[family] * 0.30))
        current = test_family_counts[family]
        remaining = max(0, test_quotas[family] - current)
        groups = {
            signature: target_ids
            for (group_family, signature), target_ids in topology_targets.items()
            if group_family == family and not (target_ids & test)
        }
        maximum = max(1, math.floor(test_quotas[family] * 0.5))
        chosen, _ = _bounded_group_selection(
            groups, min(panel_target, remaining), maximum, seed + ":structure"
        )
        for signature in chosen:
            selected_structures.add((family, signature))
            add_test(groups[signature], "unseen_regional_topology")

    remaining_precursor_degree = Counter()
    for target_id, row in by_id.items():
        if target_id not in test:
            remaining_precursor_degree.update(row["precursor_ids"])
    for family in sorted(formal):
        quota = test_quotas[family]
        candidates = sorted(
            (
                (signature, target_ids)
                for (group_family, signature), target_ids in combination_targets.items()
                if group_family == family and not (target_ids & test)
            ),
            key=lambda item: stable_order(seed + ":combination", family, item[0]),
        )
        for signature, target_ids in candidates:
            current = test_family_counts[family]
            if current >= quota or current + len(target_ids) > quota:
                continue
            removal = Counter(
                precursor_id
                for target_id in target_ids
                for precursor_id in by_id[target_id]["precursor_ids"]
            )
            if any(
                remaining_precursor_degree[precursor_id] - count < 1
                for precursor_id, count in removal.items()
            ):
                continue
            selected_combinations.add((family, signature))
            add_test(target_ids, "unseen_exact_combination")
            remaining_precursor_degree.subtract(removal)

    selected_calibration_structures = set()
    selected_calibration_combinations = set()
    protected_train_precursors = {
        precursor_id
        for family, signature in selected_combinations
        for target_id in combination_targets[(family, signature)]
        for precursor_id in by_id[target_id]["precursor_ids"]
    }
    remaining_train_degree = Counter()
    for target_id, row in by_id.items():
        if target_id not in test and row["family"] in formal:
            remaining_train_degree.update(row["precursor_ids"])

    def preserves_required_training_precursors(target_ids):
        removal = Counter(
            precursor_id
            for target_id in target_ids
            for precursor_id in by_id[target_id]["precursor_ids"]
        )
        return removal, all(
            precursor_id not in protected_train_precursors
            or remaining_train_degree[precursor_id] - count >= 1
            for precursor_id, count in removal.items()
        )

    for family in sorted(formal):
        quota = calibration_quotas[family]
        groups = {
            signature: target_ids
            for (group_family, signature), target_ids in topology_targets.items()
            if group_family == family and not (target_ids & test)
        }
        maximum = max(1, math.floor(quota * 0.5))
        chosen, _ = _bounded_group_selection(
            groups, quota, maximum, seed + ":calibration_structure"
        )
        for signature in chosen:
            targets = groups[signature]
            removal, preserves = preserves_required_training_precursors(targets)
            if (
                not preserves
                or calibration_family_counts[family] + len(set(targets) - calibration)
                > quota
            ):
                continue
            selected_calibration_structures.add((family, signature))
            add_calibration(targets)
            remaining_train_degree.subtract(removal)
        candidates = sorted(
            (
                (signature, target_ids)
                for (group_family, signature), target_ids in combination_targets.items()
                if group_family == family
                and not (target_ids & test or target_ids & calibration)
            ),
            key=lambda item: stable_order(
                seed + ":calibration_combination", family, item[0]
            ),
        )
        for signature, target_ids in candidates:
            current = calibration_family_counts[family]
            if current >= quota or current + len(target_ids) > quota:
                continue
            removal, preserves = preserves_required_training_precursors(target_ids)
            if not preserves:
                continue
            selected_calibration_combinations.add((family, signature))
            add_calibration(target_ids)
            remaining_train_degree.subtract(removal)

    if test & calibration:
        raise ValueError("test and calibration targets overlap")
    reference = set().union(
        *(target_ids for family, target_ids in families.items() if family not in formal)
    )
    train = set(by_id) - test - calibration - reference
    train_precursors = {
        precursor_id
        for target_id in train
        for precursor_id in by_id[target_id]["precursor_ids"]
    }
    if selected_precursors & train_precursors:
        raise ValueError("designated unseen precursor leaks into training")
    for family, signature in selected_structures:
        if topology_targets[(family, signature)] - test:
            raise ValueError("selected structural group leaks outside test")
    for family, signature in selected_combinations:
        targets = combination_targets[(family, signature)]
        if targets - test:
            raise ValueError("selected combination leaks outside test")
        for target_id in targets:
            if set(by_id[target_id]["precursor_ids"]) - train_precursors:
                raise ValueError(
                    "combination panel contains a precursor unseen in train"
                )
    for family, study in selected_studies:
        if studies_by_family[family][study] - test:
            raise ValueError("selected source study leaks outside test")
    for family, signature in selected_calibration_structures:
        if topology_targets[(family, signature)] - calibration:
            raise ValueError("calibration structural group leaks outside calibration")

    assignments = []
    for target_id in sorted(by_id):
        split = (
            "reference"
            if target_id in reference
            else (
                "test"
                if target_id in test
                else "calibration" if target_id in calibration else "train"
            )
        )
        assignments.append(
            {
                "target_id": target_id,
                "family": by_id[target_id]["family"],
                "split": split,
                "test_panels": sorted(panels[target_id]) if split == "test" else [],
                "combination_signature": combinations[target_id],
                "core_scaffold_signature": by_id[target_id]["core_scaffold_signature"],
                "structural_group_signature": structural_group_signature(
                    by_id[target_id]
                ),
                "regional_topology_signature": by_id[target_id][
                    "regional_topology_signature"
                ],
            }
        )
    by_family_split = Counter((row["family"], row["split"]) for row in assignments)
    return {
        "assignments": assignments,
        "selected_precursors": sorted(selected_precursors),
        "selected_structural_groups": [
            list(item) for item in sorted(selected_structures)
        ],
        "selected_combination_groups": [
            list(item) for item in sorted(selected_combinations)
        ],
        "selected_source_studies": [list(item) for item in sorted(selected_studies)],
        "selected_calibration_structural_groups": [
            list(item) for item in sorted(selected_calibration_structures)
        ],
        "selected_calibration_combination_groups": [
            list(item) for item in sorted(selected_calibration_combinations)
        ],
        "formal_evaluation_families": sorted(formal),
        "reference_only_families": sorted(set(families) - formal),
        "split_counts": dict(Counter(row["split"] for row in assignments)),
        "family_split_counts": [
            {"family": family, "split": split, "targets": count}
            for (family, split), count in sorted(by_family_split.items())
        ],
    }
