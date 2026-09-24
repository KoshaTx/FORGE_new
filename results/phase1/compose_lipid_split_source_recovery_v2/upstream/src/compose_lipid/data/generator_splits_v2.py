"""Structure-resolved grouped splits for the broad lipid generator."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from compose_lipid.data.training_corpus import digest


def stable_order(label, *parts):
    return digest([label, *parts])


def combination_signature(row):
    instances = sorted(
        (item["role"], item["component_id"])
        for item in row["component_instances"]
    )
    return digest(["complete_component_role_multiset_v2", row["family"], instances])


def morphology_group_signature(row):
    return digest(
        [
            "reaction_core_regional_morphology_v2",
            row["family"],
            row["core_scaffold_signature"],
            row["regional_morphology_signature"],
        ]
    )


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


def make_generator_splits_v2(
    records,
    *,
    seed="broad_lipid_generator_structure_split_v2",
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

    families = defaultdict(set)
    component_targets = defaultdict(set)
    morphology_targets = defaultdict(set)
    combination_targets = defaultdict(set)
    combinations = {}
    studies_by_family = defaultdict(lambda: defaultdict(set))
    global_study_targets = defaultdict(set)
    for row in records:
        target_id = row["target_id"]
        family = row["family"]
        families[family].add(target_id)
        for component_id in row["component_ids"]:
            component_targets[component_id].add(target_id)
        morphology_targets[(family, morphology_group_signature(row))].add(target_id)
        combination = combination_signature(row)
        combinations[target_id] = combination
        combination_targets[(family, combination)].add(target_id)
        for study in row.get("source_pmids") or []:
            study = "pmid:" + str(study).removeprefix("pmid:")
            studies_by_family[family][study].add(target_id)
            global_study_targets[study].add(target_id)

    formal = {
        family
        for family, target_ids in families.items()
        if len(target_ids) >= minimum_training_class
    }
    formal_targets = set().union(*(families[family] for family in formal))
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
    selected_components = set()
    selected_morphologies = set()
    selected_combinations = set()

    def add_test(target_ids, panel):
        target_ids = set(target_ids) & formal_targets
        new = target_ids - test
        test_family_counts.update(by_id[target_id]["family"] for target_id in new)
        test.update(target_ids)
        for target_id in target_ids:
            panels[target_id].add(panel)

    def add_calibration(target_ids):
        new = set(target_ids) - calibration
        calibration_family_counts.update(
            by_id[target_id]["family"] for target_id in new
        )
        calibration.update(new)

    # A selected PMID is held out globally across every formal family.  This is
    # stronger than the v8 per-(family, study) contract.
    for family in sorted(formal):
        groups = studies_by_family[family]
        if len(groups) < 2:
            continue
        target = max(1, math.floor(test_quotas[family] * 0.15))
        maximum = max(1, math.floor(test_quotas[family] * 0.5))
        added = 0
        for study in sorted(groups, key=lambda value: stable_order(seed + ":study", value)):
            if study in selected_studies:
                continue
            affected = global_study_targets[study] & formal_targets
            new = affected - test
            family_new = new & families[family]
            if not family_new or len(affected & families[family]) > maximum:
                continue
            by_family = Counter(by_id[target_id]["family"] for target_id in new)
            if any(
                test_family_counts[other] + count > test_quotas[other]
                for other, count in by_family.items()
            ):
                continue
            if added + len(family_new) > target:
                continue
            selected_studies.add(study)
            add_test(affected, "source_study_transfer_global")
            added += len(family_new)
            if added >= target:
                break

    family_component_degree = defaultdict(Counter)
    for family, target_ids in families.items():
        for target_id in target_ids:
            family_component_degree[family].update(
                set(by_id[target_id]["component_ids"])
            )
    for family in sorted(formal):
        panel_target = max(1, math.floor(test_quotas[family] * 0.30))
        added = 0
        candidates = sorted(
            family_component_degree[family],
            key=lambda component_id: stable_order(
                seed + ":component", family, component_id
            ),
        )
        for component_id in candidates:
            if component_id in selected_components:
                continue
            affected = component_targets[component_id] & formal_targets
            family_degree = family_component_degree[family][component_id]
            if family_degree < 2 or len(affected) > max(2, panel_target):
                continue
            new = affected - test
            if not new:
                continue
            by_family = Counter(by_id[target_id]["family"] for target_id in new)
            if any(
                test_family_counts[other] + count > test_quotas[other]
                for other, count in by_family.items()
            ):
                continue
            if added + len(new & families[family]) > panel_target:
                continue
            selected_components.add(component_id)
            add_test(affected, "unseen_component_structure")
            added += len(new & families[family])
            if added >= panel_target:
                break

    for family in sorted(formal):
        panel_target = max(1, math.floor(test_quotas[family] * 0.30))
        remaining = max(0, test_quotas[family] - test_family_counts[family])
        groups = {
            signature: target_ids
            for (group_family, signature), target_ids in morphology_targets.items()
            if group_family == family and not (target_ids & test)
        }
        maximum = max(1, math.floor(test_quotas[family] * 0.5))
        chosen, _ = _bounded_group_selection(
            groups, min(panel_target, remaining), maximum, seed + ":morphology"
        )
        for signature in chosen:
            selected_morphologies.add((family, signature))
            add_test(groups[signature], "unseen_regional_morphology")

    remaining_component_degree = Counter()
    for target_id, row in by_id.items():
        if target_id not in test and row["family"] in formal:
            remaining_component_degree.update(
                (row["family"], component_id)
                for component_id in set(row["component_ids"])
            )
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
                (family, component_id)
                for target_id in target_ids
                for component_id in set(by_id[target_id]["component_ids"])
            )
            if any(
                remaining_component_degree[family_component] - count < 1
                for family_component, count in removal.items()
            ):
                continue
            selected_combinations.add((family, signature))
            add_test(target_ids, "unseen_exact_component_combination")
            remaining_component_degree.subtract(removal)

    selected_calibration_morphologies = set()
    selected_calibration_combinations = set()
    protected_train_components = {
        (family, component_id)
        for family, signature in selected_combinations
        for target_id in combination_targets[(family, signature)]
        for component_id in by_id[target_id]["component_ids"]
    }
    remaining_train_degree = Counter()
    for target_id, row in by_id.items():
        if target_id not in test and row["family"] in formal:
            remaining_train_degree.update(
                (row["family"], component_id)
                for component_id in set(row["component_ids"])
            )

    def preserves_required_training_components(target_ids):
        removal = Counter(
            (by_id[target_id]["family"], component_id)
            for target_id in target_ids
            for component_id in set(by_id[target_id]["component_ids"])
        )
        return removal, all(
            family_component not in protected_train_components
            or remaining_train_degree[family_component] - count >= 1
            for family_component, count in removal.items()
        )

    for family in sorted(formal):
        quota = calibration_quotas[family]
        groups = {
            signature: target_ids
            for (group_family, signature), target_ids in morphology_targets.items()
            if group_family == family and not (target_ids & test)
        }
        maximum = max(1, math.floor(quota * 0.5))
        chosen, _ = _bounded_group_selection(
            groups, quota, maximum, seed + ":calibration_morphology"
        )
        for signature in chosen:
            targets = groups[signature]
            removal, preserves = preserves_required_training_components(targets)
            if (
                not preserves
                or calibration_family_counts[family] + len(set(targets) - calibration)
                > quota
            ):
                continue
            selected_calibration_morphologies.add((family, signature))
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
            removal, preserves = preserves_required_training_components(target_ids)
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
    train_components = {
        component_id
        for target_id in train
        for component_id in by_id[target_id]["component_ids"]
    }
    train_family_components = {
        (by_id[target_id]["family"], component_id)
        for target_id in train
        for component_id in by_id[target_id]["component_ids"]
    }
    if selected_components & train_components:
        raise ValueError("designated unseen component structure leaks into training")
    for study in selected_studies:
        if (global_study_targets[study] & formal_targets) - test:
            raise ValueError("selected source study leaks outside test")
    for family, signature in selected_morphologies:
        if morphology_targets[(family, signature)] - test:
            raise ValueError("selected morphology group leaks outside test")
    for family, signature in selected_combinations:
        targets = combination_targets[(family, signature)]
        if targets - test:
            raise ValueError("selected component combination leaks outside test")
        for target_id in targets:
            family = by_id[target_id]["family"]
            if {
                (family, component_id)
                for component_id in by_id[target_id]["component_ids"]
            } - train_family_components:
                raise ValueError(
                    "combination panel contains a component unseen in its family train set"
                )
    for family, signature in selected_calibration_morphologies:
        if morphology_targets[(family, signature)] - calibration:
            raise ValueError("calibration morphology group leaks outside calibration")
    for family, signature in selected_calibration_combinations:
        if combination_targets[(family, signature)] - calibration:
            raise ValueError("calibration combination leaks outside calibration")

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
                "morphology_group_signature": morphology_group_signature(
                    by_id[target_id]
                ),
                "regional_morphology_signature": by_id[target_id][
                    "regional_morphology_signature"
                ],
            }
        )
    return {
        "assignments": assignments,
        "selected_components": sorted(selected_components),
        "selected_morphology_groups": [
            list(item) for item in sorted(selected_morphologies)
        ],
        "selected_combination_groups": [
            list(item) for item in sorted(selected_combinations)
        ],
        "selected_source_studies": sorted(selected_studies),
        "selected_calibration_morphology_groups": [
            list(item) for item in sorted(selected_calibration_morphologies)
        ],
        "selected_calibration_combination_groups": [
            list(item) for item in sorted(selected_calibration_combinations)
        ],
        "formal_evaluation_families": sorted(formal),
        "reference_only_families": sorted(set(families) - formal),
        "split_counts": dict(Counter(row["split"] for row in assignments)),
        "family_split_counts": [
            {"family": family, "split": split, "targets": count}
            for (family, split), count in sorted(
                Counter((row["family"], row["split"]) for row in assignments).items()
            )
        ],
    }


__all__ = [
    "combination_signature",
    "make_generator_splits_v2",
    "morphology_group_signature",
]
