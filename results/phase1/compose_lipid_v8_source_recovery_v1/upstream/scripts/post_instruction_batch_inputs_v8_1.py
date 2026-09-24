"""Bound v8.1 train/calibration inputs and lazy exact-path stores."""

from __future__ import annotations

import gzip
import json
from collections import Counter, OrderedDict
from pathlib import Path

from scripts.construction_pilot import read_paths

from compose_lipid.core import load
from compose_lipid.data.assets import sha256_file
from compose_lipid.data.target_sampling import WeightedTargetSampler
from compose_lipid.models.corpus_runtime_v4 import canonical_provider


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else path.open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def output_hash(receipt, name):
    value = receipt["outputs"][name]
    return value["sha256"] if isinstance(value, dict) else value


class BoundSplitPathStore:
    """Lazy exact paths from one immutable split-specific Modal release."""

    def __init__(
        self,
        directory,
        targets,
        *,
        path_receipt_sha256,
        required_split,
        limit=16,
    ):
        self.directory = Path(directory)
        self.targets = tuple(targets)
        self.required_split = required_split
        if required_split not in {"train", "calibration"} or not 1 <= limit <= 64:
            raise ValueError("invalid bound split path-store policy")
        ids = [row["target_id"] for row in self.targets]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("target-index universe must be unique and nonempty")
        receipt_path = self.directory / "receipt.json"
        if sha256_file(receipt_path) != path_receipt_sha256:
            raise ValueError("split exact-path receipt changed")
        receipt = read(receipt_path)
        if not receipt.get("complete") or not receipt.get("exact_paths_ready"):
            raise ValueError("complete split exact paths required")
        manifest_path = self.directory / "path_manifest.jsonl.gz"
        if sha256_file(manifest_path) != output_hash(receipt, manifest_path.name):
            raise ValueError("split exact-path manifest changed")
        all_entries = rows(manifest_path)
        entries = {row["target_id"]: row for row in all_entries}
        if len(entries) != len(all_entries):
            raise ValueError("duplicate split exact-path manifest target")
        self.entries = {target_id: entries[target_id] for target_id in ids if target_id in entries}
        self.limit = limit
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.targets)

    def item(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index not in self.cache:
            target = self.targets[index]
            target_id = target["target_id"]
            entry = self.entries.get(target_id)
            if entry is None:
                raise ValueError("frozen workload requested a target without a prepared path")
            if entry["family"] != target["family"]:
                raise ValueError("prepared path family differs from target measure")
            path = (self.directory / entry["path"]).resolve()
            if not path.is_relative_to(self.directory.resolve()):
                raise ValueError("prepared path escapes its split release")
            if sha256_file(path) != entry["path_sha256"]:
                raise ValueError("prepared exact path bytes changed")
            self.cache[index] = read_paths(path)["paths"][0]
            while len(self.cache) > self.limit:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def __getitem__(self, index):
        item = self.item(index)
        return load("reference_law").PathRecord(item["target_key"], item["path"])


def validate_workload_rows(
    universe,
    selected_train,
    calibration,
    schedule,
    path_manifests,
    *,
    expected,
):
    panels = {
        "training_measure_targets": universe,
        "scheduled_training_targets": selected_train,
        "calibration_targets": calibration,
        "scheduled_training_rows": schedule,
        "training_path_targets": path_manifests["train"],
        "calibration_path_targets": path_manifests["calibration"],
    }
    for name, values in panels.items():
        if len(values) != expected[name]:
            raise ValueError(f"unexpected {name} count")
    for name, values in panels.items():
        if name == "scheduled_training_rows":
            continue
        ids = [row["target_id"] for row in values]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate {name} target")
    if [row["row"] for row in schedule] != list(range(len(schedule))):
        raise ValueError("training schedule rows are missing or reordered")

    universe_by_id = {row["target_id"]: row for row in universe}
    selected_by_id = {row["target_id"]: row for row in selected_train}
    calibration_by_id = {row["target_id"]: row for row in calibration}
    train_paths = {row["target_id"]: row for row in path_manifests["train"]}
    calibration_paths = {row["target_id"]: row for row in path_manifests["calibration"]}
    if set(train_paths) != set(universe_by_id):
        raise ValueError("training exact paths differ from the full training measure")
    if set(calibration_paths) != set(calibration_by_id):
        raise ValueError("calibration exact paths differ from the selected panel")
    if set(universe_by_id) & set(calibration_by_id):
        raise ValueError("training and calibration path targets overlap")
    occurrences = Counter(row["target_id"] for row in schedule)
    if set(occurrences) != set(selected_by_id):
        raise ValueError("scheduled unique targets differ from selected training paths")
    for target_id, count in occurrences.items():
        selected = selected_by_id[target_id]
        source = universe_by_id.get(target_id)
        if source is None or source["family"] != selected["family"]:
            raise ValueError("scheduled target differs from the training measure")
        if count != selected["training_row_occurrences"]:
            raise ValueError("recorded training occurrence count differs")
    for row in schedule:
        source = universe_by_id.get(row["target_id"])
        if source is None or source["family"] != row["family"]:
            raise ValueError("training schedule family differs from target measure")
    for split, records in path_manifests.items():
        expected_by_id = universe_by_id if split == "train" else calibration_by_id
        for row in records:
            target = expected_by_id[row["target_id"]]
            if row["family"] != target["family"]:
                raise ValueError("exact path family differs from its split target")

    universe = sorted(universe, key=lambda row: row["target_id"])
    probabilities = {row["target_id"]: row["probability"] for row in universe}
    sampler = WeightedTargetSampler([row["target_id"] for row in universe], probabilities)
    combined_manifest = {
        row["target_id"]: {**row, "split": split}
        for split, records in path_manifests.items()
        for row in records
    }
    return {
        "universe": universe,
        "selected_train": sorted(selected_train, key=lambda row: row["target_id"]),
        "calibration": sorted(calibration, key=lambda row: row["target_id"]),
        "schedule": schedule,
        "path_manifest": combined_manifest,
        "sampler": sampler,
    }


def load_inputs(root, config):
    root = Path(root).resolve()
    inputs = {}
    for name, expected in config.get("decisions", {}).items():
        path = root / name
        if sha256_file(path) != expected:
            raise ValueError(f"changed decision input: {name}")
        inputs[name] = expected

    support_name = config["model_support_receipt"]
    support_path = root / support_name
    if sha256_file(support_path) != config["model_support_receipt_sha256"]:
        raise ValueError("changed post-instruction model-support receipt")
    support_receipt = read(support_path)
    if (
        support_receipt.get("complete") is not True
        or support_receipt.get("model_support_frozen") is not True
    ):
        raise ValueError("incomplete post-instruction model-support release")
    support_root = support_path.parent
    inputs[support_name] = config["model_support_receipt_sha256"]

    target_name = config["target_measure_receipt"]
    target_path = root / target_name
    if sha256_file(target_path) != config["target_measure_receipt_sha256"]:
        raise ValueError("changed post-instruction target-measure receipt")
    target_receipt = read(target_path)
    if (
        target_receipt.get("complete") is not True
        or target_receipt.get("target_measure_frozen") is not True
    ):
        raise ValueError("incomplete post-instruction target measure")
    target_root = target_path.parent
    inputs[target_name] = config["target_measure_receipt_sha256"]

    resolved = {
        "universe": target_root / "target_measure.jsonl.gz",
    }
    target_hash = output_hash(target_receipt, "target_measure.jsonl.gz")
    if sha256_file(resolved["universe"]) != target_hash:
        raise ValueError("changed train-only target-measure rows")
    inputs[str(resolved["universe"].relative_to(root))] = target_hash
    for label, name in {
        "calibration": "calibration_targets.jsonl.gz",
        "schedule": "training_schedule.jsonl.gz",
        "ontology": "role_ontology_v3.json",
        "vocabulary": "lipid_vocabulary_v6.json",
    }.items():
        path = support_root / name
        expected_hash = output_hash(support_receipt, name)
        if sha256_file(path) != expected_hash:
            raise ValueError("changed rational-lipid workload input: " + name)
        inputs[str(path.relative_to(root))] = expected_hash
        resolved[label] = path

    volume_root = Path(config["modal_volume_root"])
    path_directories = {
        split: volume_root / config[f"{split}_path_volume_dir"]
        for split in ("train", "calibration")
    }
    path_receipt_hashes = {}
    path_manifests = {}
    for split, directory in path_directories.items():
        receipt_path = directory / "receipt.json"
        expected = config[f"{split}_path_receipt_sha256"]
        if sha256_file(receipt_path) != expected:
            raise ValueError(f"changed {split} exact-path receipt")
        receipt = read(receipt_path)
        if not receipt.get("complete") or not receipt.get("exact_paths_ready"):
            raise ValueError(f"incomplete {split} exact-path release")
        manifest_path = directory / "path_manifest.jsonl.gz"
        manifest_hash = output_hash(receipt, manifest_path.name)
        if sha256_file(manifest_path) != manifest_hash:
            raise ValueError(f"changed {split} exact-path manifest")
        inputs[str(receipt_path)] = expected
        inputs[str(manifest_path)] = manifest_hash
        path_receipt_hashes[split] = expected
        path_manifests[split] = rows(manifest_path)

    universe = rows(resolved["universe"])
    schedule = rows(resolved["schedule"])
    universe_by_id = {row["target_id"]: row for row in universe}
    occurrences = Counter(row["target_id"] for row in schedule)
    selected_train = [
        {
            **universe_by_id[target_id],
            "training_row_occurrences": count,
        }
        for target_id, count in sorted(occurrences.items())
    ]
    result = validate_workload_rows(
        universe,
        selected_train,
        rows(resolved["calibration"]),
        schedule,
        path_manifests,
        expected=config["expected"],
    )
    result.update(
        root=root,
        path_directories=path_directories,
        path_receipt_sha256=path_receipt_hashes,
        ontology_path=resolved["ontology"],
        vocabulary_path=resolved["vocabulary"],
        inputs=dict(sorted(inputs.items())),
        provider=canonical_provider(resolved["ontology"], n_slots=config["n_slots"]),
    )
    return result
