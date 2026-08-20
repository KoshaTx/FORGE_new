#!/usr/bin/env python3
"""Audit terminal-decoration representation and clean-state readout errors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from forge.corpus.r1_prime_audit import sha256_file
from forge.potency.annotations import ROLE_NAMES
from forge.model.ugi_joint_sparse_flow import (
    UgiJointSparseFlow,
    collate_ugi_joint_sparse_records,
)
from experiments.phase1.product_l1.training.ugi_training_cache import load_ugi_training_cache

REPO = Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _move(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
    }


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _decoration_representation_census(records: tuple[Any, ...]) -> dict[str, Any]:
    count_distribution: Counter[int] = Counter()
    atom_bond_pairs: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    duplicate_anchor_records = 0
    maximum_per_anchor = 0
    total_decorations = 0
    for record in records:
        anchors = [int(value) for value in record.decoration_anchors.tolist()]
        multiplicities = Counter(anchors)
        count_distribution[len(anchors)] += 1
        total_decorations += len(anchors)
        maximum_per_anchor = max(maximum_per_anchor, max(multiplicities.values(), default=0))
        duplicate_anchor_records += any(value > 1 for value in multiplicities.values())
        boundaries = np.cumsum(record.program.node_counts)
        for anchor, atom, bond in zip(
            anchors,
            record.decoration_atom_states.tolist(),
            record.decoration_bond_states.tolist(),
            strict=True,
        ):
            role_index = int(np.searchsorted(boundaries, anchor, side="right"))
            role_counts[ROLE_NAMES[role_index]] += 1
            atom_bond_pairs[f"atom_{int(atom)}:bond_{int(bond)}"] += 1
    return {
        "records": len(records),
        "total_decorations": total_decorations,
        "decoration_count_distribution": {
            str(key): value for key, value in sorted(count_distribution.items())
        },
        "maximum_decorations_on_one_support_atom": maximum_per_anchor,
        "records_with_repeated_decoration_anchor": duplicate_anchor_records,
        "fraction_with_repeated_decoration_anchor": _fraction(
            duplicate_anchor_records, len(records)
        ),
        "single_local_decoration_state_is_lossless": maximum_per_anchor <= 1,
        "decoration_counts_by_anchor_role": dict(sorted(role_counts.items())),
        "decoration_atom_bond_pair_counts": dict(sorted(atom_bond_pairs.items())),
    }


def _selected_heldout_indices(
    assignments: tuple[Any, ...], *, records_per_source: int, seed: int
) -> tuple[np.ndarray, dict[str, int]]:
    by_source: dict[str, list[int]] = {}
    for index, assignment in enumerate(assignments):
        source = str(assignment.get("source_stratum") or "unspecified")
        by_source.setdefault(source, []).append(index)
    rng = np.random.default_rng(seed)
    selected = []
    counts = {}
    for source, indices in sorted(by_source.items()):
        if len(indices) < records_per_source:
            raise ValueError(
                f"heldout source {source!r} has {len(indices)} records, fewer than requested"
            )
        values = rng.choice(np.asarray(indices), size=records_per_source, replace=False)
        selected.extend(int(value) for value in values)
        counts[source] = records_per_source
    return np.asarray(sorted(selected), dtype=np.int64), counts


def _empty_role_metrics() -> dict[str, int]:
    return {
        "target_present_slots": 0,
        "anchor_exact": 0,
        "atom_exact": 0,
        "bond_exact": 0,
        "atom_bond_exact": 0,
        "anchor_atom_bond_exact": 0,
        "predicted_none": 0,
    }


def _finalize_role_metrics(values: dict[str, int]) -> dict[str, Any]:
    denominator = values["target_present_slots"]
    return {
        **values,
        "anchor_accuracy": _fraction(values["anchor_exact"], denominator),
        "atom_accuracy": _fraction(values["atom_exact"], denominator),
        "bond_accuracy": _fraction(values["bond_exact"], denominator),
        "atom_bond_accuracy": _fraction(values["atom_bond_exact"], denominator),
        "anchor_atom_bond_accuracy": _fraction(values["anchor_atom_bond_exact"], denominator),
        "false_none_fraction": _fraction(values["predicted_none"], denominator),
    }


def _checkpoint_readout(
    checkpoint_path: Path,
    records: tuple[Any, ...],
    *,
    atom_classes: int,
    batch_size: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = dict(checkpoint["model_config"])
    architecture.pop("source_probability_floor", None)
    model = UgiJointSparseFlow(atom_classes=atom_classes, **architecture).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    totals = {
        "records": 0,
        "slots": 0,
        "target_present_slots": 0,
        "target_empty_slots": 0,
        "predicted_present_slots": 0,
        "present_anchor_exact": 0,
        "present_atom_exact": 0,
        "present_bond_exact": 0,
        "present_atom_bond_exact": 0,
        "present_anchor_atom_bond_exact": 0,
        "present_predicted_none": 0,
        "empty_predicted_none": 0,
        "exact_decoration_tensor_records": 0,
        "absolute_count_error": 0,
    }
    by_role = {role: _empty_role_metrics() for role in ROLE_NAMES}
    target_pairs: Counter[str] = Counter()
    predicted_pairs_on_target_present: Counter[str] = Counter()
    with torch.no_grad():
        for offset in range(0, len(records), batch_size):
            local = records[offset : offset + batch_size]
            batch = _move(
                collate_ugi_joint_sparse_records(
                    local,
                    maximum_nodes=max(record.node_count for record in local),
                    maximum_children=int(architecture["maximum_children"]),
                    maximum_closures=int(architecture["maximum_cycle_rank"]) * len(ROLE_NAMES),
                    maximum_decorations=int(architecture["maximum_decorations"]),
                ),
                device,
            )
            predictions = model(
                offspring=batch["offspring"],
                nodes=batch["nodes"],
                parent_bonds=batch["parent_bonds"],
                role_states=batch["role_states"],
                within_role_positions=batch["within_role_positions"],
                programs=batch["programs"],
                node_mask=batch["node_mask"],
                t=torch.ones(len(local), dtype=torch.float32, device=device),
                closure_left=batch["closure_left"],
                closure_right=batch["closure_right"],
                decoration_anchors=batch["decoration_anchors"],
                decoration_atoms=batch["decoration_atoms"],
                decoration_bonds=batch["decoration_bonds"],
            )
            target_anchor = batch["decoration_anchors"]
            target_atom = batch["decoration_atoms"]
            target_bond = batch["decoration_bonds"]
            predicted_anchor = predictions["decoration_anchors"].argmax(dim=-1)
            predicted_atom = predictions["decoration_atoms"].argmax(dim=-1)
            predicted_bond = predictions["decoration_bonds"].argmax(dim=-1)
            present = target_anchor > 0
            empty = ~present
            totals["records"] += len(local)
            totals["slots"] += int(target_anchor.numel())
            totals["target_present_slots"] += int(present.sum())
            totals["target_empty_slots"] += int(empty.sum())
            totals["predicted_present_slots"] += int((predicted_anchor > 0).sum())
            totals["present_anchor_exact"] += int(
                ((predicted_anchor == target_anchor) & present).sum()
            )
            totals["present_atom_exact"] += int(((predicted_atom == target_atom) & present).sum())
            totals["present_bond_exact"] += int(((predicted_bond == target_bond) & present).sum())
            atom_bond_exact = (predicted_atom == target_atom) & (predicted_bond == target_bond)
            totals["present_atom_bond_exact"] += int((atom_bond_exact & present).sum())
            totals["present_anchor_atom_bond_exact"] += int(
                ((predicted_anchor == target_anchor) & atom_bond_exact & present).sum()
            )
            totals["present_predicted_none"] += int(((predicted_anchor == 0) & present).sum())
            totals["empty_predicted_none"] += int(((predicted_anchor == 0) & empty).sum())
            record_exact = (
                (predicted_anchor == target_anchor)
                & ((predicted_atom == target_atom) | empty)
                & ((predicted_bond == target_bond) | empty)
            ).all(dim=1)
            totals["exact_decoration_tensor_records"] += int(record_exact.sum())
            totals["absolute_count_error"] += int(
                torch.abs((predicted_anchor > 0).sum(dim=1) - present.sum(dim=1)).sum()
            )

            rows, slots = torch.nonzero(present, as_tuple=True)
            for row, slot in zip(rows.tolist(), slots.tolist(), strict=True):
                anchor = int(target_anchor[row, slot]) - 1
                role_index = int(batch["role_states"][row, anchor])
                role = ROLE_NAMES[role_index]
                metrics = by_role[role]
                metrics["target_present_slots"] += 1
                metrics["anchor_exact"] += int(
                    predicted_anchor[row, slot] == target_anchor[row, slot]
                )
                metrics["atom_exact"] += int(predicted_atom[row, slot] == target_atom[row, slot])
                metrics["bond_exact"] += int(predicted_bond[row, slot] == target_bond[row, slot])
                pair_exact = bool(atom_bond_exact[row, slot])
                metrics["atom_bond_exact"] += int(pair_exact)
                metrics["anchor_atom_bond_exact"] += int(
                    pair_exact and predicted_anchor[row, slot] == target_anchor[row, slot]
                )
                metrics["predicted_none"] += int(predicted_anchor[row, slot] == 0)
                target_pairs[
                    f"atom_{int(target_atom[row, slot])}:bond_{int(target_bond[row, slot])}"
                ] += 1
                predicted_pairs_on_target_present[
                    f"atom_{int(predicted_atom[row, slot])}:bond_{int(predicted_bond[row, slot])}"
                ] += 1
    present_slots = totals["target_present_slots"]
    empty_slots = totals["target_empty_slots"]
    return {
        **totals,
        "present_anchor_accuracy": _fraction(totals["present_anchor_exact"], present_slots),
        "present_atom_accuracy": _fraction(totals["present_atom_exact"], present_slots),
        "present_bond_accuracy": _fraction(totals["present_bond_exact"], present_slots),
        "present_atom_bond_accuracy": _fraction(totals["present_atom_bond_exact"], present_slots),
        "present_anchor_atom_bond_accuracy": _fraction(
            totals["present_anchor_atom_bond_exact"], present_slots
        ),
        "present_false_none_fraction": _fraction(totals["present_predicted_none"], present_slots),
        "empty_slot_specificity": _fraction(totals["empty_predicted_none"], empty_slots),
        "exact_decoration_tensor_record_fraction": _fraction(
            totals["exact_decoration_tensor_records"], totals["records"]
        ),
        "mean_absolute_decoration_count_error": _fraction(
            totals["absolute_count_error"], totals["records"]
        ),
        "mean_target_decoration_count": _fraction(present_slots, totals["records"]),
        "mean_predicted_decoration_count": _fraction(
            totals["predicted_present_slots"], totals["records"]
        ),
        "by_target_anchor_role": {
            role: _finalize_role_metrics(values) for role, values in by_role.items()
        },
        "target_atom_bond_pair_counts": dict(sorted(target_pairs.items())),
        "predicted_atom_bond_pair_counts_on_target_present_slots": dict(
            sorted(predicted_pairs_on_target_present.items())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_decoration_readout_audit_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/ugi_decoration_readout_audit_v1.json",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch

    config = _load_json(args.config)
    if config.get("schema_version") != "phase1_ugi_decoration_readout_audit_config.v1":
        raise ValueError("unexpected decoration-readout audit config schema")
    verified_inputs = {}
    for name, specification in config["inputs"].items():
        path = REPO / specification["path"]
        digest = sha256_file(path)
        if digest != specification["sha256"]:
            raise ValueError(f"frozen decoration-audit input changed: {name}")
        verified_inputs[name] = {"path": specification["path"], "sha256": digest}
    cache_path = REPO / config["inputs"]["training_cache"]["path"]
    corpus, records_by_fold = load_ugi_training_cache(cache_path)
    all_records = tuple(
        record for fold in ("train", "calibration", "heldout") for record in records_by_fold[fold]
    )
    fold = str(config["design"]["fold"])
    records = records_by_fold[fold]
    assignments = corpus.assignments_by_fold[fold]
    selected_indices, selected_by_source = _selected_heldout_indices(
        assignments,
        records_per_source=int(config["design"]["records_per_source"]),
        seed=int(config["design"]["sampling_seed"]),
    )
    selected_records = tuple(records[int(index)] for index in selected_indices)
    selected_ids = [record.product_id for record in selected_records]
    selected_id_hash = hashlib.sha256("\n".join(selected_ids).encode()).hexdigest()
    device = torch.device(args.device)
    checkpoint_metrics = {}
    for step in config["design"]["checkpoints"]:
        path = REPO / config["inputs"][f"checkpoint_step_{step}"]["path"]
        checkpoint_metrics[str(step)] = _checkpoint_readout(
            path,
            selected_records,
            atom_classes=len(corpus.atom_vocabulary),
            batch_size=int(config["design"]["batch_size"]),
            device=device,
        )
    output = {
        "schema_version": "phase1_ugi_decoration_readout_audit.v1",
        "status": "complete_selection_visible_diagnostic",
        "config": {
            "path": str(args.config.relative_to(REPO)),
            "sha256": sha256_file(args.config),
        },
        "inputs": verified_inputs,
        "selection": {
            "fold": fold,
            "sampling_seed": int(config["design"]["sampling_seed"]),
            "selected_by_source": selected_by_source,
            "selected_records": len(selected_records),
            "selected_product_id_sha256": selected_id_hash,
        },
        "representation_census": _decoration_representation_census(all_records),
        "clean_time_one_checkpoint_readout": checkpoint_metrics,
        "interpretation_policy": {
            "clean_scaffold_readout_is_generous_relative_to_free_generation": True,
            "failure_here_establishes_terminal_head_or_training_error": True,
            "success_here_does_not_establish_free_generation_calibration": True,
            "architecture_or_checkpoint_promoted": False,
            "chemistry_quota_fitted": False,
        },
    }
    _atomic_json(args.output, output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_records": len(selected_records),
                "checkpoint_metrics": checkpoint_metrics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
