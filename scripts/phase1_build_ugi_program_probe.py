#!/usr/bin/env python3
"""Freeze a source- and branch-stratified Ugi morphology-program probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.design.flow.defog_feasibility import sha256_file
from forge.design.training.ugi_training_cache import load_ugi_training_cache

REPO = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _branch_class(junctions: tuple[int, int, int]) -> str:
    aldehyde = junctions[1] > 0
    isocyanide = junctions[2] > 0
    if aldehyde and isocyanide:
        return "both_tail_origins_branched"
    if aldehyde:
        return "aldehyde_origin_branched"
    if isocyanide:
        return "isocyanide_origin_branched"
    return "linear_tail_origins"


def _held_role_class(assignment: Mapping[str, Any]) -> str:
    roles = (
        "amine_head",
        "oxoester_aldehyde_body_tail",
        "isocyanide_tail",
    )
    held = tuple(role for role in roles if str(assignment.get(f"{role}_family_fold")) == "heldout")
    return "+".join(held) if held else "no_heldout_role"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--fold",
        choices=("train", "calibration", "heldout"),
        default="calibration",
    )
    parser.add_argument("--stratify-component-novelty", action="store_true")
    parser.add_argument("--stratify-held-roles", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("probe count must be positive")
    cache_path = args.cache if args.cache.is_absolute() else REPO / args.cache
    corpus, joint_records = load_ugi_training_cache(cache_path)
    assignments = corpus.assignments_by_fold[args.fold]
    records = joint_records[args.fold]
    buckets: defaultdict[tuple[str, ...], list[tuple[str, object, object]]] = defaultdict(list)
    for assignment, record in zip(assignments, records, strict=True):
        source = str(assignment.get("source_stratum") or "unspecified")
        branch_class = _branch_class(record.program.junction_budgets)
        novelty_class = str(assignment.get("component_novelty_class") or "unspecified")
        held_role_class = _held_role_class(assignment)
        digest = hashlib.sha256(f"{args.seed}|{record.product_id}".encode()).hexdigest()
        key_values = [source]
        if args.stratify_component_novelty:
            key_values.append(novelty_class)
        if args.stratify_held_roles:
            key_values.append(held_role_class)
        key_values.append(branch_class)
        key = tuple(key_values)
        buckets[key].append((digest, assignment, record))
    for values in buckets.values():
        values.sort(key=lambda value: value[0])
    selected = []
    keys_by_source: defaultdict[str, list[tuple[str, ...]]] = defaultdict(list)
    for key in sorted(buckets):
        keys_by_source[key[0]].append(key)
    sources = sorted(keys_by_source)
    source_base, source_remainder = divmod(min(args.count, len(records)), len(sources))
    for source_index, source in enumerate(sources):
        quota = source_base + int(source_index < source_remainder)
        keys = keys_by_source[source]
        while quota > 0:
            added = False
            for key in keys:
                values = buckets[key]
                if values:
                    _, assignment, record = values.pop(0)
                    selected.append((key, assignment, record))
                    quota -= 1
                    added = True
                    if quota == 0:
                        break
            if not added:
                break
    rows = []
    for _, assignment, record in selected:
        source = str(assignment.get("source_stratum") or "unspecified")
        branch_class = _branch_class(record.program.junction_budgets)
        held_role_class = _held_role_class(assignment)
        rows.append(
            {
                "product_id": record.product_id,
                "source_stratum": source,
                "branch_class": branch_class,
                "component_novelty_class": assignment.get("component_novelty_class"),
                "held_role_class": held_role_class,
                "program": {
                    "node_counts": record.program.node_counts,
                    "junction_budgets": record.program.junction_budgets,
                    "cycle_ranks": record.program.cycle_ranks,
                    "attachment_counts": record.program.attachment_counts,
                },
            }
        )
    counts: defaultdict[str, int] = defaultdict(int)
    legacy_counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        counts[f"{row['source_stratum']}|{row['held_role_class']}|{row['branch_class']}"] += 1
        legacy_counts[f"{row['source_stratum']}|{row['branch_class']}"] += 1
    _atomic_json(
        args.output,
        {
            "schema_version": "phase1_ugi_program_probe.v1",
            "seed": args.seed,
            "fold": args.fold,
            "stratify_component_novelty": args.stratify_component_novelty,
            "stratify_held_roles": args.stratify_held_roles,
            "input_cache": {
                "path": str(cache_path.relative_to(REPO)),
                "sha256": sha256_file(cache_path),
            },
            "samples": rows,
            "stratum_branch_counts": dict(sorted(legacy_counts.items())),
            "stratum_held_role_branch_counts": dict(sorted(counts.items())),
        },
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
