#!/usr/bin/env python3
"""Seal the three morphology-program strata for OOD morphology-conditioned sampling.

Stage 2 asks whether the role-structured flow can execute design programs associated with
chemistry outside the training-family regime. It does not ask whether unseen chemical families
are regenerated, and nothing built here supports that reading: a morphology program is test-time
conditioning, not a molecular identity.

Three strata, fixed in the evidence contract before any sample was drawn:

  A  train_family_derived      programs of heldout-fold products whose aldehyde family is a
                               train family. The in-regime reference.
  B  held_family_overlapping   programs of heldout-fold products whose aldehyde family is a
                               heldout family, restricted to programs that also occur in A.
  C  held_family_disjoint      the same, restricted to programs that do NOT occur in A.

C exists because four of the six held aldehyde families share no morphology program with any
train-family product. Discarding them to manufacture a matched comparison would report a
generalization result for the 8.2% of held products that happen to overlap and silently drop the
rest. They are sampled and reported as their own stratum instead.

Only heldout-fold component families count as held. Calibration families steered early stopping
and are excluded from the held strata rather than quietly merged into them.

The draw is written with its own SHA-256 and consumed by the sampler through the existing sealed
program path, so the programs cannot change between sealing and sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from forge.design.corpus.ugi_held_component_gate import held_role_class  # noqa: E402
from forge.design.training.ugi_training_cache import load_ugi_training_cache  # noqa: E402

CACHE = "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt"
ROLE = "oxoester_aldehyde_body_tail"
DRAW_SEED = 20260816


def program_key(program) -> tuple:
    return (tuple(program.node_counts), tuple(program.junction_budgets),
            tuple(program.cycle_ranks), tuple(program.attachment_counts))


def as_row(record, assignment, stratum: str) -> dict:
    program = record.program
    return {
        "product_id": record.product_id,
        "source_stratum": stratum,
        "component_novelty_class": assignment.get("component_novelty_class"),
        "held_role_class": held_role_class(assignment),
        "aldehyde_family_id": assignment[f"{ROLE}_family_id"],
        "program": {
            "node_counts": list(program.node_counts),
            "junction_budgets": list(program.junction_budgets),
            "cycle_ranks": list(program.cycle_ranks),
            "attachment_counts": list(program.attachment_counts),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-stratum", type=int, required=True,
                        help="programs to draw per stratum; frozen before sampling")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()

    corpus, records_by_fold = load_ugi_training_cache(REPO / CACHE)
    records = records_by_fold["heldout"]
    assignments = corpus.assignments_by_fold["heldout"]
    if len(records) != len(assignments) or any(
        r.product_id != a["product_id"] for r, a in zip(records, assignments, strict=True)
    ):
        raise SystemExit("heldout records and assignments are not aligned")

    train_side = [(r, a) for r, a in zip(records, assignments, strict=True)
                  if a[f"{ROLE}_family_fold"] == "train"]
    held_side = [(r, a) for r, a in zip(records, assignments, strict=True)
                 if a[f"{ROLE}_family_fold"] == "heldout"]
    train_programs = {program_key(r.program) for r, _ in train_side}
    overlapping = [(r, a) for r, a in held_side if program_key(r.program) in train_programs]
    disjoint = [(r, a) for r, a in held_side if program_key(r.program) not in train_programs]

    pools = {
        "A_train_family_derived": train_side,
        "B_held_family_overlapping": overlapping,
        "C_held_family_disjoint": disjoint,
    }
    print(f"{'stratum':28s} {'available':>10s} {'families':>9s}")
    for name, pool in pools.items():
        families = {a[f"{ROLE}_family_id"] for _, a in pool}
        print(f"{name:28s} {len(pool):10d} {len(families):9d}")
        if len(pool) < args.per_stratum:
            raise SystemExit(f"{name} has {len(pool)} products, fewer than the requested "
                             f"{args.per_stratum}; choose a smaller budget rather than "
                             f"sampling with replacement")

    rng = random.Random(DRAW_SEED)
    rows: list[dict] = []
    composition: dict[str, dict] = {}
    for name, pool in pools.items():
        ordered = sorted(pool, key=lambda pair: pair[0].product_id)
        chosen = rng.sample(ordered, args.per_stratum)
        rows.extend(as_row(record, assignment, name) for record, assignment in chosen)
        composition[name] = {
            "drawn": len(chosen),
            "available": len(pool),
            "distinct_programs": len({program_key(r.program) for r, _ in chosen}),
            "aldehyde_families": dict(Counter(a[f"{ROLE}_family_id"] for _, a in chosen)),
        }

    payload = {
        "schema_version": "phase1_forge_held_family_program_draw.v1",
        "status": "sealed_before_any_sample_was_drawn",
        "contract": "docs/FORGE_EVIDENCE_CONTRACT_v1.md section B2, amended by Amendment 2",
        "naming": "conditional sampling under held-family-derived morphology programs. A "
                  "morphology program is test-time conditioning, not a molecular identity, so "
                  "this does not show that unseen chemical families are regenerated.",
        "role": ROLE,
        "held_definition": "heldout-fold component families only; calibration families steered "
                           "early stopping and are excluded rather than merged in",
        "seed": DRAW_SEED,
        "per_stratum": args.per_stratum,
        "composition": composition,
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "elapsed_seconds": round(time.time() - started, 1)},
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"\nsealed {len(rows)} programs to {args.output}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
