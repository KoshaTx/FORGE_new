#!/usr/bin/env python3
"""Seal the combinatorial-coverage splits and test strata, before either model is trained.

The question this serves is not cross-role coupling, which the corpus audit showed is absent:
component identities pair near-independently and one role pair fully saturates its grid. It is

    does experimentally grounded semantic organization let a generator reuse molecular knowledge
    across an incompletely observed combinatorial product space?

The corpus is already an incomplete combinatorial design, which is what makes the question
answerable: the train fold holds 66,464 products occupying 13.7% of a 185 x 75 x 35 component
triple grid, with exactly one product per triple. Reducing coverage further therefore withholds
*combinations* rather than deleting chemistry, provided the subsets keep the underlying molecular
factors visible. Random product subsampling would not guarantee that, so coverage is reduced over
grid cells with the marginal component coverage checked and reported at every level.

Two reserved holdouts sit outside every training subset, so that library extrapolation can be
distinguished from combination novelty:

  identity holdout   individual component identities removed entirely, their families still
                     represented in training
  family holdout     whole component families removed, the harder structural shift

Four test strata, defined relative to each coverage level:

  S1 in-sample control        products inside the training subset. Labelled in-sample; it is a
                             fidelity control, not a generalization measurement.
  S2 new combination         the combination is absent from the subset while all three of its
                             components are present. The primary compositional-reuse stratum.
  S3 held component identity one component identity is reserved out; its family is not.
  S4 held component family   one component family is reserved out.

Nesting is enforced: the 10% subset is a subset of 25%, which is a subset of 50%, which is a
subset of 100%. Everything is written with SHA-256 before any training run exists.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ASSIGN = "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz"
ROLES = ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")
COVERAGE = (0.10, 0.25, 0.50, 1.00)
SPLIT_SEED = 20260817
# Reserved holdout sizes. Aldehyde and isocyanide carry few families in the train fold, so one
# family each is already a severe structural shift; amine families are numerous enough for 10%.
FAMILY_HOLDOUT = {"amine_head": 0.10, "oxoester_aldehyde_body_tail": 1, "isocyanide_tail": 1}
IDENTITY_HOLDOUT = 0.05
# A fixed combination test set withheld from every coverage level. Without it the primary
# compositional-reuse stratum is empty at full coverage, because "absent from the subset" and
# "the subset is everything" cannot both hold, and a learning curve needs one test set held
# constant while training size varies.
COMBINATION_TEST_FRACTION = 0.15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "results/phase1/forge_coverage_splits_v1/splits.json")
    args = parser.parse_args()
    started = time.time()
    rng = random.Random(SPLIT_SEED)

    with gzip.open(REPO / ASSIGN, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [r for r in rows if r["primary_product_fold"] == "train"]
    print(f"corpus {len(rows)} products; train fold {len(train)}")

    # ---- reserved family holdout, then identity holdout among surviving families
    held_families: dict[str, set[str]] = {}
    for role, size in FAMILY_HOLDOUT.items():
        families = sorted({r[f"{role}_family_id"] for r in train})
        count = max(1, round(len(families) * size)) if isinstance(size, float) else int(size)
        held_families[role] = set(rng.sample(families, count))
        print(f"  {role:30s} families {len(families):4d}, holding {count}")

    survivors = [r for r in train
                 if not any(r[f"{role}_family_id"] in held_families[role] for role in ROLES)]
    held_identities: dict[str, set[str]] = {}
    for role in ROLES:
        identities = sorted({r[f"{role}_smiles"] for r in survivors})
        count = max(1, round(len(identities) * IDENTITY_HOLDOUT))
        held_identities[role] = set(rng.sample(identities, count))
        print(f"  {role:30s} identities {len(identities):4d}, holding {count}")

    eligible = [r for r in survivors
                if not any(r[f"{role}_smiles"] in held_identities[role] for role in ROLES)]
    print(f"\neligible training cells after both reserved holdouts: {len(eligible)}")

    # ---- fixed combination test set, then nested coverage subsets over what remains
    order = sorted(eligible, key=lambda r: r["product_id"])
    rng.shuffle(order)
    reserved = round(len(order) * COMBINATION_TEST_FRACTION)
    combination_test = {r["product_id"] for r in order[:reserved]}
    pool = order[reserved:]
    print(f"fixed combination test set: {len(combination_test)} cells, withheld from every level")
    print(f"coverage pool: {len(pool)} cells")
    subsets: dict[str, list[str]] = {}
    for fraction in COVERAGE:
        take = round(len(pool) * fraction)
        subsets[f"{fraction:.2f}"] = [r["product_id"] for r in pool[:take]]
    for a, b in zip(COVERAGE, COVERAGE[1:], strict=False):
        assert set(subsets[f"{a:.2f}"]).issubset(set(subsets[f"{b:.2f}"])), "subsets not nested"
    print("nesting verified across all coverage levels")

    by_id = {r["product_id"]: r for r in rows}
    report: dict[str, dict] = {}
    print(f"\n{'coverage':>9s} {'train':>7s} "
          + " ".join(f"{role[:11]:>12s}" for role in ROLES)
          + f" {'S2':>7s} {'S2fix':>6s} {'S3':>6s} {'S4':>6s}")
    for fraction in COVERAGE:
        key = f"{fraction:.2f}"
        ids = set(subsets[key])
        present = {role: {by_id[i][f"{role}_smiles"] for i in ids} for role in ROLES}
        strata: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            product = row["product_id"]
            if product in ids:
                strata["S1_in_sample"].append(product)
                continue
            if any(row[f"{role}_family_id"] in held_families[role] for role in ROLES):
                strata["S4_held_component_family"].append(product)
            elif any(row[f"{role}_smiles"] in held_identities[role] for role in ROLES):
                strata["S3_held_component_identity"].append(product)
            elif all(row[f"{role}_smiles"] in present[role] for role in ROLES):
                strata["S2_new_combination"].append(product)
                if product in combination_test:
                    strata["S2_fixed_combination_test"].append(product)
            else:
                strata["S5_unseen_component_not_reserved"].append(product)
        report[key] = {
            "training_products": len(ids),
            "component_coverage": {role: len(present[role]) for role in ROLES},
            "family_coverage": {role: len({by_id[i][f"{role}_family_id"] for i in ids})
                                for role in ROLES},
            "strata_sizes": {k: len(v) for k, v in sorted(strata.items())},
        }
        print(f"{key:>9s} {len(ids):7d} "
              + " ".join(f"{len(present[role]):12d}" for role in ROLES)
              + f" {len(strata['S2_new_combination']):7d} "
              f"{len(strata['S2_fixed_combination_test']):6d} "
              f"{len(strata['S3_held_component_identity']):6d} "
              f"{len(strata['S4_held_component_family']):6d}")

    payload = {
        "schema_version": "phase1_forge_coverage_splits.v1",
        "status": "sealed_before_any_model_was_trained",
        "purpose": "does experimentally grounded semantic organization let a generator reuse "
                   "molecular knowledge across an incompletely observed combinatorial product "
                   "space",
        "not_the_purpose": "cross-role coupling, which the corpus audit showed is absent: "
                           "component identities pair near-independently and the "
                           "aldehyde-by-isocyanide grid is fully saturated",
        "inputs": {"assignments": {"path": ASSIGN, "sha256": sha256_file(REPO / ASSIGN)}},
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "elapsed_seconds": round(time.time() - started, 1)},
        "seed": SPLIT_SEED,
        "geometry": {
            "train_products": len(train),
            "train_components": {role: len({r[f"{role}_smiles"] for r in train})
                                 for role in ROLES},
            "triple_grid": 185 * 75 * 35,
            "grid_occupancy": len(train) / (185 * 75 * 35),
            "products_per_triple": 1,
        },
        "reserved_holdouts": {
            "families": {role: sorted(v) for role, v in held_families.items()},
            "identities": {role: sorted(v) for role, v in held_identities.items()},
            "policy": "excluded from every coverage subset, so library extrapolation is "
                      "separable from combination novelty",
        },
        "coverage_levels": list(COVERAGE),
        "fixed_combination_test_set": sorted(combination_test),
        "fixed_combination_test_fraction": COMBINATION_TEST_FRACTION,
        "nested": True,
        "training_product_ids": subsets,
        "per_coverage": report,
        "strata_definitions": {
            "S1_in_sample": "inside the training subset; a fidelity control, not generalization",
            "S2_new_combination": "combination absent from the subset, all three components "
                                  "present in it; grows as coverage falls",
            "S2_fixed_combination_test": "the subset of S2 lying in the fixed combination test "
                                         "set, withheld from every coverage level. This is the "
                                         "primary endpoint for the coverage curve, because it is "
                                         "the same test set at every training size.",
            "S3_held_component_identity": "one component identity reserved out, family retained",
            "S4_held_component_family": "one component family reserved out; the harder shift",
            "S5_unseen_component_not_reserved": "component absent from the subset by coverage "
                                                "reduction rather than by reservation; reported "
                                                "separately so it cannot be read as either S2 or S3",
        },
        "nonclaims": [
            "Coverage reduction withholds combinations; it does not create new chemistry. A "
            "coverage curve measures reuse of observed molecular factors, not extrapolation to "
            "chemistry absent from the corpus.",
            "S1 is in-sample and may never be reported as a generalization result.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {args.output.relative_to(REPO)}")
    print(f"sha256 {sha256_file(args.output)}")


if __name__ == "__main__":
    main()
