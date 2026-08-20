#!/usr/bin/env python3
"""What does "disjoint" mean for a Stage 2 stratum, and how many families carry it?

Two audits the Stage 2 headline needs before it can be phrased precisely.

**Anatomy.** A complete program `c = (c_a, c_d, c_i)` can be absent from a reference set while
every role subprogram in it was seen individually, just never in that combination. That is
compositional generalization over familiar role programs, which is a different and weaker claim
than extrapolation to role-level values outside training support. The two are separated here
rather than merged into one word.

**Reference correction.** The strata were defined against the train-family subset of the heldout
product fold, because that is the comparison Stage 1 needed. That is *not* the same as "unseen by
the model": the development checkpoint trained on the 66,464-product train fold, whose programs
are a different set. This audit reports both references, so "disjoint" is never allowed to imply
"never seen during fitting" unless the train-fold check says so.

**Per family.** Stratum C draws 1,024 programs from only six held source families. Rates are
reported per family with a leave-one-family-out range, because the last four weeks established
that thousands of product rows are not thousands of independent chemical regimes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from forge.design.training.ugi_training_cache import load_ugi_training_cache  # noqa: E402

CACHE = "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt"
ROLE = "oxoester_aldehyde_body_tail"
ROLES = ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tuple_key(program) -> tuple:
    return (tuple(program["node_counts"]), tuple(program["junction_budgets"]),
            tuple(program["cycle_ranks"]), tuple(program["attachment_counts"]))


def record_key(program) -> tuple:
    return (tuple(program.node_counts), tuple(program.junction_budgets),
            tuple(program.cycle_ranks), tuple(program.attachment_counts))


def role_keys(key: tuple) -> list[tuple]:
    """One subprogram per role: its own counts, budget, cycle rank and attachments."""
    return [(index, key[0][index], key[1][index], key[2][index], key[3][index])
            for index in range(3)]


def flat(key: tuple) -> list[int]:
    return [value for group in key for value in group]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()

    corpus, records_by_fold = load_ugi_training_cache(REPO / CACHE)

    # Reference 1: programs the model actually trained on.
    train_keys = {record_key(r.program) for r in records_by_fold["train"]}
    train_role_keys: set[tuple] = set()
    for key in train_keys:
        train_role_keys.update(role_keys(key))

    # Reference 2: the train-family subset of the heldout fold, which is how the strata were cut.
    heldout_records = records_by_fold["heldout"]
    heldout_assignments = corpus.assignments_by_fold["heldout"]
    stratum_reference = {record_key(r.program) for r, a
                         in zip(heldout_records, heldout_assignments, strict=True)
                         if a[f"{ROLE}_family_fold"] == "train"}
    print(f"train-fold programs {len(train_keys)}; "
          f"train-family heldout programs {len(stratum_reference)}; "
          f"train-fold role subprograms {len(train_role_keys)}")

    draw = json.loads(args.draw.read_text())
    rows = draw["samples"]
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_stratum[row["source_stratum"]].append(row)

    train_list = [flat(k) for k in train_keys]
    anatomy: dict[str, dict] = {}
    for stratum, members in sorted(by_stratum.items()):
        seen_in_train_fold = 0
        seen_in_stratum_reference = 0
        unseen_role_counts = Counter()
        distances = []
        for row in members:
            key = tuple_key(row["program"])
            seen_in_train_fold += int(key in train_keys)
            seen_in_stratum_reference += int(key in stratum_reference)
            unseen = sum(1 for sub in role_keys(key) if sub not in train_role_keys)
            unseen_role_counts[unseen] += 1
        # Nearest train-fold program in L1 over the twelve program coordinates, on a bounded
        # sample because the reference set is large and this is descriptive.
        for row in members[:128]:
            vector = flat(tuple_key(row["program"]))
            distances.append(min(sum(abs(a - b) for a, b in zip(vector, other, strict=True))
                                 for other in train_list))
        anatomy[stratum] = {
            "programs": len(members),
            "complete_tuple_present_in_train_fold": seen_in_train_fold,
            "complete_tuple_present_in_train_fold_rate": seen_in_train_fold / len(members),
            "complete_tuple_present_in_stratum_reference": seen_in_stratum_reference,
            "roles_with_a_subprogram_unseen_in_train_fold": dict(sorted(unseen_role_counts.items())),
            "median_l1_distance_to_nearest_train_fold_program": statistics.median(distances),
            "distance_sample": len(distances),
        }

    print(f"\n{'stratum':28s} {'n':>5s} {'tuple in train fold':>20s} {'roles unseen':>28s} "
          f"{'median L1':>10s}")
    for stratum, values in anatomy.items():
        print(f"{stratum:28s} {values['programs']:5d} "
              f"{values['complete_tuple_present_in_train_fold_rate']:20.3f} "
              f"{str(values['roles_with_a_subprogram_unseen_in_train_fold']):>28s} "
              f"{values['median_l1_distance_to_nearest_train_fold_program']:10.1f}")

    # ---- per-family rates within each stratum
    samples = json.loads(args.samples.read_text())["samples"]
    family_of = {row["product_id"]: row["aldehyde_family_id"] for row in rows}
    stratum_of = {row["product_id"]: row["source_stratum"] for row in rows}
    per_family: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in samples:
        product_id = row.get("product_id")
        family = family_of.get(product_id)
        if family is None:
            continue
        stratum = stratum_of[product_id]
        admitted = bool((row.get("l1_forward_verification") or {}).get(
            "exact_product_reconstructed"))
        per_family[stratum][family].append(int(admitted))

    family_report: dict[str, dict] = {}
    for stratum, families in sorted(per_family.items()):
        rates = {family: sum(values) / len(values) for family, values in sorted(families.items())}
        counts = {family: len(values) for family, values in sorted(families.items())}
        pooled = sum(sum(v) for v in families.values()) / sum(len(v) for v in families.values())
        loo = {}
        for excluded in families:
            kept = [v for f, values in families.items() if f != excluded for v in values]
            if kept:
                loo[excluded] = sum(kept) / len(kept)
        family_report[stratum] = {
            "families": len(rates), "pooled_admission": pooled,
            "per_family_admission": rates, "per_family_programs": counts,
            "leave_one_family_out_range": [min(loo.values()), max(loo.values())] if loo else None,
            "per_family_range": [min(rates.values()), max(rates.values())] if rates else None,
        }
        print(f"\n{stratum}: pooled admission {pooled:.4f} over {len(rates)} source families")
        for family, rate in sorted(rates.items(), key=lambda kv: -counts[kv[0]]):
            print(f"    {family:50s} n={counts[family]:4d}  {rate:.4f}")
        if loo:
            print(f"    leave-one-family-out range "
                  f"[{min(loo.values()):.4f}, {max(loo.values()):.4f}]")

    payload = {
        "schema_version": "phase1_forge_program_disjointness_anatomy.v1",
        "status": "complete_anatomy_and_per_family_stratification",
        "inputs": {
            "draw": {"path": str(args.draw), "sha256": sha256_file(args.draw)},
            "samples": {"path": str(args.samples), "sha256": sha256_file(args.samples)},
        },
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "elapsed_seconds": round(time.time() - started, 1)},
        "references": {
            "train_fold_programs": len(train_keys),
            "train_fold_role_subprograms": len(train_role_keys),
            "train_family_heldout_programs": len(stratum_reference),
            "why_two": "the strata were cut against the train-family subset of the heldout fold, "
                       "which is not the set the model trained on. Disjointness from the former "
                       "does not imply the program was never seen during fitting.",
        },
        "anatomy": anatomy,
        "per_family": family_report,
        "nonclaims": [
            "A program absent as a complete tuple whose role subprograms were all seen "
            "individually shows composition of familiar role programs, not extrapolation to "
            "unseen role-level values.",
            "Per-family rates are reported because stratum C draws 1,024 programs from six "
            "source families; the pooled rate has six independent chemical regimes behind it, "
            "not 1,024.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
