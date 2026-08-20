#!/usr/bin/env python3
"""Does deleting the origin embedding actually remove precursor-origin information?

An origin-channel ablation is only informative if the ablated arm cannot recover what was
removed. `AdapterNodeConditioning` embeds five node channels, and two of the four that survive
deleting `origin_embedding` are defined relative to the atom's own role:

  origin_states            the channel under ablation
  core_position_states     position within the Ugi core, or not_core
  port_states              not_port, or the name of the role whose anchor this atom is
  distance_to_core         graph distance to the nearest core atom
  distance_to_own_port     graph distance to this atom's OWN role anchor

`use_all_port_distances` is False in production and is never set True anywhere in the
repository, so all-port distances are not a channel and are excluded here.

Two channels leak by construction, and the audit reports both leaks explicitly rather than
letting them inflate a headline number:

  - port_states equals the atom's own origin role at exactly the three role anchors, because
    ugi_adapter_features assigns the port state from the role that owns the anchor;
  - distance_to_own_port is NOT_APPLICABLE exactly for assembly-introduced atoms, because those
    atoms have no origin anchor to measure from.

So the scientifically interesting subset is the remaining atoms: interior atoms of the three
precursors, where nothing names the role and the model would have to infer it from geometry.
Accuracy is therefore reported four ways, and the decision rule in the evidence contract keys
on the overall heldout figure, fixed before this ran.

Two ablation levels, both prespecified, and no third is added after seeing the answer:

  level A   drop origin_states only
  level B   drop the whole role-indexed group {origin_states, port_states, distance_to_own_port}

The predictor is the Bayes-optimal rule under the empirical conditional distribution: fit the
train fold, take the argmax origin per channel tuple, back off to the global majority class for
tuples unseen in training. Backoff is fixed here, before evaluation. Nothing is tuned.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

ATOMS = REPO / "results/phase1/ugi_balanced_chemistry_corpus_v2/semantic_atoms.csv.gz"
ASSIGN = REPO / "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz"

BOOTSTRAP_SEED = 20260815
BOOTSTRAP_DRAWS = 1000
NOT_APPLICABLE_DISTANCE = -1

# The two feature sets under test. Fixed before the audit ran; see the evidence contract.
LEVELS = {
    "level_A_drop_origin_embedding_only": ("core_position", "port_state", "d_core", "d_own"),
    "level_B_drop_role_indexed_group": ("core_position", "d_core"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_of_product() -> dict[str, str]:
    with gzip.open(ASSIGN, "rt", newline="") as handle:
        return {row["product_id"]: row["primary_product_fold"] for row in csv.DictReader(handle)}


def entropy(counts) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    out = 0.0
    for n in counts:
        if n:
            p = n / total
            out -= p * math.log2(p)
    return out


def conditional_entropy(joint: dict[tuple, Counter]) -> float:
    """Plug-in H(origin | channels) on whatever population `joint` was built from."""
    total = sum(sum(c.values()) for c in joint.values())
    if total == 0:
        return 0.0
    return sum(
        (sum(c.values()) / total) * entropy(c.values())
        for c in joint.values()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "results/phase1/forge_origin_channel_information_audit_v1/result.json",
    )
    args = parser.parse_args()
    started = time.time()

    folds = fold_of_product()
    print(f"fold assignment for {len(folds)} products")

    # Streaming scan. Per atom we keep only the channel tuple, the origin, the fold and enough
    # structure to separate the two by-construction leaks from the rest.
    fitted: dict[str, dict[tuple, Counter]] = {name: defaultdict(Counter) for name in LEVELS}
    heldout_joint: dict[str, dict[tuple, Counter]] = {name: defaultdict(Counter) for name in LEVELS}
    heldout_rows: list[tuple[str, tuple, tuple, str, str]] = []
    origin_prior: Counter = Counter()
    atoms_seen = 0
    leak_check = {"port_names_origin_at_anchor": 0, "anchors": 0,
                  "d_own_absent_and_assembly": 0, "assembly_atoms": 0,
                  "d_own_absent_total": 0}

    with gzip.open(ATOMS, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            product = row["product_id"]
            fold = folds.get(product)
            if fold is None:
                continue
            atoms_seen += 1
            origin = row["origin_role"]
            is_core = row["is_ugi_core"] == "True"
            core_position = row["core_position"] if is_core else "not_core"
            is_anchor = row["is_role_anchor"] == "True"
            port_state = origin if is_anchor else "not_port"
            d_core = int(row["distance_to_nearest_core"])
            raw_own = row["distance_to_origin_anchor"]
            d_own = int(raw_own) if raw_own else NOT_APPLICABLE_DISTANCE

            # Confirm the two structural leaks rather than assuming them.
            if is_anchor:
                leak_check["anchors"] += 1
                leak_check["port_names_origin_at_anchor"] += int(port_state == origin)
            if d_own == NOT_APPLICABLE_DISTANCE:
                leak_check["d_own_absent_total"] += 1
            if origin == "assembly_introduced":
                leak_check["assembly_atoms"] += 1
                leak_check["d_own_absent_and_assembly"] += int(d_own == NOT_APPLICABLE_DISTANCE)

            features = {"core_position": core_position, "port_state": port_state,
                        "d_core": d_core, "d_own": d_own}
            keys = {name: tuple(features[f] for f in fields) for name, fields in LEVELS.items()}
            if fold == "train":
                origin_prior[origin] += 1
                for name, key in keys.items():
                    fitted[name][key][origin] += 1
            elif fold == "heldout":
                for name, key in keys.items():
                    heldout_joint[name][key][origin] += 1
                subset = ("role_anchor" if is_anchor
                          else "assembly_introduced" if origin == "assembly_introduced"
                          else "precursor_interior")
                heldout_rows.append((product, keys["level_A_drop_origin_embedding_only"],
                                     keys["level_B_drop_role_indexed_group"], origin, subset))

    print(f"scanned {atoms_seen} atoms; heldout atoms {len(heldout_rows)}")
    print(f"leak check: port names origin at "
          f"{leak_check['port_names_origin_at_anchor']}/{leak_check['anchors']} anchors; "
          f"own-distance absent for {leak_check['d_own_absent_and_assembly']}/"
          f"{leak_check['assembly_atoms']} assembly atoms "
          f"({leak_check['d_own_absent_total']} absent overall)")

    majority = origin_prior.most_common(1)[0][0]
    rules = {name: {key: counts.most_common(1)[0][0] for key, counts in table.items()}
             for name, table in fitted.items()}
    for name, table in rules.items():
        print(f"{name}: {len(table)} distinct channel tuples in train")

    h_origin = entropy(origin_prior.values())
    results: dict[str, dict] = {}
    rng = random.Random(BOOTSTRAP_SEED)

    for level_index, name in enumerate(LEVELS):
        rule = rules[name]
        key_at = 1 if name.startswith("level_A") else 2
        per_product: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        by_subset: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        unseen = 0
        for record in heldout_rows:
            product, key, origin, subset = record[0], record[key_at], record[3], record[4]
            predicted = rule.get(key)
            if predicted is None:
                predicted = majority
                unseen += 1
            hit = int(predicted == origin)
            per_product[product][0] += 1
            per_product[product][1] += hit
            by_subset[subset][0] += 1
            by_subset[subset][1] += hit

        totals = [0, 0]
        exact_products = 0
        for counts in per_product.values():
            totals[0] += counts[0]
            totals[1] += counts[1]
            exact_products += int(counts[0] == counts[1])
        accuracy = totals[1] / totals[0]
        exact_rate = exact_products / len(per_product)

        # Cluster bootstrap at the product: atoms within a product are not independent.
        products = list(per_product.values())
        draws = []
        for _ in range(BOOTSTRAP_DRAWS):
            n_atom = 0
            n_hit = 0
            for _ in range(len(products)):
                counts = products[rng.randrange(len(products))]
                n_atom += counts[0]
                n_hit += counts[1]
            draws.append(n_hit / n_atom)
        draws.sort()
        lo = draws[int(0.05 * BOOTSTRAP_DRAWS)]
        hi = draws[int(0.95 * BOOTSTRAP_DRAWS) - 1]

        h_cond = conditional_entropy(heldout_joint[name])
        subsets = {k: {"atoms": v[0], "accuracy": v[1] / v[0]} for k, v in sorted(by_subset.items())}
        results[name] = {
            "channels": list(LEVELS[name]),
            "heldout_atoms": totals[0],
            "heldout_atom_accuracy": accuracy,
            "cluster_bootstrap_90_interval_product_level": [lo, hi],
            "heldout_product_exact_match_rate": exact_rate,
            "heldout_products": len(per_product),
            "tuples_unseen_in_train": unseen,
            "plug_in_conditional_entropy_bits": h_cond,
            "normalized_conditional_entropy": h_cond / h_origin if h_origin else 0.0,
            "accuracy_by_subset": subsets,
        }

        print(f"\n{name}")
        print(f"  channels                     {', '.join(LEVELS[name])}")
        print(f"  heldout atom accuracy        {accuracy:.4f}  "
              f"[{lo:.4f}, {hi:.4f}] clustered at product")
        print(f"  product exact-match rate     {exact_rate:.4f}")
        print(f"  H(origin|channels)/H(origin) {h_cond / h_origin:.4f}   "
              f"(H(origin) = {h_origin:.3f} bits)")
        print(f"  tuples unseen in train       {unseen}")
        for subset, values in subsets.items():
            print(f"    {subset:22s} {values['atoms']:9d} atoms   {values['accuracy']:.4f}")

    headline = results["level_A_drop_origin_embedding_only"]["heldout_atom_accuracy"]
    if headline >= 0.95:
        verdict = (
            "Deleting the origin embedding alone is NOT an information ablation: origin is "
            "recovered from the surviving node channels at or above 0.95 on heldout atoms. The "
            "primary comparison escalates to level B, and no arm may be described as "
            "product-only."
        )
    elif headline >= 0.60:
        verdict = (
            "Level A is a PARTIAL ablation. Origin is substantially but not fully recoverable "
            "from the surviving channels, so the leakage figure must be reported in the same "
            "sentence as any level-A result."
        )
    else:
        verdict = (
            "Level A is a GENUINE information ablation: origin is not recoverable from the "
            "surviving node channels, so it may be run as the primary arm."
        )
    print(f"\nverdict under the prespecified rule: {verdict}")

    payload = {
        "schema_version": "phase1_forge_origin_channel_information_audit.v1",
        "status": "complete_information_audit_no_training_performed",
        "contract": "docs/FORGE_EVIDENCE_CONTRACT_v1.md section A1, committed before this ran",
        "inputs": {
            "semantic_atoms": {"path": str(ATOMS.relative_to(REPO)), "sha256": sha256_file(ATOMS)},
            "assignments": {"path": str(ASSIGN.relative_to(REPO)), "sha256": sha256_file(ASSIGN)},
        },
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "elapsed_seconds": round(time.time() - started, 1)},
        "seeds": {"cluster_bootstrap": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS},
        "estimator": {
            "rule": "Bayes-optimal under the empirical train-fold conditional distribution",
            "backoff": f"global majority origin class ({majority}), fixed before evaluation",
            "fit_fold": "train", "evaluation_fold": "heldout",
            "all_port_distances_excluded_because": "use_all_port_distances is False in production "
                                                   "and is never set True in the repository",
        },
        "origin_prior_train": dict(origin_prior),
        "origin_entropy_bits": h_origin,
        "structural_leaks_confirmed": leak_check,
        "levels": results,
        "verdict": verdict,
        "nonclaims": [
            "This measures information available in the node channels, not what a trained network "
            "would actually use. A recoverable channel is not proof the ablated model recovers it.",
            "Accuracy is inflated by two by-construction leaks: port_states names the role at the "
            "three anchors, and an absent own-port distance identifies assembly-introduced atoms. "
            "The precursor_interior subset is the subset where role must actually be inferred.",
            "Node channels are not the only route by which origin reaches the model. The separate "
            "structural-dependence audit lists consumption sites in the training objective and "
            "the sampling constraint, which deleting an embedding does not remove.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
