#!/usr/bin/env python3
"""Do regions whose chemistry differs more get more benefit from the semantic map?

Closure section 3, on the measure declared in contract Amendment 14 **before** any association was
examined. No training. One shot. No alternative divergence will be computed if this one is weak.

For each eligible head, the role differentiation is the Jensen-Shannon divergence among the three
role-conditional empirical target distributions on the train fold, uniform mixture reference, base-2
logarithms, so it lies in [0, 1]:

    D_h = JSD( p_h(.|amine), p_h(.|aldehyde), p_h(.|isocyanide) )

Head-to-target mapping is fixed by the amendment. `decoration_anchor_ce` is excluded because its
target is a position rather than a categorical chemical state.

**The power limitation was stated in advance and stands.** Six heads. A rank correlation on six
points is very low-powered and only a near-perfect ordering reaches conventional significance. This
is descriptive and secondary whatever it shows, and may not be featured as a primary result.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from forge.design.training.ugi_training_cache import load_ugi_training_cache  # noqa: E402

CACHE = "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt"
CLOSURE = "results/phase1/forge_semantics_pilot_v1/closure.json"
# Amendment 14 mapping. decoration_anchor_ce excluded: positional target.
HEADS = ("atom_ce", "parent_bond_ce", "offspring_ce",
         "closure_bond_ce", "decoration_atom_ce", "decoration_bond_ce")
PERMUTATIONS = 100000
PERMUTATION_SEED = 20260819


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counter.values() if n)


def jensen_shannon(distributions: list[Counter]) -> float:
    """Uniform-weighted JSD in bits over however many distributions are supplied."""
    present = [d for d in distributions if sum(d.values())]
    if len(present) < 2:
        return 0.0
    weight = 1.0 / len(present)
    mixture: Counter = Counter()
    for d in present:
        total = sum(d.values())
        for key, n in d.items():
            mixture[key] += weight * (n / total)
    mixture_entropy = -sum(p * math.log2(p) for p in mixture.values() if p > 0)
    return mixture_entropy - sum(weight * entropy(d) for d in present)


def collect(records) -> dict[str, list[Counter]]:
    """Role-conditional target counts per head, over the train fold."""
    tables = {head: [Counter(), Counter(), Counter()] for head in HEADS}
    for record in records:
        blocks = np.concatenate([
            np.full(count, role, dtype=np.int64)
            for role, count in enumerate(record.program.node_counts)])
        for position, role in enumerate(blocks.tolist()):
            tables["atom_ce"][role][int(record.atom_states[position])] += 1
            tables["parent_bond_ce"][role][int(record.parent_bond_states[position])] += 1
            tables["offspring_ce"][role][int(record.offspring[position])] += 1
        for index, left in enumerate(record.closure_left.tolist()):
            role = int(blocks[int(left)])
            tables["closure_bond_ce"][role][int(record.closure_bond_states[index])] += 1
        for index, anchor in enumerate(record.decoration_anchors.tolist()):
            role = int(blocks[int(anchor)])
            tables["decoration_atom_ce"][role][int(record.decoration_atom_states[index])] += 1
            tables["decoration_bond_ce"][role][int(record.decoration_bond_states[index])] += 1
    return tables


def spearman(x: list[float], y: list[float]) -> float:
    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = shared
            i = j + 1
        return ranks

    a, b = rank(x), rank(y)
    n = len(a)
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    num = sum((ai - mean_a) * (bi - mean_b) for ai, bi in zip(a, b, strict=True))
    den = math.sqrt(sum((ai - mean_a) ** 2 for ai in a)
                    * sum((bi - mean_b) ** 2 for bi in b))
    return num / den if den else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=20000)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "results/phase1/forge_role_divergence_association_v1/result.json")
    args = parser.parse_args()
    started = time.time()

    corpus, records_by_fold = load_ugi_training_cache(REPO / CACHE)
    records = list(itertools.islice(records_by_fold["train"], args.records))
    tables = collect(records)
    divergence = {head: jensen_shannon(tables[head]) for head in HEADS}

    closure = json.loads((REPO / CLOSURE).read_text())
    per_seed = closure["per_seed"]
    utility = {}
    utility_low_t = {}
    for head in HEADS:
        values = [per_seed[tag]["delta_bar_sem"].get(head) for tag in per_seed]
        low = [per_seed[tag]["delta_sem_by_t"]["0.1"].get(head) for tag in per_seed]
        if any(v is None for v in values):
            continue
        utility[head] = statistics.fmean(values)
        utility_low_t[head] = statistics.fmean(low)

    heads = [h for h in HEADS if h in utility]
    print(f"role differentiation and semantic utility, {len(records)} train-fold records, "
          f"{len(heads)} heads")
    print(f"  {'head':22s} {'D_h (JSD bits)':>15s} {'mean delta_bar':>16s} {'delta at t=0.10':>17s}")
    for head in sorted(heads, key=lambda h: -divergence[h]):
        print(f"  {head:22s} {divergence[head]:15.4f} {utility[head]:+16.5f} "
              f"{utility_low_t[head]:+17.5f}")

    association = {}
    rng = np.random.default_rng(PERMUTATION_SEED)
    for label, target in (("delta_bar_sem", utility), ("delta_sem_at_t_0.10", utility_low_t)):
        x = [divergence[h] for h in heads]
        y = [target[h] for h in heads]
        rho = spearman(x, y)
        # Exact-ish permutation over head labels; 6! = 720 orderings, so sample generously.
        draws = 0
        for _ in range(PERMUTATIONS):
            if abs(spearman(x, list(rng.permutation(y)))) >= abs(rho):
                draws += 1
        association[label] = {
            "spearman_rho": rho,
            "permutation_p_two_sided": (draws + 1) / (PERMUTATIONS + 1),
            "heads": heads,
            "n": len(heads),
        }
        print(f"\n  {label}: Spearman rho {rho:+.4f}, permutation p "
              f"{(draws + 1) / (PERMUTATIONS + 1):.4f} over {len(heads)} heads")

    verdict = (
        "Descriptive and secondary by prior commitment. With six heads only a near-perfect ordering "
        "could reach conventional significance, so this analysis is reported for mechanism and is "
        "not featured as a primary result regardless of what it shows. No alternative divergence "
        "was computed."
    )
    print(f"\n{verdict}")

    payload = {
        "schema_version": "phase1_forge_role_divergence_association.v1",
        "status": "complete_one_shot_descriptive",
        "contract": "docs/FORGE_EVIDENCE_CONTRACT_v1.md Amendment 14",
        "measure": "Jensen-Shannon divergence among the three role-conditional empirical target "
                   "distributions, uniform mixture reference, base-2 logarithms, declared before "
                   "computation",
        "excluded_head": {"decoration_anchor_ce": "positional target, not a categorical chemical "
                                                  "state, so not comparable"},
        "inputs": {"training_cache": {"path": CACHE, "sha256": sha256_file(REPO / CACHE)},
                   "closure": {"path": CLOSURE, "sha256": sha256_file(REPO / CLOSURE)}},
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "elapsed_seconds": round(time.time() - started, 1)},
        "records_used": len(records),
        "permutations": PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "role_divergence_bits": divergence,
        "semantic_utility_mean_over_seeds": utility,
        "semantic_utility_at_low_t": utility_low_t,
        "association": association,
        "verdict": verdict,
        "nonclaims": [
            "Six heads cannot support an inferential claim. This is mechanism, not statistics.",
            "A weak association is recorded as such and is not re-measured with another metric.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
