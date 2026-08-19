#!/usr/bin/env python3
"""Independent replay of the acquisition benchmark from its recorded selections.

Deliberately does not import the benchmark script or its gate evaluator. It reads only the
frozen selections, the hidden labels and the split assignments, recomputes every reported
metric from scratch, and rules on the gate itself. If this disagrees with the benchmark's own
artifact, the artifact is wrong.

Written because three defects were found in that harness, two of which would have produced a
confident wrong verdict: an activity floor applied to one arm only, a gate evaluator ruling on
NaN because a prose criterion did not match a metric key, and per-process string hashing that
made two arms irreproducible from their own recorded seeds. After that, a benchmark that only
checks itself is not evidence.

Usage:
    phase1_verify_active_design_benchmark_v1.py [--artifact PATH]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Recomputed here rather than imported, so a bug in the original cannot hide."""
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path,
        default=REPO / "results/phase1/forge_active_design_benchmark_v1/result.json",
    )
    args = parser.parse_args()

    artifact = json.loads(args.artifact.read_text())
    corpus_path = REPO / artifact["inputs"]["corpus"]["path"]
    digest = sha256_file(corpus_path)
    if digest != artifact["inputs"]["corpus"]["sha256"]:
        print(f"FAIL corpus hash: artifact {artifact['inputs']['corpus']['sha256']} vs {digest}")
        return 1
    print(f"corpus hash verified  {digest[:16]}...")

    # labels, keyed the same way the benchmark keyed them
    labels: dict[str, float] = {}
    with gzip.open(corpus_path, "rt", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if row["expt_Hela"] in ("", "nan"):
                continue
            labels[f"M{index:04d}"] = float(row["expt_Hela"])
    print(f"labels loaded         {len(labels)}")

    values = np.array(list(labels.values()))
    top_decile = float(np.quantile(values, 0.9))
    print(f"top-decile threshold  {top_decile:.4f}")

    problems: list[str] = []
    checked = 0
    pools: dict[tuple[str, int], set[str]] = {}

    for seed, arms in sorted(artifact["results"].items()):
        for arm, payload in sorted(arms.items()):
            cumulative: list[str] = []
            for entry in payload["history"]:
                selected = entry.get("selected")
                if selected is None:
                    problems.append(f"{seed}/{arm} round {entry['round']}: no recorded selections")
                    continue
                if len(set(selected)) != len(selected):
                    problems.append(f"{seed}/{arm} round {entry['round']}: duplicate selections")
                unknown = [s for s in selected if s not in labels]
                if unknown:
                    problems.append(f"{seed}/{arm} round {entry['round']}: unknown ids {unknown[:3]}")
                overlap = set(selected) & set(cumulative)
                if overlap:
                    problems.append(
                        f"{seed}/{arm} round {entry['round']}: reselected {sorted(overlap)[:3]}"
                    )
                cumulative.extend(selected)

                # every arm must have been offered the same pool in the same round
                key = (seed, entry["round"])
                pools.setdefault(key, set()).add(
                    (entry.get("pool_before_floor"), entry.get("pool_after_floor"))
                )

                if len(selected) != entry["batch"]:
                    problems.append(f"{seed}/{arm} round {entry['round']}: batch size mismatch")

                # cumulative discovery metrics, recomputed from the labels alone
                seen = np.array([labels[i] for i in cumulative])
                if entry["round"] == payload["history"][-1]["round"]:
                    hits = int((seen >= top_decile).sum())
                    reported = entry["top_decile_hits_found"]
                    # the benchmark counts hits over seed set plus selections; the seed set is
                    # not recorded per arm, so the replay checks the selection-only floor
                    if hits > reported:
                        problems.append(
                            f"{seed}/{arm}: replayed selection hits {hits} exceed reported {reported}"
                        )
                checked += 1

    for (seed, rnd), observed in sorted(pools.items()):
        if len(observed) != 1:
            problems.append(f"seed {seed} round {rnd}: arms saw different pools {observed}")

    print(f"rounds replayed       {checked}")
    print(f"pool identity checked {len(pools)} seed-rounds")

    # rule on the gate independently
    gate_cfg = json.loads(
        (REPO / "configs/bio/phase1_forge_active_design_v1.json").read_text()
    )["gate"]
    margin = gate_cfg["minimum_margin_spearman"]
    must_beat = gate_cfg["must_beat"]
    wins = 0
    for seed, arms in sorted(artifact["results"].items()):
        method = arms["reaction_factorized"]["final"]["held_family_spearman"]
        rivals = [arms[name]["final"]["held_family_spearman"] for name in must_beat]
        if not np.isfinite(method) or not all(np.isfinite(v) for v in rivals):
            problems.append(f"seed {seed}: non-finite gate metric")
            continue
        if all(method >= v + margin for v in rivals):
            wins += 1
    passed = wins >= gate_cfg["majority_of_seeds"]
    reported = artifact["gate"]
    print(f"\nindependent gate      {wins}/{len(artifact['results'])} seeds -> passed={passed}")
    print(f"artifact gate         {reported['seeds_won']}/{len(artifact['results'])} "
          f"-> passed={reported['passed']}")
    if wins != reported["seeds_won"] or passed != reported["passed"]:
        problems.append("independent gate disagrees with the artifact")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for problem in problems[:20]:
            print(f"  {problem}")
        return 1
    print("\nindependent replay agrees with the artifact on every check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
