#!/usr/bin/env python3
"""Retrospective acquisition benchmark on the 1,100 measured AGILE lipids.

Simulates sequential experimental design under component-family-aware splits: start from a
seed set, select a batch from the remaining pool under a fixed budget, reveal its labels,
refit, repeat. Every arm sees the same pool, the same budget and the same batch size, so the
comparison isolates the selection rule.

The gate in configs/bio/phase1_forge_active_design_v1.json was frozen before this ran. If the
reaction-factorized arm does not clear it, that is the result, and no second library is
synthesised.

Usage:
    phase1_benchmark_active_design_v1.py [--seeds N] [--output PATH]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import sys
import time
import zlib
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from forge.potency.reaction_factorized_surrogate import (  # noqa: E402
    ReactionFactorizedSurrogate,
    RoleWeights,
)
from forge.potency.synthesis_aware_acquisition import (  # noqa: E402
    AcquisitionPolicy,
    Candidate,
    apply_activity_floor,
    select_batch,
    select_component_diversity,
    select_random,
    select_top_lower_bound,
    select_top_mean,
    select_uncertainty,
)

CONFIG = REPO / "configs/bio/phase1_forge_active_design_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_measured(path: Path) -> list[Candidate]:
    """The 1,100 curated single-compound AGILE lipids, with their measured HeLa endpoint."""
    from rdkit import Chem, rdBase

    def canon(smiles: str) -> str:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else ""

    rows = []
    with gzip.open(path, "rt", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if row["expt_Hela"] in ("", "nan"):
                continue
            rows.append(
                (
                    Candidate(
                        identifier=f"M{index:04d}",
                        product=canon(row["model_smiles"]),
                        amine=canon(row["A_smiles"]),
                        aldehyde=canon(row["B_smiles"]),
                        isocyanide=canon(row["C_smiles"]),
                        # cost stands in for synthetic burden; the measured library has no
                        # route dossier, so every record costs one unit and the budget binds
                        # only through batch size. Library 2 will carry real preparation counts.
                        cost=1.0,
                    ),
                    float(row["expt_Hela"]),
                )
            )
    return rows


def family_aware_split(
    records: list[tuple[Candidate, float]],
    held_role: str,
    seed_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Seed and pool indices split so a held role identity never spans both.

    A random split lets a tail in the seed set reappear in the pool, which flatters every arm
    equally and hides exactly the differences the benchmark exists to measure.
    """
    families: dict[str, list[int]] = defaultdict(list)
    for index, (candidate, _) in enumerate(records):
        families[getattr(candidate, held_role)].append(index)
    keys = sorted(families)
    rng.shuffle(keys)
    seed_target = int(round(seed_fraction * len(records)))
    seed_indices: list[int] = []
    for key in keys:
        if len(seed_indices) >= seed_target:
            break
        seed_indices.extend(families[key])
    seed_set = set(seed_indices)
    pool = [i for i in range(len(records)) if i not in seed_set]
    return sorted(seed_set), pool


def role_coverage(candidates: list[Candidate]) -> dict[str, int]:
    return {
        "amine": len({c.amine for c in candidates}),
        "aldehyde": len({c.aldehyde for c in candidates}),
        "isocyanide": len({c.isocyanide for c in candidates}),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def run_arm(
    arm: str,
    records: list[tuple[Candidate, float]],
    seed_indices: list[int],
    pool_indices: list[int],
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    bench = config["retrospective_benchmark"]
    surrogate_cfg = config["surrogate"]
    acq = config["acquisition"]
    policy = AcquisitionPolicy(
        batch_size=bench["batch_size"],
        budget=bench["budget_per_round"],
        value_weight=acq["value_weight"],
        activity_floor_quantile=acq["activity_floor_quantile"],
        cost_normalised=acq["cost_normalised"],
    )

    def fresh_surrogate() -> ReactionFactorizedSurrogate:
        return ReactionFactorizedSurrogate(
            prior_variance=surrogate_cfg["prior_variance"],
            noise_variance=surrogate_cfg["noise_variance"],
            weights=RoleWeights(**surrogate_cfg["role_weights"]),
            n_bits=surrogate_cfg["features"]["n_bits"],
            radius=surrogate_cfg["features"]["radius"],
        )

    observed = list(seed_indices)
    remaining = list(pool_indices)
    all_values = np.array([value for _, value in records])
    top_decile = float(np.quantile(all_values, 0.9))
    history: list[dict[str, Any]] = []

    for round_index in range(1, bench["rounds"] + 1):
        surrogate = fresh_surrogate().fit(
            [records[i][0].as_record() for i in observed],
            [records[i][1] for i in observed],
        )
        full_pool = [records[i][0] for i in remaining]
        target = [records[i][0].as_record() for i in remaining]
        if not full_pool:
            break

        # One floor, applied once, identical for every arm. Applying it inside a single
        # selector would confine that arm to the top of the predicted distribution while its
        # baselines sampled freely, and on a learning metric that is a handicap rather than an
        # advantage: broad coverage is exactly what improves held-family correlation.
        pool_candidates, floor_value = apply_activity_floor(
            surrogate, full_pool, acq["activity_floor_quantile"]
        )
        floored_policy = AcquisitionPolicy(
            batch_size=policy.batch_size,
            budget=policy.budget,
            value_weight=policy.value_weight,
            activity_floor_quantile=0.0,
            cost_normalised=policy.cost_normalised,
        )

        if arm == "random":
            chosen = select_random(pool_candidates, floored_policy, rng)
        elif arm == "top_mean":
            chosen = select_top_mean(surrogate, pool_candidates, floored_policy)
        elif arm == "top_lower_bound":
            chosen = select_top_lower_bound(surrogate, pool_candidates, floored_policy)
        elif arm == "uncertainty":
            chosen = select_uncertainty(surrogate, pool_candidates, floored_policy)
        elif arm == "component_diversity":
            chosen = select_component_diversity(pool_candidates, floored_policy, rng)
        elif arm == "reaction_factorized":
            chosen = select_batch(surrogate, pool_candidates, target, floored_policy).selected
        else:  # pragma: no cover - guarded by the config
            raise SystemExit(f"unknown arm {arm!r}")

        by_id = {records[i][0].identifier: i for i in remaining}
        picked = [by_id[c.identifier] for c in chosen]
        observed.extend(picked)
        remaining = [i for i in remaining if i not in set(picked)]

        final = fresh_surrogate().fit(
            [records[i][0].as_record() for i in observed],
            [records[i][1] for i in observed],
        )
        held = remaining[: min(len(remaining), 400)]
        if held:
            mu, sd = final.predict([records[i][0].as_record() for i in held])
            truth = np.array([records[i][1] for i in held])
            rmse = float(np.sqrt(np.mean((mu - truth) ** 2)))
            rho = spearman(mu, truth)
            lower = mu - 1.2816 * sd
            coverage = float(np.mean(truth >= lower))
        else:
            rmse = rho = coverage = float("nan")

        seen_values = np.array([records[i][1] for i in observed])
        history.append(
            {
                "round": round_index,
                "measurements": len(observed),
                "batch": len(chosen),
                "selected": [c.identifier for c in chosen],
                "pool_before_floor": len(full_pool),
                "pool_after_floor": len(pool_candidates),
                "activity_floor": floor_value,
                "cost": float(sum(c.cost for c in chosen)),
                "best_activity_found": float(seen_values.max()),
                "top_decile_hits_found": int((seen_values >= top_decile).sum()),
                "held_family_rmse": rmse,
                "held_family_spearman": rho,
                "coverage90": coverage,
                "posterior_variance_over_target": (
                    final.variance_over(target) if target else float("nan")
                ),
                "role_coverage": role_coverage([records[i][0] for i in observed]),
            }
        )

    return {"arm": arm, "history": history, "final": history[-1] if history else {}}


def _metric_key(criterion: str, available: Sequence[str]) -> str:
    """Resolve the config's prose criterion to the metric key the history actually carries.

    The frozen config names the criterion in words ("held_family_spearman after the final
    round"). A plain dict lookup on that string silently returns the default, which made an
    earlier run report a gate verdict computed entirely from NaN. Failing loudly is the only
    acceptable behaviour here: a gate that cannot find its own metric has not been evaluated.
    """
    for key in sorted(available, key=len, reverse=True):
        if criterion.startswith(key):
            return key
    raise SystemExit(
        f"gate criterion {criterion!r} does not name any recorded metric; "
        f"available: {sorted(available)}"
    )


def evaluate_gate(
    results: dict[int, dict[str, dict[str, Any]]], config: dict[str, Any]
) -> dict[str, Any]:
    gate = config["gate"]
    any_final = next(iter(next(iter(results.values())).values()))["final"]
    primary = _metric_key(gate["primary_criterion"], list(any_final))
    secondary = _metric_key(gate["secondary_criterion"], list(any_final))
    margin = gate["minimum_margin_spearman"]
    wins = []
    detail = []
    for seed, arms in sorted(results.items()):
        method = arms["reaction_factorized"]["final"][primary]
        rivals = {name: arms[name]["final"][primary] for name in gate["must_beat"]}
        if not np.isfinite(method) or not all(np.isfinite(v) for v in rivals.values()):
            raise SystemExit(
                f"seed {seed}: gate metric {primary!r} is not finite; refusing to rule"
            )
        beat = all(method >= value + margin for value in rivals.values())
        wins.append(beat)
        detail.append(
            {
                "seed": seed,
                "reaction_factorized": method,
                **rivals,
                "beat_all": beat,
                "secondary_reaction_factorized": arms["reaction_factorized"]["final"][secondary],
                "secondary_best_baseline": max(
                    arms[name]["final"][secondary] for name in arms if name != "reaction_factorized"
                ),
            }
        )
    passed = sum(wins) >= gate["majority_of_seeds"]
    return {
        "criterion": primary,
        "secondary_criterion": secondary,
        "minimum_margin": margin,
        "must_beat": gate["must_beat"],
        "seeds_won": int(sum(wins)),
        "seeds_required": gate["majority_of_seeds"],
        "per_seed": detail,
        "passed": bool(passed),
        "consequence": (
            "Gate passed. Library 2 may be generated under the frozen pool-construction rule."
            if passed
            else "Gate failed. Report the negative result; do not synthesise a second library."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=0, help="limit the number of frozen seeds")
    parser.add_argument("--rounds", type=int, default=0, help="override rounds (smoke runs only)")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/forge_active_design_benchmark_v1/result.json",
    )
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text())
    bench = config["retrospective_benchmark"]
    if args.rounds:
        bench["rounds"] = args.rounds
    corpus = REPO / bench["corpus"]["path"]
    digest = sha256_file(corpus)
    if digest != bench["corpus"]["sha256"]:
        raise SystemExit(
            f"corpus hash mismatch\n  expected {bench['corpus']['sha256']}\n  observed {digest}"
        )

    records = load_measured(corpus)
    if len(records) != bench["corpus"]["records"]:
        raise SystemExit(f"expected {bench['corpus']['records']} records, loaded {len(records)}")
    print(f"loaded {len(records)} measured lipids (sha256 verified)")

    seeds = bench["seeds"][: args.seeds] if args.seeds else bench["seeds"]
    started = time.time()
    results: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        seed_idx, pool_idx = family_aware_split(
            records, bench["held_role"], bench["seed_fraction"], rng
        )
        print(
            f"\nseed {seed}: {len(seed_idx)} seed / {len(pool_idx)} pool "
            f"(held role: {bench['held_role']})"
        )
        results[seed] = {}
        for arm in bench["arms"]:
            # zlib.crc32, not the builtin hash: str hashing is salted per process, so
            # the stochastic baselines drew a different stream on every run and the
            # artifact was not reproducible from its own recorded seeds.
            arm_rng = np.random.default_rng(seed + zlib.crc32(arm.encode()) % 10_000)
            outcome = run_arm(arm, records, seed_idx, pool_idx, config, arm_rng)
            results[seed][arm] = outcome
            final = outcome["final"]
            print(
                f"  {arm:22s} rho={final.get('held_family_spearman', float('nan')):.3f}  "
                f"best={final.get('best_activity_found', float('nan')):.2f}  "
                f"hits={final.get('top_decile_hits_found', 0)}  "
                f"cov90={final.get('coverage90', float('nan')):.3f}"
            )

    gate = evaluate_gate(results, config)
    payload = {
        "schema_version": "phase1_forge_active_design_benchmark.v1",
        "status": "complete_retrospective_acquisition_benchmark",
        "config": {"path": str(CONFIG.relative_to(REPO)), "sha256": sha256_file(CONFIG)},
        "inputs": {"corpus": {"path": bench["corpus"]["path"], "sha256": digest}},
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "elapsed_seconds": round(time.time() - started, 1),
            "numpy": np.__version__,
        },
        "seeds": seeds,
        "results": {
            str(seed): {arm: results[seed][arm] for arm in results[seed]} for seed in results
        },
        "gate": gate,
        "nonclaims": config["nonclaims"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))

    print(
        f"\n=== gate: {'PASSED' if gate['passed'] else 'FAILED'} "
        f"({gate['seeds_won']}/{len(seeds)} seeds, {gate['seeds_required']} required) ==="
    )
    print(gate["consequence"])
    print(f"\nwrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
