#!/usr/bin/env python3
"""Does reaction-factorized acquisition lie outside the simple-mixture frontier?

The first study asked whether the rule dominates both a score-oriented and a
diversity-oriented baseline. It does not: the gate failed 0/5. But that compared the method
against the two endpoints of a trade-off, not against the line joining them. The rule lands
between them, and a reviewer is entitled to ask whether it is an elaborate way of interpolating
between two policies anyone could combine in an afternoon.

This traces that line. Batch positions are allocated between top-mean exploitation and a
learning-oriented policy at exploit fractions 0, 0.25, 0.5, 0.75, 1, on two frontiers
(component diversity and uncertainty), and the frozen reaction-factorized rule is placed in the
same plane. Nothing about that rule changes. The seeds are fresh, because the first study's
seeds are development evidence now.

The protocol helpers are imported from the original benchmark rather than reimplemented, so
both studies split, score and measure identically.

Usage:
    phase1_benchmark_mixture_frontier_v1.py [--seeds N]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from forge.potency.oracle.reaction_factorized_surrogate import (  # noqa: E402
    ReactionFactorizedSurrogate,
    RoleWeights,
)
from experiments.archive.phase1.potency_ranking.synthesis_aware_acquisition import (  # noqa: E402
    AcquisitionPolicy,
    apply_activity_floor,
    select_batch,
    select_mixture,
    select_uncertainty_mixture,
)

CONFIG = REPO / "configs/bio/phase1_forge_mixture_frontier_v1.json"


def _load_benchmark_module():
    """Reuse the original harness's split, metric and loader, so protocols cannot diverge."""
    path = REPO / "scripts/phase1_benchmark_active_design_v1.py"
    spec = importlib.util.spec_from_file_location("forge_ad_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["forge_ad_benchmark"] = module
    spec.loader.exec_module(module)
    return module


BENCH = _load_benchmark_module()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_arm(arm: str, records, seed_indices, pool_indices, cfg, rng) -> dict[str, Any]:
    proto = cfg["unchanged_from_the_failed_study"]["protocol"]
    surrogate_cfg = cfg["unchanged_from_the_failed_study"]["surrogate"]
    acq = cfg["unchanged_from_the_failed_study"]["acquisition"]
    policy = AcquisitionPolicy(
        batch_size=proto["batch_size"],
        budget=proto["budget_per_round"],
        value_weight=acq["value_weight"],
        activity_floor_quantile=acq["activity_floor_quantile"],
        cost_normalised=acq["cost_normalised"],
    )

    def fresh() -> ReactionFactorizedSurrogate:
        return ReactionFactorizedSurrogate(
            prior_variance=surrogate_cfg["prior_variance"],
            noise_variance=surrogate_cfg["noise_variance"],
            weights=RoleWeights(**surrogate_cfg["role_weights"]),
            n_bits=surrogate_cfg["features"]["n_bits"],
            radius=surrogate_cfg["features"]["radius"],
        )

    observed = list(seed_indices)
    remaining = list(pool_indices)
    top_decile = float(np.quantile([v for _, v in records], 0.9))
    history: list[dict[str, Any]] = []

    for round_index in range(1, proto["rounds"] + 1):
        surrogate = fresh().fit(
            [records[i][0].as_record() for i in observed], [records[i][1] for i in observed]
        )
        full_pool = [records[i][0] for i in remaining]
        target = [records[i][0].as_record() for i in remaining]
        if not full_pool:
            break

        pool, _ = apply_activity_floor(surrogate, full_pool, acq["activity_floor_quantile"])
        floored = AcquisitionPolicy(
            batch_size=policy.batch_size,
            budget=policy.budget,
            value_weight=policy.value_weight,
            activity_floor_quantile=0.0,
            cost_normalised=policy.cost_normalised,
        )

        if arm == "reaction_factorized":
            chosen = select_batch(surrogate, pool, target, floored).selected
        elif arm.startswith("mix_div_"):
            chosen = select_mixture(
                surrogate, pool, floored, rng, exploit_fraction=float(arm.split("_")[-1])
            )
        elif arm.startswith("mix_unc_"):
            chosen = select_uncertainty_mixture(
                surrogate, pool, floored, exploit_fraction=float(arm.split("_")[-1])
            )
        else:  # pragma: no cover - guarded by the config
            raise SystemExit(f"unknown arm {arm!r}")

        by_id = {records[i][0].identifier: i for i in remaining}
        picked = [by_id[c.identifier] for c in chosen]
        observed.extend(picked)
        remaining = [i for i in remaining if i not in set(picked)]

        final = fresh().fit(
            [records[i][0].as_record() for i in observed], [records[i][1] for i in observed]
        )
        held = remaining[: min(len(remaining), 400)]
        if held:
            mu, _ = final.predict([records[i][0].as_record() for i in held])
            truth = np.array([records[i][1] for i in held])
            rho = BENCH.spearman(mu, truth)
        else:
            rho = float("nan")
        seen = np.array([records[i][1] for i in observed])
        history.append(
            {
                "round": round_index,
                "selected": [c.identifier for c in chosen],
                "batch": len(chosen),
                "pool_after_floor": len(pool),
                "held_family_spearman": rho,
                "top_decile_hits_found": int((seen >= top_decile).sum()),
                "best_activity_found": float(seen.max()),
            }
        )
    return {"arm": arm, "history": history, "final": history[-1] if history else {}}


def pareto_verdict(means: dict[str, tuple[float, float]]) -> dict[str, Any]:
    """Is reaction_factorized dominated by any simple mixture, as the frozen rule defines it?"""
    rho, hits = means["reaction_factorized"]
    dominators = [
        {"arm": arm, "spearman": r, "hits": h}
        for arm, (r, h) in means.items()
        if arm != "reaction_factorized" and r >= rho and h >= hits
    ]
    return {
        "reaction_factorized": {"spearman": rho, "hits": hits},
        "dominating_mixtures": dominators,
        "dominated": bool(dominators),
        "passed": not dominators,
        "consequence": (
            "Not dominated. The rule occupies a point no simple mixture reaches; a confirmatory "
            "study on fresh splits is required before any claim or second library."
            if not dominators
            else "Dominated by a simple mixture. Stop the acquisition workstream permanently: the "
            "rule adds nothing beyond combining existing policies. No Library 2, no "
            "reformulation, nothing in the paper."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/forge_mixture_frontier_v1/result.json",
    )
    args = parser.parse_args()

    cfg = json.loads(CONFIG.read_text())
    proto = cfg["unchanged_from_the_failed_study"]["protocol"]
    corpus = REPO / proto["corpus"]["path"]
    digest = sha256_file(corpus)
    if digest != proto["corpus"]["sha256"]:
        raise SystemExit(f"corpus hash mismatch: {digest}")
    records = BENCH.load_measured(corpus)
    print(f"loaded {len(records)} measured lipids (sha256 verified)")

    grid = cfg["arms"]["mixture_grid"]
    arms = [f"mix_div_{a}" for a in grid] + [f"mix_unc_{a}" for a in grid] + ["reaction_factorized"]
    seeds = (
        cfg["fresh_randomisation"]["seeds"][: args.seeds]
        if args.seeds
        else cfg["fresh_randomisation"]["seeds"]
    )

    started = time.time()
    results: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        seed_idx, pool_idx = BENCH.family_aware_split(
            records, proto["held_role"], proto["seed_fraction"], rng
        )
        print(f"\nseed {seed}: {len(seed_idx)} seed / {len(pool_idx)} pool")
        results[str(seed)] = {}
        for arm in arms:
            arm_rng = np.random.default_rng(seed + zlib.crc32(arm.encode()) % 10_000)
            out = run_arm(arm, records, seed_idx, pool_idx, cfg, arm_rng)
            results[str(seed)][arm] = out
            f = out["final"]
            print(
                f"  {arm:22s} rho={f['held_family_spearman']:6.3f}  hits={f['top_decile_hits_found']:3d}"
            )

    means = {
        arm: (
            float(np.mean([results[s][arm]["final"]["held_family_spearman"] for s in results])),
            float(np.mean([results[s][arm]["final"]["top_decile_hits_found"] for s in results])),
        )
        for arm in arms
    }
    verdict = pareto_verdict(means)

    payload = {
        "schema_version": "phase1_forge_mixture_frontier_benchmark.v1",
        "status": "complete_simple_mixture_frontier_control",
        "config": {"path": str(CONFIG.relative_to(REPO)), "sha256": sha256_file(CONFIG)},
        "inputs": {"corpus": {"path": proto["corpus"]["path"], "sha256": digest}},
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "elapsed_seconds": round(time.time() - started, 1),
        },
        "seeds": seeds,
        "arm_means": {a: {"spearman": r, "hits": h} for a, (r, h) in means.items()},
        "results": results,
        "verdict": verdict,
        "nonclaims": cfg["nonclaims"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))

    print("\n=== mean over fresh seeds ===")
    print(f"{'arm':22s} {'spearman':>9s} {'hits':>7s}")
    for arm, (r, h) in sorted(means.items(), key=lambda kv: -kv[1][0]):
        mark = "  <- method" if arm == "reaction_factorized" else ""
        print(f"{arm:22s} {r:9.3f} {h:7.1f}{mark}")
    print(f"\ndominated: {verdict['dominated']}")
    if verdict["dominating_mixtures"]:
        for d in verdict["dominating_mixtures"]:
            print(
                f"  dominated by {d['arm']}: rho {d['spearman']:.3f} >= , hits {d['hits']:.1f} >="
            )
    print(verdict["consequence"])
    print(f"\nwrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
