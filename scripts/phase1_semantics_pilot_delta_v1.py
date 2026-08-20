#!/usr/bin/env python3
"""Measure Delta_sem(t; alpha) for the semantics pilot, in the frozen reading order.

Contract amendments 11, 12 and 13. The question:

    does supplying the correct experimentally grounded role map reduce denoising loss beyond what the
    same network infers from the corrupted molecule, generic position and the global program?

    Delta_sem(t; alpha) = L_no_role(t; alpha) - L_true_role(t; alpha)

At the Bayes optimum this equals the conditional information the role map supplies, per head, and a
weighted sum of conditional mutual informations for the multi-head objective. **Finite trained
networks give an empirical semantic-utility gap, not an exact mutual information**, and the artifact
says so.

Pairing is exact rather than approximate. The collated batch is identical across flat arms, verified
in the specification suite, so one collate serves both models. Both then see identical examples in
identical order at identical t with identical noise draws. The only difference is which index the
role table is fed.

Both arms are read from `checkpoint_step_3000.pt`, never `checkpoint_best.pt`. Amendment 13: the
configs inherit calibration early stopping, so `best` lands at each arm's own step, and using it
would be per-arm checkpoint shopping that breaks the pairing.

Reading order is frozen and this script follows it:

  1. all four runs completed and losses numerically sane
  2. Delta_sem(t; alpha)
  3. is it positive coherently across the primary heads
  4. does it grow toward the corrupted end, which under q_t = t*delta_x + (1-t)*p_0 is LOW t
  5. Delta_bar_sem(0.25) against Delta_bar_sem(1.00)
  6. nothing exploratory until the above are recorded
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from forge.design.flow.ugi_joint_sparse_flow import (  # noqa: E402
    UgiJointSparseFlow,
    collate_ugi_joint_sparse_records,
    noise_ugi_joint_sparse_batch,
    ugi_joint_sparse_loss,
)
from forge.design.training.ugi_training_cache import load_ugi_training_cache  # noqa: E402

CACHE = "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt"
SPLITS = "results/phase1/forge_coverage_splits_v1/splits.json"
PILOT = "results/phase1/forge_semantics_pilot_v1"
CHECKPOINT = "checkpoint_step_3000.pt"          # Amendment 13
ARMS = ("flat_no_role", "flat_true_role")
ALPHAS = ("0.25", "1.00")
T_GRID = (0.1, 0.25, 0.5, 0.75, 0.9)
STRATA = ("S2_fixed_combination_test", "S3_held_component_identity", "S4_held_component_family")
PRIMARY_HEADS = ("total", "atom_ce", "parent_bond_ce", "offspring_ce",
                 "closure_bond_ce", "decoration_anchor_ce")
BATCH = 128
NOISE_SEED = 20260818
MAX_PER_STRATUM = 2048


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_arm(arm: str, alpha: str, atom_classes: int):
    directory = REPO / PILOT / f"{arm}_a{int(float(alpha) * 100):03d}"
    checkpoint = torch.load(directory / CHECKPOINT, map_location="cpu", weights_only=False)
    if int(checkpoint["step"]) != 3000:
        raise SystemExit(f"{arm} a={alpha} checkpoint is step {checkpoint['step']}, expected 3000")
    architecture = dict(checkpoint["model_config"])
    architecture.pop("source_probability_floor", None)
    model = UgiJointSparseFlow(atom_classes=atom_classes,
                               semantic_organization=arm, **architecture)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    sources = {k: torch.as_tensor(v, dtype=torch.float32)
               for k, v in checkpoint["source_marginals"].items()}
    result = json.loads((directory / "result.json").read_text())
    return model, sources, architecture, checkpoint, result


def evaluate(models, sources, architecture, records, arm_names):
    """One pass, both arms, identical batches and noise. Returns {arm: {t: {head: loss}}}."""
    chunks = [records[i:i + BATCH] for i in range(0, len(records), BATCH)]
    totals = {arm: defaultdict(lambda: defaultdict(float)) for arm in arm_names}
    weights = {arm: defaultdict(lambda: defaultdict(float)) for arm in arm_names}
    skipped = defaultdict(int)
    with torch.no_grad():
        for t_index, t_value in enumerate(T_GRID):
            for batch_index, chunk in enumerate(chunks):
                batch = collate_ugi_joint_sparse_records(
                    tuple(chunk), maximum_nodes=max(r.node_count for r in chunk),
                    maximum_children=int(architecture["maximum_children"]),
                    maximum_closures=int(architecture["maximum_cycle_rank"]) * 3,
                    maximum_decorations=int(architecture["maximum_decorations"]),
                    semantic_organization="flat_no_role")
                t = torch.full((len(chunk),), float(t_value))
                for arm in arm_names:
                    # Same seed per (t, batch) for every arm: identical corruption draws.
                    generator = torch.Generator().manual_seed(
                        NOISE_SEED + 1000 * t_index + batch_index)
                    noisy = noise_ugi_joint_sparse_batch(batch, sources[arm], t, generator)
                    predictions = models[arm](
                        offspring=noisy["offspring"], nodes=noisy["nodes"],
                        parent_bonds=noisy["parent_bonds"], role_states=batch["role_states"],
                        within_role_positions=batch["within_role_positions"],
                        programs=batch["programs"], node_mask=batch["node_mask"], t=t,
                        closure_left=batch["closure_left"], closure_right=batch["closure_right"],
                        decoration_anchors=noisy["decoration_anchors"],
                        decoration_atoms=noisy["decoration_atoms"],
                        decoration_bonds=noisy["decoration_bonds"])
                    _, metrics = ugi_joint_sparse_loss(
                        predictions, batch, semantic_organization=arm)
                    weight = len(chunk)
                    finite = {k: v for k, v in metrics.items()
                              if k != "total" and not np.isnan(v)}
                    for key, value in finite.items():
                        totals[arm][t_value][key] += value * weight
                        weights[arm][t_value][key] += weight
                    for key in metrics:
                        if key != "total" and key not in finite:
                            skipped[f"{arm}:{key}"] += 1
                    totals[arm][t_value]["total"] += sum(finite.values()) * weight
                    weights[arm][t_value]["total"] += weight
    out = {arm: {t: {k: totals[arm][t][k] / weights[arm][t][k]
                     for k in totals[arm][t] if weights[arm][t][k]}
                 for t in totals[arm]} for arm in arm_names}
    return out, dict(skipped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "results/phase1/forge_semantics_pilot_v1/delta_sem.json")
    args = parser.parse_args()
    started = time.time()

    corpus, records_by_fold = load_ugi_training_cache(REPO / CACHE)
    sealed = json.loads((REPO / SPLITS).read_text())
    by_id = {r.product_id: r for fold in records_by_fold.values() for r in fold}

    # ---- step 1: every run completed and is numerically sane
    completion: dict[str, dict] = {}
    loaded: dict[str, tuple] = {}
    for alpha in ALPHAS:
        for arm in ARMS:
            model, sources, architecture, checkpoint, result = load_arm(
                arm, alpha, len(corpus.atom_vocabulary))
            final = result["losses"][-1]
            sane = all(np.isfinite(v) for v in final.values() if isinstance(v, (int, float)))
            completion[f"{arm}_a{alpha}"] = {
                "completed_steps": result["selection"]["completed_steps"],
                "final_training_total": final.get("total"),
                "numerically_sane": bool(sane),
                "checkpoint_step": int(checkpoint["step"]),
                "coverage_alpha": result["semantics_pilot"]["coverage_alpha"],
                "seed": result["semantics_pilot"]["seed"],
                "training_wall_seconds": result["semantics_pilot"]["training_wall_seconds"],
            }
            loaded[f"{arm}_a{alpha}"] = (model, sources, architecture)
            if not sane or int(result["selection"]["completed_steps"]) != 3000:
                raise SystemExit(f"{arm} a={alpha} did not complete sanely")
    print("step 1: all four runs completed at 3000 steps with finite losses")
    for name, record in sorted(completion.items()):
        print(f"  {name:24s} final total {record['final_training_total']:.4f}  "
              f"{record['training_wall_seconds']:.0f}s")

    # ---- steps 2 to 5
    per_alpha: dict[str, dict] = {}
    for alpha in ALPHAS:
        _, _, architecture = loaded[f"{ARMS[0]}_a{alpha}"]
        models = {arm: loaded[f"{arm}_a{alpha}"][0] for arm in ARMS}
        sources = {arm: loaded[f"{arm}_a{alpha}"][1] for arm in ARMS}
        strata_result: dict[str, dict] = {}
        for stratum in STRATA:
            ids = sealed["per_coverage"][alpha]["strata_sizes"]
            available = [by_id[pid] for pid in
                         _stratum_ids(sealed, alpha, stratum) if pid in by_id]
            records = sorted(available, key=lambda r: r.product_id)[:MAX_PER_STRATUM]
            if not records:
                continue
            losses, skipped = evaluate(models, sources, architecture, records, ARMS)
            delta = {t: {head: losses[ARMS[0]][t][head] - losses[ARMS[1]][t][head]
                         for head in losses[ARMS[0]][t]} for t in T_GRID}
            strata_result[stratum] = {
                "records_evaluated": len(records),
                "records_available": ids.get(stratum),
                "loss_by_arm": {arm: {str(t): losses[arm][t] for t in T_GRID} for arm in ARMS},
                "delta_sem_by_t": {str(t): delta[t] for t in T_GRID},
                "delta_bar_sem": {head: statistics.fmean(delta[t][head] for t in T_GRID)
                                  for head in delta[T_GRID[0]]},
                "skipped_metric_batches": skipped,
            }
        per_alpha[alpha] = strata_result

    print(f"\nstep 2 and 3: Delta_sem(t; alpha) on {STRATA[0]}, "
          f"positive means the role map helped")
    print(f"  {'alpha':>6s} {'t':>6s} " + " ".join(f"{h[:12]:>13s}" for h in PRIMARY_HEADS))
    for alpha in ALPHAS:
        block = per_alpha[alpha].get(STRATA[0])
        if not block:
            continue
        for t in T_GRID:
            row = block["delta_sem_by_t"][str(t)]
            print(f"  {alpha:>6s} {t:6.2f} " + " ".join(
                f"{row.get(h, float('nan')):+13.5f}" for h in PRIMARY_HEADS))

    print(f"\nstep 4: does the advantage grow toward LOW t (more corrupted; "
          f"q_t = t*delta_x + (1-t)*p_0)")
    corruption = {}
    for alpha in ALPHAS:
        block = per_alpha[alpha].get(STRATA[0])
        if not block:
            continue
        low = block["delta_sem_by_t"][str(T_GRID[0])].get("total", float("nan"))
        high = block["delta_sem_by_t"][str(T_GRID[-1])].get("total", float("nan"))
        corruption[alpha] = {"delta_at_low_t": low, "delta_at_high_t": high,
                             "grows_toward_corruption": bool(low > high)}
        print(f"  alpha={alpha}: Delta at t={T_GRID[0]} is {low:+.5f}, at t={T_GRID[-1]} is "
              f"{high:+.5f}, grows toward corruption: {low > high}")

    print("\nstep 5: Delta_bar_sem(0.25) against Delta_bar_sem(1.00)")
    efficiency = {}
    for head in PRIMARY_HEADS:
        a = per_alpha["0.25"].get(STRATA[0], {}).get("delta_bar_sem", {}).get(head)
        b = per_alpha["1.00"].get(STRATA[0], {}).get("delta_bar_sem", {}).get(head)
        if a is None or b is None:
            continue
        efficiency[head] = {"alpha_025": a, "alpha_100": b, "sparse_exceeds_full": bool(a > b)}
        print(f"  {head:22s} {a:+.5f} vs {b:+.5f}   sparse > full: {a > b}")

    payload = {
        "schema_version": "phase1_forge_semantics_pilot_delta.v1",
        "status": "complete_preregistered_reading",
        "contract": "docs/FORGE_EVIDENCE_CONTRACT_v1.md amendments 11, 12 and 13",
        "estimand": {
            "definition": "Delta_sem(t; alpha) = L_no_role(t; alpha) - L_true_role(t; alpha)",
            "interpretation": "At the Bayes optimum, the reduction in categorical denoising loss "
                              "equals the conditional information supplied by the semantic role "
                              "map, per head, and a weighted sum of conditional mutual "
                              "informations for the multi-head objective. We therefore measure the "
                              "practical value of this information through the paired "
                              "denoising-loss gap. Finite trained networks give an empirical "
                              "semantic-utility gap, NOT an exact mutual information.",
            "corruption_convention": "q_t = t*delta_x + (1-t)*p_0, so low t is more corrupted",
        },
        "checkpoint_policy": "checkpoint_step_3000.pt for every arm; checkpoint_best is each arm's "
                             "own calibration-selected step and using it would be per-arm "
                             "checkpoint shopping (Amendment 13)",
        "pairing": "identical collated batches across arms, identical t grid, identical noise "
                   "generator seed per (t, batch); only the fed role index differs",
        "inputs": {"training_cache": {"path": CACHE, "sha256": sha256_file(REPO / CACHE)},
                   "sealed_splits": {"path": SPLITS, "sha256": sha256_file(REPO / SPLITS)}},
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(), "torch": torch.__version__,
                    "elapsed_seconds": round(time.time() - started, 1)},
        "protocol": {"t_grid": list(T_GRID), "batch_size": BATCH, "noise_seed": NOISE_SEED,
                     "max_records_per_stratum": MAX_PER_STRATUM,
                     "primary_stratum": STRATA[0]},
        "step_1_completion": completion,
        "per_alpha": per_alpha,
        "step_4_corruption_dependence": corruption,
        "step_5_data_efficiency": efficiency,
        "nonclaims": [
            "A positive gap is an empirical semantic-utility gap between two finite trained "
            "networks, not a measured mutual information.",
            "Nothing here speaks to generated molecular chemistry; no sample was drawn.",
            "The primary comparison is between the two flat arms only, never between either and "
            "the production checkpoint, which differs in layout, sources, objective and capacity.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {args.output.relative_to(REPO)}")


def _stratum_ids(sealed: dict, alpha: str, stratum: str) -> list[str]:
    """Recompute stratum membership from the sealed split, which stores sizes not id lists."""
    keep = set(sealed["training_product_ids"][alpha])
    combination_test = set(sealed["fixed_combination_test_set"])
    held_families = sealed["reserved_holdouts"]["families"]
    held_identities = sealed["reserved_holdouts"]["identities"]
    import csv
    import gzip
    path = REPO / "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz"
    roles = ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")
    present = {role: set() for role in roles}
    rows = []
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            if row["product_id"] in keep:
                for role in roles:
                    present[role].add(row[f"{role}_smiles"])
    out = []
    for row in rows:
        product = row["product_id"]
        if product in keep:
            continue
        if any(row[f"{r}_family_id"] in held_families[r] for r in roles):
            if stratum == "S4_held_component_family":
                out.append(product)
        elif any(row[f"{r}_smiles"] in held_identities[r] for r in roles):
            if stratum == "S3_held_component_identity":
                out.append(product)
        elif all(row[f"{r}_smiles"] in present[r] for r in roles):
            if stratum == "S2_fixed_combination_test" and product in combination_test:
                out.append(product)
    return out


if __name__ == "__main__":
    main()
