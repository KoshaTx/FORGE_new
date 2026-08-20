#!/usr/bin/env python3
"""Close the semantics branch: replicate across seeds, and run the misaligned control.

Sections 1A and 2 of the closure directive, contract amendments 11 to 14.

Two things this does that the pilot could not:

**Replication with the right unit of uncertainty.** Three independent paired training seeds at
alpha = 1.00. Uncertainty is reported **across paired seeds**, never across the t-by-head cells,
which are not independent training replicates. Sixty correlated cells from one seed would give a
spuriously tight interval and that is exactly the error to avoid.

**The misaligned control.** A third arm whose role channel carries labels permuted across nodes with
each label's count preserved, deterministic per example. It separates "correct scientific alignment
matters" from "an extra regional embedding helps". The interpretive hierarchy was fixed before the
run and is applied mechanically:

    L_true < L_none <= L_misaligned   alignment carries the effect
    L_true < L_misaligned < L_none    true semantics best, generic regional structure also helps
    L_true ~ L_misaligned < L_none    narrow the claim to regional side-information; do NOT call it
                                      experiment-semantic

Every arm is read at `checkpoint_step_3000.pt` (Amendment 13). Pairing is exact: one collate serves
all arms, the noise generator is seeded per (t, batch) rather than per arm, so every arm sees the
same examples in the same order at the same t with the same corruption draws.
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

sys.path.insert(0, str(REPO / "scripts"))
from phase1_semantics_pilot_delta_v1 import _stratum_ids  # noqa: E402

CACHE = "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt"
SPLITS = "results/phase1/forge_coverage_splits_v1/splits.json"
PILOT = REPO / "results/phase1/forge_semantics_pilot_v1"
CHECKPOINT = "checkpoint_step_3000.pt"
T_GRID = (0.1, 0.25, 0.5, 0.75, 0.9)
STRATUM = "S2_fixed_combination_test"
PRIMARY_HEADS = ("total", "atom_ce", "parent_bond_ce", "offspring_ce",
                 "closure_bond_ce", "decoration_anchor_ce")
BATCH = 128
NOISE_SEED = 20260818
MAX_RECORDS = 2048

# Three paired seeds at alpha 1.00, plus the misaligned control on the first seed.
SEED_PAIRS = (
    ("s1", "flat_no_role_a100", "flat_true_role_a100"),
    ("s2", "flat_no_role_a100_s2", "flat_true_role_a100_s2"),
    ("s3", "flat_no_role_a100_s3", "flat_true_role_a100_s3"),
)
MISALIGNED = ("s1", "flat_misaligned_role_a100_s1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(directory: str, atom_classes: int):
    path = PILOT / directory
    checkpoint = torch.load(path / CHECKPOINT, map_location="cpu", weights_only=False)
    if int(checkpoint["step"]) != 3000:
        raise SystemExit(f"{directory} is step {checkpoint['step']}, expected 3000")
    architecture = dict(checkpoint["model_config"])
    architecture.pop("source_probability_floor", None)
    organization = json.loads((path / "result.json").read_text())["semantics_pilot"][
        "semantic_organization"]
    model = UgiJointSparseFlow(atom_classes=atom_classes,
                              semantic_organization=organization, **architecture)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    sources = {k: torch.as_tensor(v, dtype=torch.float32)
               for k, v in checkpoint["source_marginals"].items()}
    return model, sources, architecture, organization


def losses_for(models, sources, organizations, architecture, records):
    """One pass over all arms with identical batches, t and corruption draws."""
    chunks = [records[i:i + BATCH] for i in range(0, len(records), BATCH)]
    totals = {a: defaultdict(lambda: defaultdict(float)) for a in models}
    weights = {a: defaultdict(lambda: defaultdict(float)) for a in models}
    with torch.no_grad():
        for t_index, t_value in enumerate(T_GRID):
            for batch_index, chunk in enumerate(chunks):
                base = collate_ugi_joint_sparse_records(
                    tuple(chunk), maximum_nodes=max(r.node_count for r in chunk),
                    maximum_children=int(architecture["maximum_children"]),
                    maximum_closures=int(architecture["maximum_cycle_rank"]) * 3,
                    maximum_decorations=int(architecture["maximum_decorations"]),
                    semantic_organization="flat_no_role")
                misaligned = collate_ugi_joint_sparse_records(
                    tuple(chunk), maximum_nodes=max(r.node_count for r in chunk),
                    maximum_children=int(architecture["maximum_children"]),
                    maximum_closures=int(architecture["maximum_cycle_rank"]) * 3,
                    maximum_decorations=int(architecture["maximum_decorations"]),
                    semantic_organization="flat_misaligned_role")
                t = torch.full((len(chunk),), float(t_value))
                for arm, model in models.items():
                    batch = (misaligned if organizations[arm] == "flat_misaligned_role" else base)
                    generator = torch.Generator().manual_seed(
                        NOISE_SEED + 1000 * t_index + batch_index)
                    noisy = noise_ugi_joint_sparse_batch(batch, sources[arm], t, generator)
                    predictions = model(
                        offspring=noisy["offspring"], nodes=noisy["nodes"],
                        parent_bonds=noisy["parent_bonds"], role_states=batch["role_states"],
                        within_role_positions=batch["within_role_positions"],
                        programs=batch["programs"], node_mask=batch["node_mask"], t=t,
                        closure_left=batch["closure_left"], closure_right=batch["closure_right"],
                        decoration_anchors=noisy["decoration_anchors"],
                        decoration_atoms=noisy["decoration_atoms"],
                        decoration_bonds=noisy["decoration_bonds"])
                    _, metrics = ugi_joint_sparse_loss(
                        predictions, batch, semantic_organization=organizations[arm])
                    finite = {k: v for k, v in metrics.items()
                              if k != "total" and not np.isnan(v)}
                    for key, value in finite.items():
                        totals[arm][t_value][key] += value * len(chunk)
                        weights[arm][t_value][key] += len(chunk)
                    totals[arm][t_value]["total"] += sum(finite.values()) * len(chunk)
                    weights[arm][t_value]["total"] += len(chunk)
    return {a: {t: {k: totals[a][t][k] / weights[a][t][k]
                    for k in totals[a][t] if weights[a][t][k]}
                for t in totals[a]} for a in models}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PILOT / "closure.json")
    args = parser.parse_args()
    started = time.time()

    corpus, records_by_fold = load_ugi_training_cache(REPO / CACHE)
    sealed = json.loads((REPO / SPLITS).read_text())
    by_id = {r.product_id: r for fold in records_by_fold.values() for r in fold}
    ids = _stratum_ids(sealed, "1.00", STRATUM)
    records = sorted((by_id[i] for i in ids if i in by_id),
                     key=lambda r: r.product_id)[:MAX_RECORDS]
    print(f"{STRATUM}: {len(records)} records at alpha 1.00")

    # ---------------------------------------------------- section 2: replication across seeds
    per_seed: dict[str, dict] = {}
    for tag, none_dir, true_dir in SEED_PAIRS:
        models, sources, organizations = {}, {}, {}
        architecture = None
        for arm, directory in (("none", none_dir), ("true", true_dir)):
            model, source, architecture, organization = load(
                directory, len(corpus.atom_vocabulary))
            models[arm], sources[arm], organizations[arm] = model, source, organization
        losses = losses_for(models, sources, organizations, architecture, records)
        delta = {t: {h: losses["none"][t][h] - losses["true"][t][h]
                     for h in losses["none"][t]} for t in T_GRID}
        per_seed[tag] = {
            "arms": {"none": none_dir, "true": true_dir},
            "loss": {arm: {str(t): losses[arm][t] for t in T_GRID} for arm in models},
            "delta_sem_by_t": {str(t): delta[t] for t in T_GRID},
            "delta_bar_sem": {h: statistics.fmean(delta[t][h] for t in T_GRID)
                              for h in delta[T_GRID[0]]},
        }
        print(f"  seed {tag}: delta_bar total {per_seed[tag]['delta_bar_sem']['total']:+.5f}")

    print("\nsection 2: replication, uncertainty ACROSS PAIRED SEEDS (n=3), not across cells")
    print(f"  {'head':22s} {'mean':>10s} {'sd':>9s} {'per-seed':>34s} {'all positive':>13s}")
    replication = {}
    for head in PRIMARY_HEADS:
        values = [per_seed[tag]["delta_bar_sem"].get(head) for tag, _, _ in SEED_PAIRS]
        if any(v is None for v in values):
            continue
        mean = statistics.fmean(values)
        sd = statistics.stdev(values)
        replication[head] = {"per_seed": values, "mean": mean, "sd": sd,
                             "all_seeds_positive": all(v > 0 for v in values),
                             "seeds": 3}
        print(f"  {head:22s} {mean:+10.5f} {sd:9.5f} "
              f"{'  '.join(f'{v:+.4f}' for v in values):>34s} "
              f"{str(all(v > 0 for v in values)):>13s}")

    print("\n  corruption dependence per seed, total head")
    corruption = {}
    for tag, _, _ in SEED_PAIRS:
        low = per_seed[tag]["delta_sem_by_t"][str(T_GRID[0])]["total"]
        high = per_seed[tag]["delta_sem_by_t"][str(T_GRID[-1])]["total"]
        corruption[tag] = {"low_t": low, "high_t": high, "ratio": low / high if high else None}
        print(f"    seed {tag}: t=0.10 {low:+.5f}   t=0.90 {high:+.5f}   ratio {low / high:.2f}x")

    # ------------------------------------------------- section 1A: the misaligned control
    tag, misaligned_dir = MISALIGNED
    none_dir, true_dir = next((n, t) for k, n, t in SEED_PAIRS if k == tag)
    models, sources, organizations = {}, {}, {}
    architecture = None
    for arm, directory in (("none", none_dir), ("true", true_dir),
                           ("misaligned", misaligned_dir)):
        model, source, architecture, organization = load(directory, len(corpus.atom_vocabulary))
        models[arm], sources[arm], organizations[arm] = model, source, organization
    three = losses_for(models, sources, organizations, architecture, records)
    control = {}
    print(f"\nsection 1A: misaligned control on seed {tag}, absolute loss (lower is better)")
    print(f"  {'head':22s} {'true':>10s} {'none':>10s} {'misaligned':>11s} {'verdict':>34s}")
    for head in PRIMARY_HEADS:
        bars = {arm: statistics.fmean(three[arm][t][head] for t in T_GRID)
                for arm in models if head in three[arm][T_GRID[0]]}
        if len(bars) != 3:
            continue
        true, none, mis = bars["true"], bars["none"], bars["misaligned"]
        if true < none <= mis:
            verdict = "alignment carries the effect"
        elif true < mis < none:
            verdict = "true best; regional structure also helps"
        elif abs(true - mis) < 0.02 * max(abs(true), 1e-9) and mis < none:
            verdict = "regional side-information only"
        else:
            verdict = "does not match any frozen pattern"
        control[head] = {"true": true, "none": none, "misaligned": mis, "verdict": verdict,
                         "true_minus_misaligned": true - mis,
                         "none_minus_misaligned": none - mis}
        print(f"  {head:22s} {true:10.5f} {none:10.5f} {mis:11.5f} {verdict:>34s}")

    payload = {
        "schema_version": "phase1_forge_semantics_closure.v1",
        "status": "complete",
        "contract": "docs/FORGE_EVIDENCE_CONTRACT_v1.md amendments 11 to 14",
        "stratum": STRATUM,
        "records_evaluated": len(records),
        "checkpoint_policy": "checkpoint_step_3000.pt for every arm (Amendment 13)",
        "uncertainty_unit": "paired training seeds, n=3. The t-by-head cells are NOT independent "
                            "training replicates and are never used as the unit of uncertainty.",
        "pairing": "one collate per organization shared across arms; noise generator seeded per "
                   "(t, batch) not per arm",
        "inputs": {"training_cache": {"path": CACHE, "sha256": sha256_file(REPO / CACHE)},
                   "sealed_splits": {"path": SPLITS, "sha256": sha256_file(REPO / SPLITS)}},
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(), "torch": torch.__version__,
                    "elapsed_seconds": round(time.time() - started, 1)},
        "protocol": {"t_grid": list(T_GRID), "batch_size": BATCH, "noise_seed": NOISE_SEED,
                     "max_records": MAX_RECORDS},
        "per_seed": per_seed,
        "replication_across_seeds": replication,
        "corruption_dependence_per_seed": corruption,
        "misaligned_control": control,
        "nonclaims": [
            "An empirical semantic-utility gap between finite trained networks, not a measured "
            "mutual information.",
            "No sample was drawn, so nothing here speaks to generated molecular chemistry.",
            "Three seeds support a direction and a rough magnitude, not a precise effect size.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
