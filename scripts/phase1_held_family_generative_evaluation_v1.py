#!/usr/bin/env python3
"""Does the generator model component families it never saw, or only recombine familiar ones?

This is the analysis that carries the fidelity and generalization claim. Global ester
prevalence does not carry it, and the chemotype audits do not carry it.

The evaluation checkpoint is the development run, trained on the 66,464-product train fold with
the calibration fold used only for early stopping. The heldout fold is untouched by both, and
every component of every train-fold product comes from a train-fold family, so the model saw
zero held families. No retraining happens here.

The corpus partitions each precursor role into component families and assigns families to
folds. Within the heldout product fold there are 6 held aldehyde families against 5 train ones,
40 held amine families against 129 train ones, and 1 held isocyanide family against 7. So the
clean contrast lives INSIDE the heldout fold: every product there is an unseen product, and the
only thing that differs between the groups is whether its component family was also unseen. No
heldout product is built entirely from train families, so an all-familiar out-of-sample
reference does not exist and is not invented here.

Two estimands, because they answer different questions and the marginal one is confounded:

  marginal    all heldout products, held-family against train-family. Confounded, because the
              two groups occupy different morphology programs: they share only 1,080 programs,
              covering 11.5% of held-family products and 36.8% of train-family products.
  ω-matched   exact morphology-program matching. For every program present in both groups, the
              same number of products is drawn from each side, so the two groups carry
              identical program multisets and the pooled difference is standardized by
              construction rather than by reweighting.

The matched comparison is paired all the way down: the two groups are ordered by program, so
corresponding batches share morphology, the same t values from a fixed grid, and the same noise
generator seed. Only the clean states differ.

Per-family losses are listed individually rather than summarized by a bootstrap. With 6 held
aldehyde families, leave-one-out already showed that an interval over so few clusters says
little that the individual values do not say better.

Prespecified in docs/FORGE_EVIDENCE_CONTRACT_v1.md section B.
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

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from forge.product.ugi_joint_sparse_flow import (  # noqa: E402
    UgiJointSparseFlow,
    collate_ugi_joint_sparse_records,
    noise_ugi_joint_sparse_batch,
    ugi_joint_sparse_loss,
)
from forge.product.ugi_training_cache import load_ugi_training_cache  # noqa: E402

CHECKPOINT = "results/phase1/ugi_joint_sparse_balanced_v2_full/checkpoint_step_1000.pt"
CACHE = "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt"
DEV_RESULT = "results/phase1/ugi_joint_sparse_balanced_v2_full/result.json"

ROLES = ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")
T_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
BATCH = 128
NOISE_SEED = 20260816
PER_FAMILY_CAP = 512
REPORTED_METRICS = ("total", "atom_ce", "parent_bond_ce", "offspring_ce", "closure_bond_ce")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def program_key(program) -> tuple:
    return (tuple(program.node_counts), tuple(program.junction_budgets),
            tuple(program.cycle_ranks), tuple(program.attachment_counts))


class Evaluator:
    """Teacher-forced denoising loss under a fixed t grid and reproducible noise."""

    def __init__(self, model, sources, architecture) -> None:
        self.model = model
        self.sources = sources
        self.architecture = architecture

    def __call__(self, records) -> dict[str, float]:
        """Weighted mean per metric, skipping batches where that metric is undefined.

        A batch in which no record carries a decoration gives an empty-tensor cross-entropy,
        so `decoration_atom_ce` comes back NaN and poisons the aggregate `total`. Such a batch
        carries no decoration evidence and must contribute no weight to decoration metrics
        rather than destroying them. Skips are counted and reported.
        """
        if not records:
            return {}
        totals: dict[str, float] = defaultdict(float)
        weights: dict[str, float] = defaultdict(float)
        skipped: dict[str, int] = defaultdict(int)
        chunks = [records[i:i + BATCH] for i in range(0, len(records), BATCH)]
        with torch.no_grad():
            for t_index, t_value in enumerate(T_GRID):
                for batch_index, chunk in enumerate(chunks):
                    batch = collate_ugi_joint_sparse_records(
                        tuple(chunk),
                        maximum_nodes=max(r.node_count for r in chunk),
                        maximum_children=int(self.architecture["maximum_children"]),
                        maximum_closures=int(self.architecture["maximum_cycle_rank"]) * 3,
                        maximum_decorations=int(self.architecture["maximum_decorations"]),
                    )
                    # Seeded only by position in the grid, never by which group this is, so
                    # matched groups are denoised against identical noise draws.
                    generator = torch.Generator().manual_seed(
                        NOISE_SEED + 1000 * t_index + batch_index)
                    t = torch.full((len(chunk),), float(t_value))
                    noisy = noise_ugi_joint_sparse_batch(batch, self.sources, t, generator)
                    predictions = self.model(
                        offspring=noisy["offspring"], nodes=noisy["nodes"],
                        parent_bonds=noisy["parent_bonds"], role_states=batch["role_states"],
                        within_role_positions=batch["within_role_positions"],
                        programs=batch["programs"], node_mask=batch["node_mask"], t=t,
                        closure_left=batch["closure_left"], closure_right=batch["closure_right"],
                        decoration_anchors=noisy["decoration_anchors"],
                        decoration_atoms=noisy["decoration_atoms"],
                        decoration_bonds=noisy["decoration_bonds"],
                    )
                    _, metrics = ugi_joint_sparse_loss(predictions, batch)
                    weight = len(chunk)
                    finite = {k: v for k, v in metrics.items()
                              if k != "total" and not np.isnan(v)}
                    for key in metrics:
                        if key == "total":
                            continue
                        if key in finite:
                            totals[key] += finite[key] * weight
                            weights[key] += weight
                        else:
                            skipped[key] += 1
                    # The loss defines total as exactly the sum of its *_ce terms, so one
                    # undefined term poisons it. Rebuild it from the defined terms for every
                    # batch, not only the poisoned ones, so all groups share one definition.
                    totals["total"] += sum(finite.values()) * weight
                    weights["total"] += weight
        out = {key: float(totals[key] / weights[key]) for key in totals if weights[key]}
        if skipped:
            out["batches_with_an_undefined_metric"] = {k: v for k, v in sorted(skipped.items())}
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "results/phase1/forge_held_family_generative_evaluation_v1/result.json",
    )
    args = parser.parse_args()
    started = time.time()
    torch.manual_seed(NOISE_SEED)

    dev = json.loads((REPO / DEV_RESULT).read_text())
    if dev["corpus"] != {"train": 66464, "calibration": 15800, "heldout": 30122}:
        raise SystemExit("development corpus does not match the expected fold counts")
    print(f"development run: best_step {dev['selection']['best_step']} by calibration loss, "
          f"stopped at {dev['selection']['completed_steps']}; evaluating the checkpoint the "
          f"production duration contract names")

    corpus, records_by_fold = load_ugi_training_cache(REPO / CACHE)
    checkpoint = torch.load(REPO / CHECKPOINT, map_location="cpu", weights_only=False)
    architecture = dict(checkpoint["model_config"])
    architecture.pop("source_probability_floor", None)
    model = UgiJointSparseFlow(atom_classes=len(corpus.atom_vocabulary), **architecture)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    sources = {key: torch.as_tensor(value, dtype=torch.float32)
               for key, value in checkpoint["source_marginals"].items()}
    evaluate = Evaluator(model, sources, architecture)
    print(f"checkpoint step {checkpoint['step']}, t grid {T_GRID}, batch {BATCH}")

    heldout = records_by_fold["heldout"]
    assignments = corpus.assignments_by_fold["heldout"]
    if len(heldout) != len(assignments) or any(
        record.product_id != assignment["product_id"]
        for record, assignment in zip(heldout, assignments, strict=True)
    ):
        raise SystemExit("heldout records and assignments are not aligned")

    train_records = records_by_fold["train"]
    train_assignments = corpus.assignments_by_fold["train"]
    for role in ROLES:
        folds = {a[f"{role}_family_fold"] for a in train_assignments}
        if folds != {"train"}:
            raise SystemExit(f"train fold contains non-train {role} families: {sorted(folds)}")
    print("verified: every component family of every train-fold product is a train family")

    paired = list(zip(heldout, assignments, strict=True))

    # ------------------------------------------------------ in-sample reference
    reference = evaluate([r for r, _ in sorted(
        zip(train_records, train_assignments), key=lambda p: p[1]["product_id"])][:2048])
    print(f"\nIN-SAMPLE train-fold reference (2,048 records): "
          f"total {reference['total']:.4f}  atom {reference['atom_ce']:.4f}")

    results: dict[str, dict] = {"in_sample_train_reference": reference}

    # ----------------------------------------------------------- per family
    per_role: dict[str, dict] = {}
    for role in ROLES:
        by_family: dict[str, list] = defaultdict(list)
        fold_of: dict[str, str] = {}
        for record, assignment in paired:
            family = assignment[f"{role}_family_id"]
            by_family[family].append(record)
            fold_of[family] = assignment[f"{role}_family_fold"]
        families = sorted(by_family, key=lambda f: (-len(by_family[f]), f))
        reported = [f for f in families if len(by_family[f]) >= 200][:16]
        rows = []
        for family in reported:
            members = sorted(by_family[family], key=lambda r: r.product_id)[:PER_FAMILY_CAP]
            metrics = evaluate(members)
            rows.append({"family_id": family, "family_fold": fold_of[family],
                         "heldout_products": len(by_family[family]),
                         "evaluated": len(members),
                         **{k: metrics[k] for k in REPORTED_METRICS if k in metrics}})
        print(f"\n{role}: per-family loss on heldout products "
              f"(families with >= 200 products, capped at {PER_FAMILY_CAP} evaluated)")
        print(f"  {'family':50s} {'fold':12s} {'n':>6s} {'total':>8s} {'atom':>8s}")
        for row in sorted(rows, key=lambda r: (r["family_fold"], -r["heldout_products"])):
            print(f"  {row['family_id']:50s} {row['family_fold']:12s} "
                  f"{row['heldout_products']:6d} {row['total']:8.4f} {row['atom_ce']:8.4f}")
        summary: dict[str, object] = {"families_reported": len(rows)}
        for fold in ("train", "calibration", "heldout"):
            subset = [r for r in rows if r["family_fold"] == fold]
            if not subset:
                continue
            summary[f"{fold}_family_total_mean"] = statistics.fmean(r["total"] for r in subset)
            summary[f"{fold}_family_total_range"] = [min(r["total"] for r in subset),
                                                     max(r["total"] for r in subset)]
            summary[f"{fold}_families"] = len(subset)
            print(f"  {fold + '-fold families':22s} {len(subset):3d}  "
                  f"mean {summary[f'{fold}_family_total_mean']:.4f}  "
                  f"range {summary[f'{fold}_family_total_range'][0]:.4f}"
                  f"-{summary[f'{fold}_family_total_range'][1]:.4f}")
        per_role[role] = {"families": rows, "summary": summary}

    # ------------------------------------------- marginal and ω-matched contrasts
    #
    # "Held" is split by fold rather than lumped as "not train". Calibration families steered
    # early stopping, so they are weakly seen; heldout families are the genuinely unseen ones.
    # Lumping them hid that the single unseen isocyanide family behaves like a train family
    # while a calibration family is seven times worse.
    def matched_pair(group: list, reference_by_program: dict[tuple, list]) -> dict:
        by_program: dict[tuple, list] = defaultdict(list)
        for record in group:
            by_program[program_key(record.program)].append(record)
        shared = sorted(set(by_program) & set(reference_by_program))
        left: list = []
        right: list = []
        for key in shared:
            take = min(len(by_program[key]), len(reference_by_program[key]))
            left.extend(sorted(by_program[key], key=lambda r: r.product_id)[:take])
            right.extend(sorted(reference_by_program[key], key=lambda r: r.product_id)[:take])
        if not left:
            return {"shared_programs": 0, "matched_products_per_side": 0}
        assert Counter(program_key(r.program) for r in left) == \
               Counter(program_key(r.program) for r in right), "program multisets differ"
        left_metrics = evaluate(left)
        right_metrics = evaluate(right)
        return {
            "shared_programs": len(shared),
            "matched_products_per_side": len(left),
            "coverage_of_group": len(left) / len(group),
            "group": left_metrics, "train_family_reference": right_metrics,
            "difference_total": left_metrics["total"] - right_metrics["total"],
            "difference_atom_ce": left_metrics["atom_ce"] - right_metrics["atom_ce"],
        }

    contrasts: dict[str, dict] = {}
    for role in ROLES:
        groups: dict[str, list] = defaultdict(list)
        for record, assignment in paired:
            groups[assignment[f"{role}_family_fold"]].append(record)
        if "train" not in groups:
            contrasts[role] = {"skipped": "no train-family products in the heldout fold"}
            continue
        train_by_program: dict[tuple, list] = defaultdict(list)
        for record in groups["train"]:
            train_by_program[program_key(record.program)].append(record)

        marginal_train = evaluate(sorted(groups["train"], key=lambda r: r.product_id)[:4096])
        role_result: dict[str, dict] = {"train_family_marginal": marginal_train,
                                        "train_family_products": len(groups["train"])}
        print(f"\n{role}: denoising loss against train-family products in the same fold")
        print(f"  {'group':34s} {'n':>6s} {'marginal':>9s} {'Δ marg':>8s} "
              f"{'ω-match':>8s} {'Δ ω':>8s} {'cover':>6s}")
        print(f"  {'train-family reference':34s} {len(groups['train']):6d} "
              f"{marginal_train['total']:9.4f}")
        for fold in ("calibration", "heldout"):
            if fold not in groups:
                continue
            marginal = evaluate(sorted(groups[fold], key=lambda r: r.product_id)[:4096])
            matched = matched_pair(groups[fold], train_by_program)
            role_result[f"{fold}_family"] = {
                "products": len(groups[fold]), "marginal": marginal,
                "marginal_difference_total": marginal["total"] - marginal_train["total"],
                "omega_matched": matched,
            }
            print(f"  {fold + '-fold families':34s} {len(groups[fold]):6d} "
                  f"{marginal['total']:9.4f} "
                  f"{marginal['total'] - marginal_train['total']:+8.4f} "
                  f"{matched.get('group', {}).get('total', float('nan')):8.4f} "
                  f"{matched.get('difference_total', float('nan')):+8.4f} "
                  f"{matched.get('coverage_of_group', float('nan')):6.3f}")

        # Per held family, because one family moved the composition result by half a point and
        # a single pooled number cannot show that.
        per_family_matched = []
        by_family: dict[str, list] = defaultdict(list)
        for record, assignment in paired:
            if assignment[f"{role}_family_fold"] == "heldout":
                by_family[assignment[f"{role}_family_id"]].append(record)
        for family in sorted(by_family, key=lambda f: -len(by_family[f]))[:8]:
            matched = matched_pair(by_family[family], train_by_program)
            if not matched["matched_products_per_side"]:
                continue
            per_family_matched.append({"family_id": family,
                                       "products": len(by_family[family]), **matched})
        if per_family_matched:
            print(f"  per held family, ω-matched against train-family products:")
            for entry in per_family_matched:
                print(f"    {entry['family_id']:50s} n={entry['products']:5d} "
                      f"matched={entry['matched_products_per_side']:5d} "
                      f"Δ={entry['difference_total']:+.4f}")
            spread = [e["difference_total"] for e in per_family_matched]
            role_result["per_held_family_matched_difference_range"] = [min(spread), max(spread)]
            print(f"    range over held families: [{min(spread):+.4f}, {max(spread):+.4f}]")
        role_result["per_held_family_matched"] = per_family_matched
        contrasts[role] = role_result

    payload = {
        "schema_version": "phase1_forge_held_family_generative_evaluation.v1",
        "status": "stage_1_teacher_forced_complete_sampling_stage_not_run",
        "contract": "docs/FORGE_EVIDENCE_CONTRACT_v1.md section B",
        "inputs": {
            "checkpoint": {"path": CHECKPOINT, "sha256": sha256_file(REPO / CHECKPOINT),
                           "step": int(checkpoint["step"]),
                           "trained_on": "train fold only; calibration used for early stopping"},
            "training_cache": {"path": CACHE, "sha256": sha256_file(REPO / CACHE)},
        },
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "elapsed_seconds": round(time.time() - started, 1)},
        "protocol": {
            "t_grid": list(T_GRID), "batch_size": BATCH, "noise_seed": NOISE_SEED,
            "per_family_cap": PER_FAMILY_CAP,
            "pairing": "matched groups are ordered by morphology program, so corresponding "
                       "batches share morphology, t, and noise generator seed",
            "source_marginals": "taken from the checkpoint, not recomputed",
        },
        "verified": {
            "train_fold_uses_only_train_families": True,
            "heldout_records_aligned_to_assignments": True,
            "no_heldout_product_is_built_entirely_from_train_families": True,
        },
        "in_sample_train_reference": reference,
        "per_role_families": per_role,
        "contrasts": contrasts,
        "nonclaims": [
            "This is teacher-forced denoising loss, not generation. It measures how well the "
            "model predicts clean states from noised ones, not whether sampling produces valid, "
            "admitted, unique products under held-family morphology. That is the second stage "
            "and it has not been run.",
            "The marginal contrast is confounded by morphology: held-family and train-family "
            "products occupy substantially different program supports, sharing only the "
            "programs reported. The ω-matched contrast is the standardized estimand.",
            "Per-family values are listed rather than pooled into an interval. With 6 held "
            "aldehyde families an interval over clusters would be wide enough to be "
            "uninformative, and leave-one-out has already shown how much a single family moves.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
