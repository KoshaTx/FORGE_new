#!/usr/bin/env python3
"""Evaluate a trained Phase 1 whole-lipid checkpoint with matched multi-seed sampling."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from forge.data.r0_splits import sha256_file
from forge.design.flow.defog_feasibility import graph_to_molecule
from forge.design.sampling.phase1_evaluation import evaluate_product_samples
from forge.design.flow.phase1_flow import (
    Phase1FlowError,
    _degree_continuation_log_prior,
    _training_records,
    build_corpus_from_config,
)
from forge.design.sampling.phase1_generation import (
    load_product_checkpoint,
    maximum_likelihood_count_distributions,
    sample_product_endpoints,
)

try:
    from rdkit.Chem import Draw
except ModuleNotFoundError:  # pragma: no cover - required by the chem extra
    Draw = None

REPO = Path(__file__).resolve().parents[1]


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_product_pretrain_v2.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-seed", type=int, default=256)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3101, 3102, 3103, 3104])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--grid-output", type=Path)
    parser.add_argument("--grid-molecules", type=int, default=24)
    parser.add_argument(
        "--derive-topology-priors-from-r0",
        action="store_true",
        help=(
            "derive newly added sampling-only topology priors from frozen R0_train; "
            "intended only for a labeled historical-checkpoint ablation"
        ),
    )
    parser.add_argument("--degree-prior-strength", type=float)
    parser.add_argument("--closure-ring-size-prior-strength", type=float)
    return parser.parse_args()


def _atomic_json(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parse_args()
    if (
        args.samples_per_seed <= 0
        or args.sample_steps < 2
        or args.batch_size <= 0
        or args.grid_molecules <= 0
    ):
        raise SystemExit("sampling counts must be positive and sample steps must be at least two")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("evaluation seeds must be nonempty and unique")

    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config, _, corpus = build_corpus_from_config(config_path, REPO, smoke=False)
    model, vocabulary, node_marginal, bond_marginal, package = load_product_checkpoint(
        checkpoint_path,
        device=args.device,
    )
    if package["config_sha256"] != sha256_file(config_path):
        if not args.derive_topology_priors_from_r0:
            raise Phase1FlowError("evaluation config does not match the checkpoint contract")
        sampling_only_keys = {
            "degree_prior_maximum_degree",
            "degree_continuation_probability_floor",
            "degree_continuation_prior_strength",
        }
        checkpoint_model = dict(package["model_config"])
        evaluation_model = dict(config["model"])
        for key in sampling_only_keys:
            checkpoint_model.pop(key, None)
            evaluation_model.pop(key, None)
        if checkpoint_model != evaluation_model:
            raise Phase1FlowError(
                "historical checkpoint differs from evaluation config beyond "
                "declared sampling-only topology priors"
            )
    train_records = _training_records(corpus)
    degree_prior_strength = (
        float(args.degree_prior_strength)
        if args.degree_prior_strength is not None
        else float(config["model"].get("degree_continuation_prior_strength", 0.0))
    )
    closure_ring_size_prior_strength = (
        float(args.closure_ring_size_prior_strength)
        if args.closure_ring_size_prior_strength is not None
        else float(config["model"].get("closure_ring_size_prior_strength", 0.0))
    )
    if (
        min(
            degree_prior_strength,
            closure_ring_size_prior_strength,
        )
        < 0.0
    ):
        raise Phase1FlowError("sampling-prior strengths must be nonnegative")
    if args.derive_topology_priors_from_r0:
        degree_continuation_log_prior = _degree_continuation_log_prior(
            train_records,
            region_classes=int(config["model"].get("region_classes", 0)),
            maximum_degree=int(config["model"].get("degree_prior_maximum_degree", 0)),
            probability_floor=float(
                config["model"].get("degree_continuation_probability_floor", 0.01)
            ),
        )
    else:
        degree_continuation_log_prior = (
            np.asarray(
                package["degree_continuation_log_prior"],
                dtype=np.float64,
            )
            if package.get("degree_continuation_log_prior") is not None
            else None
        )
    node_distribution, closure_distribution = maximum_likelihood_count_distributions(
        [record.node_count for record in train_records],
        [record.closure_count for record in train_records],
        maximum_heavy_atoms=int(config["model"]["maximum_heavy_atoms"]),
        maximum_closures=int(config["model"]["maximum_closure_slots"]),
    )

    all_samples = []
    seed_profiles = []
    repair_totals: Counter[str] = Counter()
    for seed in args.seeds:
        samples, profile = sample_product_endpoints(
            model,
            vocabulary,
            node_marginal,
            bond_marginal,
            sample_count=args.samples_per_seed,
            sample_steps=args.sample_steps,
            batch_size=args.batch_size,
            seed=seed,
            device=args.device,
            node_count_distribution=node_distribution,
            closure_count_distribution=closure_distribution,
            closure_ring_size_prior_strength=float(closure_ring_size_prior_strength),
            region_marginal=(
                np.asarray(package["region_marginal"], dtype=np.float64)
                if package.get("region_marginal") is not None
                else None
            ),
            region_atom_marginal=(
                np.asarray(package["region_atom_marginal"], dtype=np.float64)
                if package.get("region_atom_marginal") is not None
                else None
            ),
            aromatic_cycle_sizes=tuple(
                int(size) for size in config["model"].get("aromatic_cycle_sizes", [5, 6])
            ),
            aromatic_cycle_probability_threshold=float(
                config["model"].get("aromatic_cycle_probability_threshold", 0.5)
            ),
            degree_continuation_log_prior=(degree_continuation_log_prior),
            degree_continuation_prior_strength=float(degree_prior_strength),
        )
        all_samples.extend(samples)
        repair_totals.update(profile["terminal_constraint_events"])
        seed_profiles.append({"seed": seed, "sampling": profile})

    evaluation = evaluate_product_samples(all_samples, train_records, vocabulary)
    grid_record = None
    if args.grid_output is not None:
        if Draw is None:
            raise Phase1FlowError("sample-grid rendering requires RDKit drawing support")
        molecules = []
        legends = []
        for index, (nodes, edges) in enumerate(all_samples):
            try:
                molecule = graph_to_molecule(nodes, edges, vocabulary)
            except (ValueError, RuntimeError):
                continue
            molecules.append(molecule)
            legends.append(f"sample {index + 1}")
            if len(molecules) >= args.grid_molecules:
                break
        if not molecules:
            raise Phase1FlowError("sample-grid rendering found no valid molecules")
        grid_path = args.grid_output.resolve()
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        image = Draw.MolsToGridImage(
            molecules,
            molsPerRow=4,
            subImgSize=(320, 240),
            legends=legends,
            useSVG=False,
        )
        image.save(grid_path)
        grid_record = {
            "path": _portable(grid_path),
            "sha256": sha256_file(grid_path),
            "molecules": len(molecules),
        }
    result = {
        "schema_version": "phase1_product_generation_evaluation.v1",
        "status": "complete",
        "scope": (
            "broad whole-lipid endpoint quality without biological or synthesis guidance; "
            "node and closure counts use the frozen R0_train categorical MLE"
        ),
        "checkpoint": {
            "path": _portable(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "model_state_sha256": package["model_state_sha256"],
            "step": int(package["step"]),
        },
        "config": {
            "path": _portable(config_path),
            "sha256": sha256_file(config_path),
        },
        "data_manifest": {
            "path": config["inputs"]["phase1_manifest"]["path"],
            "sha256": package["data_manifest_sha256"],
        },
        "sampling_contract": {
            "seeds": args.seeds,
            "samples_per_seed": args.samples_per_seed,
            "total_samples": len(all_samples),
            "sample_steps": args.sample_steps,
            "batch_size": args.batch_size,
            "device": args.device,
            "count_calibration": "exact categorical MLE from frozen R0_train",
            "degree_coordination": "R0_train region-conditioned continuation prior",
            "cycle_coordination": "R0_train multiplicity-corrected ring-size prior",
            "historical_checkpoint_topology_prior_ablation": bool(
                args.derive_topology_priors_from_r0
            ),
            "degree_prior_strength": degree_prior_strength,
            "closure_ring_size_prior_strength": closure_ring_size_prior_strength,
        },
        "seed_profiles": seed_profiles,
        "terminal_constraint_events": dict(sorted(repair_totals.items())),
        "rendered_grid": grid_record,
        "evaluation": evaluation,
        "interpretation_limits": [
            "This evaluation does not establish biological activity.",
            "This evaluation does not establish synthesis route closure.",
            "Count fidelity is a sampler-contract check and is not credited to the neural model.",
            "A relative architecture claim requires the same sampling contract for every checkpoint.",
        ],
    }
    _atomic_json(args.output.resolve(), result)
    print(json.dumps({"status": "complete", "output": str(args.output), "evaluation": evaluation}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
