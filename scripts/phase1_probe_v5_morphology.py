#!/usr/bin/env python3
"""Train and checkpoint a bounded V5 program-conditioned morphology flow."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from forge.design.audit.canonical_representation_audit import load_atom_vocabulary
from forge.design.flow.defog_feasibility import set_determinism, sha256_file
from forge.design.flow.v5_morphology_flow import (
    V5MorphologyFlow,
    V5MorphologySample,
    collate_v5_morphology_records,
    morphology_source_marginals,
    noise_v5_morphology_batch,
    sample_v5_morphologies,
    v5_morphology_flow_loss,
    v5_morphology_statistics,
)
from forge.design.flow.v5_morphology_program import program_from_sparse_record
from forge.design.flow.v5_sparse_representation import tensorize_v5_sparse_row

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - required by this script
    torch = None

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r0",
        type=Path,
        default=REPO / "results/m0_03/r0_constitutional.csv.gz",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=REPO / "data/splits/m0_03_constitutional/r0_fold_assignments.csv",
    )
    parser.add_argument(
        "--atom-vocabulary",
        type=Path,
        default=REPO / "results/phase1/product_v3_atom_vocabulary.json",
    )
    parser.add_argument(
        "--architecture-config",
        type=Path,
        default=REPO / "configs/model/phase1_product_morphology_v5.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--records", type=int, default=128)
    parser.add_argument("--maximum-record-atoms", type=int, default=64)
    parser.add_argument("--maximum-junction-budget", type=int, default=12)
    parser.add_argument("--maximum-children", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def _atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_train_ids(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        return {
            row["r0_structure_id"]
            for row in csv.DictReader(handle)
            if row["source_study_fold"] == "R0_train"
        }


def _load_records(args: argparse.Namespace) -> tuple[tuple[Any, ...], dict[str, int]]:
    vocabulary = load_atom_vocabulary(args.atom_vocabulary.resolve())
    atom_to_index = {state: index for index, state in enumerate(vocabulary)}
    train_ids = _load_train_ids(args.assignments.resolve())
    eligible = []
    coverage = {
        "r0_train_pretraining_eligible": 0,
        "over_atom_gate": 0,
        "over_child_gate": 0,
        "over_junction_gate": 0,
        "bounded_gate_eligible": 0,
    }
    with gzip.open(args.r0.resolve(), "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["r0_structure_id"] not in train_ids or row["r0_pretraining_eligible"] != "True":
                continue
            coverage["r0_train_pretraining_eligible"] += 1
            record = tensorize_v5_sparse_row(
                row,
                atom_to_index,
                preserve_aromaticity=True,
                root_strategy="lipid_polar",
                region_scheme="polar_structural_v2",
                tree_traversal="breadth_first_tree_preorder",
            )
            program = program_from_sparse_record(record)
            if record.node_count > args.maximum_record_atoms:
                coverage["over_atom_gate"] += 1
                continue
            if record.offspring.max(initial=0) > args.maximum_children:
                coverage["over_child_gate"] += 1
                continue
            if program.junction_budget_total > args.maximum_junction_budget:
                coverage["over_junction_gate"] += 1
                continue
            eligible.append(record)
            coverage["bounded_gate_eligible"] += 1
    if len(eligible) < args.records:
        raise SystemExit(
            f"requested {args.records} records but only {len(eligible)} satisfy the bounded gate"
        )
    generator = np.random.default_rng(args.seed)
    indices = generator.choice(len(eligible), size=args.records, replace=False)
    return tuple(eligible[int(index)] for index in indices), coverage


def _move(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def _reference_samples(records: tuple[Any, ...]) -> list[V5MorphologySample]:
    return [
        V5MorphologySample(
            program=program_from_sparse_record(record),
            offspring=record.offspring.copy(),
            regions=record.region_states.copy(),
            parents=record.parents,
        )
        for record in records
    ]


def _evaluate(
    model: Any,
    records: tuple[Any, ...],
    marginals: dict[str, np.ndarray],
    *,
    args: argparse.Namespace,
    step: int,
) -> dict[str, Any]:
    local = records[: min(args.eval_samples, len(records))]
    programs = tuple(program_from_sparse_record(record) for record in local)
    generated, sampling = sample_v5_morphologies(
        model,
        programs,
        marginals,
        sample_steps=args.sample_steps,
        batch_size=min(args.batch_size, len(programs)),
        seed=args.seed + 10_000 + step,
        device=args.device,
    )
    return {
        "step": step,
        "sampling": sampling,
        "reference": v5_morphology_statistics(_reference_samples(local)),
        "generated": v5_morphology_statistics(generated),
        "sample_programs": [program.__dict__ for program in programs],
        "sample_offspring": [sample.offspring.tolist() for sample in generated],
        "sample_regions": [sample.regions.tolist() for sample in generated],
    }


def main() -> int:
    args = parse_args()
    if torch is None:
        raise SystemExit("V5 morphology probe requires torch")
    if (
        args.records < 16
        or args.maximum_record_atoms < 2
        or args.maximum_junction_budget < 0
        or args.maximum_children < 1
        or args.steps < 1
        or args.batch_size < 1
        or args.eval_every < 1
        or args.eval_samples < 1
        or args.sample_steps < 2
    ):
        raise SystemExit("invalid bounded probe arguments")
    set_determinism(args.seed, min(8, os.cpu_count() or 1))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but unavailable")

    records, coverage = _load_records(args)
    split = max(1, int(round(0.8 * len(records))))
    train_records = records[:split]
    validation_records = records[split:]
    if not validation_records:
        raise SystemExit("bounded probe requires validation records")
    marginals = morphology_source_marginals(
        train_records,
        maximum_children=args.maximum_children,
    )
    source_tensors = {
        field: torch.as_tensor(values, dtype=torch.float32, device=device)
        for field, values in marginals.items()
    }
    model = V5MorphologyFlow(
        maximum_children=args.maximum_children,
        maximum_heavy_atoms=args.maximum_record_atoms,
        maximum_junction_budget=args.maximum_junction_budget,
        maximum_cycle_rank=12,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    torch_generator = torch.Generator(device=device).manual_seed(args.seed)
    numpy_generator = np.random.default_rng(args.seed + 1)
    losses: list[dict[str, float | int]] = []
    evaluations = [_evaluate(model, validation_records, marginals, args=args, step=0)]
    model.train()
    for step in range(1, args.steps + 1):
        indices = numpy_generator.choice(
            len(train_records),
            size=min(args.batch_size, len(train_records)),
            replace=False,
        )
        batch_records = tuple(train_records[int(index)] for index in indices)
        batch = _move(
            collate_v5_morphology_records(
                batch_records,
                maximum_nodes=max(record.node_count for record in batch_records),
                maximum_children=args.maximum_children,
            ),
            device,
        )
        t = torch.rand(
            len(batch_records),
            generator=torch_generator,
            device=device,
        ).clamp(0.02, 0.98)
        noisy = noise_v5_morphology_batch(batch, source_tensors, t, torch_generator)
        predictions = model(
            noisy["offspring"],
            noisy["regions"],
            batch["programs"],
            t,
            batch["node_mask"],
        )
        loss, components = v5_morphology_flow_loss(predictions, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(
            {
                "step": step,
                **components,
                "gradient_norm": float(gradient_norm.detach()),
            }
        )
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            evaluations.append(
                _evaluate(model, validation_records, marginals, args=args, step=step)
            )
            model.train()
            progress = {
                "schema_version": "phase1_v5_morphology_probe_progress.v1",
                "status": "running" if step < args.steps else "complete",
                "latest_loss": losses[-1],
                "evaluations": evaluations,
            }
            if args.progress is not None:
                _atomic_json(args.progress.resolve(), progress)
            if args.checkpoint is not None:
                _atomic_checkpoint(
                    args.checkpoint.resolve(),
                    {
                        "schema_version": "phase1_v5_morphology_probe_checkpoint.v1",
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "source_marginals": marginals,
                    },
                )

    result = {
        "schema_version": "phase1_v5_morphology_probe.v1",
        "status": "complete",
        "scope": (
            "bounded R0_train morphology flow with exact global tree and region programs; "
            "atom chemistry, bond orders, learned closure endpoints, Ugi L1, and guidance deferred"
        ),
        "inputs": {
            "r0": {"path": str(args.r0.resolve()), "sha256": sha256_file(args.r0)},
            "assignments": {
                "path": str(args.assignments.resolve()),
                "sha256": sha256_file(args.assignments),
            },
            "atom_vocabulary": {
                "path": str(args.atom_vocabulary.resolve()),
                "sha256": sha256_file(args.atom_vocabulary),
            },
            "architecture_config": {
                "path": str(args.architecture_config.resolve()),
                "sha256": sha256_file(args.architecture_config),
            },
        },
        "config": {
            "seed": args.seed,
            "device": args.device,
            "records": len(records),
            "train_records": len(train_records),
            "validation_records": len(validation_records),
            "maximum_record_atoms": args.maximum_record_atoms,
            "maximum_junction_budget": args.maximum_junction_budget,
            "maximum_children": args.maximum_children,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "learning_rate": args.learning_rate,
            "eval_every": args.eval_every,
            "eval_samples": args.eval_samples,
            "sample_steps": args.sample_steps,
        },
        "bounded_selection_coverage": coverage,
        "optimization": {
            "initial": losses[0],
            "final": losses[-1],
            "total_loss_reduction_fraction": (losses[0]["total"] - losses[-1]["total"])
            / losses[0]["total"],
        },
        "evaluations": evaluations,
    }
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
