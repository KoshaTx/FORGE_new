#!/usr/bin/env python3
"""Run a bounded R0-only probe of the exact BFS offspring tree flow."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from forge.design.flow.defog_feasibility import set_determinism
from forge.design.flow.phase1_flow import (
    _training_records,
    build_corpus_from_config,
)
from forge.design.flow.phase1_tree_topology_flow import (
    OffspringTreeFlow,
    TreeTopologyFlowError,
    TreeTopologySample,
    collate_tree_records,
    noise_tree_batch,
    offspring_marginal,
    offspring_to_parents,
    record_offspring_counts,
    sample_tree_topologies,
    tree_flow_loss,
    tree_topology_statistics,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - required by the model extra
    torch = None

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_product_pretrain_v3.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--records", type=int, default=64)
    parser.add_argument("--maximum-record-atoms", type=int, default=64)
    parser.add_argument("--maximum-children", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def _atomic_json(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
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


def _count_distribution(values: list[int], maximum: int) -> np.ndarray:
    counts = np.bincount(values, minlength=maximum + 1).astype(np.float64)
    if counts.size > maximum + 1 or counts.sum() == 0:
        raise TreeTopologyFlowError("invalid count observations")
    return counts / counts.sum()


def _move(batch: dict[str, object], device: object) -> dict[str, object]:
    return {key: value.to(device) for key, value in batch.items()}


def main() -> int:
    args = parse_args()
    if torch is None:
        raise SystemExit("tree topology probe requires torch")
    if (
        args.records < 2
        or args.maximum_record_atoms < 2
        or args.maximum_children < 1
        or args.steps < 1
        or args.batch_size < 1
        or args.sample_count < 1
        or args.sample_steps < 2
    ):
        raise SystemExit("invalid positive probe count")
    set_determinism(args.seed, min(8, os.cpu_count() or 1))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA tree probe requested without CUDA")

    config, _, corpus = build_corpus_from_config(
        args.config.resolve(),
        REPO,
        smoke=False,
    )
    eligible = tuple(
        record
        for record in _training_records(corpus)
        if record.node_count <= args.maximum_record_atoms
        and record_offspring_counts(record).max(initial=0) <= args.maximum_children
    )
    if len(eligible) < args.records:
        raise TreeTopologyFlowError(
            f"requested {args.records} records but only {len(eligible)} are eligible"
        )
    rng = np.random.default_rng(args.seed)
    selected_indices = rng.choice(len(eligible), size=args.records, replace=False)
    records = tuple(eligible[int(index)] for index in selected_indices)
    maximum_nodes = max(record.node_count for record in records)
    source_marginal = offspring_marginal(
        records,
        maximum_children=args.maximum_children,
    )
    source = torch.as_tensor(
        source_marginal,
        dtype=torch.float32,
        device=device,
    )
    model = OffspringTreeFlow(
        maximum_children=args.maximum_children,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        attention_heads=args.attention_heads,
        maximum_heavy_atoms=int(config["model"]["maximum_heavy_atoms"]),
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    losses = []
    model.train()
    for step in range(args.steps):
        indices = rng.choice(
            len(records),
            size=min(args.batch_size, len(records)),
            replace=False,
        )
        batch_records = tuple(records[int(index)] for index in indices)
        clean = _move(
            collate_tree_records(
                batch_records,
                maximum_nodes=max(record.node_count for record in batch_records),
                maximum_children=args.maximum_children,
            ),
            device,
        )
        t = torch.rand(
            len(batch_records),
            generator=generator,
            device=device,
        ).clamp(0.02, 0.98)
        noisy = noise_tree_batch(clean, source, t, generator)
        predictions = model(
            noisy["offspring"],
            t,
            clean["node_mask"],
        )
        loss, components = tree_flow_loss(predictions, clean)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(
            {
                "step": step + 1,
                **components,
                "gradient_norm": float(gradient_norm.detach()),
            }
        )

    node_distribution = _count_distribution(
        [record.node_count for record in records],
        int(config["model"]["maximum_heavy_atoms"]),
    )
    generated, sampling = sample_tree_topologies(
        model,
        source_marginal,
        node_distribution,
        sample_count=args.sample_count,
        sample_steps=args.sample_steps,
        batch_size=min(args.batch_size, args.sample_count),
        seed=args.seed + 1,
        device=args.device,
    )
    reference = [
        TreeTopologySample(
            offspring=record_offspring_counts(record),
            parents=offspring_to_parents(record_offspring_counts(record)),
        )
        for record in records
    ]
    result = {
        "schema_version": "phase1_tree_topology_probe.v1",
        "status": "complete",
        "scope": (
            "R0-only bounded tree-topology flow without regions, closures, "
            "atom identities, bond orders or guidance"
        ),
        "config": {
            "source": str(args.config.resolve()),
            "seed": args.seed,
            "device": args.device,
            "records": len(records),
            "maximum_record_atoms": args.maximum_record_atoms,
            "observed_maximum_nodes": maximum_nodes,
            "maximum_children": args.maximum_children,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "attention_heads": args.attention_heads,
            "learning_rate": args.learning_rate,
        },
        "optimization": {
            "initial": losses[0],
            "final": losses[-1],
            "total_loss_reduction_fraction": (losses[0]["total"] - losses[-1]["total"])
            / losses[0]["total"],
        },
        "sampling": sampling,
        "reference": tree_topology_statistics(
            reference,
            maximum_children=args.maximum_children,
        ),
        "generated": tree_topology_statistics(
            generated,
            maximum_children=args.maximum_children,
        ),
    }
    _atomic_json(args.output.resolve(), result)
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": "phase1_tree_topology_probe_checkpoint.v1",
                "model_state_dict": model.state_dict(),
                "config": result["config"],
                "offspring_marginal": source_marginal,
                "node_count_distribution": node_distribution,
            },
            checkpoint_path,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
