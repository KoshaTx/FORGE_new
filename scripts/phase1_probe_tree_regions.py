#!/usr/bin/env python3
"""Run a bounded R0-only probe of region flow on settled lipid trees."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from forge.product.defog_feasibility import set_determinism
from forge.product.phase1_flow import (
    _training_records,
    build_corpus_from_config,
)
from forge.product.phase1_tree_region_flow import (
    ConditionalTreeRegionFlow,
    TreeRegionFlowError,
    TreeRegionSample,
    collate_region_records,
    noise_region_batch,
    region_depth_marginal,
    region_flow_loss,
    region_transition_marginal,
    sample_tree_regions,
    tree_region_statistics,
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
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument(
        "--transition-strengths",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
    )
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


def _move(batch: dict[str, object], device: object) -> dict[str, object]:
    return {key: value.to(device) for key, value in batch.items()}


def main() -> int:
    args = parse_args()
    if torch is None:
        raise SystemExit("tree-region probe requires torch")
    if (
        args.records < 2
        or args.maximum_record_atoms < 2
        or args.steps < 1
        or args.batch_size < 1
        or args.sample_steps < 2
        or not args.transition_strengths
        or min(args.transition_strengths) < 0
    ):
        raise SystemExit("invalid region probe configuration")
    set_determinism(args.seed, min(8, os.cpu_count() or 1))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA tree-region probe requested without CUDA")

    config, _, corpus = build_corpus_from_config(
        args.config.resolve(),
        REPO,
        smoke=False,
    )
    eligible = tuple(
        record
        for record in _training_records(corpus)
        if record.node_count <= args.maximum_record_atoms
    )
    if len(eligible) < args.records:
        raise TreeRegionFlowError(
            f"requested {args.records} records but only {len(eligible)} are eligible"
        )
    rng = np.random.default_rng(args.seed)
    selected_indices = rng.choice(len(eligible), size=args.records, replace=False)
    records = tuple(eligible[int(index)] for index in selected_indices)
    maximum_nodes = max(record.node_count for record in records)
    region_classes = int(config["model"]["region_classes"])
    depth_buckets = 32
    depth_marginal = region_depth_marginal(
        records,
        depth_buckets=depth_buckets,
        region_classes=region_classes,
    )
    transition_marginal = region_transition_marginal(
        records,
        region_classes=region_classes,
    )
    depth_source = torch.as_tensor(
        depth_marginal,
        dtype=torch.float32,
        device=device,
    )
    model = ConditionalTreeRegionFlow(
        region_classes=region_classes,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        attention_heads=args.attention_heads,
        maximum_heavy_atoms=int(config["model"]["maximum_heavy_atoms"]),
        depth_buckets=depth_buckets,
        maximum_degree=8,
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
            collate_region_records(
                batch_records,
                maximum_nodes=max(record.node_count for record in batch_records),
            ),
            device,
        )
        t = torch.rand(
            len(batch_records),
            generator=generator,
            device=device,
        ).clamp(0.02, 0.98)
        noisy = noise_region_batch(
            clean,
            depth_source,
            t,
            generator,
        )
        logits = model(
            noisy,
            clean["parents"],
            t,
            clean["node_mask"],
            clean["child_mask"],
        )
        loss, components = region_flow_loss(logits, clean)
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

    parents = [record.parents for record in records]
    generation = {}
    for index, strength in enumerate(args.transition_strengths):
        samples, runtime = sample_tree_regions(
            model,
            parents,
            depth_marginal,
            transition_marginal,
            sample_steps=args.sample_steps,
            batch_size=min(args.batch_size, len(records)),
            transition_strength=strength,
            seed=args.seed + 100 + index,
            device=args.device,
        )
        generation[str(strength)] = {
            "sampling": runtime,
            "statistics": tree_region_statistics(samples),
        }
    reference = [
        TreeRegionSample(
            regions=record.region_states.copy(),
            parents=record.parents.copy(),
        )
        for record in records
    ]
    result = {
        "schema_version": "phase1_tree_region_probe.v1",
        "status": "complete",
        "scope": (
            "R0-only region flow on settled reference trees without closures, "
            "atom identities, bond orders or guidance"
        ),
        "config": {
            "source": str(args.config.resolve()),
            "seed": args.seed,
            "device": args.device,
            "records": len(records),
            "maximum_record_atoms": args.maximum_record_atoms,
            "observed_maximum_nodes": maximum_nodes,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "attention_heads": args.attention_heads,
            "learning_rate": args.learning_rate,
            "transition_strengths": args.transition_strengths,
        },
        "optimization": {
            "initial": losses[0],
            "final": losses[-1],
            "total_loss_reduction_fraction": (losses[0]["total"] - losses[-1]["total"])
            / losses[0]["total"],
        },
        "reference": tree_region_statistics(reference),
        "generation": generation,
    }
    _atomic_json(args.output.resolve(), result)
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": "phase1_tree_region_probe_checkpoint.v1",
                "model_state_dict": model.state_dict(),
                "config": result["config"],
                "depth_marginal": depth_marginal,
                "transition_marginal": transition_marginal,
            },
            checkpoint_path,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
