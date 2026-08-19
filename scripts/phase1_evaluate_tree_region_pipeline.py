#!/usr/bin/env python3
"""Compose trained tree and region probes into one morphology endpoint audit."""

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
    TreeRegionSample,
    sample_tree_regions,
    tree_region_statistics,
)
from forge.product.phase1_tree_topology_flow import (
    OffspringTreeFlow,
    TreeTopologySample,
    offspring_to_parents,
    record_offspring_counts,
    sample_tree_topologies,
    tree_topology_statistics,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - required by the model extra
    torch = None

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-checkpoint", type=Path, required=True)
    parser.add_argument("--region-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_product_pretrain_v3.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--transition-strength", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260801)
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


def main() -> int:
    args = parse_args()
    if torch is None:
        raise SystemExit("composed morphology evaluation requires torch")
    if (
        args.sample_count < 1
        or args.sample_steps < 2
        or args.batch_size < 1
        or args.transition_strength < 0
    ):
        raise SystemExit("invalid composed morphology evaluation")
    set_determinism(args.seed, min(8, os.cpu_count() or 1))
    device = torch.device(args.device)
    tree_package = torch.load(
        args.tree_checkpoint.resolve(),
        map_location=device,
        weights_only=False,
    )
    region_package = torch.load(
        args.region_checkpoint.resolve(),
        map_location=device,
        weights_only=False,
    )
    if (
        tree_package.get("schema_version") != "phase1_tree_topology_probe_checkpoint.v1"
        or region_package.get("schema_version") != "phase1_tree_region_probe_checkpoint.v1"
    ):
        raise SystemExit("incompatible morphology probe checkpoint")

    tree_config = tree_package["config"]
    tree_model = OffspringTreeFlow(
        maximum_children=int(tree_config["maximum_children"]),
        hidden_dim=int(tree_config["hidden_dim"]),
        layers=int(tree_config["layers"]),
        attention_heads=int(tree_config["attention_heads"]),
        maximum_heavy_atoms=len(tree_package["node_count_distribution"]) - 1,
        dropout=0.0,
    ).to(device)
    tree_model.load_state_dict(tree_package["model_state_dict"])
    region_config = region_package["config"]
    depth_marginal = np.asarray(
        region_package["depth_marginal"],
        dtype=np.float64,
    )
    transition_marginal = np.asarray(
        region_package["transition_marginal"],
        dtype=np.float64,
    )
    region_model = ConditionalTreeRegionFlow(
        region_classes=transition_marginal.shape[0],
        hidden_dim=int(region_config["hidden_dim"]),
        layers=int(region_config["layers"]),
        attention_heads=int(region_config["attention_heads"]),
        maximum_heavy_atoms=tree_model.maximum_heavy_atoms,
        depth_buckets=depth_marginal.shape[0],
        maximum_degree=8,
        dropout=0.0,
    ).to(device)
    region_model.load_state_dict(region_package["model_state_dict"])

    trees, tree_runtime = sample_tree_topologies(
        tree_model,
        np.asarray(tree_package["offspring_marginal"], dtype=np.float64),
        np.asarray(
            tree_package["node_count_distribution"],
            dtype=np.float64,
        ),
        sample_count=args.sample_count,
        sample_steps=args.sample_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    regions, region_runtime = sample_tree_regions(
        region_model,
        [sample.parents for sample in trees],
        depth_marginal,
        transition_marginal,
        sample_steps=args.sample_steps,
        batch_size=args.batch_size,
        transition_strength=args.transition_strength,
        seed=args.seed + 1,
        device=args.device,
    )

    config, _, corpus = build_corpus_from_config(
        args.config.resolve(),
        REPO,
        smoke=False,
    )
    eligible = tuple(
        record
        for record in _training_records(corpus)
        if record.node_count <= int(tree_config["maximum_record_atoms"])
        and record_offspring_counts(record).max(initial=0) <= int(tree_config["maximum_children"])
    )
    reference_rng = np.random.default_rng(int(tree_config["seed"]))
    reference_indices = reference_rng.choice(
        len(eligible),
        size=int(tree_config["records"]),
        replace=False,
    )
    reference_records = tuple(eligible[int(index)] for index in reference_indices)
    reference_trees = [
        TreeTopologySample(
            offspring=record_offspring_counts(record),
            parents=offspring_to_parents(record_offspring_counts(record)),
        )
        for record in reference_records
    ]
    reference_regions = [
        TreeRegionSample(
            regions=record.region_states.copy(),
            parents=record.parents.copy(),
        )
        for record in reference_records
    ]
    result = {
        "schema_version": "phase1_tree_region_pipeline_evaluation.v1",
        "status": "complete",
        "scope": (
            "composed R0-only tree and region morphology endpoint without "
            "closures, atom identities, bond orders or guidance"
        ),
        "config": {
            "tree_checkpoint": str(args.tree_checkpoint.resolve()),
            "region_checkpoint": str(args.region_checkpoint.resolve()),
            "product_config": str(args.config.resolve()),
            "device": args.device,
            "sample_count": args.sample_count,
            "sample_steps": args.sample_steps,
            "batch_size": args.batch_size,
            "transition_strength": args.transition_strength,
            "seed": args.seed,
            "maximum_heavy_atoms": int(config["model"]["maximum_heavy_atoms"]),
        },
        "runtime": {
            "tree": tree_runtime,
            "region": region_runtime,
        },
        "reference": {
            "tree": tree_topology_statistics(
                reference_trees,
                maximum_children=tree_model.maximum_children,
            ),
            "regions": tree_region_statistics(reference_regions),
        },
        "generated": {
            "tree": tree_topology_statistics(
                trees,
                maximum_children=tree_model.maximum_children,
            ),
            "regions": tree_region_statistics(regions),
        },
    }
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
