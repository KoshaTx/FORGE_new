#!/usr/bin/env python3
"""Re-evaluate a Ugi morphology checkpoint on one fixed calibration draw."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forge.corpus.ugi_morphology_corpus import (
    load_expanded_ugi_morphology_corpus,
    sample_family_balanced_records,
)
from forge.model.ugi_morphology_flow import UgiMorphologyFlow
from experiments.phase1.product_l1.training.ugi_morphology_training import _validation_loss

REPO = Path(__file__).resolve().parents[3]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _model(model_config: dict[str, Any]) -> UgiMorphologyFlow:
    return UgiMorphologyFlow(
        maximum_children=int(model_config["maximum_children"]),
        maximum_component_atoms=int(model_config["maximum_component_atoms"]),
        maximum_total_atoms=int(model_config["maximum_total_atoms"]),
        maximum_junction_budget=int(model_config["maximum_junction_budget"]),
        maximum_cycle_rank=int(model_config["maximum_cycle_rank"]),
        maximum_attachment_count=(
            int(model_config["maximum_attachment_count"])
            if "maximum_attachment_count" in model_config
            else None
        ),
        hidden_dim=int(model_config["hidden_dim"]),
        layers=int(model_config["layers"]),
        dropout=float(model_config["dropout"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=16)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    seed = int(config["seed"])
    inputs = {label: REPO / record["path"] for label, record in config["inputs"].items()}
    corpus = load_expanded_ugi_morphology_corpus(
        inputs["component_exemplar_ledger"],
        inputs["semantic_products"],
        inputs["semantic_atoms"],
        inputs["atom_vocabulary"],
    )
    calibration_records = sample_family_balanced_records(
        corpus,
        fold="calibration",
        count=int(config["sampling"]["calibration_pool_size"]),
        rng=np.random.default_rng(seed + 400),
        id_prefix="fixed-calibration",
    )
    package = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = dict(package["model_config"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    initial_model = _model(model_config)
    trained_model = _model(model_config)
    trained_model.load_state_dict(package["model_state_dict"])
    source_tensors = torch.as_tensor(package["source_marginals"], dtype=torch.float32)
    validation_seed = seed + 20_000
    common = {
        "records": calibration_records,
        "source_tensors": source_tensors,
        "maximum_children": int(model_config["maximum_children"]),
        "batch_size": int(config["full"]["batch_size"]),
        "batches": args.batches,
        "seed": validation_seed,
        "device": torch.device("cpu"),
        "label_smoothing": float(config["full"].get("label_smoothing", 0.0)),
    }
    initial = _validation_loss(initial_model, **common)
    trained = _validation_loss(trained_model, **common)
    result = {
        "schema_version": "phase1_ugi_morphology_fixed_validation_audit.v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(package["step"]),
        "calibration_records": len(calibration_records),
        "validation_batches": args.batches,
        "validation_seed": validation_seed,
        "initial": initial,
        "trained": trained,
        "total_loss_change": trained["total"] - initial["total"],
        "improved_on_fixed_validation": trained["total"] < initial["total"],
    }
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
