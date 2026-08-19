#!/usr/bin/env python3
"""Which role channel does the model that is actually trained receive, and is it removable?

This supersedes the applicability of the earlier origin-channel information audit, which
measured `AdapterNodeConditioning`. That module carries a five-state `origin_embedding` plus
port, core-position and distance channels, and it lives in `UgiChemistryFlow`. The production
generator is `UgiJointSparseFlow`, it does not instantiate `AdapterNodeConditioning`, and it has
no `origin_embedding` at all. The earlier audit therefore characterised a module the production
model does not use, and its 0.552 recoverability figure does not apply to the production
architecture. That number stands for what it measured and is withdrawn as a statement about the
production model.

The production model's role channel is `role_embedding`, of shape (3, hidden_dim). This audit
asks the same question of it that the earlier one asked of the wrong module: if the embedding
were deleted, would the model still know each node's precursor role?

The answer is structural rather than statistical. The conditioning program declares per-role node
counts, and the sparse serialization lays nodes out in contiguous role blocks in that order. So
role is not merely predictable from the surviving inputs, it is exactly reconstructible from the
conditioning the model always receives. `within_role_positions` marks the same boundaries a
second time by resetting to zero at each block.

Consequences, both recorded before any ablation run:

  - under the prespecified rule in the evidence contract, recoverability at or above 0.95 means
    deleting the embedding is not an information ablation. This is recoverability of 1.0;
  - a genuine role-information ablation would have to change the state layout so that role is no
    longer implied by position, which changes the object being modelled rather than removing a
    channel from it.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from forge.product.ugi_joint_sparse_flow import (  # noqa: E402
    UgiJointSparseFlow,
    collate_ugi_joint_sparse_records,
)
from forge.product.ugi_training_cache import load_ugi_training_cache  # noqa: E402

CHECKPOINT = "results/phase1/ugi_joint_sparse_balanced_v2_full/checkpoint_step_1000.pt"
PRODUCTION = "results/phase1/ugi_joint_sparse_production_refit_full/checkpoint_step_1700.pt"
CACHE = "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt"
RECORDS_PER_FOLD = 3000
BATCH = 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "results/phase1/forge_production_role_channel_audit_v1/result.json",
    )
    args = parser.parse_args()
    started = time.time()

    corpus, records_by_fold = load_ugi_training_cache(REPO / CACHE)
    checkpoint = torch.load(REPO / CHECKPOINT, map_location="cpu", weights_only=False)
    architecture = dict(checkpoint["model_config"])
    architecture.pop("source_probability_floor", None)
    model = UgiJointSparseFlow(atom_classes=len(corpus.atom_vocabulary), **architecture)

    names = {name for name, _ in model.named_modules() if name}
    parameters = {name: tuple(p.shape) for name, p in model.named_parameters()}
    total_parameters = sum(p.numel() for p in model.parameters())
    role_parameters = {n: p.numel() for n, p in model.named_parameters()
                       if n.startswith("role_embedding")}
    architecture_facts = {
        "model_class": "UgiJointSparseFlow",
        "instantiates_adapter_node_conditioning": any("adapter" in n.lower() for n in names),
        "has_origin_embedding": any("origin" in n.lower() for n in parameters),
        "role_channel": "role_embedding",
        "role_embedding_shape": parameters.get("role_embedding.weight"),
        "within_role_position_shape": parameters.get("position_embedding.weight"),
        "total_parameters": total_parameters,
        "role_embedding_parameters": sum(role_parameters.values()),
        "role_embedding_share_of_model": sum(role_parameters.values()) / total_parameters,
    }
    print(f"model class                          {architecture_facts['model_class']}")
    print(f"instantiates AdapterNodeConditioning {architecture_facts['instantiates_adapter_node_conditioning']}")
    print(f"has an origin_embedding              {architecture_facts['has_origin_embedding']}")
    print(f"role channel                         role_embedding "
          f"{architecture_facts['role_embedding_shape']}")
    print(f"total parameters                     {total_parameters:,}")
    print(f"role_embedding parameters            {sum(role_parameters.values()):,} "
          f"({architecture_facts['role_embedding_share_of_model'] * 100:.4f}% of the model)")

    # ---- is role reconstructible from the conditioning the model always receives?
    checked = 0
    role_mismatch = 0
    position_mismatch = 0
    count_mismatch = 0
    for fold in ("train", "calibration", "heldout"):
        records = list(itertools.islice(records_by_fold[fold], RECORDS_PER_FOLD))
        for start in range(0, len(records), BATCH):
            chunk = tuple(records[start:start + BATCH])
            batch = collate_ugi_joint_sparse_records(
                chunk, maximum_nodes=max(r.node_count for r in chunk),
                maximum_children=int(architecture["maximum_children"]),
                maximum_closures=int(architecture["maximum_cycle_rank"]) * 3,
                maximum_decorations=int(architecture["maximum_decorations"]),
            )
            roles = batch["role_states"]
            positions = batch["within_role_positions"]
            mask = batch["node_mask"]
            for index, record in enumerate(chunk):
                checked += 1
                expected_roles: list[int] = []
                expected_positions: list[int] = []
                for role_index, count in enumerate(record.program.node_counts):
                    expected_roles += [role_index] * count
                    expected_positions += list(range(count))
                if sum(record.program.node_counts) != record.node_count:
                    count_mismatch += 1
                if roles[index][mask[index]].tolist() != expected_roles:
                    role_mismatch += 1
                if positions[index][mask[index]].tolist() != expected_positions:
                    position_mismatch += 1

    deterministic = role_mismatch == 0 and position_mismatch == 0 and count_mismatch == 0
    print(f"\nrecords checked across all three folds  {checked}")
    print(f"role_states differ from program blocks  {role_mismatch}")
    print(f"within_role_positions differ            {position_mismatch}")
    print(f"program node counts disagree with size  {count_mismatch}")

    verdict = (
        "role_states and within_role_positions are EXACT deterministic functions of the "
        "conditioning program, which the model always receives. Recoverability is 1.0, not the "
        "0.552 measured for the different module in the earlier audit. Under the contract rule "
        "for recoverability at or above 0.95, deleting role_embedding is NOT an information "
        "ablation of the production model: it removes a redundant re-encoding, not the role "
        "information itself. A genuine role-information ablation would require changing the "
        "state layout so role is no longer implied by position, which changes the modelled "
        "object rather than removing a channel."
        if deterministic else
        "role_states are not exactly reconstructible from the program; an embedding ablation "
        "does remove information and the contract's original framing applies."
    )
    print(f"\n{verdict}")

    payload = {
        "schema_version": "phase1_forge_production_role_channel_audit.v1",
        "status": "complete_architecture_and_determinism_audit_no_training_performed",
        "supersedes_applicability_of":
            "results/phase1/forge_origin_channel_information_audit_v1/result.json, which measured "
            "AdapterNodeConditioning, a module the production generator does not instantiate",
        "inputs": {
            "development_checkpoint": {"path": CHECKPOINT, "sha256": sha256_file(REPO / CHECKPOINT)},
            "training_cache": {"path": CACHE, "sha256": sha256_file(REPO / CACHE)},
        },
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(), "torch": torch.__version__,
                    "elapsed_seconds": round(time.time() - started, 1)},
        "architecture": architecture_facts,
        "determinism_check": {
            "records_checked": checked,
            "folds": ["train", "calibration", "heldout"],
            "records_per_fold_cap": RECORDS_PER_FOLD,
            "role_states_mismatches": role_mismatch,
            "within_role_position_mismatches": position_mismatch,
            "program_node_count_mismatches": count_mismatch,
            "role_is_deterministic_from_conditioning": deterministic,
        },
        "verdict": verdict,
        "nonclaims": [
            "This does not say the role-resolved formulation is unimportant. It says the opposite: "
            "role structure is carried by the conditioning program and the state layout, not by a "
            "removable embedding, so it cannot be toggled off without changing the modelled object.",
            "Deleting role_embedding remains a runnable experiment. It would test whether an "
            "explicit re-encoding of a deterministic function of the conditioning aids "
            "optimization. It would not test whether precursor-role information helps.",
            "The earlier audit's numbers are correct for AdapterNodeConditioning. Only their "
            "application to the production generator is withdrawn.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
