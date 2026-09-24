"""Bounded matched warm-start comparison of serialization and exact repeat supervision."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from forge.core.hashing import sha256_file
from forge.core.io import write_json
from forge.corpus.qualified_program_cache import _vocabulary
from forge.model.compose_lipid_sampling import sample
from forge.model.compose_lipid_training import compose_lipid_forward_loss
from forge.model.synthesis_program_training import build_synthesis_program_flow, move_tensors

FIELDS = ("nodes", "parents", "parent_bonds", "closure_left", "closure_right", "closure_bonds")


def setup(payload, device, seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed)
    model = build_synthesis_program_flow(
        vocabulary=_vocabulary(payload["vocabulary"]),
        node_classes=payload["node_classes"],
        model_config=payload["config"]["model"],
        device=device,
    ).float()
    model.load_state_dict(payload["model"], strict=True)
    node = torch.tensor(payload["noise"]["node"], device=device, dtype=torch.float32)
    bond = torch.tensor(payload["noise"]["bond"], device=device, dtype=torch.float32)
    return model, node, bond


def update(model, optimizer, batch, node, bond, generator, weights, repeat_mode):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    times = torch.rand(len(batch["nodes"]), device=batch["nodes"].device, generator=generator)
    loss, metrics = compose_lipid_forward_loss(
        model,
        batch,
        architecture="reaction_program_graph_transformer",
        node_marginal=node,
        bond_marginal=bond,
        times=times,
        generator=generator,
        semantic_weights=weights,
        repeat_supervision=repeat_mode,
    )
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    values = {key: float(value.detach()) for key, value in metrics.items()}
    values.update(loss=float(loss.detach()), gradient_norm=float(norm))
    if not np.isfinite(list(values.values())).all():
        raise ValueError("Nonfinite repeat-pilot update")
    return values


def run_study(payload, protocol, output: Path, commit):
    if protocol["optimizer_steps_per_arm"] != 512 or protocol["replicates"] != 4:
        raise ValueError("The frozen pilot requires 512 updates and four evaluation replicates")
    if protocol["automatic_retries"] != 0 or protocol["production_checkpoint_promotion"]:
        raise ValueError("The pilot cannot retry automatically or promote its checkpoints")
    if len(payload["batches"]) != 128 or any(len(b["nodes"]) != 32 for b in payload["batches"]):
        raise ValueError("Pilot must retain all 4096 presampled TRAIN draws")
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    model, node, bond = setup(payload, device, protocol["seed"])
    reports = {}
    start = time.monotonic()

    def evaluate(name):
        model.eval()
        rows = []
        with torch.inference_mode():
            for replicate in range(protocol["replicates"]):
                for entry in payload["evaluation"]:
                    batch = move_tensors(entry["batch"], device)
                    seed = protocol["generation_seed"] + replicate * 1000 + entry["offset"]
                    terminal = sample(model, batch, node, bond, steps=64, seed=seed)
                    for index, record in enumerate(entry["records"]):
                        rows.append(
                            dict(
                                record,
                                attempt=len(rows),
                                replicate=replicate,
                                seed=seed,
                                terminal={
                                    key: terminal[key][
                                        index,
                                        : (
                                            record["closure_count"]
                                            if key.startswith("closure")
                                            else record["node_count"]
                                        ),
                                    ]
                                    .cpu()
                                    .tolist()
                                    for key in FIELDS
                                },
                            )
                        )
        write_json(output / (name + "-generation.json"), rows)
        commit()
        return rows

    reports["frozen"] = {"generation": evaluate("frozen"), "optimizer_steps": 0}
    for arm in protocol["arms"]:
        name = arm["name"]
        model.load_state_dict(payload["model"], strict=True)
        # All arms are warm starts from identical weights with identical fresh optimizer state.
        optimizer = torch.optim.AdamW(model.parameters(), **payload["config"]["optimizer"])
        torch.manual_seed(protocol["seed"])
        generator = torch.Generator(device=device).manual_seed(protocol["seed"])
        weights = dict(
            payload["config"]["semantic_weights"], repeat_consistency_weight=arm["repeat_weight"]
        )
        history = []
        arm_start = time.monotonic()
        for step in range(1, 513):
            batch = move_tensors(payload["batches"][(step - 1) % 128], device)
            metrics = update(
                model, optimizer, batch, node, bond, generator, weights, arm["repeat_supervision"]
            )
            history.append(dict(step=step, **metrics))
            if step % 128 == 0:
                # Publish checkpoints before evaluation; a lost client cannot discard fitting.
                destination = output / f"{name}-checkpoint-{step}.pt"
                temporary = destination.with_suffix(".tmp")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                        "generator": generator.get_state(),
                        "completed_steps": step,
                        "protocol": protocol,
                        "arm": arm,
                    },
                    temporary,
                )
                temporary.replace(destination)
                write_json(
                    output / (name + "-progress.json"),
                    {"completed_steps": step, "metrics": history},
                )
                commit()
                for old in sorted(output.glob(name + "-checkpoint-*.pt"))[:-2]:
                    old.unlink()
        checkpoint = output / f"{name}-checkpoint-512.pt"
        reports[name] = {
            "optimizer_steps": 512,
            "draws": 16384,
            "seconds": time.monotonic() - arm_start,
            "metrics": history,
            "checkpoint": {"name": checkpoint.name, "sha256": str(sha256_file(checkpoint))},
            "generation": evaluate(name),
        }
        del optimizer
        write_json(output / "completed-arms.json", reports)
        commit()
    result = {
        "schema_version": "forge.compose_lipid_repeat_pilot_result.v1",
        "completed": True,
        "protocol": protocol,
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "precision": "float32",
        "deterministic": True,
        "seconds": time.monotonic() - start,
        "arms": reports,
        "promoted": False,
    }
    write_json(output / "result.json", result)
    commit()
    return result
