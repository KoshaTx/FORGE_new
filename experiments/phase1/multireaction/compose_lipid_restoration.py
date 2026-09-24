"""Matched bounded restoration study, with durable optimizer/RNG checkpoints."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.phase1.multireaction.compose_lipid_repeat_pilot import FIELDS, sample
from forge.core.hashing import sha256_file
from forge.core.io import write_json
from forge.corpus.qualified_program_cache import _vocabulary
from forge.model.compose_lipid_training import compose_lipid_backward
from forge.model.synthesis_program_sampling import decode_synthesis_program_strict_argmax
from forge.model.synthesis_program_training import build_synthesis_program_flow, move_tensors
from forge.model.training_restart import (
    atomic_torch_save,
    capture_training_random_state,
    restore_training_random_state,
)


def setup(payload, arm, device, seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed)
    config = payload["configs"][arm]
    model = build_synthesis_program_flow(
        vocabulary=_vocabulary(payload["vocabulary"]),
        node_classes=len(payload["atoms"]),
        model_config=config["model"],
        device=device,
    ).float()
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    allowed = (
        "program_encoder.role_morphology_embeddings.",
        "offspring_output.",
        "terminal_chemistry_adapter.",
        "closure_output_adapter.",
    )
    if unexpected or any(not name.startswith(allowed) for name in missing):
        raise ValueError(f"Unexpected warm-start mismatch: {missing}, {unexpected}")
    # Added semantic embeddings and residual output adapters start as the identity.
    # Their parameters remain trainable; the auxiliary offspring head is newly initialized.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in missing and (
                "role_morphology_embeddings" in name or ".experts." in name and ".3." in name
            ):
                p.zero_()
    noise = payload["noise"][arm]
    return (
        model,
        torch.tensor(noise["node"], device=device, dtype=torch.float32),
        torch.tensor(noise["bond"], device=device, dtype=torch.float32),
        missing,
    )


def update(model, optimizer, clean, node, bond, config, generator):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = compose_lipid_backward(
        model,
        clean,
        architecture=config["model"]["architecture"],
        node_marginal=node,
        bond_marginal=bond,
        times=torch.rand(len(clean["nodes"]), device=clean["nodes"].device, generator=generator),
        generator=generator,
        semantic_weights=config["semantic_weights"],
        repeat_supervision="exact_fragment",
        objective=config.get("objective"),
    )
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True
    )
    optimizer.step()
    values = {k: float(v.detach()) for k, v in metrics.items()}
    values.update(loss=float(loss.detach()), gradient_norm=float(norm))
    if not np.isfinite(list(values.values())).all():
        raise ValueError("Nonfinite restoration update")
    return values


def generate(model, node, bond, payload, protocol, device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for repeat in range(protocol["replicates"]):
            for group_index, group in enumerate(payload["evaluation"]):
                # Every model receives the same separately sampled core and coarse layout.
                batch = move_tensors(group["batch"], device)
                raw, predictions = sample(
                    model,
                    batch,
                    node,
                    bond,
                    steps=protocol["sample_steps"],
                    seed=protocol["generation_seed"] + repeat * 1000 + group_index,
                    return_predictions=True,
                )
                strict, reasons = decode_synthesis_program_strict_argmax(
                    predictions,
                    batch,
                    group["records"],
                    payload["atoms"],
                    qualified_core_units=group["core_units"],
                    confine_origin_edges=True,
                    reserve_fixed_closures=True,
                )
                for index, record in enumerate(group["records"]):

                    def packed(state):
                        return {
                            k: state[k][
                                index,
                                : (
                                    record.graph.closure_count
                                    if k.startswith("closure")
                                    else record.node_count
                                ),
                            ]
                            .cpu()
                            .tolist()
                            for k in FIELDS
                        }

                    rows.append(
                        dict(
                            target_id=record.graph.structure_id,
                            program_id=record.program_id,
                            node_count=record.node_count,
                            closure_count=record.graph.closure_count,
                            repeat=repeat,
                            raw=packed(raw),
                            strict=packed(strict),
                            abstention=reasons[index],
                            family=group["families"][index],
                            scope=group["scopes"][index],
                        )
                    )
    return rows


def run_study(payload, protocol, output: Path, commit):
    if output.exists():
        raise FileExistsError("Paid study is never retried automatically")
    output.mkdir(parents=True)
    device = torch.device("cuda")
    start = time.monotonic()
    results = {}
    for arm in ("baseline", "restored"):
        model, node, bond, missing = setup(payload, arm, device, protocol["seed"])
        optimizer = torch.optim.AdamW(model.parameters(), **payload["configs"][arm]["optimizer"])
        generator = torch.Generator(device=device).manual_seed(protocol["seed"])
        rng = np.random.default_rng(protocol["seed"])
        batches = [move_tensors(b[arm], device) for b in payload["batches"]]
        metrics = []
        learning_curve = []
        for step in range(1, protocol["optimizer_steps_per_arm"] + 1):
            stamp = time.monotonic()
            values = update(
                model,
                optimizer,
                batches[(step - 1) % len(batches)],
                node,
                bond,
                payload["configs"][arm],
                generator,
            )
            values["seconds"] = time.monotonic() - stamp
            metrics.append(values)
            if step in protocol["learning_curve_steps"]:
                learning_curve.append(
                    dict(
                        step=step,
                        mean_recent_loss=float(np.mean([m["loss"] for m in metrics[-64:]])),
                    )
                )
            if (
                step % protocol["checkpoint_interval"] == 0
                or step == protocol["optimizer_steps_per_arm"]
            ):
                path = output / f"{arm}-checkpoint-{step}.pt"
                state = capture_training_random_state(rng, generator, device=device)
                atomic_torch_save(
                    path,
                    dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        random=state,
                        protocol=protocol,
                        arm=arm,
                        step=step,
                        config=payload["configs"][arm],
                    ),
                )
                write_json(
                    output / "progress.json",
                    dict(arm=arm, step=step, checkpoint=path.name, completed=False),
                )
                commit()
                restore_training_random_state(state, rng, generator, device=device)
                for old in sorted(
                    output.glob(f"{arm}-checkpoint-*.pt"),
                    key=lambda p: int(p.stem.rsplit("-", 1)[1]),
                )[:-2]:
                    old.unlink()
                print(
                    f'{arm}: {step}/{protocol["optimizer_steps_per_arm"]} loss={values["loss"]:.4f}',
                    flush=True,
                )
        del batches
        checkpoint = output / f'{arm}-checkpoint-{protocol["optimizer_steps_per_arm"]}.pt'
        results[arm] = dict(
            metrics=metrics,
            learning_curve=learning_curve,
            new_parameters=missing,
            checkpoint=dict(path=checkpoint.name, sha256=str(sha256_file(checkpoint))),
            generation=generate(model, node, bond, payload, protocol, device),
        )
        write_json(output / f"{arm}-result.json", results[arm])
        commit()
        del model, optimizer
        torch.cuda.empty_cache()
    result = dict(completed=True, protocol=protocol, arms=results, seconds=time.monotonic() - start)
    write_json(output / "result.json", result)
    commit()
    return result
