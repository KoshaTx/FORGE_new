"""Bounded padding benchmark on frozen TRAIN batches; no production checkpoint output."""

from __future__ import annotations

import os
import platform
import statistics
import time
from typing import Any


def run_benchmark(payload: dict, *, device: str = "cpu", profile: bool = False) -> dict:
    import torch

    from forge.corpus.qualified_program_cache import _vocabulary
    from forge.model.compose_lipid_training import compose_lipid_forward_loss
    from forge.model.reaction_program_flow import noise_synthesis_program_batch
    from forge.model.reaction_program_transformer import reaction_program_transformer_loss
    from forge.model.synthesis_program_training import (
        _synthesis_program_predict,
        build_synthesis_program_flow,
        move_tensors,
    )

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    target = torch.device(device)
    config = payload["config"]
    weights = config["semantic_weights"]
    node = torch.tensor(payload["noise"]["node"], device=target)
    bond = torch.tensor(payload["noise"]["bond"], device=target)
    torch.manual_seed(payload["seed"])
    model = build_synthesis_program_flow(
        vocabulary=_vocabulary(payload["vocabulary"]),
        node_classes=payload["node_classes"],
        model_config=config["model"],
        device=target,
    ).float()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def sync():
        if target.type == "cuda":
            torch.cuda.synchronize()

    def decode(values):
        return {
            k: torch.tensor(v["values"], dtype=getattr(torch, v["dtype"]))
            for k, v in values.items()
        }

    reports = []
    for group in payload["groups"]:
        batches = {mode: decode(group[mode]) for mode in ("fixed", "batch")}
        full = move_tensors(batches["fixed"], target)
        small = move_tensors(batches["batch"], target)
        width = small["nodes"].shape[1]
        # Corrupt once, then compare identical active inputs. Padding changes RNG consumption
        # in production, so equal seeds alone would not be a numerical-equivalence test.
        times = torch.linspace(0.0, 1.0, full["nodes"].shape[0], device=target)
        noisy = noise_synthesis_program_batch(
            full, node, bond, times, torch.Generator(device=target).manual_seed(payload["seed"])
        )
        trimmed = {
            k: v[:, :width] if k in {"nodes", "parents", "parent_bonds"} else v
            for k, v in noisy.items()
        }
        reference: dict[str, Any] | None = None
        equivalence = {}
        for mode, clean, state in (("fixed", full, noisy), ("batch", small, trimmed)):
            model.load_state_dict(initial)
            model.eval()  # Disable stochastic dropout for the mathematical equivalence check.
            model.zero_grad(set_to_none=True)
            output = _synthesis_program_predict(model, clean, state, times)
            loss, metrics = reaction_program_transformer_loss(
                output, clean, **weights, materialize_metrics=False
            )
            loss.backward()
            observed = {
                "loss": loss.detach().clone(),
                "metrics": {k: v.detach().clone() for k, v in metrics.items()},
                "gradients": {
                    k: p.grad.detach().clone()
                    for k, p in model.named_parameters()
                    if p.grad is not None
                },
            }
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
            optimizer.step()
            observed["updated"] = {k: p.detach().clone() for k, p in model.named_parameters()}
            if reference is None:
                reference = observed
            else:
                torch.testing.assert_close(
                    observed["loss"], reference["loss"], rtol=1e-5, atol=1e-5
                )
                for field, atol, rtol in (
                    ("metrics", 1e-5, 1e-5),
                    ("gradients", 1e-5, 2e-4),
                    ("updated", 2e-5, 2e-4),
                ):
                    assert observed[field].keys() == reference[field].keys()
                    errors = []
                    for name in observed[field]:
                        torch.testing.assert_close(
                            observed[field][name],
                            reference[field][name],
                            atol=atol,
                            rtol=rtol,
                            msg=lambda m: name + ": " + m,
                        )
                        errors.append(
                            float((observed[field][name] - reference[field][name]).abs().max())
                        )
                    equivalence[field + "_maximum_absolute_error"] = max(errors)
                equivalence["loss_absolute_error"] = float(
                    (observed["loss"] - reference["loss"]).abs()
                )
        del reference, observed, optimizer, output, loss, metrics, full, small, noisy, trimmed
        timings = {"fixed": [], "batch": []}
        peaks = {"fixed": [], "batch": []}
        profiles = {}
        repeats = payload["repeats"]
        # Alternate order to reduce clock/temperature/order bias. Reset model and optimizer
        # before each timed update; include corruption, transfers, backward, clipping, AdamW,
        # and scalar metrics. Initialization and collation are outside the timing boundary.
        for repeat in range(-payload["warmups"], repeats):
            for mode in (("fixed", "batch") if repeat % 2 == 0 else ("batch", "fixed")):
                model.load_state_dict(initial)
                model.train()
                optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
                # Preallocate Adam state outside steady-state timing, without an update.
                for parameter in model.parameters():
                    optimizer.state[parameter] = {
                        "step": torch.tensor(1.0),
                        "exp_avg": torch.zeros_like(parameter),
                        "exp_avg_sq": torch.zeros_like(parameter),
                    }
                torch.manual_seed(payload["seed"] + repeat)
                generator = torch.Generator(device=target).manual_seed(payload["seed"] + repeat)
                model.zero_grad(set_to_none=True)
                if target.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                sync()
                start = time.perf_counter()
                clean = move_tensors(batches[mode], target)
                times = torch.rand(len(clean["nodes"]), generator=generator, device=target)
                loss, metrics = compose_lipid_forward_loss(
                    model,
                    clean,
                    architecture=config["model"]["architecture"],
                    node_marginal=node,
                    bond_marginal=bond,
                    times=times,
                    generator=generator,
                    semantic_weights=weights,
                )
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                scalars = {k: float(v.detach()) for k, v in metrics.items()}
                scalars.update(loss=float(loss.detach()), gradient_norm=float(norm))
                sync()
                elapsed = time.perf_counter() - start
                if repeat >= 0:
                    timings[mode].append(elapsed)
                    if target.type == "cuda":
                        peaks[mode].append(torch.cuda.max_memory_allocated())
                if profile and repeat == 0 and group["name"] == "median":
                    model.zero_grad(set_to_none=True)
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU]
                    ) as prof:
                        value, _ = compose_lipid_forward_loss(
                            model,
                            clean,
                            architecture=config["model"]["architecture"],
                            node_marginal=node,
                            bond_marginal=bond,
                            times=times,
                            generator=generator,
                            semantic_weights=weights,
                        )
                        value.backward()
                    profiles[mode] = prof.key_averages().table(
                        sort_by="self_cpu_time_total", row_limit=15
                    )
                del optimizer, loss, metrics, clean
        reports.append(
            {
                "name": group["name"],
                "batch_size": len(batches["fixed"]["nodes"]),
                "fixed_width": config["model"]["maximum_heavy_atoms"],
                "batch_width": width,
                "equivalence": equivalence,
                "seconds": timings,
                "peak_allocated_bytes": peaks,
                "median_speedup": statistics.median(timings["fixed"])
                / statistics.median(timings["batch"]),
                "profile": profiles,
            }
        )
    return {
        "schema_version": "forge.compose_lipid_padding_benchmark.v1",
        "passed": True,
        "inputs": payload["inputs"],
        "seed": payload["seed"],
        "device": str(target),
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "hardware": torch.cuda.get_device_name() if target.type == "cuda" else platform.platform(),
        "precision": "float32",
        "threads": 4,
        "deterministic": True,
        "warmups": payload["warmups"],
        "repeats": payload["repeats"],
        "weights": "fresh_seeded_model_disposable_updates",
        "production_checkpoint_modified": False,
        "equivalence_policy": "matched corruption and dropout disabled; production RNG trajectory changes",
        "timing_boundary": "transfer, corruption, forward, backward, clipping, AdamW, scalar metrics; excludes collation, initialization, persistence",
        "groups": reports,
    }
