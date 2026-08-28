"""Reproducible CPU benchmark for shared synthesis-program sampling throughput.

This measures the *execution* cost of the frozen sampling contract.  It changes no scientific
setting: 32 flow steps, batch 128, the ``strict_valence_topology_argmax`` terminal policy, the
production Transformer architecture and the frozen factorized layout prior fitted to the real
production cache.

Model weights are deterministically initialized rather than loaded, because the authenticated
production checkpoints live on the remote accelerator.  Weights change *which* molecule is decoded;
they do not change the shape of the work.  To keep the decoder and RDKit workload representative
instead of degenerate, the terminal chemistry output biases are set so that argmax over the
valence-masked candidates lands on chemically ordinary states.  Without that, every random-weight
sample decodes to an unparseable graph, RDKit canonicalization is never reached, and the benchmark
silently stops measuring one of the real costs.  ``--raw-init`` disables the bias for comparison.

Byte-identity evidence
----------------------
Every run emits three SHA-256 digests:

``sample_ledger_sha256``   the complete per-sample output rows,
``receipt_sha256``         the sampling receipt,
``flow_trace_sha256``      the six categorical state tensors at *every* model call.

The trace digest is the RNG-stream contract: any change that shifts which random numbers land where
diverges on some flow step long before it could reach the terminal decoder, so a matching trace is
evidence about the sampler's stochastic path and not only about its final molecules.

Reproduce::

    .venv/bin/python -m experiments.phase1.multireaction.sampling_throughput_benchmark \\
        --programs 384 --batch-size 128 --sample-steps 32 --seed 101 --threads 8 \\
        --output results/phase1/ugi_sampling_cpu_throughput_v1/before.json
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import pstats
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PRODUCTION_CACHE = REPO / "results/phase1/shared_synthesis_program_production_cache_v1/cache.npz"

# Frozen production architecture (configs/multireaction shared production design v1).
PRODUCTION_ARCHITECTURE: dict[str, Any] = {
    "hidden_dim": 192,
    "layers": 6,
    "heads": 8,
    "expert_count": 3,
    "adapter_dim": 64,
    "maximum_closures": 3,
    "maximum_heavy_atoms": 194,
    "dropout": 0.1,
    "bond_classes": 4,
    "repeat_group_conditioning": True,
    "role_morphology_conditioning": True,
}


class _Digest:
    """Order-sensitive rolling SHA-256 over tensor bytes."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256()
        self.calls = 0

    def update_tensors(self, values: dict[str, Any], fields: tuple[str, ...]) -> None:
        self.calls += 1
        for field in fields:
            value = values.get(field)
            if value is None:
                self._hash.update(b"<none>")
                continue
            array = value.detach().to("cpu").contiguous().numpy()
            self._hash.update(str(array.dtype).encode())
            self._hash.update(str(array.shape).encode())
            self._hash.update(array.tobytes())

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


STATE_FIELDS = (
    "nodes",
    "parents",
    "parent_bonds",
    "closure_left",
    "closure_right",
    "closure_bonds",
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_model(cache: Any, *, raw_init: bool) -> Any:
    import torch

    from forge.model.reaction_program_transformer import ReactionProgramGraphTransformer

    torch.manual_seed(0)
    model = ReactionProgramGraphTransformer(
        vocabulary=cache.vocabulary,
        node_classes=len(cache.atom_vocabulary),
        **PRODUCTION_ARCHITECTURE,
    )
    if not raw_init:
        symbols = [state.symbol for state in cache.atom_vocabulary]
        aromatic = [state.aromatic for state in cache.atom_vocabulary]
        charges = [state.formal_charge for state in cache.atom_vocabulary]
        with torch.no_grad():
            node_bias = model.state.node_output.bias
            node_bias.zero_()
            for index, symbol in enumerate(symbols):
                if aromatic[index] or charges[index]:
                    continue
                if symbol == "C":
                    node_bias[index] = 12.0
                elif symbol in {"N", "O"}:
                    node_bias[index] = 7.0
            for head in (model.state.backbone_bond_output, model.state.closure_bond_output):
                final = head[-1] if isinstance(head, torch.nn.Sequential) else head
                final.bias.zero_()
                final.bias[0] = 12.0
    model.eval()
    return model


def _marginals(cache: Any) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    node_marginal = rng.random(len(cache.atom_vocabulary)) + 0.05
    node_marginal /= node_marginal.sum()
    bond_marginal = rng.random(PRODUCTION_ARCHITECTURE["bond_classes"]) + 0.05
    bond_marginal /= bond_marginal.sum()
    return node_marginal, bond_marginal


class _NullArm:
    """No-op context manager so both timing arms run through identical Python structure."""

    def __enter__(self) -> _NullArm:
        return self

    def __exit__(self, *_: Any) -> None:
        return None


class _UnoptimizedArm:
    """Restore the pre-optimization call shape inside one process, for interleaved A/B timing.

    Dropping ``program_memory`` from the sampler's frozen conditioning makes every model call
    encode the reaction program and project its cross-attention memory again, which is exactly the
    work the optimization removes.  Restoring the per-record decoder and audit helpers likewise
    reinstates their original per-record loops.  Nothing else differs, so the two arms are a
    controlled comparison rather than two separate runs on a noisy host.
    """

    def __init__(self) -> None:
        from forge.model import synthesis_program_sampling as module

        self.module = module
        self.saved: dict[str, Any] = {}

    def __enter__(self) -> _UnoptimizedArm:
        module = self.module
        original = module._program_conditioning

        def unoptimized_conditioning(model: Any, layout: Any) -> dict[str, Any]:
            conditioning = original(model, layout)
            conditioning.pop("program_memory", None)
            return conditioning

        self.saved["_program_conditioning"] = original
        module._program_conditioning = unoptimized_conditioning
        return self

    def __exit__(self, *_: Any) -> None:
        for name, value in self.saved.items():
            setattr(self.module, name, value)
        self.saved.clear()


def run_benchmark(
    *,
    programs: int,
    batch_size: int,
    sample_steps: int,
    seed: int,
    layout_seed: int,
    program_id: str,
    threads: int | None,
    repeats: int,
    raw_init: bool,
    profile: bool,
    arm: str = "optimized",
    interleave: bool = False,
) -> dict[str, Any]:
    import torch

    from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
    from forge.model import synthesis_program_sampling as sampling_module
    from forge.model.synthesis_program_layout import SynthesisProgramLayoutPrior

    if threads is not None:
        torch.set_num_threads(threads)

    timings: dict[str, float] = {}
    start = time.perf_counter()
    cache = SynthesisProgramProductionCache(PRODUCTION_CACHE)
    timings["cache_open_seconds"] = time.perf_counter() - start

    try:
        start = time.perf_counter()
        prior = SynthesisProgramLayoutPrior(cache)
        timings["layout_prior_build_seconds"] = time.perf_counter() - start

        start = time.perf_counter()
        layouts = prior.sample(
            program_id,
            sample_count=programs,
            seed=layout_seed,
            role_morphology_conditioning=PRODUCTION_ARCHITECTURE["role_morphology_conditioning"],
        )
        timings["layout_sampling_seconds"] = time.perf_counter() - start

        model = _build_model(cache, raw_init=raw_init)
        node_marginal, bond_marginal = _marginals(cache)

        traces: list[_Digest] = [_Digest()]
        original_forward = type(model).forward

        def traced_forward(self: Any, **kwargs: Any) -> Any:
            traces[-1].update_tensors(kwargs, STATE_FIELDS)
            return original_forward(self, **kwargs)

        type(model).forward = traced_forward  # type: ignore[method-assign]

        def one_pass() -> tuple[float, Any, Any]:
            begin = time.perf_counter()
            rows, receipt = sampling_module.sample_synthesis_program_products(
                model,
                layouts,
                cache.atom_vocabulary,
                node_marginal,
                bond_marginal,
                samples_per_program=1,
                sample_steps=sample_steps,
                batch_size=batch_size,
                seed=seed,
                device="cpu",
                conditioning_mode="program",
                terminal_decode_policy="strict_valence_topology_argmax",
            )
            return time.perf_counter() - begin, rows, receipt

        try:
            wall_times: list[float] = []
            baseline_times: list[float] = []
            baseline_identity: dict[str, str] | None = None
            profile_text = None
            for repeat in range(repeats):
                if repeat:
                    traces.append(_Digest())
                if interleave:
                    # Alternate arms inside one process so both see the same contended host.
                    baseline_trace = _Digest()
                    traces.append(baseline_trace)
                    with _UnoptimizedArm():
                        elapsed, baseline_rows, baseline_receipt = one_pass()
                    baseline_times.append(elapsed)
                    baseline_identity = {
                        "sample_ledger_sha256": _canonical_sha256(baseline_rows),
                        "receipt_sha256": _canonical_sha256(baseline_receipt),
                        "flow_trace_sha256": baseline_trace.hexdigest(),
                    }
                    traces.remove(baseline_trace)
                if profile and repeat == repeats - 1:
                    profiler = cProfile.Profile()
                    profiler.enable()
                with _UnoptimizedArm() if arm == "baseline" else _NullArm():
                    elapsed, rows, receipt = one_pass()
                wall_times.append(elapsed)
                if profile and repeat == repeats - 1:
                    profiler.disable()
                    stream = io.StringIO()
                    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(45)
                    profile_text = stream.getvalue()
        finally:
            type(model).forward = original_forward  # type: ignore[method-assign]

        node_counts = [record.node_count for record in layouts]
        best = min(wall_times)
        result: dict[str, Any] = {
            "schema_version": "forge.ugi_sampling_cpu_throughput_benchmark.v1",
            "request": {
                "programs": programs,
                "batch_size": batch_size,
                "sample_steps": sample_steps,
                "seed": seed,
                "layout_seed": layout_seed,
                "program_id": program_id,
                "repeats": repeats,
                "raw_init": raw_init,
                "terminal_decode_policy": "strict_valence_topology_argmax",
                "device": "cpu",
                "torch_threads": torch.get_num_threads(),
            },
            "environment": {
                "torch_version": torch.__version__,
                "production_cache": str(PRODUCTION_CACHE.relative_to(REPO)),
                "model_parameters": int(sum(p.numel() for p in model.parameters())),
                "architecture": PRODUCTION_ARCHITECTURE,
            },
            "layouts": {
                "count": len(layouts),
                "node_count_min": int(min(node_counts)),
                "node_count_max": int(max(node_counts)),
                "node_count_mean": float(sum(node_counts) / len(node_counts)),
            },
            "timings": timings,
            "throughput": {
                "arm": arm,
                "wall_seconds_per_repeat": wall_times,
                "wall_seconds_best": best,
                "samples_per_second_best": programs / best,
                "samples_per_second_mean": programs / (sum(wall_times) / len(wall_times)),
            },
            "outcome": {
                "samples": receipt["samples"],
                "valid": receipt["valid"],
                "fixed_state_failures": receipt["fixed_state_failures"],
                "strict_constraint_abstentions": receipt["strict_constraint_abstentions"],
                "strict_constraint_abstention_reasons": receipt[
                    "strict_constraint_abstention_reasons"
                ],
            },
            "identity": {
                "sample_ledger_sha256": _canonical_sha256(rows),
                "receipt_sha256": _canonical_sha256(receipt),
                "flow_trace_sha256": traces[0].hexdigest(),
                "flow_trace_model_calls": traces[0].calls,
                "flow_trace_stable_across_repeats": len({digest.hexdigest() for digest in traces})
                == 1,
            },
        }
        if interleave and baseline_times:
            baseline_best = min(baseline_times)
            result["interleaved_baseline"] = {
                "wall_seconds_per_repeat": baseline_times,
                "wall_seconds_best": baseline_best,
                "samples_per_second_best": programs / baseline_best,
                "speedup_best": baseline_best / best,
                "identity": baseline_identity,
                "identical_to_optimized": (
                    baseline_identity is not None
                    and baseline_identity["sample_ledger_sha256"]
                    == result["identity"]["sample_ledger_sha256"]
                    and baseline_identity["receipt_sha256"] == result["identity"]["receipt_sha256"]
                    and baseline_identity["flow_trace_sha256"]
                    == result["identity"]["flow_trace_sha256"]
                ),
            }
        if profile_text is not None:
            result["profile"] = profile_text
        return result
    finally:
        cache.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--programs", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--layout-seed", type=int, default=11)
    parser.add_argument("--program-id", type=str, default="ugi_3cr_agile")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--raw-init", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--arm", choices=("optimized", "baseline"), default="optimized")
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="alternate the unoptimized and optimized arms in one process for a controlled A/B",
    )
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    result = run_benchmark(
        programs=arguments.programs,
        batch_size=arguments.batch_size,
        sample_steps=arguments.sample_steps,
        seed=arguments.seed,
        layout_seed=arguments.layout_seed,
        program_id=arguments.program_id,
        threads=arguments.threads,
        repeats=arguments.repeats,
        raw_init=arguments.raw_init,
        profile=arguments.profile,
        arm=arguments.arm,
        interleave=arguments.interleave,
    )
    profile_text = result.pop("profile", None)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if profile_text is not None:
        print(profile_text)


if __name__ == "__main__":
    main()
