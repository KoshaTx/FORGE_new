"""Reproducible CPU benchmark for the reaction-program Transformer training hot path."""

from __future__ import annotations

import argparse
import copy
import math
import platform
import statistics
import time
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.phase1.multireaction.production_training import (
    _compile_stratified_program_sampler,
)
from forge.core.hashing import artifact_record
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.reaction_program_transformer import (
    balanced_pcgrad_backward,
    per_program_transformer_losses,
)
from forge.model.sparse_topology_feasibility import (
    _endpoint_candidate_mask,
    _parent_candidate_mask,
)
from forge.model.synthesis_program_training import (
    build_synthesis_program_flow,
    collate_synthesis_program_training_batch,
)

try:
    import torch
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]

SCHEMA = "forge.transformer_training_performance_benchmark.v1"


class TransformerTrainingBenchmarkError(ValueError):
    """The benchmark input or equivalence contract is invalid."""


def _reference_attention_forward(
    attention: Any,
    query: Any,
    memory: Any,
    *,
    query_mask: Any,
    memory_mask: Any,
    attention_bias: Any | None = None,
) -> Any:
    """Pre-optimization materialized score/probability attention."""

    batch, queries, hidden_dim = query.shape
    keys = memory.shape[1]
    q = (
        attention.query(query)
        .reshape(batch, queries, attention.heads, attention.head_dim)
        .transpose(1, 2)
    )
    k = (
        attention.key(memory)
        .reshape(batch, keys, attention.heads, attention.head_dim)
        .transpose(1, 2)
    )
    v = (
        attention.value(memory)
        .reshape(batch, keys, attention.heads, attention.head_dim)
        .transpose(1, 2)
    )
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(attention.head_dim)
    if attention_bias is not None:
        scores = scores + attention_bias
    scores = scores.masked_fill(~memory_mask[:, None, None, :], -1e9)
    probabilities = torch.softmax(scores, dim=-1)
    context = torch.einsum("bhqk,bhkd->bhqd", probabilities, v)
    context = context.transpose(1, 2).reshape(batch, queries, hidden_dim)
    return attention.output(context) * query_mask[:, :, None]


def _prepare_attention_implementation(model: Any, *, reference: bool) -> None:
    """Disable attention dropout for a paired kernel-equivalence benchmark."""

    for module in model.modules():
        if module.__class__.__name__ != "_MaskedMultiheadAttention":
            continue
        module.dropout.p = 0.0
        if reference:
            module.forward = types.MethodType(_reference_attention_forward, module)


def _reference_graph_bias(
    model: Any,
    parents: Any,
    closure_left: Any,
    closure_right: Any,
    child_mask: Any,
    closure_mask: Any,
) -> Any:
    """Pre-optimization one-hot/einsum relation construction."""

    batch, nodes = parents.shape
    relations = torch.zeros((batch, nodes, nodes), dtype=torch.long, device=parents.device)
    diagonal = torch.arange(nodes, device=parents.device)
    relations[:, diagonal, diagonal] = 1
    parent_edges = functional.one_hot(parents, num_classes=nodes).to(torch.bool)
    parent_edges &= child_mask[:, :, None]
    relations[parent_edges] = 2
    relations[parent_edges.transpose(1, 2)] = 3
    left = functional.one_hot(closure_left, num_classes=nodes).to(torch.bool)
    right = functional.one_hot(closure_right, num_classes=nodes).to(torch.bool)
    closures = torch.einsum(
        "bkn,bkm->bnm",
        left.to(torch.float32) * closure_mask[:, :, None],
        right.to(torch.float32),
    ).to(torch.bool)
    relations[closures | closures.transpose(1, 2)] = 4
    return model.relation_bias(relations).permute(0, 3, 1, 2)


def _reference_masked_cross_entropy(logits: Any, targets: Any, mask: Any, zero: Any) -> Any:
    if bool(mask.any()):
        return functional.cross_entropy(logits[mask], targets[mask])
    return zero


def _reference_balanced_state_cross_entropy(logits: Any, targets: Any, mask: Any) -> Any:
    losses = []
    for state in torch.unique(targets[mask], sorted=True):
        state_mask = mask & (targets == state)
        losses.append(functional.cross_entropy(logits[state_mask], targets[state_mask]))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _reference_repeat_consistency(predictions: Mapping[str, Any], clean: Mapping[str, Any]) -> Any:
    groups = clean["repeat_group_states"]
    positions = clean["component_position_states"]
    instances = clean["component_instance_states"]
    active = (groups > 0) & clean["node_mask"] & (clean["core_position_states"] == 1)
    pair_mask = (
        active[:, :, None]
        & active[:, None, :]
        & (groups[:, :, None] == groups[:, None, :])
        & (positions[:, :, None] == positions[:, None, :])
        & (instances[:, :, None] != instances[:, None, :])
    )
    nodes = groups.shape[1]
    pair_mask &= torch.triu(
        torch.ones((nodes, nodes), dtype=torch.bool, device=groups.device), diagonal=1
    )[None]
    node_p = torch.softmax(predictions["nodes"], dim=-1)
    node_distance = (node_p[:, :, None, :] - node_p[:, None, :, :]).square().mean(dim=-1)
    node_loss = (node_distance * pair_mask).sum() / pair_mask.sum().clamp(min=1)
    bond_mask = pair_mask & clean["child_mask"][:, :, None] & clean["child_mask"][:, None, :]
    bond_p = torch.softmax(predictions["parent_bonds"], dim=-1)
    bond_distance = (bond_p[:, :, None, :] - bond_p[:, None, :, :]).square().mean(dim=-1)
    bond_loss = (bond_distance * bond_mask).sum() / bond_mask.sum().clamp(min=1)
    return node_loss + bond_loss


def _repeat_state_work(
    clean: Mapping[str, Any], *, node_classes: int, bond_classes: int
) -> dict[str, int | float]:
    """Count class-valued pair differences evaluated by dense and sparse formulations."""

    groups = clean["repeat_group_states"]
    positions = clean["component_position_states"]
    instances = clean["component_instance_states"]
    active = (groups > 0) & clean["node_mask"] & (clean["core_position_states"] == 1)
    pairs = (
        active[:, :, None]
        & active[:, None, :]
        & (groups[:, :, None] == groups[:, None, :])
        & (positions[:, :, None] == positions[:, None, :])
        & (instances[:, :, None] != instances[:, None, :])
    )
    nodes = groups.shape[1]
    pairs &= torch.triu(
        torch.ones((nodes, nodes), dtype=torch.bool, device=groups.device), diagonal=1
    )[None]
    node_pairs = int(pairs.sum())
    bond_pairs = int((pairs & clean["child_mask"][:, :, None] & clean["child_mask"][:, None]).sum())
    batch = groups.shape[0]
    dense = int(batch * nodes * nodes * (node_classes + bond_classes))
    sparse = int(node_pairs * node_classes + bond_pairs * bond_classes)
    return {
        "admitted_node_pairs": node_pairs,
        "admitted_bond_pairs": bond_pairs,
        "dense_class_valued_pair_elements": dense,
        "sparse_class_valued_pair_elements": sparse,
        "work_reduction_fraction": 1.0 - sparse / dense,
    }


def _slice_batch(values: Mapping[str, Any], indices: Any) -> dict[str, Any]:
    batch = int(indices.shape[0])
    return {
        key: (
            value[indices]
            if hasattr(value, "shape") and value.ndim > 0 and value.shape[0] == batch
            else value
        )
        for key, value in values.items()
    }


def _reference_transformer_loss(
    predictions: Mapping[str, Any],
    clean: Mapping[str, Any],
    *,
    role_weight: float,
    core_weight: float,
    repeat_weight: float,
) -> Any:
    parent_logits = predictions["parents"].masked_fill(
        ~_parent_candidate_mask(clean["node_mask"]), -1e9
    )
    endpoint_candidates = _endpoint_candidate_mask(
        clean["node_mask"], clean["closure_left"].shape[1]
    )
    left_logits = predictions["closure_left"].masked_fill(~endpoint_candidates, -1e9)
    right_logits = predictions["closure_right"].masked_fill(~endpoint_candidates, -1e9)
    zero = predictions["nodes"].sum() * 0.0
    base = sum(
        (
            _reference_masked_cross_entropy(
                predictions["nodes"], clean["nodes"], clean["atom_variable_mask"], zero
            ),
            _reference_masked_cross_entropy(
                parent_logits, clean["parents"], clean["parent_variable_mask"], zero
            ),
            _reference_masked_cross_entropy(
                predictions["parent_bonds"],
                clean["parent_bonds"],
                clean["parent_bond_variable_mask"],
                zero,
            ),
            _reference_masked_cross_entropy(
                left_logits,
                clean["closure_left"],
                clean["closure_endpoint_variable_mask"],
                zero,
            ),
            _reference_masked_cross_entropy(
                right_logits,
                clean["closure_right"],
                clean["closure_endpoint_variable_mask"],
                zero,
            ),
            _reference_masked_cross_entropy(
                predictions["closure_bonds"],
                clean["closure_bonds"],
                clean["closure_bond_variable_mask"],
                zero,
            ),
        ),
        start=zero,
    )
    role = _reference_balanced_state_cross_entropy(
        predictions["role_states"], clean["role_states"], clean["node_mask"]
    )
    core = _reference_balanced_state_cross_entropy(
        predictions["core_position_states"],
        clean["core_position_states"],
        clean["node_mask"],
    )
    repeat = _reference_repeat_consistency(predictions, clean)
    return base + role_weight * role + core_weight * core + repeat_weight * repeat


def _reference_family_losses(
    predictions: Mapping[str, Any],
    clean: Mapping[str, Any],
    *,
    program_states: Sequence[int],
    role_weight: float,
    core_weight: float,
    repeat_weight: float,
) -> dict[int, Any]:
    source_programs = clean["source_program_states"]
    return {
        program_state: _reference_transformer_loss(
            _slice_batch(predictions, source_programs == program_state),
            _slice_batch(clean, source_programs == program_state),
            role_weight=role_weight,
            core_weight=core_weight,
            repeat_weight=repeat_weight,
        )
        for program_state in program_states
    }


def _reference_pcgrad_backward(losses: Mapping[int, Any], model: Any) -> int:
    """Pre-optimization per-parameter PCGrad implementation."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    ordered = sorted(losses)
    raw = [
        torch.autograd.grad(
            losses[program], parameters, retain_graph=index + 1 < len(ordered), allow_unused=True
        )
        for index, program in enumerate(ordered)
    ]
    projected = []
    conflicts = 0
    for left_index, gradients in enumerate(raw):
        current = [None if gradient is None else gradient.clone() for gradient in gradients]
        for right_index, reference in enumerate(raw):
            if left_index == right_index:
                continue
            dot = sum(
                (
                    left_gradient.mul(right_gradient).sum()
                    for left_gradient, right_gradient in zip(current, reference, strict=True)
                    if left_gradient is not None and right_gradient is not None
                ),
                start=losses[ordered[0]].new_zeros(()),
            )
            if float(dot.detach()) < 0.0:
                conflicts += 1
                denominator = sum(
                    (gradient.square().sum() for gradient in reference if gradient is not None),
                    start=losses[ordered[0]].new_zeros(()),
                ).clamp(min=1e-12)
                coefficient = dot / denominator
                current = [
                    (
                        left_gradient
                        if left_gradient is None or right_gradient is None
                        else left_gradient - coefficient * right_gradient
                    )
                    for left_gradient, right_gradient in zip(current, reference, strict=True)
                ]
        projected.append(current)
    for parameter_index, parameter in enumerate(parameters):
        available = [
            gradients[parameter_index]
            for gradients in projected
            if gradients[parameter_index] is not None
        ]
        if available:
            parameter.grad = torch.stack(available).sum(dim=0).div(len(projected)).detach().clone()
    return conflicts


def _flat_model_gradients(model: Any) -> Any:
    return torch.cat(
        [
            (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).reshape(
                -1
            )
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
    )


def run_transformer_training_benchmark(
    *,
    repo: Path,
    config_path: Path,
    cache_path: Path,
    output_path: Path,
    repeats: int,
    seed: int,
    cpu_threads: int,
) -> dict[str, Any]:
    """Compare the prior and optimized forward/loss/PCGrad path on one production-sized batch."""

    if torch is None or repeats < 3 or cpu_threads < 1:
        raise TransformerTrainingBenchmarkError(
            "benchmark requires torch, at least three repeats and positive CPU threads"
        )
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    config = read_json_object(
        config_path,
        error=TransformerTrainingBenchmarkError,
        label="Transformer production design",
    )
    model_config = dict(config["model"])
    if model_config.get("architecture") != "reaction_program_graph_transformer":
        raise TransformerTrainingBenchmarkError("benchmark requires the production Transformer")
    cache = SynthesisProgramProductionCache(cache_path)
    try:
        arm = config["training"]["arms"]["shared_three_program_conditioned"]
        measure = cache.training_measure(arm["program_mass"])
        sampler = _compile_stratified_program_sampler(cache, measure)
        rng = np.random.default_rng(seed + 2)
        records = None
        clean = None
        for _ in range(1000):
            candidate = cache.records(sampler.sample(12, rng))
            candidate_clean = collate_synthesis_program_training_batch(
                candidate,
                maximum_closures=int(model_config["maximum_closures"]),
                conditioning="program",
                vocabulary=cache.vocabulary,
            )
            if candidate_clean["nodes"].shape[1] >= 130:
                records = candidate
                clean = candidate_clean
                break
        if records is None or clean is None:
            raise TransformerTrainingBenchmarkError(
                "could not construct the declared production-sized benchmark batch"
            )
        vocabulary = cache.vocabulary
        node_classes = len(cache.atom_vocabulary)
    finally:
        cache.close()

    model_inputs = {
        key: clean[key]
        for key in (
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
            "node_mask",
            "child_mask",
            "closure_mask",
            "program_states",
            "role_states",
            "core_position_states",
            "program_depths",
            "component_instance_states",
            "component_position_states",
            "repeat_group_states",
            "role_morphology_states",
            "adapter_mask",
        )
    }
    model_inputs["t"] = torch.full((len(records),), 0.5)
    objective = model_config["semantic_objective"]
    role_weight = float(objective["role_consistency_weight"])
    core_weight = float(objective["core_consistency_weight"])
    repeat_weight = float(objective.get("repeat_consistency_weight", 0.0))

    def build_model() -> Any:
        return build_synthesis_program_flow(
            vocabulary=vocabulary,
            node_classes=node_classes,
            model_config=model_config,
            device=torch.device("cpu"),
        ).train()

    base_model = build_model()
    state = copy.deepcopy(base_model.state_dict())
    graph_arguments = (
        clean["parents"],
        clean["closure_left"],
        clean["closure_right"],
        clean["child_mask"],
        clean["closure_mask"],
    )
    reference_graph_bias = _reference_graph_bias(base_model, *graph_arguments)
    optimized_graph_bias = base_model._graph_bias(*graph_arguments)
    _reference_graph_bias(base_model, *graph_arguments)
    base_model._graph_bias(*graph_arguments)
    graph_reference_times = []
    graph_optimized_times = []
    for _ in range(repeats * 5):
        started = time.perf_counter()
        _reference_graph_bias(base_model, *graph_arguments)
        graph_reference_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        base_model._graph_bias(*graph_arguments)
        graph_optimized_times.append(time.perf_counter() - started)

    def run_once(reference: bool, run_seed: int, collect: bool) -> dict[str, Any]:
        model = build_model()
        model.load_state_dict(state, strict=True)
        _prepare_attention_implementation(model, reference=reference)
        if reference:
            model._graph_bias = types.MethodType(_reference_graph_bias, model)
        torch.manual_seed(run_seed)
        started = time.perf_counter()
        predictions = model(**model_inputs)
        if reference:
            losses = _reference_family_losses(
                predictions,
                clean,
                program_states=sampler.program_states,
                role_weight=role_weight,
                core_weight=core_weight,
                repeat_weight=repeat_weight,
            )
            conflicts = _reference_pcgrad_backward(losses, model)
        else:
            losses, _ = per_program_transformer_losses(
                predictions,
                clean,
                role_weight=role_weight,
                core_weight=core_weight,
                repeat_consistency_weight=repeat_weight,
                program_states=sampler.program_states,
                materialize_metrics=False,
            )
            diagnostic = balanced_pcgrad_backward(losses, model, materialize_diagnostics=False)
            conflicts = int(diagnostic["projected_conflicts"])
        elapsed = time.perf_counter() - started
        return {
            "seconds": elapsed,
            "losses": [float(loss.detach()) for loss in losses.values()] if collect else None,
            "gradients": _flat_model_gradients(model) if collect else None,
            "conflicts": conflicts,
        }

    run_once(True, seed + 10, False)
    run_once(False, seed + 10, False)
    reference_times = []
    optimized_times = []
    reference_check = None
    optimized_check = None
    for index in range(repeats):
        collect = index == 0
        reference = run_once(True, seed + 100 + index, collect)
        optimized = run_once(False, seed + 100 + index, collect)
        reference_times.append(float(reference["seconds"]))
        optimized_times.append(float(optimized["seconds"]))
        if collect:
            reference_check = reference
            optimized_check = optimized
    if reference_check is None or optimized_check is None:
        raise TransformerTrainingBenchmarkError("benchmark produced no equivalence pair")
    loss_difference = max(
        abs(left - right)
        for left, right in zip(reference_check["losses"], optimized_check["losses"], strict=True)
    )
    gradient_difference = float(
        (reference_check["gradients"] - optimized_check["gradients"]).abs().max()
    )

    def run_pcgrad_once(reference: bool, run_seed: int, collect: bool) -> dict[str, Any]:
        model = build_model()
        model.load_state_dict(state, strict=True)
        torch.manual_seed(run_seed)
        predictions = model(**model_inputs)
        losses, _ = per_program_transformer_losses(
            predictions,
            clean,
            role_weight=role_weight,
            core_weight=core_weight,
            repeat_consistency_weight=repeat_weight,
            program_states=sampler.program_states,
            materialize_metrics=False,
        )
        started = time.perf_counter()
        if reference:
            conflicts = _reference_pcgrad_backward(losses, model)
        else:
            diagnostic = balanced_pcgrad_backward(losses, model, materialize_diagnostics=False)
            conflicts = int(diagnostic["projected_conflicts"])
        elapsed = time.perf_counter() - started
        return {
            "seconds": elapsed,
            "gradients": _flat_model_gradients(model) if collect else None,
            "conflicts": conflicts,
        }

    run_pcgrad_once(True, seed + 20, False)
    run_pcgrad_once(False, seed + 20, False)
    pcgrad_reference_times = []
    pcgrad_optimized_times = []
    pcgrad_reference_check = None
    pcgrad_optimized_check = None
    for index in range(repeats):
        collect = index == 0
        reference = run_pcgrad_once(True, seed + 200 + index, collect)
        optimized = run_pcgrad_once(False, seed + 200 + index, collect)
        pcgrad_reference_times.append(float(reference["seconds"]))
        pcgrad_optimized_times.append(float(optimized["seconds"]))
        if collect:
            pcgrad_reference_check = reference
            pcgrad_optimized_check = optimized
    if pcgrad_reference_check is None or pcgrad_optimized_check is None:
        raise TransformerTrainingBenchmarkError("benchmark produced no PCGrad equivalence pair")

    reference_median = statistics.median(reference_times)
    optimized_median = statistics.median(optimized_times)
    graph_reference_median = statistics.median(graph_reference_times)
    graph_optimized_median = statistics.median(graph_optimized_times)
    pcgrad_reference_median = statistics.median(pcgrad_reference_times)
    pcgrad_optimized_median = statistics.median(pcgrad_optimized_times)
    repeat_work = _repeat_state_work(
        clean,
        node_classes=node_classes,
        bond_classes=int(model_config["bond_classes"]),
    )
    gates = {
        "losses_numerically_equivalent": bool(
            np.allclose(
                reference_check["losses"],
                optimized_check["losses"],
                rtol=1e-6,
                atol=1e-7,
            )
        ),
        "gradients_numerically_equivalent": bool(
            torch.allclose(
                reference_check["gradients"],
                optimized_check["gradients"],
                rtol=1e-5,
                atol=1e-7,
            )
        ),
        "conflict_count_identical": (reference_check["conflicts"] == optimized_check["conflicts"]),
        "graph_bias_bit_identical": torch.equal(reference_graph_bias, optimized_graph_bias),
        "pcgrad_gradients_bit_identical": torch.equal(
            pcgrad_reference_check["gradients"], pcgrad_optimized_check["gradients"]
        ),
        "pcgrad_conflict_count_identical": (
            pcgrad_reference_check["conflicts"] == pcgrad_optimized_check["conflicts"]
        ),
        "graph_bias_median_faster": graph_optimized_median < graph_reference_median,
        "pcgrad_median_faster": pcgrad_optimized_median < pcgrad_reference_median,
        "repeat_state_work_reduced": (
            repeat_work["sparse_class_valued_pair_elements"]
            < repeat_work["dense_class_valued_pair_elements"]
        ),
    }
    result = {
        "schema_version": SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "inputs": {
            "config": artifact_record(config_path),
            "cache": artifact_record(cache_path),
            "implementation": artifact_record(Path(__file__)),
            "transformer": artifact_record(repo / "forge/model/reaction_program_transformer.py"),
            "training_loop": artifact_record(
                repo / "experiments/phase1/multireaction/production_training.py"
            ),
            "shared_attention": artifact_record(repo / "forge/model/structural_attention.py"),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_threads": cpu_threads,
        },
        "benchmark": {
            "seed": seed,
            "repeats": repeats,
            "batch_size": len(records),
            "padded_nodes": int(clean["nodes"].shape[1]),
            "parameters": sum(parameter.numel() for parameter in base_model.parameters()),
            "scope": "forward_loss_and_equal_family_pcgrad_backward_with_attention_dropout_zero",
            "attention_dropout_probability": 0.0,
            "reference_seconds": reference_times,
            "optimized_seconds": optimized_times,
            "reference_median_seconds": reference_median,
            "optimized_median_seconds": optimized_median,
            "median_speedup": reference_median / optimized_median,
            "observed_outcome": (
                "faster" if optimized_median < reference_median else "neutral_or_slower"
            ),
        },
        "components": {
            "graph_bias": {
                "reference_seconds": graph_reference_times,
                "optimized_seconds": graph_optimized_times,
                "reference_median_seconds": graph_reference_median,
                "optimized_median_seconds": graph_optimized_median,
                "median_speedup": graph_reference_median / graph_optimized_median,
            },
            "pcgrad_backward": {
                "reference_seconds": pcgrad_reference_times,
                "optimized_seconds": pcgrad_optimized_times,
                "reference_median_seconds": pcgrad_reference_median,
                "optimized_median_seconds": pcgrad_optimized_median,
                "median_speedup": pcgrad_reference_median / pcgrad_optimized_median,
            },
            "repeat_consistency": repeat_work,
        },
        "equivalence": {
            "maximum_absolute_loss_difference": loss_difference,
            "maximum_absolute_gradient_difference": gradient_difference,
            "reference_conflicts": reference_check["conflicts"],
            "optimized_conflicts": optimized_check["conflicts"],
            "loss_tolerance": {"rtol": 1e-6, "atol": 1e-7},
            "gradient_tolerance": {"rtol": 1e-5, "atol": 1e-7},
        },
        "gates": gates,
        "nonclaims": [
            "This CPU benchmark does not establish CUDA throughput.",
            "The benchmark does not change the frozen data mixture or scientific objective.",
            "A current exact-GPU preflight remains required before any paid full launch.",
        ],
    }
    write_json(output_path, result)
    if result["status"] != "pass":
        raise TransformerTrainingBenchmarkError(f"performance benchmark failed: {gates}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multireaction/shared_production_comparison_transformer_v1.json"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("results/phase1/shared_synthesis_program_production_cache_v1/cache.npz"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_transformer_training_benchmark(
        repo=args.repo.resolve(),
        config_path=args.config.resolve(),
        cache_path=args.cache.resolve(),
        output_path=args.output.resolve(),
        repeats=args.repeats,
        seed=args.seed,
        cpu_threads=args.cpu_threads,
    )
    print(f"{result['status']}: {result['benchmark']['median_speedup']:.3f}x median speedup")


if __name__ == "__main__":
    main()


__all__ = [
    "SCHEMA",
    "TransformerTrainingBenchmarkError",
    "run_transformer_training_benchmark",
]
