"""Reproducible CPU profile of the frozen mixed-program Transformer optimizer step.

The exact-H100 reference for this step is 0.35855 s and 351.63 examples/s at the frozen 32 x 4
geometry.  A CPU cannot reproduce that number, so this module measures the three quantities that a
CPU can establish about the same step and that carry to an accelerator:

* whether an execution change leaves the losses and the parameter gradients unchanged,
* how many operator dispatches and host synchronizations the step issues, which is what an eager
  variable-shape step spends its accelerator time waiting on, and
* how much of the step is padding, which is set by the packing of the micro-batch and not by the
  device.

Timing on a shared workstation is unreliable in absolute terms, so every timed comparison here is
paired: the two arms run the same micro-batch back to back and the reported quantity is the ratio
of that pair, repeated and reduced by median and minimum.  Absolute seconds are reported too, but
only the ratios are claims.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.phase1.multireaction.production_training import (
    _compile_stratified_program_sampler,
)
from forge.core.hashing import artifact_record, sha256_file
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.reaction_program_flow import noise_synthesis_program_batch
from forge.model.reaction_program_transformer import (
    balanced_pcgrad_backward,
    per_program_transformer_losses,
)
from forge.model.synthesis_program_training import (
    build_synthesis_program_flow,
    collate_synthesis_program_training_batch,
    move_tensors,
    synthesis_program_fixed_state_exact_tensor,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]

SCHEMA = "forge.synthesis_program_cpu_training_step_profile.v1"

# The frozen exact-H100 baseline this profile is written against.  It is context, not a measurement
# this module can reproduce or update.
H100_REFERENCE = {
    "median_wall_seconds_per_optimizer_step": 0.35855,
    "examples_per_second": 351.63,
    "peak_reserved_bytes": 1_836_871_680,
    "source": "results/phase1/shared_synthesis_program_mixed_h100_profile_v1/result.json",
}

# The prespecified float32 equivalence gate from the frozen H100 profiling config.  An execution
# change is admissible only inside it; the changes measured here are expected to be bit-identical
# and are asserted as such.
EQUIVALENCE_TOLERANCE = {
    "loss_max_absolute_difference": 1e-05,
    "loss_max_relative_difference": 1e-06,
    "gradient_relative_l2_difference": 1e-04,
    "gradient_max_absolute_difference": 1e-06,
    "gradient_cosine_similarity": 0.999999,
}

# Dispatches that only re-describe an existing allocation.  They issue no accelerator kernel, so
# they are separated from the count that stands in for launch overhead.
_VIEW_DISPATCHES = frozenset(
    {
        "aten.view.default",
        "aten._unsafe_view.default",
        "aten.t.default",
        "aten.transpose.int",
        "aten.permute.default",
        "aten.expand.default",
        "aten.unsqueeze.default",
        "aten.squeeze.dim",
        "aten.select.int",
        "aten.slice.Tensor",
        "aten.detach.default",
        "aten.as_strided.default",
        "aten.alias.default",
        "aten.reshape.default",
    }
)

# Dispatches whose result size is only known on the host.  On an accelerator each one drains the
# queue before the next kernel can be launched.  Boolean advanced indexing belongs here as well:
# `aten.index` and `aten.index_put_` resolve a boolean mask by calling `nonzero` inside the kernel,
# which does not appear as its own dispatch, so they are classified by their index dtypes instead.
_SYNCHRONIZING_DISPATCHES = frozenset(
    {
        "aten.nonzero.default",
        "aten._local_scalar_dense.default",
        "aten.item.default",
        "aten._unique2.default",
        "aten.unique_consecutive.default",
        "aten.masked_select.default",
        "aten.equal.default",
    }
)
_BOOLEAN_INDEX_DISPATCHES = frozenset(
    {
        "aten.index.Tensor",
        "aten.index_put_.default",
        "aten.index_put.default",
        "aten._index_put_impl_.default",
    }
)


def _is_synchronizing(name: str, args: tuple[Any, ...]) -> bool:
    if name in _SYNCHRONIZING_DISPATCHES:
        return True
    if name not in _BOOLEAN_INDEX_DISPATCHES:
        return False
    for argument in args[1:]:
        candidates = argument if isinstance(argument, (list, tuple)) else (argument,)
        for candidate in candidates:
            if torch.is_tensor(candidate) and candidate.dtype == torch.bool:
                return True
    return False


class CpuTrainingStepProfileError(ValueError):
    """The profiling request violates the frozen training contract."""


def _reference_graph_bias(
    model: Any,
    parents: Any,
    closure_left: Any,
    closure_right: Any,
    child_mask: Any,
    closure_mask: Any,
) -> Any:
    """Pre-optimization relation construction: one boolean gather per selected coordinate."""

    batch, nodes = parents.shape
    relations = torch.zeros((batch, nodes, nodes), dtype=torch.long, device=parents.device)
    diagonal = torch.arange(nodes, device=parents.device)
    relations[:, diagonal, diagonal] = 1
    batch_nodes = torch.arange(batch, device=parents.device)[:, None].expand(batch, nodes)
    child_nodes = diagonal[None].expand(batch, nodes)
    relations[batch_nodes[child_mask], child_nodes[child_mask], parents[child_mask]] = 2
    relations[batch_nodes[child_mask], parents[child_mask], child_nodes[child_mask]] = 3
    closure_slots = closure_left.shape[1]
    closure_batches = torch.arange(batch, device=parents.device)[:, None].expand(
        batch, closure_slots
    )
    active_batches = closure_batches[closure_mask]
    active_left = closure_left[closure_mask]
    active_right = closure_right[closure_mask]
    relations[active_batches, active_left, active_right] = 4
    relations[active_batches, active_right, active_left] = 4
    return model.relation_bias(relations).permute(0, 3, 1, 2)


def _reference_attention_forward(
    attention: Any,
    query: Any,
    memory: Any,
    *,
    query_mask: Any,
    memory_mask: Any,
    attention_bias: Any | None = None,
    memory_key_value: tuple[Any, Any] | None = None,
    attention_bias_is_masked: bool = False,
) -> Any:
    """Pre-optimization attention: rebuild the absent-key fill inside every layer.

    ``attention_bias_is_masked`` is deliberately ignored, because rebuilding that fill per layer
    is exactly the behaviour this reference exists to compare against.  ``memory_key_value`` is
    honoured rather than ignored: reusing a projected memory is separately established as
    bit-exact, so it belongs to the baseline this reference is pre-optimization *with respect to*,
    not to the change under test.
    """

    del attention_bias_is_masked
    batch, queries, hidden_dim = query.shape
    q = attention.query(query).reshape(batch, queries, attention.heads, attention.head_dim)
    q = q.transpose(1, 2)
    if memory_key_value is None:
        keys = memory.shape[1]
        k = attention.key(memory).reshape(batch, keys, attention.heads, attention.head_dim)
        v = attention.value(memory).reshape(batch, keys, attention.heads, attention.head_dim)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
    else:
        k, v = memory_key_value
    if attention_bias is not None:
        attention_mask = attention_bias.masked_fill(~memory_mask[:, None, None, :], -torch.inf)
    else:
        attention_mask = memory_mask[:, None, None, :]
    context = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        dropout_p=attention.dropout.p if attention.training else 0.0,
    )
    context = context.transpose(1, 2).reshape(batch, queries, hidden_dim)
    return attention.output(context) * query_mask[:, :, None]


def _reference_balanced_state_cross_entropy(logits: Any, targets: Any, mask: Any) -> Any:
    """Pre-optimization state-balanced cross entropy: one boolean gather per operand."""

    selected_targets = targets[mask]
    point_losses = torch.nn.functional.cross_entropy(
        logits[mask], selected_targets, reduction="none"
    )
    classes = logits.shape[-1]
    state_sums = logits.new_zeros(classes).scatter_add(0, selected_targets, point_losses)
    state_counts = torch.bincount(selected_targets, minlength=classes)
    present = state_counts > 0
    state_means = state_sums / state_counts.clamp(min=1)
    return (state_means * present).sum() / present.sum().clamp(min=1)


def _reference_slice_batch(
    values: Mapping[str, Any], indices: Any, *, batch_size: int | None = None
) -> dict[str, Any]:
    """Pre-optimization family selection: resolve the boolean mask once per entry."""

    if indices.dtype != torch.bool:
        membership = torch.zeros(
            (batch_size,), dtype=torch.bool, device=indices.device
        ).index_fill_(0, indices, True)
    else:
        membership = indices
    batch = int(membership.shape[0])
    output: dict[str, Any] = {}
    for key, value in values.items():
        if hasattr(value, "shape") and value.ndim > 0 and value.shape[0] == batch:
            output[key] = value[membership]
        else:
            output[key] = value
    return output


def _reference_masked_cross_entropy(logits: Any, targets: Any, mask: Any) -> Any:
    """Pre-optimization masked chemistry cross entropy: one boolean gather per operand."""

    selected = torch.nn.functional.cross_entropy(logits[mask], targets[mask], reduction="sum")
    return selected / mask.sum().clamp(min=1)


def _reference_group_balanced_masked_cross_entropy(
    logits: Any, targets: Any, mask: Any, groups: Any
) -> Any:
    """Pre-optimization role-balanced chemistry cross entropy."""

    selected_groups = groups[mask]
    point_losses = torch.nn.functional.cross_entropy(logits[mask], targets[mask], reduction="none")
    group_count = int(groups.max()) + 1
    sums = logits.new_zeros(group_count).scatter_add(0, selected_groups, point_losses)
    counts = torch.bincount(selected_groups, minlength=group_count)
    present = counts > 0
    means = sums / counts.clamp(min=1)
    return (means * present).sum() / present.sum().clamp(min=1)


def _reference_fixed_state_exact_tensor(state: Mapping[str, Any], clean: Mapping[str, Any]) -> Any:
    """Pre-optimization fixed-state check: gather the protected positions of every field."""

    checks = [
        (state[field][clean[mask]] == clean[field][clean[mask]]).all()
        for field, mask in {
            "nodes": "fixed_atom_mask",
            "parents": "fixed_parent_mask",
            "parent_bonds": "fixed_parent_bond_mask",
            "closure_left": "fixed_closure_endpoint_mask",
            "closure_right": "fixed_closure_endpoint_mask",
            "closure_bonds": "fixed_closure_bond_mask",
        }.items()
    ]
    exact = checks[0]
    for check in checks[1:]:
        exact = exact & check
    return exact


def _install_reference_execution(model: Any) -> list[tuple[Any, str, Any]]:
    """Restore the pre-optimization execution of every changed hot-path function."""

    from forge.model import reaction_program_flow as flow
    from forge.model import reaction_program_transformer as module

    undo: list[tuple[Any, str, Any]] = [
        (type(model), "_graph_bias", type(model)._graph_bias),
        (module, "_balanced_state_cross_entropy", module._balanced_state_cross_entropy),
        (module, "_slice_batch", module._slice_batch),
        (flow, "_masked_cross_entropy", flow._masked_cross_entropy),
        (
            flow,
            "_group_balanced_masked_cross_entropy",
            flow._group_balanced_masked_cross_entropy,
        ),
    ]
    type(model)._graph_bias = _reference_graph_bias
    module._balanced_state_cross_entropy = _reference_balanced_state_cross_entropy
    module._slice_batch = _reference_slice_batch
    flow._masked_cross_entropy = _reference_masked_cross_entropy
    flow._group_balanced_masked_cross_entropy = _reference_group_balanced_masked_cross_entropy
    for candidate in model.modules():
        if candidate.__class__.__name__ == "_MaskedMultiheadAttention":
            undo.append((candidate.__class__, "forward", candidate.__class__.forward))
            candidate.__class__.forward = _reference_attention_forward
            break
    return undo


def _restore(undo: Sequence[tuple[Any, str, Any]]) -> None:
    for owner, name, original in reversed(undo):
        setattr(owner, name, original)


class _DispatchCounter:
    """Count aten dispatches without changing what any of them computes."""

    def __init__(self) -> None:
        from torch.utils._python_dispatch import TorchDispatchMode

        counter = self

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
                name = str(func)
                counter.counts[name] += 1
                if _is_synchronizing(name, args):
                    counter.synchronizing += 1
                return func(*args, **(kwargs or {}))

        self.counts: Counter[str] = Counter()
        self.synchronizing = 0
        self._mode = _Mode()

    def __enter__(self) -> _DispatchCounter:
        self._mode.__enter__()
        return self

    def __exit__(self, *exception: object) -> None:
        self._mode.__exit__(*exception)

    def summary(self, *, micro_batches: int) -> dict[str, Any]:
        total = sum(self.counts.values())
        kernels = sum(c for n, c in self.counts.items() if n not in _VIEW_DISPATCHES)
        syncs = self.synchronizing
        return {
            "dispatches_per_optimizer_step": total * micro_batches,
            "kernel_launching_dispatches_per_optimizer_step": kernels * micro_batches,
            "host_synchronizing_dispatches_per_optimizer_step": syncs * micro_batches,
            "most_common": self.counts.most_common(12),
        }


def _set_determinism(seed: int, cpu_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)


def _predict(model: Any, clean: Mapping[str, Any], state: Mapping[str, Any], t: Any) -> Any:
    return model(
        nodes=state["nodes"],
        parents=state["parents"],
        parent_bonds=state["parent_bonds"],
        closure_left=state["closure_left"],
        closure_right=state["closure_right"],
        closure_bonds=state["closure_bonds"],
        t=t,
        node_mask=clean["node_mask"],
        child_mask=clean["child_mask"],
        closure_mask=clean["closure_mask"],
        program_states=clean["program_states"],
        role_states=clean["role_states"],
        core_position_states=clean["core_position_states"],
        program_depths=clean["program_depths"],
        adapter_mask=clean["adapter_mask"],
        repeat_group_states=clean["repeat_group_states"],
        component_position_states=clean["component_position_states"],
        component_instance_states=clean["component_instance_states"],
        role_morphology_states=clean["role_morphology_states"],
    )


class _Workload:
    """One frozen optimizer step of micro-batches, materialized once and reused by every arm."""

    def __init__(
        self,
        *,
        cache: SynthesisProgramProductionCache,
        design: Mapping[str, Any],
        arm_id: str,
        seed: int,
        micro_batch_size: int,
        gradient_accumulation_steps: int,
        quantile_bins: int,
    ) -> None:
        self.model_config = dict(design["model"])
        self.arm = design["training"]["arms"][arm_id]
        self.objective = self.model_config["semantic_objective"]
        self.cache = cache
        measure = cache.training_measure(self.arm["program_mass"])
        self.sampler = _compile_stratified_program_sampler(cache, measure)
        node_marginal, bond_marginal = cache.source_marginals(
            measure,
            node_classes=len(cache.atom_vocabulary),
            bond_classes=int(self.model_config["bond_classes"]),
            probability_floor=float(self.model_config["source_probability_floor"]),
        )
        node_p0 = torch.as_tensor(node_marginal, dtype=torch.float32)
        bond_p0 = torch.as_tensor(bond_marginal, dtype=torch.float32)
        rng = np.random.default_rng(seed + 2)
        generator = torch.Generator().manual_seed(seed + 1)
        self.batches = []
        node_counts = np.diff(cache.arrays["node_offsets"])
        self.true_node_counts: list[int] = []
        for _ in range(gradient_accumulation_steps):
            selected = self.sampler.sample(
                micro_batch_size, rng, padding_aware_quantile_bins=quantile_bins
            )
            self.true_node_counts.extend(int(value) for value in node_counts[selected])
            clean = move_tensors(
                collate_synthesis_program_training_batch(
                    cache.records(selected),
                    maximum_closures=int(self.model_config["maximum_closures"]),
                    conditioning=str(self.arm["conditioning"]),
                    vocabulary=cache.vocabulary,
                    program_id_mapping=self.arm.get("program_id_mapping"),
                ),
                torch.device("cpu"),
            )
            t = torch.rand(len(selected), generator=generator).clamp(0.02, 0.98)
            noisy = noise_synthesis_program_batch(clean, node_p0, bond_p0, t, generator)
            self.batches.append((clean, noisy, t))
        self.padded_widths = [int(clean["nodes"].shape[1]) for clean, _, _ in self.batches]

    def padding_summary(self) -> dict[str, Any]:
        counts = np.asarray(self.true_node_counts, dtype=np.float64)
        widths = np.asarray(self.padded_widths, dtype=np.float64)
        rows = len(counts) / len(widths)
        return {
            "padded_widths": self.padded_widths,
            "mean_padded_width": float(widths.mean()),
            "mean_true_heavy_atoms": float(counts.mean()),
            "maximum_true_heavy_atoms": int(counts.max()),
            "node_linear_utilization": float(counts.sum() / (rows * widths.sum())),
            "attention_quadratic_utilization": float(
                (counts**2).sum() / (rows * (widths**2).sum())
            ),
        }

    def micro_batch(self, index: int) -> tuple[Any, Any, Any]:
        return self.batches[index]

    def __len__(self) -> int:
        return len(self.batches)


def _hot_path(
    workload: _Workload,
    model: Any,
    batch: tuple[Any, Any, Any],
    *,
    backend: str,
    scale: float,
) -> Any:
    clean, noisy, t = batch
    predictions = _predict(model, clean, noisy, t)
    objective = workload.objective
    family_losses, _ = per_program_transformer_losses(
        predictions,
        clean,
        role_weight=float(objective["role_consistency_weight"]),
        core_weight=float(objective["core_consistency_weight"]),
        repeat_consistency_weight=float(objective.get("repeat_consistency_weight", 0.0)),
        program_states=workload.sampler.program_states,
        materialize_metrics=False,
    )
    loss = torch.stack(list(family_losses.values())).mean().detach().clone()
    balanced_pcgrad_backward(
        family_losses,
        model,
        scale=scale,
        materialize_diagnostics=False,
        backend=backend,
    )
    synthesis_program_fixed_state_exact_tensor(noisy, clean)
    return loss


def _flat_gradients(model: Any) -> Any:
    return torch.cat(
        [
            (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).reshape(
                -1
            )
            for parameter in model.parameters()
        ]
    ).clone()


def _clear_gradients(model: Any) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def _paired_execution_comparison(
    workload: _Workload,
    model: Any,
    *,
    repeats: int,
    backend: str,
) -> dict[str, Any]:
    """Time the optimized and pre-optimization execution of the same micro-batch, back to back."""

    scale = 1.0 / len(workload)
    rng_state = torch.get_rng_state()
    captured: dict[str, Any] = {}
    ratios: list[float] = []
    walls: dict[str, list[float]] = {"reference": [], "optimized": []}

    def run(arm: str) -> tuple[float, list[Any]]:
        _clear_gradients(model)
        undo = _install_reference_execution(model) if arm == "reference" else []
        try:
            torch.set_rng_state(rng_state)
            losses = []
            started = time.perf_counter()
            for index in range(len(workload)):
                losses.append(
                    _hot_path(
                        workload,
                        model,
                        workload.micro_batch(index),
                        backend=backend,
                        scale=scale,
                    )
                )
            elapsed = time.perf_counter() - started
        finally:
            _restore(undo)
        return elapsed, losses

    for repeat in range(repeats + 1):
        pair: dict[str, float] = {}
        order = ("reference", "optimized") if repeat % 2 == 0 else ("optimized", "reference")
        for arm in order:
            elapsed, losses = run(arm)
            pair[arm] = elapsed
            captured[arm] = (torch.stack(losses), _flat_gradients(model))
        if repeat:
            walls["reference"].append(pair["reference"])
            walls["optimized"].append(pair["optimized"])
            ratios.append(pair["reference"] / pair["optimized"])

    reference_losses, reference_gradients = captured["reference"]
    optimized_losses, optimized_gradients = captured["optimized"]
    loss_delta = (optimized_losses - reference_losses).abs().double()
    gradient_delta = (optimized_gradients - reference_gradients).double()
    reference_norm = float(torch.linalg.vector_norm(reference_gradients.double()))
    equivalence = {
        "losses_bitwise_exact": bool(torch.equal(reference_losses, optimized_losses)),
        "gradients_bitwise_exact": bool(torch.equal(reference_gradients, optimized_gradients)),
        "loss_max_absolute_difference": float(loss_delta.max()),
        "loss_max_relative_difference": float(
            (loss_delta / reference_losses.abs().double().clamp(min=1e-12)).max()
        ),
        "gradient_max_absolute_difference": float(gradient_delta.abs().max()),
        "gradient_relative_l2_difference": float(torch.linalg.vector_norm(gradient_delta))
        / max(reference_norm, 1e-12),
        "gradient_cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                reference_gradients.double()[None],
                optimized_gradients.double()[None],
                dim=1,
                eps=1e-12,
            )[0]
        ),
    }
    equivalence["passes_frozen_float32_tolerance"] = bool(
        equivalence["loss_max_absolute_difference"]
        <= EQUIVALENCE_TOLERANCE["loss_max_absolute_difference"]
        and equivalence["loss_max_relative_difference"]
        <= EQUIVALENCE_TOLERANCE["loss_max_relative_difference"]
        and equivalence["gradient_relative_l2_difference"]
        <= EQUIVALENCE_TOLERANCE["gradient_relative_l2_difference"]
        and equivalence["gradient_max_absolute_difference"]
        <= EQUIVALENCE_TOLERANCE["gradient_max_absolute_difference"]
        and equivalence["gradient_cosine_similarity"]
        >= EQUIVALENCE_TOLERANCE["gradient_cosine_similarity"]
    )
    return {
        "repeats": repeats,
        "reference_step_seconds": walls["reference"],
        "optimized_step_seconds": walls["optimized"],
        "reference_min_step_seconds": min(walls["reference"]),
        "optimized_min_step_seconds": min(walls["optimized"]),
        "paired_ratios": ratios,
        "speedup_paired_median": statistics.median(ratios),
        "speedup_paired_minimum": min(walls["reference"]) / min(walls["optimized"]),
        "equivalence": equivalence,
    }


def _pcgrad_walk_headroom(workload: _Workload, model: Any, *, repeats: int) -> dict[str, Any]:
    """Compare three per-family backward walks with one walk of their sum.

    The three family losses depend on disjoint batch rows, so one walk of their sum carries exactly
    the activation gradients of all three.  Three walks buy only the per-family split of the
    parameter reduction, and this measures what that split costs.
    """

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    clean, noisy, t = workload.micro_batch(0)
    objective = workload.objective
    rng_state = torch.get_rng_state()

    def losses() -> dict[int, Any]:
        torch.set_rng_state(rng_state)
        predictions = _predict(model, clean, noisy, t)
        family_losses, _ = per_program_transformer_losses(
            predictions,
            clean,
            role_weight=float(objective["role_consistency_weight"]),
            core_weight=float(objective["core_consistency_weight"]),
            repeat_consistency_weight=float(objective.get("repeat_consistency_weight", 0.0)),
            program_states=workload.sampler.program_states,
            materialize_metrics=False,
        )
        return family_losses

    def three_walks() -> float:
        family_losses = losses()
        ordered = sorted(family_losses)
        started = time.perf_counter()
        for index, program in enumerate(ordered):
            torch.autograd.grad(
                family_losses[program],
                parameters,
                retain_graph=index + 1 < len(ordered),
                allow_unused=True,
            )
        return time.perf_counter() - started

    def one_walk() -> float:
        family_losses = losses()
        started = time.perf_counter()
        summed = torch.stack([family_losses[key] for key in sorted(family_losses)]).sum()
        torch.autograd.grad(summed, parameters, allow_unused=True)
        return time.perf_counter() - started

    def forward_only() -> float:
        started = time.perf_counter()
        losses()
        return time.perf_counter() - started

    samples: dict[str, list[float]] = {"three_walks": [], "one_walk": [], "forward": []}
    measurements = {
        "three_walks": three_walks,
        "one_walk": one_walk,
        "forward": forward_only,
    }
    for repeat in range(repeats + 1):
        for name, measure in measurements.items():
            value = measure()
            if repeat:
                samples[name].append(value)
    _clear_gradients(model)
    report = {
        name: {"min": min(values), "median": statistics.median(values)}
        for name, values in samples.items()
    }
    report["three_walks_over_one_walk_min"] = (
        report["three_walks"]["min"] / report["one_walk"]["min"]
    )
    report["projected_step_speedup_if_one_walk"] = (
        report["forward"]["min"] + report["three_walks"]["min"]
    ) / (report["forward"]["min"] + report["one_walk"]["min"])
    return report


def _packing_comparison(
    *,
    cache: SynthesisProgramProductionCache,
    design: Mapping[str, Any],
    arm_id: str,
    seed: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    quantile_bins: int,
    model: Any,
    repeats: int,
    backend: str,
) -> dict[str, Any]:
    """Compare the frozen packing with the padding-aware quantile-bin packing.

    Both draws come from the same source measure.  Only the packing of a micro-batch differs, so
    the difference is the cost of the padding each packing leaves behind.  The two arms hold
    different records, so their losses are not comparable and no equivalence is claimed.
    """

    arms = {
        "quantile_bins_1": _Workload(
            cache=cache,
            design=design,
            arm_id=arm_id,
            seed=seed,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            quantile_bins=1,
        ),
        f"quantile_bins_{quantile_bins}": _Workload(
            cache=cache,
            design=design,
            arm_id=arm_id,
            seed=seed,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            quantile_bins=quantile_bins,
        ),
    }
    names = list(arms)
    best = {name: [math.inf] * gradient_accumulation_steps for name in names}
    rng_state = torch.get_rng_state()
    for repeat in range(repeats + 1):
        for index in range(gradient_accumulation_steps):
            order = names if (repeat + index) % 2 == 0 else names[::-1]
            for name in order:
                _clear_gradients(model)
                torch.set_rng_state(rng_state)
                started = time.perf_counter()
                _hot_path(
                    arms[name],
                    model,
                    arms[name].micro_batch(index),
                    backend=backend,
                    scale=1.0 / gradient_accumulation_steps,
                )
                elapsed = time.perf_counter() - started
                if repeat:
                    best[name][index] = min(best[name][index], elapsed)
    _clear_gradients(model)
    step_seconds = {name: sum(values) for name, values in best.items()}
    return {
        "padding": {name: arm.padding_summary() for name, arm in arms.items()},
        "per_micro_batch_best_seconds": best,
        "optimizer_step_seconds_from_best": step_seconds,
        "examples_per_second_from_best": {
            name: micro_batch_size * gradient_accumulation_steps / value
            for name, value in step_seconds.items()
        },
        "speedup": step_seconds[names[0]] / step_seconds[names[1]],
    }


def _dispatch_accounting(workload: _Workload, model: Any, *, backend: str) -> dict[str, Any]:
    rng_state = torch.get_rng_state()
    report: dict[str, Any] = {}
    for arm in ("reference", "optimized"):
        _clear_gradients(model)
        undo = _install_reference_execution(model) if arm == "reference" else []
        try:
            torch.set_rng_state(rng_state)
            counter = _DispatchCounter()
            with counter:
                _hot_path(
                    workload,
                    model,
                    workload.micro_batch(0),
                    backend=backend,
                    scale=1.0 / len(workload),
                )
        finally:
            _restore(undo)
        report[arm] = counter.summary(micro_batches=len(workload))
    _clear_gradients(model)
    return report


def run_cpu_training_step_profile(
    *,
    repo: Path,
    design_path: Path,
    cache_path: Path,
    output_path: Path,
    arm_id: str,
    seed: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    quantile_bins: int,
    pcgrad_backend: str,
    repeats: int,
    cpu_threads: int,
) -> dict[str, Any]:
    """Profile the frozen optimizer step on CPU and adjudicate execution equivalence."""

    if torch is None:
        raise CpuTrainingStepProfileError("CPU training-step profile requires torch")
    design = read_json_object(
        design_path, error=CpuTrainingStepProfileError, label="production design"
    )
    frozen = design["training"]
    if micro_batch_size * gradient_accumulation_steps != int(frozen["effective_batch_size"]):
        raise CpuTrainingStepProfileError(
            "profiling geometry does not reproduce the frozen effective batch size"
        )
    if frozen["precision"] != "float32" or frozen["deterministic_algorithms"] is not True:
        raise CpuTrainingStepProfileError("the frozen design is not deterministic float32")
    if pcgrad_backend not in {"sequential", "batched_vjp"}:
        raise CpuTrainingStepProfileError(f"unsupported PCGrad backend: {pcgrad_backend!r}")

    _set_determinism(seed, cpu_threads)
    cache = SynthesisProgramProductionCache(cache_path)
    try:
        workload = _Workload(
            cache=cache,
            design=design,
            arm_id=arm_id,
            seed=seed,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            quantile_bins=1,
        )
        model = build_synthesis_program_flow(
            vocabulary=cache.vocabulary,
            node_classes=len(cache.atom_vocabulary),
            model_config=workload.model_config,
            device=torch.device("cpu"),
        )
        model.train()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        execution = _paired_execution_comparison(
            workload, model, repeats=repeats, backend=pcgrad_backend
        )
        dispatch = _dispatch_accounting(workload, model, backend=pcgrad_backend)
        headroom = _pcgrad_walk_headroom(workload, model, repeats=repeats)
        packing = _packing_comparison(
            cache=cache,
            design=design,
            arm_id=arm_id,
            seed=seed,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            quantile_bins=quantile_bins,
            model=model,
            repeats=repeats,
            backend=pcgrad_backend,
        )
        backend_comparison = {}
        for backend in ("sequential", "batched_vjp"):
            _clear_gradients(model)
            rng_state = torch.get_rng_state()
            samples = []
            for _ in range(repeats + 1):
                torch.set_rng_state(rng_state)
                _clear_gradients(model)
                started = time.perf_counter()
                _hot_path(
                    workload,
                    model,
                    workload.micro_batch(0),
                    backend=backend,
                    scale=1.0 / gradient_accumulation_steps,
                )
                samples.append(time.perf_counter() - started)
            backend_comparison[backend] = {
                "min_micro_batch_seconds": min(samples[1:]),
                "median_micro_batch_seconds": statistics.median(samples[1:]),
            }
        _clear_gradients(model)
    finally:
        cache.close()

    examples = micro_batch_size * gradient_accumulation_steps
    gates = {
        "execution_change_is_bitwise_exact": bool(
            execution["equivalence"]["losses_bitwise_exact"]
            and execution["equivalence"]["gradients_bitwise_exact"]
        ),
        "execution_change_passes_frozen_tolerance": bool(
            execution["equivalence"]["passes_frozen_float32_tolerance"]
        ),
        "frozen_geometry_preserved": micro_batch_size == int(frozen["micro_batch_size"])
        and gradient_accumulation_steps == int(frozen["gradient_accumulation_steps"]),
        "three_source_balanced_programs_active": len(workload.sampler.program_states) == 3,
        "full_194_atom_support_declared": int(workload.model_config["maximum_heavy_atoms"]) == 194,
        "host_synchronizations_not_increased": int(
            dispatch["optimized"]["host_synchronizing_dispatches_per_optimizer_step"]
        )
        <= int(dispatch["reference"]["host_synchronizing_dispatches_per_optimizer_step"]),
    }
    result = {
        "schema_version": SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "seed": seed,
        "arm_id": arm_id,
        "inputs": {
            "production_design": {
                "path": str(design_path.relative_to(repo)),
                "sha256": str(sha256_file(design_path)),
            },
            "production_cache": artifact_record(cache_path),
        },
        "runtime": {
            "device": "cpu",
            "precision": "float32",
            "deterministic_algorithms": True,
            "cpu_threads": cpu_threads,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "scientific_contract": {
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": examples,
            "maximum_heavy_atoms": int(workload.model_config["maximum_heavy_atoms"]),
            "source_balancing": "stratified_equal_reaction_family_mass",
            "gradient_balancing": workload.model_config["gradient_balancing"],
            "parameter_count": parameter_count,
            "pcgrad_backend": pcgrad_backend,
        },
        "h100_reference": H100_REFERENCE,
        "frozen_packing_padding": workload.padding_summary(),
        "execution_equivalence_and_speed": execution,
        "dispatch_accounting": dispatch,
        "pcgrad_walk_headroom": headroom,
        "packing_comparison": packing,
        "pcgrad_backend_comparison": backend_comparison,
        "gates": gates,
        "calls": {"route": 0, "oracle": 0},
        "candidate_selection": False,
        "nonclaims": [
            "This is a CPU execution profile. It does not establish accelerator throughput, "
            "model quality, or a production launch decision.",
            "Absolute seconds on a shared workstation are not comparable across runs; only the "
            "paired ratios and the dispatch counts are claims.",
            "The packing comparison changes which records share a micro-batch. It preserves the "
            "source measure but is not a bitwise-equivalent execution change.",
        ],
    }
    write_json(output_path, result)
    if result["status"] != "pass":
        raise CpuTrainingStepProfileError(f"CPU training-step profile gates failed: {gates}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=repo / "results/phase1/shared_synthesis_program_mixed_design_v1/design.json",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=repo / "results/phase1/shared_synthesis_program_mixed_cache_v1/cache.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "results/phase1/shared_synthesis_program_cpu_step_profile_v1/result.json",
    )
    parser.add_argument("--arm-id", default="shared_three_program_conditioned")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--quantile-bins", type=int, default=32)
    parser.add_argument("--pcgrad-backend", default="sequential")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=8)
    arguments = parser.parse_args(argv)
    result = run_cpu_training_step_profile(
        repo=repo,
        design_path=arguments.design,
        cache_path=arguments.cache,
        output_path=arguments.output,
        arm_id=arguments.arm_id,
        seed=arguments.seed,
        micro_batch_size=arguments.micro_batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        quantile_bins=arguments.quantile_bins,
        pcgrad_backend=arguments.pcgrad_backend,
        repeats=arguments.repeats,
        cpu_threads=arguments.cpu_threads,
    )
    print(json.dumps({key: result[key] for key in ("status", "gates")}, indent=1))
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - command entry point
    raise SystemExit(main())


__all__ = [
    "CpuTrainingStepProfileError",
    "EQUIVALENCE_TOLERANCE",
    "H100_REFERENCE",
    "SCHEMA",
    "run_cpu_training_step_profile",
]
