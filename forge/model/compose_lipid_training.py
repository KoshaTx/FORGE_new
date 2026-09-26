"""Source-aware COMPOSE batches through the existing graph-flow training primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from forge.corpus.compose_lipid_training_data import ComposeLipidTrainingData
from forge.model.reaction_program_flow import synthesis_program_flow_loss
from forge.model.reaction_program_transformer import (
    balanced_pcgrad_backward,
    reaction_program_transformer_loss,
)
from forge.model.repeat_supervision import aligned_repeat_consistency
from forge.model.synthesis_program_training import (
    move_tensors,
    synthesis_program_fixed_state_exact,
    synthesis_program_forward,
    synthesis_program_paired_topology_forward,
)


def validate_objective(objective: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep restored objectives explicit and reject misspelled or partial contracts."""
    if objective is None:
        return {}
    required = {
        "offspring_weight",
        "junction_consistency_weight",
        "topology_conditioned_chemistry_weight",
        "chemistry_loss_balancing",
        "gradient_balancing",
        "pcgrad_backend",
    }
    optional = {"parent_group_loss_weight"}
    if not required <= set(objective) or set(objective) - required - optional:
        raise ValueError("Restored objective requires every declared objective setting")
    for key in required:
        if key.endswith("_weight") and (not math.isfinite(objective[key]) or objective[key] < 0):
            raise ValueError(f"Invalid objective.{key}")
    group_weight = objective.get("parent_group_loss_weight", 0.0)
    if (
        type(group_weight) not in (float, int)
        or not math.isfinite(group_weight)
        or group_weight < 0
    ):
        raise ValueError("Invalid objective.parent_group_loss_weight")
    if objective["chemistry_loss_balancing"] not in ("pooled", "equal_present_role_mass"):
        raise ValueError("Invalid chemistry loss balancing")
    if objective["gradient_balancing"] not in ("pooled", "equal_family_mean", "pcgrad"):
        raise ValueError("Invalid gradient balancing")
    if objective["pcgrad_backend"] not in ("sequential", "batched_vjp"):
        raise ValueError("Invalid PCGrad backend")
    return dict(objective)


def compose_lipid_forward_loss(
    model: Any,
    clean: Mapping[str, Any],
    *,
    architecture: str,
    node_marginal: torch.Tensor,
    bond_marginal: torch.Tensor,
    times: torch.Tensor,
    generator: torch.Generator,
    semantic_weights: Mapping[str, float],
    repeat_supervision: str = "serialization",
    objective: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply the existing masked objectives without discarding source coordinates.

    Sampling already applies the graph measure, so probabilities are not applied
    to the loss a second time. Semantic weights are explicit; this function does
    not select a new scientific objective or fit noise marginals.
    """
    if architecture not in {"sparse_mpnn", "reaction_program_graph_transformer"}:
        raise ValueError(f"Unsupported COMPOSE architecture: {architecture}")
    if repeat_supervision not in ("serialization", "exact_fragment"):
        raise ValueError("Unsupported repeat supervision")
    if (
        repeat_supervision == "exact_fragment"
        and architecture != "reaction_program_graph_transformer"
    ):
        raise ValueError("Exact fragment repeat supervision requires the transformer")
    required = {"role_weight", "core_weight", "repeat_consistency_weight"}
    if set(semantic_weights) != required or any(
        not math.isfinite(v) or v < 0 for v in semantic_weights.values()
    ):
        raise ValueError("Explicit finite nonnegative semantic loss weights are required")
    if architecture == "sparse_mpnn" and any(semantic_weights.values()):
        raise ValueError("Sparse MPNN has no auxiliary semantic prediction heads")
    settings = validate_objective(objective)
    if settings and architecture != "reaction_program_graph_transformer":
        raise ValueError("Restored objectives require the transformer")
    topology = None
    if settings.get("topology_conditioned_chemistry_weight", 0) > 0:
        predictions, noisy, topology = synthesis_program_paired_topology_forward(
            model, clean, node_marginal, bond_marginal, times, generator
        )
    else:
        predictions, noisy = synthesis_program_forward(
            model, clean, node_marginal, bond_marginal, times, generator
        )
    if not synthesis_program_fixed_state_exact(noisy, clean):
        raise ValueError("Noising changed adapter-fixed graph states")
    if architecture == "reaction_program_graph_transformer":
        weights = dict(semantic_weights)
        if repeat_supervision == "exact_fragment":
            weights["repeat_consistency_weight"] = 0.0
        loss, metrics = reaction_program_transformer_loss(
            predictions,
            clean,
            **weights,
            materialize_metrics=False,
            **{
                k: v
                for k, v in settings.items()
                if k not in ("gradient_balancing", "pcgrad_backend")
            },
            topology_conditioned_predictions=topology,
        )
        if repeat_supervision == "exact_fragment":
            repeat, atom_pairs, bond_pairs = aligned_repeat_consistency(
                predictions, clean["repeat_atom_groups"], clean["repeat_bond_groups"]
            )
            loss = loss + semantic_weights["repeat_consistency_weight"] * repeat
            metrics.update(
                repeat_consistency_mse=repeat.detach(),
                repeat_consistency_pairs=atom_pairs.detach(),
                repeat_bond_pairs=bond_pairs.detach(),
                semantic_total=loss.detach(),
            )
    else:
        loss, metrics = synthesis_program_flow_loss(predictions, clean, materialize_metrics=False)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite COMPOSE training objective")
    return loss, metrics


def trim_family_padding(clean: Mapping[str, Any]) -> dict[str, Any]:
    """Trim trailing absent nodes from a collated family, retaining every closure slot.

    Collation guarantees prefix-contiguous node masks. Fields are identified by their
    semantic axis, since node and closure widths can coincide for small molecules.
    """
    width = int(clean["node_mask"].sum(1).max())
    node_fields = {
        "nodes",
        "parents",
        "parent_bonds",
        "node_mask",
        "child_mask",
        "role_states",
        "core_position_states",
        "component_instance_states",
        "component_position_states",
        "repeat_group_states",
        "role_morphology_states",
        "adapter_mask",
        "fixed_atom_mask",
        "fixed_parent_mask",
        "fixed_parent_bond_mask",
        "atom_variable_mask",
        "parent_variable_mask",
        "parent_bond_variable_mask",
        "repeat_atom_groups",
        "repeat_bond_groups",
    }
    return {key: value[:, :width] if key in node_fields else value for key, value in clean.items()}


def compose_lipid_backward(
    model: Any,
    clean: Mapping[str, Any],
    *,
    objective: Mapping[str, Any] | None = None,
    family_padding: bool = False,
    **forward_kwargs: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Balance formal families, including families with multiple program bindings.

    Per-family forwards bound activation memory when the broad batch contains many
    families. Source IDs never enter the model. The ordinary path stays unchanged.
    """
    settings = validate_objective(objective)
    method = settings.get("gradient_balancing", "pooled")
    if family_padding and method == "pooled":
        raise ValueError("Family padding requires family-balanced gradients")
    if method == "pooled":
        loss, metrics = compose_lipid_forward_loss(
            model, clean, objective=objective, **forward_kwargs
        )
        loss.backward()
        return loss, metrics
    if "family_states" not in clean:
        raise ValueError("Formal family labels are required for family-balanced gradients")
    losses, metrics = {}, {}
    count = len(clean["nodes"])
    families = torch.unique(clean["family_states"], sorted=True).tolist()
    for family in families:
        selected = (clean["family_states"] == family).nonzero(as_tuple=True)[0]
        batch = {
            key: (
                value[selected]
                if torch.is_tensor(value) and value.ndim and value.shape[0] == count
                else value
            )
            for key, value in clean.items()
        }
        kwargs = dict(forward_kwargs, times=forward_kwargs["times"][selected])
        if family_padding:
            batch = trim_family_padding(batch)
        loss, values = compose_lipid_forward_loss(model, batch, objective=objective, **kwargs)
        losses[int(family)] = loss
        for key, value in values.items():
            metrics[key] = metrics.get(key, 0) + value / len(families)
    loss = torch.stack(list(losses.values())).mean()
    if method == "pcgrad" and len(losses) > 1:
        diagnostic = balanced_pcgrad_backward(
            losses, model, backend=settings["pcgrad_backend"], materialize_diagnostics=False
        )
        metrics["projected_conflicts"] = diagnostic["projected_conflicts"]
    else:
        loss.backward()
    return loss, metrics


def compose_lipid_training_step(
    data: ComposeLipidTrainingData,
    model: Any,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    maximum_nodes: int,
    maximum_closures: int,
    architecture: str,
    device: torch.device,
    rng: np.random.Generator,
    generator: torch.Generator,
    node_marginal: torch.Tensor,
    bond_marginal: torch.Tensor,
    semantic_weights: Mapping[str, float],
    gradient_clip_norm: float,
    node_padding: str = "model",
    repeat_supervision: str = "serialization",
    core_conditioning: str = "adapter",
    objective: Mapping[str, Any] | None = None,
    families_per_batch: int | None = None,
    family_selection: Any = None,
    tensor_cache: Any = None,
    prepared_batch: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Draw admitted graphs, preserve source semantics, and perform one finite update.

    The caller owns model/optimizer/RNG state and the restartable run lifecycle.
    Opening ``data`` requires final admission before this update is reachable.
    Batch padding preserves the sampling measure and objective but changes RNG
    consumption in corruption/dropout; it must be fixed in the run configuration.
    The runner validates prepared storage. For a prefetched batch it commits the
    associated sampler state only after this update returns successfully.
    """
    if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be finite and positive")
    if prepared_batch is None:
        selected = (
            data.sample_indices(
                batch_size,
                rng,
                families_per_batch=families_per_batch,
                family_selection=family_selection,
            )
            if family_selection is not None
            else (
                data.sample_indices(batch_size, rng)
                if families_per_batch is None
                else data.sample_indices(batch_size, rng, families_per_batch=families_per_batch)
            )
        )
        prepared_batch = (
            tensor_cache.batch(selected)
            if tensor_cache is not None
            else data.batch(
                selected,
                maximum_nodes=maximum_nodes,
                maximum_closures=maximum_closures,
                node_padding="batch" if node_padding == "family" else node_padding,
                repeat_supervision=repeat_supervision,
                core_conditioning=core_conditioning,
            )
        )
    if len(prepared_batch["nodes"]) != batch_size:
        raise ValueError("Prepared batch size differs from the training contract")
    clean = move_tensors(prepared_batch, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    times = torch.rand(batch_size, generator=generator, device=device)
    loss, metrics = compose_lipid_backward(
        model,
        clean,
        architecture=architecture,
        node_marginal=node_marginal,
        bond_marginal=bond_marginal,
        times=times,
        generator=generator,
        semantic_weights=semantic_weights,
        repeat_supervision=repeat_supervision,
        objective=objective,
        family_padding=node_padding == "family",
    )
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), gradient_clip_norm, error_if_nonfinite=True
    )
    optimizer.step()
    return {
        **{name: float(value.detach()) for name, value in metrics.items()},
        "loss": float(loss.detach()),
        "gradient_norm": float(norm),
    }
