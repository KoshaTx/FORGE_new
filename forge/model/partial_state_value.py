"""Small role-aware value heads over frozen FORGE partial-state representations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

try:
    import torch
    from torch import nn
    from torch.nn import functional
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]


class PartialStateValueError(ValueError):
    """A partial-state value tensor or training objective is malformed."""


if nn is not None:

    def role_aware_graph_summary(
        hidden: Any,
        node_mask: Any,
        role_states: Any,
        flow_time: Any,
        *,
        role_count: int,
    ) -> Any:
        """Pool one partial graph globally and by declared precursor role.

        The summary contains no component identifier or stored precursor graph.  Empty roles are
        represented by zeros, and all reductions ignore padded nodes.
        """

        if (
            hidden.ndim != 3
            or node_mask.shape != hidden.shape[:2]
            or role_states.shape != hidden.shape[:2]
            or flow_time.ndim != 1
            or flow_time.shape[0] != hidden.shape[0]
            or role_count < 2
        ):
            raise PartialStateValueError("partial-state tensors have incompatible shapes")
        if role_states.numel() and (
            int(role_states.min().item()) < 0 or int(role_states.max().item()) >= role_count
        ):
            raise PartialStateValueError("partial-state role index is outside the vocabulary")

        mask = node_mask.to(dtype=hidden.dtype)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        global_mean = (hidden * mask[:, :, None]).sum(dim=1) / count
        masked_hidden = hidden.masked_fill(~node_mask[:, :, None], -torch.inf)
        global_max = masked_hidden.max(dim=1).values
        global_max = torch.where(torch.isfinite(global_max), global_max, 0.0)

        role_means = []
        role_presence = []
        for role in range(1, role_count):
            role_mask = node_mask & (role_states == role)
            role_float = role_mask.to(dtype=hidden.dtype)
            role_count_tensor = role_float.sum(dim=1, keepdim=True)
            role_means.append(
                (hidden * role_float[:, :, None]).sum(dim=1) / role_count_tensor.clamp(min=1.0)
            )
            role_presence.append((role_count_tensor > 0).to(dtype=hidden.dtype))

        time_features = torch.stack(
            (
                flow_time,
                torch.sin(math.pi * flow_time),
                torch.cos(math.pi * flow_time),
            ),
            dim=1,
        )
        return torch.cat(
            (global_mean, global_max, *role_means, *role_presence, time_features), dim=1
        )

    class RoleAwarePartialStateValueHead(nn.Module):
        """Predict a scalar endpoint value directly from a frozen partial graph."""

        def __init__(
            self,
            *,
            hidden_dim: int,
            role_count: int,
            value_hidden_dim: int = 128,
        ) -> None:
            super().__init__()
            if hidden_dim < 1 or role_count < 2 or value_hidden_dim < 2:
                raise PartialStateValueError("partial-state value dimensions must be positive")
            self.hidden_dim = int(hidden_dim)
            self.role_count = int(role_count)
            self.value_hidden_dim = int(value_hidden_dim)
            self.summary_dim = hidden_dim * (role_count + 1) + (role_count - 1) + 3
            self.network = nn.Sequential(
                nn.LayerNorm(self.summary_dim),
                nn.Linear(self.summary_dim, value_hidden_dim),
                nn.GELU(),
                nn.Linear(value_hidden_dim, value_hidden_dim),
                nn.GELU(),
                nn.Linear(value_hidden_dim, 1),
            )

        def summarize(self, hidden: Any, node_mask: Any, role_states: Any, flow_time: Any) -> Any:
            return role_aware_graph_summary(
                hidden,
                node_mask,
                role_states,
                flow_time,
                role_count=self.role_count,
            )

        def forward_summary(self, summary: Any) -> Any:
            if summary.ndim != 2 or summary.shape[1] != self.summary_dim:
                raise PartialStateValueError("pooled partial-state summary changed shape")
            return self.network(summary).squeeze(-1)

        def forward(self, hidden: Any, node_mask: Any, role_states: Any, flow_time: Any) -> Any:
            return self.forward_summary(self.summarize(hidden, node_mask, role_states, flow_time))

    class LayerwiseRoleAttentionValueHead(nn.Module):
        """Predict a scalar value without discarding node-level or role-local structure."""

        def __init__(
            self,
            *,
            hidden_dim: int,
            layer_count: int,
            atom_classes: int,
            bond_classes: int,
            role_count: int,
            attention_role_indices: tuple[int, int, int],
            value_hidden_dim: int = 256,
        ) -> None:
            super().__init__()
            if (
                hidden_dim < 1
                or layer_count < 1
                or atom_classes < 2
                or bond_classes < 2
                or role_count < 4
                or value_hidden_dim < 2
                or len(set(attention_role_indices)) != 3
                or any(index <= 0 or index >= role_count for index in attention_role_indices)
            ):
                raise PartialStateValueError("layerwise attention dimensions or roles are invalid")
            self.hidden_dim = int(hidden_dim)
            self.layer_count = int(layer_count)
            self.atom_classes = int(atom_classes)
            self.bond_classes = int(bond_classes)
            self.role_count = int(role_count)
            self.attention_role_indices = tuple(int(index) for index in attention_role_indices)
            self.value_hidden_dim = int(value_hidden_dim)

            self.layer_logits = nn.Parameter(torch.zeros(self.layer_count))
            self.attention_queries = nn.Parameter(torch.empty(4, self.hidden_dim))
            nn.init.normal_(self.attention_queries, mean=0.0, std=self.hidden_dim**-0.5)
            self.atom_projection = nn.Linear(self.atom_classes, self.hidden_dim, bias=False)
            self.bond_projection = nn.Linear(self.bond_classes, self.hidden_dim, bias=False)
            self.closure_projection = nn.Linear(self.bond_classes, self.hidden_dim, bias=False)
            self.node_norm = nn.LayerNorm(self.hidden_dim)
            # Four learned readouts, three explicit pairwise interactions, one closure summary and
            # three flow-time coordinates.  No component identity enters this representation.
            self.summary_dim = 8 * self.hidden_dim + 3
            self.network = nn.Sequential(
                nn.LayerNorm(self.summary_dim),
                nn.Linear(self.summary_dim, self.value_hidden_dim),
                nn.GELU(),
                nn.Linear(self.value_hidden_dim, self.value_hidden_dim),
                nn.GELU(),
                nn.Linear(self.value_hidden_dim, 1),
            )

        def _attend(self, nodes: Any, mask: Any, query: Any) -> Any:
            present = mask.any(dim=1, keepdim=True)
            scores = torch.einsum("bnd,d->bn", nodes, query) / math.sqrt(self.hidden_dim)
            scores = scores.masked_fill(~mask, -torch.inf)
            scores = torch.where(present, scores, torch.zeros_like(scores))
            weights = torch.softmax(scores, dim=1) * mask.to(dtype=nodes.dtype)
            weights /= weights.sum(dim=1, keepdim=True).clamp(min=1.0)
            return torch.einsum("bn,bnd->bd", weights, nodes)

        def summarize(
            self,
            hidden_layers: Any,
            atom_probabilities: Any,
            parent_bond_probabilities: Any,
            closure_bond_probabilities: Any,
            node_mask: Any,
            closure_mask: Any,
            role_states: Any,
            flow_time: Any,
        ) -> Any:
            if (
                hidden_layers.ndim != 4
                or hidden_layers.shape[1] != self.layer_count
                or hidden_layers.shape[-1] != self.hidden_dim
                or atom_probabilities.shape
                != (*hidden_layers.shape[:1], hidden_layers.shape[2], self.atom_classes)
                or parent_bond_probabilities.shape
                != (*hidden_layers.shape[:1], hidden_layers.shape[2], self.bond_classes)
                or closure_bond_probabilities.ndim != 3
                or closure_bond_probabilities.shape[0] != hidden_layers.shape[0]
                or closure_bond_probabilities.shape[2] != self.bond_classes
                or node_mask.shape != hidden_layers.shape[:1] + hidden_layers.shape[2:3]
                or role_states.shape != node_mask.shape
                or closure_mask.shape != closure_bond_probabilities.shape[:2]
                or flow_time.shape != hidden_layers.shape[:1]
            ):
                raise PartialStateValueError(
                    "layerwise partial-state tensors have incompatible shapes"
                )
            tensors = (
                hidden_layers,
                atom_probabilities,
                parent_bond_probabilities,
                closure_bond_probabilities,
                flow_time,
            )
            if any(not bool(torch.isfinite(value).all()) for value in tensors):
                raise PartialStateValueError("layerwise partial-state tensors must be finite")
            if role_states.numel() and (
                int(role_states.min().item()) < 0
                or int(role_states.max().item()) >= self.role_count
            ):
                raise PartialStateValueError("partial-state role index is outside the vocabulary")

            layer_weights = torch.softmax(self.layer_logits, dim=0)
            nodes = torch.einsum("l,blnd->bnd", layer_weights, hidden_layers)
            nodes = self.node_norm(
                nodes
                + self.atom_projection(atom_probabilities)
                + self.bond_projection(parent_bond_probabilities)
            )
            nodes = nodes * node_mask[:, :, None]

            pooled = [self._attend(nodes, node_mask, self.attention_queries[0])]
            for query_index, role_index in enumerate(self.attention_role_indices, start=1):
                role_mask = node_mask & (role_states == role_index)
                pooled.append(self._attend(nodes, role_mask, self.attention_queries[query_index]))
            interactions = (
                pooled[1] * pooled[2],
                pooled[1] * pooled[3],
                pooled[2] * pooled[3],
            )

            closure_float = closure_mask.to(dtype=closure_bond_probabilities.dtype)
            closure_summary = (closure_bond_probabilities * closure_float[:, :, None]).sum(
                dim=1
            ) / closure_float.sum(dim=1, keepdim=True).clamp(min=1.0)
            closure_summary = self.closure_projection(closure_summary)
            time_features = torch.stack(
                (
                    flow_time,
                    torch.sin(math.pi * flow_time),
                    torch.cos(math.pi * flow_time),
                ),
                dim=1,
            )
            return torch.cat((*pooled, *interactions, closure_summary, time_features), dim=1)

        def forward_summary(self, summary: Any) -> Any:
            if summary.ndim != 2 or summary.shape[1] != self.summary_dim:
                raise PartialStateValueError("attention-pooled partial-state summary changed shape")
            return self.network(summary).squeeze(-1)

        def forward(
            self,
            hidden_layers: Any,
            atom_probabilities: Any,
            parent_bond_probabilities: Any,
            closure_bond_probabilities: Any,
            node_mask: Any,
            closure_mask: Any,
            role_states: Any,
            flow_time: Any,
        ) -> Any:
            return self.forward_summary(
                self.summarize(
                    hidden_layers,
                    atom_probabilities,
                    parent_bond_probabilities,
                    closure_bond_probabilities,
                    node_mask,
                    closure_mask,
                    role_states,
                    flow_time,
                )
            )

    def direct_value_loss(
        predictions: Any,
        targets: Any,
        *,
        rank_weight: float,
        huber_delta: float,
    ) -> tuple[Any, Mapping[str, Any]]:
        """Combine calibrated scalar regression with a within-batch ordinal contrast."""

        if (
            predictions.ndim != 1
            or targets.shape != predictions.shape
            or rank_weight < 0.0
            or huber_delta <= 0.0
            or not bool(torch.isfinite(predictions).all())
            or not bool(torch.isfinite(targets).all())
        ):
            raise PartialStateValueError("direct value loss inputs are malformed")
        regression = functional.huber_loss(
            predictions, targets, reduction="mean", delta=float(huber_delta)
        )
        target_difference = targets[:, None] - targets[None, :]
        prediction_difference = predictions[:, None] - predictions[None, :]
        upper = torch.triu(torch.ones_like(target_difference, dtype=torch.bool), diagonal=1)
        comparable = upper & (target_difference.abs() > 1e-8)
        if bool(comparable.any()):
            direction = torch.sign(target_difference[comparable])
            ranking = functional.softplus(-direction * prediction_difference[comparable]).mean()
        else:
            ranking = predictions.new_zeros(())
        total = regression + float(rank_weight) * ranking
        return total, {"regression": regression, "ranking": ranking}

else:  # pragma: no cover

    class RoleAwarePartialStateValueHead:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise PartialStateValueError("partial-state value prediction requires torch")

    class LayerwiseRoleAttentionValueHead:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise PartialStateValueError("partial-state value prediction requires torch")

    def role_aware_graph_summary(*_: Any, **__: Any) -> Any:
        raise PartialStateValueError("partial-state value prediction requires torch")

    def direct_value_loss(*_: Any, **__: Any) -> Any:
        raise PartialStateValueError("partial-state value prediction requires torch")


__all__ = [
    "PartialStateValueError",
    "LayerwiseRoleAttentionValueHead",
    "RoleAwarePartialStateValueHead",
    "direct_value_loss",
    "role_aware_graph_summary",
]
