"""Bounded complete-candidate graph corrections and multiplicity-preserving class loss.

The caller owns candidate support, frozen node features, and fixed node-color definitions.
This module owns no backbone, chemistry policy, candidate enumeration, or sampling logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

ROLE_COUNT = 2
MESSAGE_PASSING_ROUNDS = 2
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


class CandidateGraphResidualError(ValueError):
    """A candidate graph, frozen feature, or candidate-class grouping is malformed."""


def _finite_tensor(value: Tensor, label: str) -> None:
    if not isinstance(value, Tensor) or value.is_complex() or not bool(torch.isfinite(value).all()):
        raise CandidateGraphResidualError(f"{label} must be a finite real tensor")


def _positive_integer(value: int, label: str) -> None:
    if type(value) is not int or value < 1:
        raise CandidateGraphResidualError(f"{label} must be a positive integer")


class CandidateGraphResidual(nn.Module):
    """Score a complete candidate through two rounds of color-aware message passing.

    Projection order is hidden features, flow time, two role indicators, node colors, raw degree.
    Each message round computes ``ReLU(self_linear(h) + neighbor_linear(mean(h_neighbors)))``.
    Self transforms have biases; neighbor transforms do not. Masked mean pooling feeds a scalar
    linear projection initialized to exact zero. All supplied inputs are detached.

    ``use_adjacency=False`` retains the same parameters, initialization, node colors, and original
    candidate degrees, replacing adjacency by zero only for message passing. This control can
    learn local degree/context interactions but cannot distinguish equal-degree edge rewiring.
    Neighbor parameters have zero data gradients: parameter counts match, but effective feature
    use and effective trained capacity differ. This is an adjacency-message ablation.
    Construction and forward calls preserve the ambient random-number stream.
    """

    def __init__(
        self,
        hidden_dim: int,
        node_color_dim: int,
        width: int = 32,
        *,
        initialization_seed: int,
        use_adjacency: bool = True,
    ) -> None:
        super().__init__()
        for label, value in (
            ("hidden_dim", hidden_dim),
            ("node_color_dim", node_color_dim),
            ("width", width),
        ):
            _positive_integer(value, label)
        if type(initialization_seed) is not int or not 0 <= initialization_seed < 2**63:
            raise CandidateGraphResidualError("initialization_seed must lie in [0, 2**63)")
        if not isinstance(use_adjacency, bool):
            raise CandidateGraphResidualError("use_adjacency must be boolean")
        self.hidden_dim = hidden_dim
        self.node_color_dim = node_color_dim
        self.width = width
        self.initialization_seed = initialization_seed
        self.use_adjacency = use_adjacency
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Linear(
                hidden_dim + 1 + ROLE_COUNT + node_color_dim + 1,
                width,
                device="cpu",
                dtype=torch.float32,
            )
            self.self_updates = nn.ModuleList(
                nn.Linear(width, width, device="cpu", dtype=torch.float32)
                for _ in range(MESSAGE_PASSING_ROUNDS)
            )
            self.neighbor_updates = nn.ModuleList(
                nn.Linear(width, width, bias=False, device="cpu", dtype=torch.float32)
                for _ in range(MESSAGE_PASSING_ROUNDS)
            )
            self.output = nn.Linear(width, 1, device="cpu", dtype=torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        for layer in (self.projection, *self.self_updates, *self.neighbor_updates):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=generator)
            if layer.bias is not None:
                bound = 1 / math.sqrt(layer.in_features)
                nn.init.uniform_(layer.bias, -bound, bound, generator=generator)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        hidden: Tensor,
        adjacency: Tensor,
        node_colors: Tensor,
        node_mask: Tensor,
        flow_time: Tensor,
        role_index: Tensor,
    ) -> Tensor:
        """Return ``[B]`` corrections for finite simple undirected candidate graphs.

        Hidden features and node colors are floating tensors of shape ``[B,N,H]`` and ``[B,N,F]``.
        Adjacency is binary ``[B,N,N]`` with no self edges. Padded nodes and incident edges are
        ignored; each graph must retain at least one real node. Role indices are integers in [0,2).
        The constructor defaults to CPU float32; moving the module requires matching input devices
        and matching hidden/color dtypes. The original tensors are never modified.
        """

        _finite_tensor(hidden, "hidden")
        if (
            not hidden.is_floating_point()
            or hidden.ndim != 3
            or hidden.shape[-1] != self.hidden_dim
            or min(hidden.shape[:2]) < 1
        ):
            raise CandidateGraphResidualError("hidden must be floating [batch, nodes, hidden_dim]")
        batch, nodes, _ = hidden.shape
        _finite_tensor(node_colors, "node_colors")
        if not node_colors.is_floating_point() or node_colors.shape != (
            batch,
            nodes,
            self.node_color_dim,
        ):
            raise CandidateGraphResidualError(
                "node_colors must be floating [batch, nodes, node_color_dim]"
            )
        if (
            not isinstance(node_mask, Tensor)
            or node_mask.dtype != torch.bool
            or node_mask.shape != (batch, nodes)
        ):
            raise CandidateGraphResidualError("node_mask must be boolean [batch, nodes]")
        if not bool(node_mask.any(dim=1).all()):
            raise CandidateGraphResidualError("every candidate graph must contain a real node")
        _finite_tensor(adjacency, "adjacency")
        if adjacency.shape != (batch, nodes, nodes):
            raise CandidateGraphResidualError("adjacency must have shape [batch, nodes, nodes]")
        if not bool(((adjacency == 0) | (adjacency == 1)).all()):
            raise CandidateGraphResidualError("adjacency must be unweighted binary")
        if not torch.equal(adjacency, adjacency.transpose(1, 2)) or bool(
            adjacency.diagonal(dim1=1, dim2=2).any()
        ):
            raise CandidateGraphResidualError("adjacency must be symmetric without self edges")
        _finite_tensor(flow_time, "flow_time")
        if not flow_time.is_floating_point() or flow_time.shape != (batch,):
            raise CandidateGraphResidualError("flow_time must be floating [batch]")
        if not bool(((flow_time >= 0) & (flow_time <= 1)).all()):
            raise CandidateGraphResidualError("flow_time must lie in [0, 1]")
        if (
            not isinstance(role_index, Tensor)
            or role_index.dtype not in _INTEGER_DTYPES
            or role_index.shape != (batch,)
        ):
            raise CandidateGraphResidualError("role_index must be integer [batch]")
        if not bool(((role_index >= 0) & (role_index < ROLE_COUNT)).all()):
            raise CandidateGraphResidualError("role_index must lie in [0, 2)")
        if any(
            value.device != hidden.device
            for value in (
                adjacency,
                node_colors,
                node_mask,
                flow_time,
                role_index,
                self.projection.weight,
            )
        ):
            raise CandidateGraphResidualError("module and all inputs must use the same device")
        if hidden.dtype != self.projection.weight.dtype or node_colors.dtype != hidden.dtype:
            raise CandidateGraphResidualError(
                "hidden and node_colors dtype must match module dtype"
            )
        dtype = hidden.dtype
        role = functional.one_hot(role_index.detach().to(torch.long), num_classes=ROLE_COUNT).to(
            dtype
        )
        edges = adjacency.detach().to(dtype)
        edges = edges * (node_mask[:, :, None] & node_mask[:, None, :]).to(dtype)
        original_degree = edges.sum(-1, keepdim=True)
        features = torch.cat(
            (
                hidden.detach(),
                flow_time.detach().to(dtype)[:, None, None].expand(-1, nodes, -1),
                role[:, None, :].expand(-1, nodes, -1),
                node_colors.detach(),
                original_degree,
            ),
            dim=-1,
        )
        mask = node_mask[..., None]
        values = torch.where(mask, torch.relu(self.projection(features)), 0.0)
        if not self.use_adjacency:
            edges = torch.zeros_like(edges)
        degree = edges.sum(-1, keepdim=True).clamp_min(1.0)
        for self_update, neighbor_update in zip(
            self.self_updates, self.neighbor_updates, strict=True
        ):
            neighbors = torch.bmm(edges, values) / degree
            values = torch.where(
                mask, torch.relu(self_update(values) + neighbor_update(neighbors)), 0.0
            )
        pooled = values.sum(1) / node_mask.sum(1, keepdim=True).to(dtype)
        correction = self.output(pooled).squeeze(-1)
        _finite_tensor(correction, "candidate correction")
        return correction

    def trainable_parameter_receipt(self) -> dict[str, Any]:
        """Describe and authenticate the complete head-only trainable parameter boundary."""
        expected = {"projection.weight", "projection.bias", "output.weight", "output.bias"}
        for index in range(MESSAGE_PASSING_ROUNDS):
            expected.update(
                (
                    f"self_updates.{index}.weight",
                    f"self_updates.{index}.bias",
                    f"neighbor_updates.{index}.weight",
                )
            )
        parameters = dict(self.named_parameters())
        if set(parameters) != expected or not all(
            value.requires_grad for value in parameters.values()
        ):
            raise CandidateGraphResidualError(
                "trainable parameters violate the candidate-head boundary"
            )
        return {
            "schema_version": "forge.ugi_candidate_graph_residual_parameters.v1",
            "owns_backbone": False,
            "hidden_dim": self.hidden_dim,
            "node_color_dim": self.node_color_dim,
            "width": self.width,
            "message_passing_rounds": MESSAGE_PASSING_ROUNDS,
            "role_count": ROLE_COUNT,
            "initialization_seed": self.initialization_seed,
            "use_adjacency": self.use_adjacency,
            "candidate_degree_feature": "raw degree over unmasked nodes, retained in both arms",
            "trainable_parameter_allowlist": sorted(expected),
            "parameter_shapes": {name: list(value.shape) for name, value in parameters.items()},
            "trainable_parameter_count": sum(value.numel() for value in parameters.values()),
            "detached_inputs": ["hidden", "adjacency", "node_colors", "flow_time", "role_index"],
            "ablation_note": "Zero adjacency messages retain node colors, actual degrees and identical parameters; neighbor weights have zero data gradients and effective capacity differs.",
        }


@dataclass(frozen=True)
class CandidateClassLoss:
    """Differentiable loss plus explicit accounting for every original candidate group.

    ``loss`` is the mean over rows containing a positive candidate, including zero-loss singleton
    and all-positive rows. ``per_row_loss.mean()`` instead retains the mass of every original row,
    assigning unlearnable rows zero contribution. Callers must state which denominator they use.
    ``absent_target_rows`` excludes empty rows; their union is the unlearnable population.
    ``zero_signal_rows`` includes every row without both a positive and a negative candidate.
    """

    loss: Tensor
    per_row_loss: Tensor
    candidate_counts: Tensor
    positive_counts: Tensor
    empty_rows: Tensor
    absent_target_rows: Tensor
    singleton_rows: Tensor
    all_positive_rows: Tensor
    optimizable_rows: Tensor
    zero_signal_rows: Tensor

    @property
    def counts(self) -> dict[str, int]:
        """Materialize logging counts without detaching the differentiable losses."""
        return {
            "rows": self.per_row_loss.numel(),
            "candidates": int(self.candidate_counts.sum()),
            "positive_candidates": int(self.positive_counts.sum()),
            "empty_rows": int(self.empty_rows.sum()),
            "absent_target_rows": int(self.absent_target_rows.sum()),
            "singleton_rows": int(self.singleton_rows.sum()),
            "all_positive_rows": int(self.all_positive_rows.sum()),
            "optimizable_rows": int(self.optimizable_rows.sum()),
            "unlearnable_rows": int((~self.optimizable_rows).sum()),
            "zero_signal_rows": int(self.zero_signal_rows.sum()),
            "informative_rows": int((~self.zero_signal_rows).sum()),
        }


def _group_logsumexp(values: Tensor, row_indices: Tensor, rows: int) -> Tensor:
    """Stable grouped reduction with explicit zero for empty groups and no padded candidate axis."""
    maximum = values.new_full((rows,), -torch.inf)
    maximum.scatter_reduce_(0, row_indices, values.detach(), reduce="amax", include_self=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    sums = values.new_zeros(rows).scatter_add(
        0, row_indices, torch.exp(values - maximum[row_indices])
    )
    return torch.where(
        sums > 0, maximum + sums.clamp_min(torch.finfo(values.dtype).tiny).log(), 0.0
    )


def candidate_class_loss(
    scores: Tensor, candidate_offsets: Tensor, positive_mask: Tensor
) -> CandidateClassLoss:
    """Compute logsumexp(all) - logsumexp(positives) without deduplicating candidates.

    ``scores`` is flat floating ``[C]``; integer offsets ``[B+1]`` start at zero, end at C and are
    nondecreasing. Every candidate occupies one original slot, including duplicate positive or
    negative representatives. Empty/no-positive groups are reported and contribute zero to
    ``per_row_loss``. Singleton-positive and all-positive groups have exactly zero loss and gradient.
    Inputs may be CPU or GPU; reductions are vectorized and use no dense candidate padding.
    """
    _finite_tensor(scores, "scores")
    if scores.ndim != 1 or not scores.is_floating_point():
        raise CandidateGraphResidualError("scores must be a floating vector")
    if (
        not isinstance(candidate_offsets, Tensor)
        or candidate_offsets.dtype not in _INTEGER_DTYPES
        or candidate_offsets.ndim != 1
        or candidate_offsets.numel() < 2
    ):
        raise CandidateGraphResidualError(
            "candidate_offsets must be an integer vector of length >= 2"
        )
    if (
        not isinstance(positive_mask, Tensor)
        or positive_mask.dtype != torch.bool
        or positive_mask.shape != scores.shape
    ):
        raise CandidateGraphResidualError("positive_mask must be boolean with the scores shape")
    if candidate_offsets.device != scores.device or positive_mask.device != scores.device:
        raise CandidateGraphResidualError("scores, offsets and positive mask must share a device")
    offsets = candidate_offsets.to(torch.long)
    if (
        int(offsets[0]) != 0
        or int(offsets[-1]) != len(scores)
        or not bool((offsets[1:] >= offsets[:-1]).all())
    ):
        raise CandidateGraphResidualError("offsets must be nondecreasing and span scores exactly")
    lengths = offsets[1:] - offsets[:-1]
    rows = len(lengths)
    row_indices = torch.repeat_interleave(torch.arange(rows, device=scores.device), lengths)
    positives = torch.zeros_like(lengths).scatter_add(0, row_indices, positive_mask.to(torch.long))
    empty = lengths == 0
    optimizable = positives > 0
    all_positive = optimizable & (positives == lengths)
    informative = optimizable & ~all_positive
    all_scores = _group_logsumexp(scores, row_indices, rows)
    positive_scores = _group_logsumexp(scores[positive_mask], row_indices[positive_mask], rows)
    per_row = torch.where(informative, all_scores - positive_scores, 0.0) + scores.mul(0.0).sum()
    loss = per_row.sum() / optimizable.sum().clamp_min(1).to(scores.dtype)
    _finite_tensor(per_row, "per-row candidate loss")
    _finite_tensor(loss, "candidate loss")
    return CandidateClassLoss(
        loss=loss,
        per_row_loss=per_row,
        candidate_counts=lengths,
        positive_counts=positives,
        empty_rows=empty,
        absent_target_rows=~empty & ~optimizable,
        singleton_rows=lengths == 1,
        all_positive_rows=all_positive,
        optimizable_rows=optimizable,
        zero_signal_rows=~informative,
    )
