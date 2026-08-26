"""Reaction-program cross-attentive graph Transformer for vocabulary-free product flow."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

from forge.model.phase1_flow import SparseWholeLipidFlow, _gather_training_nodes
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.reaction_program_flow import synthesis_program_flow_loss

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]


class ReactionProgramTransformerError(ValueError):
    """Transformer inputs or objective violate the declared semantic contract."""


if nn is not None:

    class _MaskedMultiheadAttention(nn.Module):
        def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
            super().__init__()
            if hidden_dim % heads:
                raise ReactionProgramTransformerError("hidden_dim must be divisible by heads")
            self.heads = heads
            self.head_dim = hidden_dim // heads
            self.query = nn.Linear(hidden_dim, hidden_dim)
            self.key = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            query: Any,
            memory: Any,
            *,
            query_mask: Any,
            memory_mask: Any,
            attention_bias: Any | None = None,
        ) -> Any:
            batch, queries, hidden_dim = query.shape
            keys = memory.shape[1]
            q = self.query(query).reshape(batch, queries, self.heads, self.head_dim).transpose(1, 2)
            k = self.key(memory).reshape(batch, keys, self.heads, self.head_dim).transpose(1, 2)
            v = self.value(memory).reshape(batch, keys, self.heads, self.head_dim).transpose(1, 2)
            scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(self.head_dim)
            if attention_bias is not None:
                if attention_bias.shape != scores.shape:
                    raise ReactionProgramTransformerError("attention bias shape is inconsistent")
                scores = scores + attention_bias
            scores = scores.masked_fill(~memory_mask[:, None, None, :], -1e9)
            probabilities = self.dropout(torch.softmax(scores, dim=-1))
            context = torch.einsum("bhqk,bhkd->bhqd", probabilities, v)
            context = context.transpose(1, 2).reshape(batch, queries, hidden_dim)
            return self.output(context) * query_mask[:, :, None]

    class ReactionProgramTokenEncoder(nn.Module):
        """Encode global and atom-aligned reaction-program semantics as reusable memory tokens."""

        def __init__(
            self,
            vocabulary: ReactionProgramVocabulary,
            hidden_dim: int,
            heads: int,
            dropout: float,
            maximum_heavy_atoms: int,
            repeat_group_conditioning: bool = False,
        ) -> None:
            super().__init__()
            self.vocabulary = vocabulary
            self.program = nn.Embedding(len(vocabulary.program_states), hidden_dim)
            self.role = nn.Embedding(len(vocabulary.role_states), hidden_dim)
            self.core = nn.Embedding(len(vocabulary.core_position_states), hidden_dim)
            self.depth = nn.Embedding(vocabulary.maximum_steps + 1, hidden_dim)
            self.token_type = nn.Embedding(3, hidden_dim)
            self.position = nn.Embedding(maximum_heavy_atoms, hidden_dim)
            self.repeat_group_conditioning = repeat_group_conditioning
            self.repeat_group = (
                nn.Embedding(len(vocabulary.role_states), hidden_dim)
                if repeat_group_conditioning
                else None
            )
            self.component_position = (
                nn.Embedding(maximum_heavy_atoms + 1, hidden_dim)
                if repeat_group_conditioning
                else None
            )
            self.self_attention = _MaskedMultiheadAttention(hidden_dim, heads, dropout)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
            self.norm2 = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            *,
            program_states: Any,
            role_states: Any,
            core_position_states: Any,
            program_depths: Any,
            adapter_mask: Any,
            role_isolated_attention: bool = False,
            repeat_group_states: Any | None = None,
            component_position_states: Any | None = None,
        ) -> tuple[Any, Any, Any]:
            batch, nodes = role_states.shape
            if (
                program_states.shape != (batch,)
                or core_position_states.shape != (batch, nodes)
                or program_depths.shape != (batch,)
                or adapter_mask.shape != (batch, nodes)
                or adapter_mask.dtype != torch.bool
            ):
                raise ReactionProgramTransformerError("reaction-program token shapes disagree")
            program_token = self.program(program_states) + self.token_type.weight[0]
            depth_token = self.depth(program_depths) + self.token_type.weight[1]
            positions = torch.arange(nodes, device=role_states.device)
            node_tokens = (
                self.role(role_states)
                + self.core(core_position_states)
                + self.position(positions)[None]
                + self.token_type.weight[2]
            )
            if self.repeat_group_conditioning:
                if (
                    repeat_group_states is None
                    or component_position_states is None
                    or repeat_group_states.shape != (batch, nodes)
                    or component_position_states.shape != (batch, nodes)
                ):
                    raise ReactionProgramTransformerError(
                        "repeat-aware reaction-program token shapes disagree"
                    )
                if self.repeat_group is None or self.component_position is None:
                    raise ReactionProgramTransformerError(
                        "repeat-aware embeddings are unexpectedly absent"
                    )
                node_tokens = (
                    node_tokens
                    + self.repeat_group(repeat_group_states)
                    + self.component_position(component_position_states)
                )
            tokens = torch.cat((program_token[:, None], depth_token[:, None], node_tokens), dim=1)
            token_mask = torch.cat(
                (
                    torch.ones((batch, 2), dtype=torch.bool, device=adapter_mask.device),
                    adapter_mask,
                ),
                dim=1,
            )
            normalized = self.norm1(tokens)
            attention_bias = None
            if role_isolated_attention:
                # Global program/depth tokens may not pool role-local state: otherwise they become
                # a hidden B/C -> A communication channel in the factorized control.  A role token
                # may read the shared globals and tokens from its own role only.
                token_roles = torch.cat(
                    (
                        role_states.new_full((batch, 2), -1),
                        role_states,
                    ),
                    dim=1,
                )
                query_roles = token_roles[:, :, None]
                memory_roles = token_roles[:, None, :]
                allowed = torch.where(
                    query_roles < 0,
                    memory_roles < 0,
                    (memory_roles < 0) | (memory_roles == query_roles),
                )
                attention_bias = torch.where(
                    allowed[:, None],
                    tokens.new_zeros(()),
                    tokens.new_full((), -1e9),
                ).expand(-1, self.self_attention.heads, -1, -1)
            attended = self.self_attention(
                normalized,
                normalized,
                query_mask=token_mask,
                memory_mask=token_mask,
                attention_bias=attention_bias,
            )
            tokens = tokens + attended
            tokens = tokens + self.ffn(self.norm2(tokens)) * token_mask[:, :, None]
            return tokens, token_mask, tokens[:, 0]

    class _RoutedAdapter(nn.Module):
        def __init__(
            self, hidden_dim: int, expert_count: int, adapter_dim: int, dropout: float
        ) -> None:
            super().__init__()
            if expert_count < 2 or adapter_dim < 1:
                raise ReactionProgramTransformerError("routed adapter support is invalid")
            self.experts = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(hidden_dim, adapter_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(adapter_dim, hidden_dim),
                )
                for _ in range(expert_count)
            )
            self.gate = nn.Linear(hidden_dim, expert_count)

        def forward(self, hidden: Any, program_summary: Any) -> tuple[Any, Any]:
            weights = torch.softmax(self.gate(program_summary), dim=-1)
            expert_values = torch.stack([expert(hidden) for expert in self.experts], dim=2)
            update = torch.einsum("be,bned->bnd", weights, expert_values)
            return update, weights

    class _SharedAdapter(nn.Module):
        """One residual adapter used by the no-routing mechanism control."""

        def __init__(self, hidden_dim: int, adapter_dim: int, dropout: float) -> None:
            super().__init__()
            self.adapter = nn.Sequential(
                nn.Linear(hidden_dim, adapter_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(adapter_dim, hidden_dim),
            )

        def forward(self, hidden: Any, program_summary: Any) -> tuple[Any, Any]:
            weights = program_summary.new_ones((program_summary.shape[0], 1))
            return self.adapter(hidden), weights

    class ReactionProgramTransformerBlock(nn.Module):
        """Graph self-attention, program cross-attention and program-routed adaptation."""

        def __init__(
            self,
            *,
            hidden_dim: int,
            heads: int,
            expert_count: int,
            adapter_dim: int,
            dropout: float,
            program_cross_attention: bool,
            routed_adapters: bool,
            role_isolated_attention: bool,
        ) -> None:
            super().__init__()
            self.self_norm = nn.LayerNorm(hidden_dim)
            self.self_attention = _MaskedMultiheadAttention(hidden_dim, heads, dropout)
            self.cross_norm = nn.LayerNorm(hidden_dim)
            self.cross_attention = _MaskedMultiheadAttention(hidden_dim, heads, dropout)
            self.ffn_norm = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
            self.program_cross_attention = program_cross_attention
            self.role_isolated_attention = role_isolated_attention
            self.routed_adapter = (
                _RoutedAdapter(hidden_dim, expert_count, adapter_dim, dropout)
                if routed_adapters
                else _SharedAdapter(hidden_dim, adapter_dim, dropout)
            )
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            hidden: Any,
            *,
            node_mask: Any,
            program_tokens: Any,
            program_mask: Any,
            program_summary: Any,
            graph_bias: Any,
            role_states: Any,
        ) -> tuple[Any, Any]:
            normalized = self.self_norm(hidden)
            if self.role_isolated_attention:
                # Role zero is adapter-owned/unassigned state.  It remains isolated as well: a
                # shared assembly coordinate must not become a hidden cross-role message bus.
                same_role = role_states[:, :, None] == role_states[:, None, :]
                role_bias = torch.where(
                    same_role[:, None],
                    graph_bias.new_zeros(()),
                    graph_bias.new_full((), -1e9),
                )
                graph_bias = graph_bias + role_bias
            hidden = hidden + self.dropout(
                self.self_attention(
                    normalized,
                    normalized,
                    query_mask=node_mask,
                    memory_mask=node_mask,
                    attention_bias=graph_bias,
                )
            )
            if self.program_cross_attention:
                cross_bias = None
                if self.role_isolated_attention:
                    batch, nodes = role_states.shape
                    token_roles = torch.cat(
                        (
                            role_states.new_full((batch, 2), -1),
                            role_states,
                        ),
                        dim=1,
                    )
                    allowed = (token_roles[:, None, :] < 0) | (
                        token_roles[:, None, :] == role_states[:, :, None]
                    )
                    cross_bias = torch.where(
                        allowed[:, None],
                        hidden.new_zeros(()),
                        hidden.new_full((), -1e9),
                    ).expand(-1, self.cross_attention.heads, -1, -1)
                hidden = hidden + self.dropout(
                    self.cross_attention(
                        self.cross_norm(hidden),
                        program_tokens,
                        query_mask=node_mask,
                        memory_mask=program_mask,
                        attention_bias=cross_bias,
                    )
                )
            normalized = self.ffn_norm(hidden)
            routed, weights = self.routed_adapter(normalized, program_summary)
            hidden = hidden + self.dropout(self.ffn(normalized) + routed)
            return hidden * node_mask[:, :, None], weights

    class ReactionProgramGraphTransformer(nn.Module):
        """Sparse whole-product flow conditioned on reaction-program memory at every layer."""

        def __init__(
            self,
            *,
            vocabulary: ReactionProgramVocabulary,
            node_classes: int,
            hidden_dim: int,
            layers: int,
            heads: int,
            expert_count: int,
            adapter_dim: int,
            maximum_closures: int,
            maximum_heavy_atoms: int,
            dropout: float,
            bond_classes: int = 4,
            layerwise_program_cross_attention: bool = True,
            routed_adapters: bool = True,
            role_isolated_attention: bool = False,
            role_specific_parameters: bool = False,
            repeat_group_conditioning: bool = False,
        ) -> None:
            super().__init__()
            if layers < 1:
                raise ReactionProgramTransformerError("Transformer requires at least one layer")
            self.vocabulary = vocabulary
            self.hidden_dim = hidden_dim
            self.heads = heads
            self.maximum_heavy_atoms = maximum_heavy_atoms
            self.maximum_closures = maximum_closures
            self.layers = layers
            self.layerwise_program_cross_attention = layerwise_program_cross_attention
            self.routed_adapters = routed_adapters
            self.role_isolated_attention = role_isolated_attention
            self.role_specific_parameters = role_specific_parameters
            self.repeat_group_conditioning = repeat_group_conditioning
            if role_specific_parameters and not role_isolated_attention:
                raise ReactionProgramTransformerError(
                    "role-specific parameters require role-isolated attention"
                )
            self.program_encoder = ReactionProgramTokenEncoder(
                vocabulary,
                hidden_dim,
                heads,
                dropout,
                maximum_heavy_atoms,
                repeat_group_conditioning=repeat_group_conditioning,
            )
            # Reuse the qualified sparse state embeddings and output parameterization, but not its
            # message-passing blocks.
            self.state = SparseWholeLipidFlow(
                node_classes=node_classes,
                hidden_dim=hidden_dim,
                layers=0,
                maximum_closures=maximum_closures,
                maximum_heavy_atoms=maximum_heavy_atoms,
                dropout=dropout,
                bond_classes=bond_classes,
                use_position_embedding=True,
            )
            self.relation_bias = nn.Embedding(5, heads)
            self.blocks = (
                nn.ModuleList(
                    ReactionProgramTransformerBlock(
                        hidden_dim=hidden_dim,
                        heads=heads,
                        expert_count=expert_count,
                        adapter_dim=adapter_dim,
                        dropout=dropout,
                        # The input-only control receives the program once, in the first block.
                        program_cross_attention=(layerwise_program_cross_attention or index == 0),
                        routed_adapters=routed_adapters,
                        role_isolated_attention=role_isolated_attention,
                    )
                    for index in range(layers)
                )
                if not role_specific_parameters
                else nn.ModuleList()
            )
            self.role_blocks = (
                nn.ModuleList(
                    nn.ModuleList(
                        ReactionProgramTransformerBlock(
                            hidden_dim=hidden_dim,
                            heads=heads,
                            expert_count=expert_count,
                            adapter_dim=adapter_dim,
                            dropout=dropout,
                            program_cross_attention=(
                                layerwise_program_cross_attention or index == 0
                            ),
                            routed_adapters=routed_adapters,
                            role_isolated_attention=True,
                        )
                        for index in range(layers)
                    )
                    for _ in vocabulary.role_states
                )
                if role_specific_parameters
                else None
            )
            self.role_output = nn.Linear(hidden_dim, len(vocabulary.role_states))
            self.core_output = nn.Linear(hidden_dim, len(vocabulary.core_position_states))

        def _graph_bias(
            self,
            parents: Any,
            closure_left: Any,
            closure_right: Any,
            child_mask: Any,
            closure_mask: Any,
        ) -> Any:
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
            return self.relation_bias(relations).permute(0, 3, 1, 2)

        def forward(
            self,
            *,
            nodes: Any,
            parents: Any,
            parent_bonds: Any,
            closure_left: Any,
            closure_right: Any,
            closure_bonds: Any,
            t: Any,
            node_mask: Any,
            child_mask: Any,
            closure_mask: Any,
            program_states: Any,
            role_states: Any,
            core_position_states: Any,
            program_depths: Any,
            adapter_mask: Any,
            repeat_group_states: Any | None = None,
            component_position_states: Any | None = None,
            component_instance_states: Any | None = None,
        ) -> dict[str, Any]:
            del component_instance_states  # Loss-only coordinate; never a learned identity token.
            program_tokens, program_mask, program_summary = self.program_encoder(
                program_states=program_states,
                role_states=role_states,
                core_position_states=core_position_states,
                program_depths=program_depths,
                adapter_mask=adapter_mask,
                role_isolated_attention=self.role_isolated_attention,
                repeat_group_states=repeat_group_states,
                component_position_states=component_position_states,
            )
            hidden = (
                self.state.node_embedding(nodes) + self.state.time_embedding(t[:, None])[:, None]
            )
            positions = torch.arange(nodes.shape[1], device=nodes.device)
            position_embedding = self.state.position_embedding
            if position_embedding is None:
                raise ReactionProgramTransformerError("position embedding is unexpectedly absent")
            hidden = hidden + position_embedding(positions)[None]
            hidden = hidden + self.state.bond_embedding(parent_bonds) * child_mask[:, :, None]
            hidden = hidden * node_mask[:, :, None]
            graph_bias = self._graph_bias(
                parents, closure_left, closure_right, child_mask, closure_mask
            )
            expert_weights = []
            if self.role_blocks is None:
                for block in self.blocks:
                    hidden, weights = block(
                        hidden,
                        node_mask=node_mask,
                        program_tokens=program_tokens,
                        program_mask=program_mask,
                        program_summary=program_summary,
                        graph_bias=graph_bias,
                        role_states=role_states,
                    )
                    expert_weights.append(weights)
            else:
                # Each semantic role has its own full-depth denoiser. Roles are scattered back only
                # onto their own nodes, so no role-specific parameters can alter another role.
                for layer_index in range(self.layers):
                    layer_weights = []
                    for role_index, role_stack in enumerate(self.role_blocks):
                        role_stack = cast(Any, role_stack)
                        role_mask = node_mask & (role_states == role_index)
                        candidate, weights = role_stack[layer_index](
                            hidden,
                            node_mask=role_mask,
                            program_tokens=program_tokens,
                            program_mask=program_mask,
                            program_summary=program_summary,
                            graph_bias=graph_bias,
                            role_states=role_states,
                        )
                        hidden = torch.where(role_mask[:, :, None], candidate, hidden)
                        layer_weights.append(weights)
                    expert_weights.append(torch.stack(layer_weights, dim=1).mean(dim=1))

            parent_logits = torch.einsum(
                "bid,bjd->bij", self.state.parent_query(hidden), self.state.parent_key(hidden)
            ) / math.sqrt(self.hidden_dim)
            if self.role_isolated_attention:
                same_role = role_states[:, :, None] == role_states[:, None, :]
                # Independent precursor regions are joined only at declared reaction-core atoms.
                # Exterior cross-role parent choices remain forbidden.
                core_attachment = (core_position_states[:, :, None] > 1) & (
                    core_position_states[:, None, :] > 1
                )
                parent_logits = parent_logits.masked_fill(~(same_role | core_attachment), -1e9)
            parent_hidden = _gather_training_nodes(hidden, parents)
            parent_bond_logits = self.state.backbone_bond_output(
                torch.cat((hidden, parent_hidden), dim=-1)
            )
            batch, maximum_closures = closure_left.shape
            slots = self.state.closure_slots(torch.arange(maximum_closures, device=nodes.device))[
                None
            ].expand(batch, -1, -1)
            global_hidden = (hidden * node_mask[:, :, None]).sum(dim=1)
            global_hidden /= node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            left_hidden = _gather_training_nodes(hidden, closure_left)
            right_hidden = _gather_training_nodes(hidden, closure_right)
            closure_bond_hidden = self.state.bond_embedding(closure_bonds)
            closure_bond_hidden *= closure_mask[:, :, None]
            closure_hidden = self.state.closure_update(
                torch.cat(
                    (
                        slots,
                        global_hidden[:, None].expand_as(slots),
                        left_hidden,
                        right_hidden,
                        closure_bond_hidden,
                    ),
                    dim=-1,
                )
            )
            closure_key = self.state.closure_node_key(hidden)
            left_logits = torch.einsum(
                "bkd,bnd->bkn", self.state.closure_left_query(closure_hidden), closure_key
            ) / math.sqrt(self.hidden_dim)
            right_logits = torch.einsum(
                "bkd,bnd->bkn", self.state.closure_right_query(closure_hidden), closure_key
            ) / math.sqrt(self.hidden_dim)
            return {
                "nodes": self.state.node_output(hidden),
                "parents": parent_logits,
                "parent_bonds": parent_bond_logits,
                "closure_left": left_logits,
                "closure_right": right_logits,
                "closure_bonds": self.state.closure_bond_output(closure_hidden),
                "node_count": self.state.node_count_logits[None].expand(batch, -1),
                "closure_count": self.state.closure_count_logits[None].expand(batch, -1),
                "role_states": self.role_output(hidden),
                "core_position_states": self.core_output(hidden),
                "expert_weights": torch.stack(expert_weights, dim=1),
            }

else:  # pragma: no cover

    class ReactionProgramGraphTransformer:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise ReactionProgramTransformerError("graph Transformer requires torch")


def _balanced_state_cross_entropy(logits: Any, targets: Any, mask: Any) -> Any:
    """Average semantic classification loss across present states, not atom frequency."""

    selected_targets = targets[mask]
    point_losses = functional.cross_entropy(logits[mask], selected_targets, reduction="none")
    classes = logits.shape[-1]
    state_sums = logits.new_zeros(classes).scatter_add(0, selected_targets, point_losses)
    state_counts = torch.bincount(selected_targets, minlength=classes)
    present = state_counts > 0
    state_means = state_sums / state_counts.clamp(min=1)
    return (state_means * present).sum() / present.sum().clamp(min=1)


def _repeat_component_consistency(
    predictions: Mapping[str, Any], clean: Mapping[str, Any]
) -> tuple[Any, Any]:
    """Align matched positions across repeated components without assigning step identities."""

    repeat_groups = clean.get("repeat_group_states")
    component_positions = clean.get("component_position_states")
    component_instances = clean.get("component_instance_states")
    if repeat_groups is None or component_positions is None or component_instances is None:
        zero = predictions["nodes"].new_zeros(())
        return zero, zero.to(dtype=torch.int64)
    active = (repeat_groups > 0) & clean["node_mask"] & (clean["core_position_states"] == 1)
    pair_mask = (
        active[:, :, None]
        & active[:, None, :]
        & (repeat_groups[:, :, None] == repeat_groups[:, None, :])
        & (component_positions[:, :, None] == component_positions[:, None, :])
        & (component_instances[:, :, None] != component_instances[:, None, :])
    )
    nodes = repeat_groups.shape[1]
    upper = torch.triu(
        torch.ones((nodes, nodes), dtype=torch.bool, device=repeat_groups.device),
        diagonal=1,
    )
    pair_mask &= upper[None]
    node_probabilities = torch.softmax(predictions["nodes"], dim=-1)
    node_distance = (
        (node_probabilities[:, :, None, :] - node_probabilities[:, None, :, :])
        .square()
        .mean(dim=-1)
    )
    pair_count = pair_mask.sum()
    node_loss = (node_distance * pair_mask).sum() / pair_count.clamp(min=1)

    bond_pair_mask = pair_mask & clean["child_mask"][:, :, None] & clean["child_mask"][:, None, :]
    bond_probabilities = torch.softmax(predictions["parent_bonds"], dim=-1)
    bond_distance = (
        (bond_probabilities[:, :, None, :] - bond_probabilities[:, None, :, :])
        .square()
        .mean(dim=-1)
    )
    bond_count = bond_pair_mask.sum()
    bond_loss = (bond_distance * bond_pair_mask).sum() / bond_count.clamp(min=1)
    return node_loss + bond_loss, pair_count


def reaction_program_transformer_loss(
    predictions: Mapping[str, Any],
    clean: Mapping[str, Any],
    *,
    role_weight: float,
    core_weight: float,
    repeat_consistency_weight: float = 0.0,
    materialize_metrics: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Combine graph flow with state-balanced precursor-role and reaction-core consistency."""

    if role_weight < 0.0 or core_weight < 0.0 or repeat_consistency_weight < 0.0:
        raise ReactionProgramTransformerError("semantic loss weights must be nonnegative")
    base, metrics = synthesis_program_flow_loss(
        predictions, clean, materialize_metrics=materialize_metrics
    )
    role = _balanced_state_cross_entropy(
        predictions["role_states"], clean["role_states"], clean["node_mask"]
    )
    core = _balanced_state_cross_entropy(
        predictions["core_position_states"],
        clean["core_position_states"],
        clean["node_mask"],
    )
    repeat_consistency, repeat_pairs = _repeat_component_consistency(predictions, clean)
    total = (
        base
        + role_weight * role
        + core_weight * core
        + repeat_consistency_weight * repeat_consistency
    )
    semantic_metrics = {
        "role_consistency_ce": role.detach(),
        "core_consistency_ce": core.detach(),
        "repeat_consistency_mse": repeat_consistency.detach(),
        "repeat_consistency_pairs": repeat_pairs.detach(),
        "semantic_total": total.detach(),
    }
    if materialize_metrics:
        semantic_metrics = {key: float(value) for key, value in semantic_metrics.items()}
    return total, {**metrics, **semantic_metrics}


def _slice_batch(values: Mapping[str, Any], indices: Any) -> dict[str, Any]:
    batch = int(indices.shape[0])
    output: dict[str, Any] = {}
    for key, value in values.items():
        if hasattr(value, "shape") and value.ndim > 0 and value.shape[0] == batch:
            output[key] = value[indices]
        else:
            output[key] = value
    return output


def per_program_transformer_losses(
    predictions: Mapping[str, Any],
    clean: Mapping[str, Any],
    *,
    role_weight: float,
    core_weight: float,
    repeat_consistency_weight: float = 0.0,
    program_states: tuple[int, ...] | None = None,
    materialize_metrics: bool = True,
) -> tuple[dict[int, Any], dict[str, Any]]:
    """Return one equally weighted objective per source reaction program."""

    source_programs = clean.get("source_program_states")
    if source_programs is None:
        raise ReactionProgramTransformerError(
            "source program identities are required for balancing"
        )
    losses: dict[int, Any] = {}
    metrics: dict[str, Any] = {}
    observed_programs = (
        tuple(int(value) for value in torch.unique(source_programs, sorted=True).tolist())
        if program_states is None
        else program_states
    )
    for program_state in observed_programs:
        indices = source_programs == int(program_state)
        loss, values = reaction_program_transformer_loss(
            _slice_batch(predictions, indices),
            _slice_batch(clean, indices),
            role_weight=role_weight,
            core_weight=core_weight,
            repeat_consistency_weight=repeat_consistency_weight,
            materialize_metrics=materialize_metrics,
        )
        losses[int(program_state)] = loss
        for key, value in values.items():
            metrics[f"program_{int(program_state)}_{key}"] = value
    return losses, metrics


def balanced_pcgrad_backward(
    losses: Mapping[int, Any],
    model: Any,
    *,
    scale: float = 1.0,
    materialize_diagnostics: bool = True,
) -> dict[str, Any]:
    """Give families equal loss mass and deterministically project conflicting gradients.

    Families are balanced by stratified sampling plus an equal-weight gradient mean.  Deliberately
    do not normalize each gradient to a common norm: that operation amplifies numerical noise from
    an already converged family and can destabilize the remaining objectives.
    """

    if len(losses) < 2 or scale <= 0.0:
        raise ReactionProgramTransformerError(
            "PCGrad requires at least two programs and positive scale"
        )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    ordered = sorted(losses)
    raw = [
        torch.autograd.grad(
            losses[program], parameters, retain_graph=index + 1 < len(ordered), allow_unused=True
        )
        for index, program in enumerate(ordered)
    ]
    availability = [[gradient is not None for gradient in gradients] for gradients in raw]
    flat_gradients = torch.stack(
        [
            torch.cat(
                [
                    (gradient if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
                    for gradient, parameter in zip(gradients, parameters, strict=True)
                ]
            )
            for gradients in raw
        ]
    )
    del raw
    norms = torch.linalg.vector_norm(flat_gradients, dim=1)
    conflicts = torch.zeros((), dtype=torch.int64, device=flat_gradients.device)
    pairwise_dots = flat_gradients @ flat_gradients.transpose(0, 1)
    off_diagonal = ~torch.eye(len(ordered), dtype=torch.bool, device=flat_gradients.device)
    if not bool(((pairwise_dots < 0.0) & off_diagonal).any()):
        combined = flat_gradients.mean(dim=0)
    else:
        scalar_availability = torch.as_tensor(
            availability, dtype=torch.bool, device=flat_gradients.device
        ).repeat_interleave(
            torch.as_tensor(
                [parameter.numel() for parameter in parameters], device=flat_gradients.device
            ),
            dim=1,
        )
        denominators = pairwise_dots.diagonal().clamp(min=1e-12)
        combined = torch.zeros_like(flat_gradients[0])
        for left_index, gradients in enumerate(flat_gradients):
            current = gradients.clone()
            for right_index, reference in enumerate(flat_gradients):
                if left_index == right_index:
                    continue
                dot = torch.dot(current, reference)
                negative = dot < 0.0
                conflicts.add_(negative)
                coefficient = torch.where(
                    negative,
                    dot / denominators[right_index],
                    dot.new_zeros(()),
                )
                current = current - coefficient * reference * (
                    scalar_availability[left_index] & scalar_availability[right_index]
                )
            combined.add_(current, alpha=1.0 / len(ordered))
    combined.mul_(scale).detach_()
    additions: list[Any] = []
    existing: list[Any] = []
    offset = 0
    for parameter_index, parameter in enumerate(parameters):
        elements = parameter.numel()
        update = combined[offset : offset + elements].view_as(parameter)
        offset += elements
        if not any(row[parameter_index] for row in availability):
            continue
        if parameter.grad is None:
            parameter.grad = update
        else:
            existing.append(parameter.grad)
            additions.append(update)
    if existing:
        torch._foreach_add_(existing, additions)
    if materialize_diagnostics:
        values = torch.cat((norms, conflicts.to(norms.dtype)[None])).detach().cpu().tolist()
        raw_gradient_norms: Any = values[:-1]
        projected_conflicts: Any = int(values[-1])
    else:
        raw_gradient_norms = norms.detach()
        projected_conflicts = conflicts.detach()
    return {
        "program_states": ordered,
        "raw_gradient_norms": raw_gradient_norms,
        "family_weighting": "equal_loss_mass_without_norm_amplification",
        "projected_conflicts": projected_conflicts,
    }


__all__ = [
    "ReactionProgramGraphTransformer",
    "ReactionProgramTransformerError",
    "balanced_pcgrad_backward",
    "per_program_transformer_losses",
    "reaction_program_transformer_loss",
]
