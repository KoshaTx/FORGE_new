"""Conditional discrete flow over lipid regions on a settled rooted tree."""

from __future__ import annotations

import time
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from forge.generate.sampling import rstar_step as _rstar_step

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    functional = None


class TreeRegionFlowError(RuntimeError):
    """Raised when conditional lipid-region generation violates its contract."""


@dataclass(frozen=True)
class TreeRegionSample:
    """One rooted tree with generated structural regions."""

    regions: np.ndarray
    parents: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.regions.size)


def _rooted_depths(parents: Any, node_mask: Any) -> Any:
    depths = torch.zeros_like(parents)
    for child in range(1, parents.shape[1]):
        active = node_mask[:, child]
        if active.any():
            parent = parents[:, child].clamp(min=0, max=child - 1)
            depths[:, child] = (
                depths.gather(
                    1,
                    parent[:, None],
                ).squeeze(1)
                + 1
            )
    return depths * node_mask


def collate_region_records(
    records: Sequence[Any],
    *,
    maximum_nodes: int,
) -> dict[str, Any]:
    """Collate clean regions and their settled BFS parent trees."""

    if torch is None:
        raise TreeRegionFlowError("region collation requires torch")
    if not records or maximum_nodes < max(record.node_count for record in records):
        raise TreeRegionFlowError("invalid region collation capacity")
    regions = torch.zeros((len(records), maximum_nodes), dtype=torch.long)
    parents = torch.zeros_like(regions)
    node_mask = torch.zeros((len(records), maximum_nodes), dtype=torch.bool)
    for index, record in enumerate(records):
        if record.region_states is None:
            raise TreeRegionFlowError("region record lacks supervision")
        regions[index, : record.node_count] = torch.from_numpy(record.region_states.copy())
        parents[index, : record.node_count] = torch.from_numpy(record.parents.copy())
        node_mask[index, : record.node_count] = True
    child_mask = node_mask.clone()
    child_mask[:, 0] = False
    return {
        "regions": regions,
        "parents": parents,
        "node_mask": node_mask,
        "child_mask": child_mask,
    }


def region_depth_marginal(
    records: Sequence[Any],
    *,
    depth_buckets: int,
    region_classes: int,
    probability_floor: float = 1e-6,
) -> np.ndarray:
    """Estimate region source probabilities by settled root distance."""

    if depth_buckets < 2 or region_classes < 2 or not 0 < probability_floor < 1:
        raise TreeRegionFlowError("invalid region-depth support")
    counts = np.full(
        (depth_buckets, region_classes),
        probability_floor,
        dtype=np.float64,
    )
    for record in records:
        depths = np.zeros(record.node_count, dtype=np.int64)
        for child in range(1, record.node_count):
            depths[child] = depths[int(record.parents[child])] + 1
        buckets = np.minimum(depths, depth_buckets - 1)
        np.add.at(counts, (buckets, record.region_states), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def region_transition_marginal(
    records: Sequence[Any],
    *,
    region_classes: int,
    probability_floor: float = 1e-6,
) -> np.ndarray:
    """Estimate child-region probabilities conditional on parent region."""

    if region_classes < 2 or not 0 < probability_floor < 1:
        raise TreeRegionFlowError("invalid region transition support")
    counts = np.full(
        (region_classes, region_classes),
        probability_floor,
        dtype=np.float64,
    )
    for record in records:
        for child in range(1, record.node_count):
            parent_region = int(record.region_states[int(record.parents[child])])
            child_region = int(record.region_states[child])
            counts[parent_region, child_region] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def region_source_probabilities(
    parents: Any,
    node_mask: Any,
    depth_marginal: Any,
) -> Any:
    """Expand a depth-conditioned region source over atom slots."""

    depths = _rooted_depths(parents, node_mask).clamp(max=depth_marginal.shape[0] - 1)
    probabilities = depth_marginal[depths]
    probabilities = probabilities * node_mask[:, :, None]
    return probabilities


def _tree_degrees(parents: Any, node_mask: Any, child_mask: Any) -> Any:
    node_count = parents.shape[1]
    assignment = functional.one_hot(
        parents,
        num_classes=node_count,
    ).to(torch.float32)
    assignment *= child_mask[:, :, None]
    return (assignment.sum(dim=1) + child_mask).long() * node_mask


if nn is not None:

    def _gather(hidden: Any, indices: Any) -> Any:
        return hidden.gather(
            1,
            indices[:, :, None].expand(-1, -1, hidden.shape[-1]),
        )

    class RegionTreeBlock(nn.Module):
        """Combine parent-child messages with global whole-tree attention."""

        def __init__(
            self,
            hidden_dim: int,
            attention_heads: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.local_update = nn.Sequential(
                nn.Linear(3 * hidden_dim, 2 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.local_norm = nn.LayerNorm(hidden_dim)
            self.attention = nn.MultiheadAttention(
                hidden_dim,
                attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(hidden_dim)
            self.feedforward = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
            self.output_norm = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            hidden: Any,
            parents: Any,
            node_mask: Any,
            child_mask: Any,
        ) -> Any:
            node_count = hidden.shape[1]
            parent_hidden = _gather(hidden, parents)
            assignment = functional.one_hot(
                parents,
                num_classes=node_count,
            ).to(hidden.dtype)
            assignment *= child_mask[:, :, None]
            child_sum = torch.einsum("bcp,bcd->bpd", assignment, hidden)
            child_count = assignment.sum(dim=1)
            child_mean = child_sum / child_count[:, :, None].clamp(min=1)
            hidden = self.local_norm(
                hidden + self.local_update(torch.cat((hidden, parent_hidden, child_mean), dim=-1))
            )
            attended, _ = self.attention(
                hidden,
                hidden,
                hidden,
                key_padding_mask=~node_mask,
                need_weights=False,
            )
            hidden = self.attention_norm(hidden + attended)
            hidden = self.output_norm(hidden + self.feedforward(hidden))
            return hidden * node_mask[:, :, None]

    class ConditionalTreeRegionFlow(nn.Module):
        """Generate head, interface and tail state conditional on one tree."""

        def __init__(
            self,
            *,
            region_classes: int,
            hidden_dim: int,
            layers: int,
            attention_heads: int,
            maximum_heavy_atoms: int,
            depth_buckets: int,
            maximum_degree: int,
            dropout: float,
        ) -> None:
            super().__init__()
            if (
                region_classes < 2
                or hidden_dim < 8
                or layers < 1
                or attention_heads < 1
                or hidden_dim % attention_heads
                or maximum_heavy_atoms < 2
                or depth_buckets < 2
                or maximum_degree < 2
            ):
                raise TreeRegionFlowError("invalid conditional region architecture")
            self.region_classes = region_classes
            self.maximum_heavy_atoms = maximum_heavy_atoms
            self.depth_buckets = depth_buckets
            self.maximum_degree = maximum_degree
            self.region_embedding = nn.Embedding(region_classes, hidden_dim)
            self.position_embedding = nn.Embedding(
                maximum_heavy_atoms,
                hidden_dim,
            )
            self.depth_embedding = nn.Embedding(depth_buckets, hidden_dim)
            self.degree_embedding = nn.Embedding(
                maximum_degree + 1,
                hidden_dim,
            )
            self.time_embedding = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.blocks = nn.ModuleList(
                RegionTreeBlock(hidden_dim, attention_heads, dropout) for _ in range(layers)
            )
            self.output = nn.Linear(hidden_dim, region_classes)

        def forward(
            self,
            regions: Any,
            parents: Any,
            t: Any,
            node_mask: Any,
            child_mask: Any,
        ) -> Any:
            positions = torch.arange(
                regions.shape[1],
                device=regions.device,
            )
            depths = _rooted_depths(parents, node_mask).clamp(max=self.depth_buckets - 1)
            degrees = _tree_degrees(
                parents,
                node_mask,
                child_mask,
            ).clamp(max=self.maximum_degree)
            hidden = (
                self.region_embedding(regions)
                + self.position_embedding(positions)[None, :, :]
                + self.depth_embedding(depths)
                + self.degree_embedding(degrees)
                + self.time_embedding(t[:, None])[:, None, :]
            )
            hidden *= node_mask[:, :, None]
            for block in self.blocks:
                hidden = block(
                    hidden,
                    parents,
                    node_mask,
                    child_mask,
                )
            return self.output(hidden)

else:  # pragma: no cover

    class ConditionalTreeRegionFlow:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise TreeRegionFlowError("conditional region flow requires torch")


def noise_region_batch(
    clean: dict[str, Any],
    depth_marginal: Any,
    t: Any,
    generator: Any,
) -> Any:
    """Corrupt nonroot regions under the fixed-tree depth source."""

    source = region_source_probabilities(
        clean["parents"],
        clean["node_mask"],
        depth_marginal,
    )
    active = clean["child_mask"]
    examples = torch.arange(
        active.shape[0],
        device=active.device,
    )[:, None].expand_as(active)[active]
    probabilities = source[active] * (1.0 - t[examples, None])
    probabilities.scatter_add_(
        1,
        clean["regions"][active][:, None],
        t[examples, None],
    )
    noisy = clean["regions"].clone()
    noisy[active] = torch.multinomial(
        probabilities,
        1,
        generator=generator,
    ).squeeze(1)
    noisy[:, 0] = 0
    return noisy


def region_flow_loss(
    logits: Any,
    clean: dict[str, Any],
) -> tuple[Any, dict[str, float]]:
    """Score generated nonroot regions; the selected polar root is fixed head."""

    loss = functional.cross_entropy(
        logits[clean["child_mask"]],
        clean["regions"][clean["child_mask"]],
    )
    return loss, {
        "total": float(loss.detach()),
        "region_ce": float(loss.detach()),
    }


def _sample_terminal_regions(
    logits: Any,
    parents: np.ndarray,
    transition_marginal: Any,
    *,
    transition_strength: float,
    generator: Any,
) -> np.ndarray:
    if transition_strength < 0:
        raise TreeRegionFlowError("region transition strength cannot be negative")
    regions = np.zeros(parents.size, dtype=np.int64)
    log_transition = transition_marginal.clamp(min=1e-12).log()
    for child in range(1, parents.size):
        parent_region = int(regions[int(parents[child])])
        probabilities = (
            logits[child] + transition_strength * log_transition[parent_region]
        ).softmax(dim=-1)
        regions[child] = int(
            torch.multinomial(
                probabilities,
                1,
                generator=generator,
            )
        )
    return regions


def sample_tree_regions(
    model: Any,
    parents: Sequence[np.ndarray],
    depth_marginal: np.ndarray,
    transition_marginal: np.ndarray,
    *,
    sample_steps: int,
    batch_size: int,
    transition_strength: float,
    seed: int,
    device: str,
) -> tuple[list[TreeRegionSample], dict[str, Any]]:
    """Generate structural regions conditional on settled whole-tree topology."""

    if not parents or sample_steps < 2 or batch_size < 1:
        raise TreeRegionFlowError("invalid conditional region sampling request")
    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device).manual_seed(seed)
    cpu_generator = torch.Generator().manual_seed(seed + 1)
    depth_source = torch.as_tensor(
        depth_marginal,
        dtype=torch.float32,
        device=resolved_device,
    )
    transition_source = torch.as_tensor(
        transition_marginal,
        dtype=torch.float32,
    )
    samples = []
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(parents), batch_size):
            local = parents[offset : offset + batch_size]
            maximum_nodes = max(value.size for value in local)
            parent_batch = torch.zeros(
                (len(local), maximum_nodes),
                dtype=torch.long,
                device=resolved_device,
            )
            node_mask = torch.zeros_like(parent_batch, dtype=torch.bool)
            for index, value in enumerate(local):
                parent_batch[index, : value.size] = torch.from_numpy(value.copy()).to(
                    resolved_device
                )
                node_mask[index, : value.size] = True
            child_mask = node_mask.clone()
            child_mask[:, 0] = False
            source = region_source_probabilities(
                parent_batch,
                node_mask,
                depth_source,
            )
            state = torch.zeros_like(parent_batch)
            state[child_mask] = torch.multinomial(
                source[child_mask],
                1,
                generator=generator,
            ).squeeze(1)
            for step in range(sample_steps):
                t_value = step / sample_steps
                t = torch.full(
                    (len(local),),
                    t_value,
                    dtype=torch.float32,
                    device=resolved_device,
                )
                logits = model(
                    state,
                    parent_batch,
                    t,
                    node_mask,
                    child_mask,
                )
                state = _rstar_step(
                    state,
                    logits.softmax(dim=-1),
                    source,
                    t_value,
                    1.0 / sample_steps,
                    child_mask,
                    generator,
                )
                state[:, 0] = 0
            terminal = (
                model(
                    state,
                    parent_batch,
                    torch.ones(
                        len(local),
                        dtype=torch.float32,
                        device=resolved_device,
                    ),
                    node_mask,
                    child_mask,
                )
                .detach()
                .cpu()
            )
            for index, value in enumerate(local):
                samples.append(
                    TreeRegionSample(
                        regions=_sample_terminal_regions(
                            terminal[index, : value.size],
                            value,
                            transition_source,
                            transition_strength=transition_strength,
                            generator=cpu_generator,
                        ),
                        parents=value.copy(),
                    )
                )
    elapsed = time.perf_counter() - start
    return samples, {
        "samples": len(samples),
        "sample_steps": sample_steps,
        "transition_strength": transition_strength,
        "wall_seconds": elapsed,
        "graph_steps_per_second": len(samples) * sample_steps / elapsed,
        "terminal_interventions": {},
    }


def tree_region_statistics(
    samples: Sequence[TreeRegionSample],
) -> dict[str, Any]:
    """Summarize regional organization on generated settled trees."""

    if not samples:
        raise TreeRegionFlowError("tree-region statistics require samples")
    atom_counts: Counter[int] = Counter()
    branch_counts: Counter[int] = Counter()
    component_counts: Counter[int] = Counter()
    tail_path_like = []
    tail_junctions = []
    maximum_depths = []
    for sample in samples:
        adjacency = [set() for _ in range(sample.node_count)]
        depths = np.zeros(sample.node_count, dtype=np.int64)
        for child in range(1, sample.node_count):
            parent = int(sample.parents[child])
            adjacency[child].add(parent)
            adjacency[parent].add(child)
            depths[child] = depths[parent] + 1
        maximum_depths.append(int(depths.max(initial=0)))
        degrees = np.asarray([len(neighbors) for neighbors in adjacency])
        for region in range(3):
            selected = {index for index, value in enumerate(sample.regions) if int(value) == region}
            components = _selected_components(adjacency, selected)
            atom_counts[region] += len(selected)
            branch_counts[region] += int(np.count_nonzero(degrees[list(selected)] >= 3))
            component_counts[region] += len(components)
            if region == 2:
                for component in components:
                    internal_degrees = [len(adjacency[node] & component) for node in component]
                    internal_edges = sum(internal_degrees) // 2
                    cycle_rank = max(
                        0,
                        internal_edges - len(component) + 1,
                    )
                    tail_path_like.append(
                        int(cycle_rank == 0 and max(internal_degrees, default=0) <= 2)
                    )
                    tail_junctions.append(sum(degree >= 3 for degree in internal_degrees))
    total_atoms = sum(atom_counts.values())
    names = ("head", "interface", "tail")
    return {
        "samples": len(samples),
        "mean_maximum_root_distance": float(np.mean(maximum_depths)),
        "regions": {
            name: {
                "atom_fraction": atom_counts[index] / max(1, total_atoms),
                "branch_atom_fraction": branch_counts[index] / max(1, atom_counts[index]),
                "components_per_molecule": component_counts[index] / len(samples),
            }
            for index, name in enumerate(names)
        },
        "tail_path_like_component_fraction": sum(tail_path_like) / max(1, len(tail_path_like)),
        "tail_junction_atoms_per_component": sum(tail_junctions) / max(1, len(tail_junctions)),
    }


def _selected_components(
    adjacency: Sequence[set[int]],
    selected: set[int],
) -> list[set[int]]:
    components = []
    remaining = set(selected)
    while remaining:
        start = min(remaining)
        component = {start}
        queue = deque([start])
        remaining.remove(start)
        while queue:
            node = queue.popleft()
            for candidate in adjacency[node] & remaining:
                remaining.remove(candidate)
                component.add(candidate)
                queue.append(candidate)
        components.append(component)
    return components
