"""Program-conditioned discrete flow for sparse whole-lipid morphology.

This bounded Phase 1 model generates atom-level tree offspring and structural
region states.  It operates on an O(N) token sequence and is conditioned on the
minimal V5 global morphology program.  Exact constrained decoders in
``v5_morphology_program`` turn terminal local scores into one valid tree and
one exact regional allocation without fragment enumeration or graph repair.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from forge.product.defog_feasibility import _rstar_step
from forge.product.lipid_context import HEAD_REGION
from forge.product.phase1_tree_topology_flow import preorder_offspring_to_parents
from forge.product.v5_morphology_program import (
    V5GlobalMorphologyProgram,
    program_from_sparse_record,
    sample_morphology_with_exact_program,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    functional = None
    pack_padded_sequence = None
    pad_packed_sequence = None


class MorphologyFlowError(RuntimeError):
    """Raised when morphology-flow data or execution violates its contract."""


@dataclass(frozen=True)
class V5MorphologySample:
    """One exact program-conditioned sparse lipid morphology sample."""

    program: V5GlobalMorphologyProgram
    offspring: np.ndarray
    regions: np.ndarray
    parents: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.offspring.size)


_PROGRAM_FIELDS = (
    "n_head",
    "n_interface",
    "n_tail",
    "junction_budget_head_interface",
    "junction_budget_tail",
    "cycle_rank",
)


def program_to_array(program: V5GlobalMorphologyProgram) -> np.ndarray:
    """Serialize the fixed global program in its declared field order."""

    return np.asarray([getattr(program, field) for field in _PROGRAM_FIELDS], dtype=np.int64)


def collate_v5_morphology_records(
    records: Sequence[Any],
    *,
    maximum_nodes: int,
    maximum_children: int,
) -> dict[str, Any]:
    """Collate exact V5 morphology targets without atom or bond chemistry."""

    if torch is None:
        raise MorphologyFlowError("morphology collation requires torch")
    if not records or maximum_nodes < max(record.node_count for record in records):
        raise MorphologyFlowError("invalid morphology collation capacity")
    offspring = torch.zeros((len(records), maximum_nodes), dtype=torch.long)
    regions = torch.zeros_like(offspring)
    node_mask = torch.zeros_like(offspring, dtype=torch.bool)
    programs = torch.zeros((len(records), len(_PROGRAM_FIELDS)), dtype=torch.long)
    for index, record in enumerate(records):
        if record.region_states is None:
            raise MorphologyFlowError("morphology record lacks region supervision")
        if record.offspring.max(initial=0) > maximum_children:
            raise MorphologyFlowError("offspring count exceeds declared support")
        program = program_from_sparse_record(record)
        offspring[index, : record.node_count] = torch.from_numpy(record.offspring.copy())
        regions[index, : record.node_count] = torch.from_numpy(record.region_states.copy())
        node_mask[index, : record.node_count] = True
        programs[index] = torch.from_numpy(program_to_array(program))
    region_mask = node_mask.clone()
    region_mask[:, 0] = False
    return {
        "offspring": offspring,
        "regions": regions,
        "node_mask": node_mask,
        "region_mask": region_mask,
        "programs": programs,
    }


def morphology_source_marginals(
    records: Sequence[Any],
    *,
    maximum_children: int,
    region_classes: int = 3,
    probability_floor: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Estimate clean-corpus categorical sources for the bounded flow."""

    if not records or maximum_children < 1 or region_classes != 3 or not 0 < probability_floor < 1:
        raise MorphologyFlowError("invalid morphology marginal request")
    offspring = np.full(maximum_children + 1, probability_floor, dtype=np.float64)
    regions = np.full(region_classes, probability_floor, dtype=np.float64)
    for record in records:
        if record.region_states is None:
            raise MorphologyFlowError("morphology record lacks region supervision")
        if record.offspring.max(initial=0) > maximum_children:
            raise MorphologyFlowError("offspring count exceeds declared support")
        np.add.at(offspring, record.offspring, 1.0)
        np.add.at(regions, record.region_states, 1.0)
    return {
        "offspring": offspring / offspring.sum(),
        "regions": regions / regions.sum(),
    }


def _noisy_pending_balance(offspring: Any, node_mask: Any, maximum_nodes: int) -> Any:
    active_delta = (offspring - 1) * node_mask
    balance = 1 + torch.cumsum(active_delta, dim=1)
    return balance.clamp(min=-maximum_nodes, max=maximum_nodes) + maximum_nodes


if nn is not None:

    class V5MorphologyFlow(nn.Module):
        """Bidirectional linear-sequence denoiser for V5 morphology tokens."""

        def __init__(
            self,
            *,
            maximum_children: int,
            maximum_heavy_atoms: int,
            maximum_junction_budget: int,
            maximum_cycle_rank: int,
            hidden_dim: int,
            layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            if (
                maximum_children < 1
                or maximum_heavy_atoms < 2
                or maximum_junction_budget < 0
                or maximum_cycle_rank < 0
                or hidden_dim < 16
                or hidden_dim % 2
                or layers < 1
                or not 0 <= dropout < 1
            ):
                raise MorphologyFlowError("invalid V5 morphology architecture")
            self.maximum_children = maximum_children
            self.maximum_heavy_atoms = maximum_heavy_atoms
            self.maximum_junction_budget = maximum_junction_budget
            self.maximum_cycle_rank = maximum_cycle_rank
            self.offspring_embedding = nn.Embedding(maximum_children + 1, hidden_dim)
            self.region_embedding = nn.Embedding(3, hidden_dim)
            self.position_embedding = nn.Embedding(maximum_heavy_atoms, hidden_dim)
            self.pending_embedding = nn.Embedding(2 * maximum_heavy_atoms + 1, hidden_dim)
            self.count_embeddings = nn.ModuleList(
                nn.Embedding(maximum_heavy_atoms + 1, hidden_dim) for _ in range(3)
            )
            self.junction_embeddings = nn.ModuleList(
                nn.Embedding(maximum_junction_budget + 1, hidden_dim) for _ in range(2)
            )
            self.cycle_embedding = nn.Embedding(maximum_cycle_rank + 1, hidden_dim)
            self.time_embedding = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.input_norm = nn.LayerNorm(hidden_dim)
            self.sequence = nn.GRU(
                hidden_dim,
                hidden_dim // 2,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
                bidirectional=True,
            )
            self.output = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.offspring_output = nn.Linear(hidden_dim, maximum_children + 1)
            self.region_output = nn.Linear(hidden_dim, 3)

        def _program_context(self, programs: Any) -> Any:
            if programs.ndim != 2 or programs.shape[1] != len(_PROGRAM_FIELDS):
                raise MorphologyFlowError("program tensor must be [batch, 6]")
            counts = programs[:, :3]
            junctions = programs[:, 3:5]
            cycles = programs[:, 5]
            if (
                torch.any(counts < 0)
                or torch.any(counts > self.maximum_heavy_atoms)
                or torch.any(junctions < 0)
                or torch.any(junctions > self.maximum_junction_budget)
                or torch.any(cycles < 0)
                or torch.any(cycles > self.maximum_cycle_rank)
            ):
                raise MorphologyFlowError("program value exceeds model support")
            context = self.cycle_embedding(cycles)
            for field, embedding in enumerate(self.count_embeddings):
                context = context + embedding(counts[:, field])
            for field, embedding in enumerate(self.junction_embeddings):
                context = context + embedding(junctions[:, field])
            return context

        def forward(
            self,
            offspring: Any,
            regions: Any,
            programs: Any,
            t: Any,
            node_mask: Any,
        ) -> dict[str, Any]:
            if offspring.shape != regions.shape or offspring.shape != node_mask.shape:
                raise MorphologyFlowError("morphology state shapes do not agree")
            positions = torch.arange(offspring.shape[1], device=offspring.device)
            pending = _noisy_pending_balance(
                offspring,
                node_mask,
                self.maximum_heavy_atoms,
            )
            hidden = (
                self.offspring_embedding(offspring)
                + self.region_embedding(regions)
                + self.position_embedding(positions)[None, :, :]
                + self.pending_embedding(pending)
                + self._program_context(programs)[:, None, :]
                + self.time_embedding(t[:, None])[:, None, :]
            )
            hidden = self.input_norm(hidden) * node_mask[:, :, None]
            lengths = node_mask.sum(dim=1).to("cpu")
            packed = pack_padded_sequence(
                hidden,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.sequence(packed)
            hidden, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=offspring.shape[1],
            )
            hidden = (hidden + self.output(hidden)) * node_mask[:, :, None]
            return {
                "offspring": self.offspring_output(hidden),
                "regions": self.region_output(hidden),
            }

else:  # pragma: no cover

    class V5MorphologyFlow:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise MorphologyFlowError("V5 morphology flow requires torch")


def noise_v5_morphology_batch(
    clean: dict[str, Any],
    sources: dict[str, Any],
    t: Any,
    generator: Any,
) -> dict[str, Any]:
    """Corrupt offspring and nonroot regions under categorical flow sources."""

    output: dict[str, Any] = {}
    for field, mask_name in (("offspring", "node_mask"), ("regions", "region_mask")):
        mask = clean[mask_name]
        example_indices = torch.arange(mask.shape[0], device=mask.device)[:, None].expand_as(mask)[
            mask
        ]
        probabilities = sources[field][None, :].repeat(int(mask.sum()), 1)
        probabilities *= 1.0 - t[example_indices, None]
        probabilities.scatter_add_(
            1,
            clean[field][mask][:, None],
            t[example_indices, None],
        )
        noisy = clean[field].clone()
        noisy[mask] = torch.multinomial(
            probabilities,
            1,
            generator=generator,
        ).squeeze(1)
        output[field] = noisy
    output["regions"][:, 0] = HEAD_REGION
    return output


def v5_morphology_flow_loss(
    predictions: dict[str, Any],
    clean: dict[str, Any],
) -> tuple[Any, dict[str, float]]:
    """Compute masked clean-state cross-entropies for both token fields."""

    offspring_loss = functional.cross_entropy(
        predictions["offspring"][clean["node_mask"]],
        clean["offspring"][clean["node_mask"]],
    )
    region_loss = functional.cross_entropy(
        predictions["regions"][clean["region_mask"]],
        clean["regions"][clean["region_mask"]],
    )
    total = offspring_loss + region_loss
    return total, {
        "total": float(total.detach()),
        "offspring_ce": float(offspring_loss.detach()),
        "region_ce": float(region_loss.detach()),
    }


def sample_v5_morphologies(
    model: Any,
    programs: Sequence[V5GlobalMorphologyProgram],
    source_marginals: dict[str, np.ndarray],
    *,
    sample_steps: int,
    batch_size: int,
    seed: int,
    device: str,
) -> tuple[list[V5MorphologySample], dict[str, Any]]:
    """Run discrete flow and decode every terminal state under exact programs."""

    if torch is None:
        raise MorphologyFlowError("V5 morphology sampling requires torch")
    if not programs or sample_steps < 2 or batch_size < 1:
        raise MorphologyFlowError("invalid V5 morphology sampling request")
    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device).manual_seed(seed)
    cpu_generator = torch.Generator().manual_seed(seed + 1)
    sources = {
        field: torch.as_tensor(values, dtype=torch.float32, device=resolved_device)
        for field, values in source_marginals.items()
    }
    samples: list[V5MorphologySample] = []
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(programs), batch_size):
            local = programs[offset : offset + batch_size]
            maximum_nodes = max(program.node_count for program in local)
            node_mask = torch.zeros(
                (len(local), maximum_nodes),
                dtype=torch.bool,
                device=resolved_device,
            )
            program_tensor = torch.as_tensor(
                np.stack([program_to_array(program) for program in local]),
                dtype=torch.long,
                device=resolved_device,
            )
            for index, program in enumerate(local):
                node_mask[index, : program.node_count] = True
            region_mask = node_mask.clone()
            region_mask[:, 0] = False
            offspring = torch.zeros_like(node_mask, dtype=torch.long)
            regions = torch.zeros_like(offspring)
            offspring[node_mask] = torch.multinomial(
                sources["offspring"],
                int(node_mask.sum()),
                replacement=True,
                generator=generator,
            )
            regions[region_mask] = torch.multinomial(
                sources["regions"],
                int(region_mask.sum()),
                replacement=True,
                generator=generator,
            )
            for step in range(sample_steps):
                t_value = step / sample_steps
                t = torch.full(
                    (len(local),),
                    t_value,
                    dtype=torch.float32,
                    device=resolved_device,
                )
                predictions = model(offspring, regions, program_tensor, t, node_mask)
                offspring = _rstar_step(
                    offspring,
                    predictions["offspring"].softmax(dim=-1),
                    sources["offspring"],
                    t_value,
                    1.0 / sample_steps,
                    node_mask,
                    generator,
                )
                regions = _rstar_step(
                    regions,
                    predictions["regions"].softmax(dim=-1),
                    sources["regions"],
                    t_value,
                    1.0 / sample_steps,
                    region_mask,
                    generator,
                )
                regions[:, 0] = HEAD_REGION
            terminal = model(
                offspring,
                regions,
                program_tensor,
                torch.ones(len(local), dtype=torch.float32, device=resolved_device),
                node_mask,
            )
            for index, program in enumerate(local):
                node_count = program.node_count
                sampled_offspring, sampled_regions = sample_morphology_with_exact_program(
                    terminal["offspring"][index, :node_count].detach().cpu(),
                    terminal["regions"][index, :node_count].detach().cpu(),
                    program,
                    generator=cpu_generator,
                )
                samples.append(
                    V5MorphologySample(
                        program=program,
                        offspring=sampled_offspring,
                        regions=sampled_regions,
                        parents=preorder_offspring_to_parents(sampled_offspring),
                    )
                )
    elapsed = time.perf_counter() - start
    return samples, {
        "samples": len(samples),
        "sample_steps": sample_steps,
        "wall_seconds": elapsed,
        "graph_steps_per_second": len(samples) * sample_steps / elapsed,
        "terminal_tree_repairs": 0,
        "terminal_region_repairs": 0,
    }


def v5_morphology_statistics(samples: Sequence[V5MorphologySample]) -> dict[str, Any]:
    """Summarize nontrivial topology organization beyond guaranteed budgets."""

    if not samples:
        raise MorphologyFlowError("morphology statistics require samples")
    degrees_by_region: dict[int, Counter[int]] = {region: Counter() for region in range(3)}
    transitions: Counter[tuple[int, int]] = Counter()
    maximum_depths: list[int] = []
    leaf_fractions: list[float] = []
    tail_branch_fractions: list[float] = []
    signatures: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for sample in samples:
        if sample.node_count != sample.program.node_count:
            raise MorphologyFlowError("sample size does not match its global program")
        degrees = sample.offspring.astype(np.int64, copy=True)
        if sample.node_count > 1:
            degrees[1:] += 1
        depths = np.zeros(sample.node_count, dtype=np.int64)
        for child in range(1, sample.node_count):
            parent = int(sample.parents[child])
            depths[child] = depths[parent] + 1
            transitions[(int(sample.regions[parent]), int(sample.regions[child]))] += 1
        for region in range(3):
            local = degrees[sample.regions == region]
            degrees_by_region[region].update(int(value) for value in local)
        tail = sample.regions == 2
        maximum_depths.append(int(depths.max(initial=0)))
        leaf_fractions.append(float(np.mean(sample.offspring == 0)))
        tail_branch_fractions.append(float(np.mean(degrees[tail] >= 3)) if tail.any() else 0.0)
        signatures.add((tuple(sample.offspring.tolist()), tuple(sample.regions.tolist())))
    transition_total = sum(transitions.values())
    return {
        "samples": len(samples),
        "mean_maximum_root_distance": float(np.mean(maximum_depths)),
        "mean_leaf_fraction": float(np.mean(leaf_fractions)),
        "mean_tail_branch_atom_fraction": float(np.mean(tail_branch_fractions)),
        "unique_topology_region_fraction": len(signatures) / len(samples),
        "degree_fractions_by_region": {
            str(region): {
                str(degree): count / max(1, sum(histogram.values()))
                for degree, count in sorted(histogram.items())
            }
            for region, histogram in degrees_by_region.items()
        },
        "region_transition_fractions": {
            f"{parent}->{child}": count / max(1, transition_total)
            for (parent, child), count in sorted(transitions.items())
        },
    }
