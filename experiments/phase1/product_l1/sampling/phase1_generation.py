"""Deterministic endpoint sampling for the Phase 1 sparse whole-lipid flow."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.flow.rstar import rstar_step as _rstar_step
from forge.model.defog_feasibility import AtomState, _model_state_sha256
from forge.model.lipid_context import tree_pair_ring_sizes
from forge.model.phase1_flow import (
    Phase1FlowError,
    SparseWholeLipidFlow,
    _model_architecture_kwargs,
)
from forge.model.sparse_topology_feasibility import (
    BOND_VALENCE_UNITS,
    INDEX_TO_DENSE_BOND,
    _endpoint_candidate_mask,
    _maximum_valence_units,
    _parent_candidate_mask,
    _sample_from_logits,
    pointer_rstar_step,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None


def load_product_checkpoint(
    checkpoint_path: Path,
    *,
    device: str,
) -> tuple[Any, tuple[AtomState, ...], np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a trusted, locally generated Phase 1 checkpoint for inference."""

    if torch is None:
        raise Phase1FlowError("Phase 1 product generation requires torch")
    if not checkpoint_path.is_file():
        raise Phase1FlowError(f"product checkpoint not found: {checkpoint_path}")
    try:
        package = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise Phase1FlowError(f"product checkpoint could not be loaded: {checkpoint_path}") from exc
    if (
        not isinstance(package, dict)
        or package.get("schema_version") != "phase1_product_pretrain_checkpoint.v2"
        or package.get("trusted_local_checkpoint") is not True
    ):
        raise Phase1FlowError("product checkpoint is not a trusted Phase 1 checkpoint")
    model_config = package.get("model_config")
    vocabulary_rows = package.get("atom_vocabulary")
    if not isinstance(model_config, dict) or not isinstance(vocabulary_rows, list):
        raise Phase1FlowError("product checkpoint lacks its frozen model contract")
    vocabulary = tuple(
        AtomState(
            str(row["symbol"]),
            int(row["formal_charge"]),
            bool(row["aromatic"]),
            int(row.get("explicit_hydrogens", 0)),
        )
        for row in vocabulary_rows
    )
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise Phase1FlowError("CUDA generation was requested but no CUDA device is available")
    model = SparseWholeLipidFlow(
        node_classes=len(vocabulary),
        hidden_dim=int(model_config["hidden_dim"]),
        layers=int(model_config["layers"]),
        maximum_closures=int(model_config["maximum_closure_slots"]),
        maximum_heavy_atoms=int(model_config["maximum_heavy_atoms"]),
        dropout=float(model_config["dropout"]),
        **_model_architecture_kwargs(model_config),
    )
    model.load_state_dict(package["model_state_dict"], strict=True)
    model.to(resolved_device)
    if _model_state_sha256(model) != package.get("model_state_sha256"):
        raise Phase1FlowError("product checkpoint model-state hash mismatch")
    node_marginal = np.asarray(package["node_marginal"], dtype=np.float64)
    bond_marginal = np.asarray(package["bond_marginal"], dtype=np.float64)
    if (
        node_marginal.shape != (len(vocabulary),)
        or bond_marginal.shape != (int(model_config.get("bond_classes", 3)),)
        or not np.isclose(node_marginal.sum(), 1.0)
        or not np.isclose(bond_marginal.sum(), 1.0)
    ):
        raise Phase1FlowError("product checkpoint contains invalid endpoint marginals")
    region_marginal = package.get("region_marginal")
    region_classes = int(model_config.get("region_classes", 0))
    if region_classes:
        region_array = np.asarray(region_marginal, dtype=np.float64)
        if (
            region_array.shape != (region_classes,)
            or not np.isfinite(region_array).all()
            or np.any(region_array <= 0)
            or not np.isclose(region_array.sum(), 1.0)
        ):
            raise Phase1FlowError("product checkpoint contains invalid region marginal")
        region_atom_array = np.asarray(
            package.get("region_atom_marginal"),
            dtype=np.float64,
        )
        if (
            region_atom_array.shape != (region_classes, len(vocabulary))
            or not np.isfinite(region_atom_array).all()
            or np.any(region_atom_array <= 0)
            or not np.allclose(region_atom_array.sum(axis=1), 1.0)
        ):
            raise Phase1FlowError(
                "product checkpoint contains invalid region-conditioned atom marginals"
            )
        if float(model_config.get("degree_continuation_prior_strength", 0.0)) > 0.0:
            degree_prior = np.asarray(
                package.get("degree_continuation_log_prior"),
                dtype=np.float64,
            )
            maximum_degree = int(model_config.get("degree_prior_maximum_degree", 0))
            if (
                degree_prior.shape != (region_classes, maximum_degree + 1)
                or np.isnan(degree_prior).any()
                or np.any(degree_prior[:, :-1] > 0.0)
            ):
                raise Phase1FlowError(
                    "product checkpoint contains invalid degree-continuation prior"
                )
    elif region_marginal is not None:
        raise Phase1FlowError("region marginal exists for a region-free checkpoint")
    elif package.get("region_atom_marginal") is not None:
        raise Phase1FlowError(
            "region-conditioned atom marginals exist for a region-free checkpoint"
        )
    elif package.get("degree_continuation_log_prior") is not None:
        raise Phase1FlowError("degree-continuation prior exists for a region-free checkpoint")
    model.eval()
    return model, vocabulary, node_marginal, bond_marginal, package


def _sample_counts(
    model: Any,
    *,
    sample_count: int,
    generator: Any,
    node_count_distribution: np.ndarray | None,
    closure_count_distribution: np.ndarray | None,
) -> tuple[Any, Any]:
    if node_count_distribution is None:
        node_probabilities = model.node_count_logits.detach().softmax(dim=0).clone()
        node_probabilities[0] = 0.0
        node_probabilities /= node_probabilities.sum()
    else:
        node_probabilities = torch.tensor(
            node_count_distribution,
            dtype=torch.float32,
            device=model.node_count_logits.device,
        )
    if closure_count_distribution is None:
        closure_probabilities = model.closure_count_logits.detach().softmax(dim=0)
    else:
        closure_probabilities = torch.tensor(
            closure_count_distribution,
            dtype=torch.float32,
            device=model.closure_count_logits.device,
        )
    node_counts = torch.multinomial(
        node_probabilities,
        sample_count,
        replacement=True,
        generator=generator,
    )
    maximum_closures = closure_probabilities.shape[0] - 1
    closure_counts = []
    for node_count in node_counts.tolist():
        maximum_edges = node_count * (node_count - 1) // 2
        maximum_cycle_rank = max(0, maximum_edges - max(0, node_count - 1))
        allowed = min(maximum_closures, maximum_cycle_rank)
        local_probabilities = closure_probabilities[: allowed + 1]
        local_probabilities = local_probabilities / local_probabilities.sum()
        closure_counts.append(
            int(
                torch.multinomial(
                    local_probabilities,
                    1,
                    generator=generator,
                )
            )
        )
    return node_counts, torch.tensor(closure_counts, device=node_counts.device)


def maximum_likelihood_count_distributions(
    node_counts: Sequence[int],
    closure_counts: Sequence[int],
    *,
    maximum_heavy_atoms: int,
    maximum_closures: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the exact categorical MLE for the model's unconditional count variables."""

    if not node_counts or len(node_counts) != len(closure_counts):
        raise Phase1FlowError("count calibration requires paired nonempty count observations")
    if min(node_counts) < 1 or max(node_counts) > maximum_heavy_atoms:
        raise Phase1FlowError("node-count calibration lies outside the declared support")
    if min(closure_counts) < 0 or max(closure_counts) > maximum_closures:
        raise Phase1FlowError("closure-count calibration lies outside the declared support")
    node_distribution = np.bincount(
        np.asarray(node_counts, dtype=np.int64),
        minlength=maximum_heavy_atoms + 1,
    ).astype(np.float64)
    closure_distribution = np.bincount(
        np.asarray(closure_counts, dtype=np.int64),
        minlength=maximum_closures + 1,
    ).astype(np.float64)
    node_distribution /= node_distribution.sum()
    closure_distribution /= closure_distribution.sum()
    return node_distribution, closure_distribution


def _construct_product_terminal_graph(
    predictions: Mapping[str, Any],
    index: int,
    node_count: int,
    requested_closures: int,
    atom_vocabulary: Sequence[AtomState],
    maximum_closures: int,
    generator: Any,
    closure_ring_size_log_probabilities: Any | None = None,
    closure_ring_size_prior_strength: float = 0.0,
    aromatic_cycle_sizes: Sequence[int] = (5, 6),
    aromatic_cycle_probability_threshold: float = 0.5,
    degree_continuation_log_prior: Any | None = None,
    degree_continuation_prior_strength: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Decode atom states before topology so valence masking does not enrich high-valence atoms."""

    if closure_ring_size_prior_strength < 0.0:
        raise Phase1FlowError("closure ring-size prior strength cannot be negative")
    if degree_continuation_prior_strength < 0.0:
        raise Phase1FlowError("degree-continuation prior strength cannot be negative")
    if (
        not aromatic_cycle_sizes
        or any(size < 3 for size in aromatic_cycle_sizes)
        or not 0.0 < aromatic_cycle_probability_threshold < 1.0
    ):
        raise Phase1FlowError("aromatic cycle policy is invalid")
    repairs: Counter[str] = Counter()
    tensor_device = predictions["nodes"].device
    sampled_regions: np.ndarray | None = None
    if "regions" in predictions:
        sampled_regions = np.zeros(node_count, dtype=np.int64)
        for node in range(node_count):
            region, _ = _sample_from_logits(
                predictions["regions"][index, node],
                torch.ones(
                    predictions["regions"].shape[-1],
                    dtype=torch.bool,
                    device=tensor_device,
                ),
                generator,
            )
            sampled_regions[node] = region
            repairs[f"sampled_region_{region}"] += 1
    if degree_continuation_log_prior is not None and sampled_regions is None:
        raise Phase1FlowError("degree-continuation prior requires sampled regions")
    node_states = np.zeros(node_count, dtype=np.int64)
    capacities = np.zeros(node_count, dtype=np.int64)
    for node in range(node_count):
        state, _ = _sample_from_logits(
            predictions["nodes"][index, node],
            torch.ones(
                len(atom_vocabulary),
                dtype=torch.bool,
                device=tensor_device,
            ),
            generator,
        )
        node_states[node] = state
        capacities[node] = _maximum_valence_units(atom_vocabulary[state])

    parents = np.zeros(node_count, dtype=np.int64)
    used_units = np.zeros(node_count, dtype=np.int64)
    for child in range(1, node_count):
        minimum_parent = int(parents[child - 1]) if child > 1 else 0
        valid = torch.zeros(node_count, dtype=torch.bool, device=tensor_device)
        valid[minimum_parent:child] = torch.tensor(
            used_units[minimum_parent:child] + 2 <= capacities[minimum_parent:child],
            device=tensor_device,
        )
        parent_logits = predictions["parents"][index, child, :node_count]
        if degree_continuation_log_prior is not None and degree_continuation_prior_strength > 0.0:
            current_degrees = used_units // 2
            parent_logits = parent_logits + _degree_continuation_log_bias(
                sampled_regions,
                current_degrees,
                degree_continuation_log_prior,
                strength=degree_continuation_prior_strength,
                device=tensor_device,
            )
        parent, repaired = _sample_from_logits(
            parent_logits,
            valid,
            generator,
        )
        if repaired:
            parent = min(
                range(minimum_parent, child),
                key=lambda candidate: used_units[candidate],
            )
            required = int(used_units[parent] + 2)
            valid_states = torch.tensor(
                [_maximum_valence_units(state) >= required for state in atom_vocabulary],
                dtype=torch.bool,
                device=tensor_device,
            )
            replacement, state_repaired = _sample_from_logits(
                predictions["nodes"][index, parent],
                valid_states,
                generator,
            )
            if state_repaired:
                replacement = max(
                    range(len(atom_vocabulary)),
                    key=lambda candidate: _maximum_valence_units(atom_vocabulary[candidate]),
                )
            node_states[parent] = replacement
            capacities[parent] = _maximum_valence_units(atom_vocabulary[replacement])
            repairs["parent_node_state_resampled_for_capacity"] += 1
        parents[child] = parent
        used_units[child] += 2
        used_units[parent] += 2

    edges = np.zeros((node_count, node_count), dtype=np.int64)
    for child in range(1, node_count):
        parent = int(parents[child])
        spare = min(
            capacities[child] - used_units[child],
            capacities[parent] - used_units[parent],
        )
        bond_valence_units = BOND_VALENCE_UNITS[: predictions["parent_bonds"].shape[-1]]
        valid_bonds = bond_valence_units <= 2 + max(0, spare)
        bond, repaired = _sample_from_logits(
            predictions["parent_bonds"][index, child],
            valid_bonds,
            generator,
        )
        if repaired:
            bond = 0
            repairs["backbone_bond_fallback"] += 1
        extra = int(bond_valence_units[bond]) - 2
        used_units[child] += extra
        used_units[parent] += extra
        dense_bond = INDEX_TO_DENSE_BOND[bond]
        edges[child, parent] = edges[parent, child] = dense_bond

    realized_closures = 0
    realized_closure_slots: list[tuple[int, int, int]] = []
    for slot in range(min(requested_closures, maximum_closures)):
        left_logits = predictions["closure_left"][index, slot, :node_count]
        right_logits = predictions["closure_right"][index, slot, :node_count]
        pair_logits = left_logits[:, None] + right_logits[None, :]
        valid_pairs = torch.triu(
            torch.ones(
                (node_count, node_count),
                dtype=torch.bool,
                device=tensor_device,
            ),
            diagonal=1,
        )
        valid_pairs &= torch.as_tensor(edges == 0, device=tensor_device)
        spare_nodes = torch.as_tensor(
            capacities - used_units >= 2,
            device=tensor_device,
        )
        valid_pairs &= spare_nodes[:, None] & spare_nodes[None, :]
        if not valid_pairs.any():
            repairs["closure_omitted_no_valence_pair"] += requested_closures - slot
            break
        if (
            closure_ring_size_log_probabilities is not None
            and closure_ring_size_prior_strength > 0.0
        ):
            pair_logits = pair_logits + _closure_pair_log_bias(
                parents,
                valid_pairs,
                closure_ring_size_log_probabilities,
                strength=closure_ring_size_prior_strength,
            )
        flat_pair, _ = _sample_from_logits(
            pair_logits.flatten(),
            valid_pairs.flatten(),
            generator,
        )
        left, right = divmod(flat_pair, node_count)
        spare = min(
            capacities[left] - used_units[left],
            capacities[right] - used_units[right],
        )
        bond_valence_units = BOND_VALENCE_UNITS[: predictions["closure_bonds"].shape[-1]]
        valid_bonds = bond_valence_units <= spare
        bond, repaired = _sample_from_logits(
            predictions["closure_bonds"][index, slot],
            valid_bonds,
            generator,
        )
        if repaired:
            bond = 0
            repairs["closure_bond_fallback"] += 1
        units = int(bond_valence_units[bond])
        used_units[left] += units
        used_units[right] += units
        dense_bond = INDEX_TO_DENSE_BOND[bond]
        edges[left, right] = edges[right, left] = dense_bond
        realized_closures += 1
        realized_closure_slots.append((left, right, slot))
        ring_size = int(tree_pair_ring_sizes(parents)[left, right])
        repairs[f"realized_ring_size_{ring_size}"] += 1
    _enforce_aromatic_cycle_consistency(
        node_states,
        edges,
        parents,
        realized_closure_slots,
        predictions,
        index,
        atom_vocabulary,
        generator,
        allowed_cycle_sizes=frozenset(int(size) for size in aromatic_cycle_sizes),
        probability_threshold=aromatic_cycle_probability_threshold,
        repairs=repairs,
    )
    repairs["requested_closures"] = requested_closures
    repairs["realized_closures"] = realized_closures
    return node_states, edges, dict(repairs)


def _tree_path(parents: np.ndarray, left: int, right: int) -> list[int]:
    left_path = []
    node = left
    while True:
        left_path.append(node)
        if int(parents[node]) == node:
            break
        node = int(parents[node])
    right_path = []
    node = right
    left_positions = {value: index for index, value in enumerate(left_path)}
    while node not in left_positions:
        right_path.append(node)
        node = int(parents[node])
    common = node
    return left_path[: left_positions[common] + 1] + list(reversed(right_path))


def _enforce_aromatic_cycle_consistency(
    node_states: np.ndarray,
    edges: np.ndarray,
    parents: np.ndarray,
    realized_closures: Sequence[tuple[int, int, int]],
    predictions: Mapping[str, Any],
    prediction_index: int,
    atom_vocabulary: Sequence[AtomState],
    generator: Any,
    *,
    allowed_cycle_sizes: frozenset[int],
    probability_threshold: float,
    repairs: Counter[str],
) -> None:
    """Decode aromaticity jointly over complete cycles, never isolated atoms."""

    aromatic_state_mask = torch.tensor(
        [state.aromatic for state in atom_vocabulary],
        dtype=torch.bool,
        device=predictions["nodes"].device,
    )
    if not aromatic_state_mask.any():
        return
    aromatic_nodes: set[int] = set()
    aromatic_edges: set[tuple[int, int]] = set()
    for left, right, slot in realized_closures:
        cycle_nodes = _tree_path(parents, left, right)
        ring_size = len(cycle_nodes)
        if ring_size not in allowed_cycle_sizes:
            continue
        node_probabilities = predictions["nodes"][
            prediction_index,
            cycle_nodes,
        ].softmax(dim=-1)
        atom_evidence = float(node_probabilities[:, aromatic_state_mask].sum(dim=-1).mean())
        path_children = []
        for first, second in zip(cycle_nodes[:-1], cycle_nodes[1:], strict=True):
            path_children.append(first if int(parents[first]) == second else second)
        bond_probabilities = [
            predictions["parent_bonds"][prediction_index, child].softmax(dim=-1)[3]
            for child in path_children
        ]
        bond_probabilities.append(
            predictions["closure_bonds"][prediction_index, slot].softmax(dim=-1)[3]
        )
        bond_evidence = float(torch.stack(bond_probabilities).mean())
        if min(atom_evidence, bond_evidence) < probability_threshold:
            continue
        replacements: dict[int, int] = {}
        for node in cycle_nodes:
            current = atom_vocabulary[int(node_states[node])]
            valid = torch.tensor(
                [
                    state.aromatic
                    and state.symbol == current.symbol
                    and state.formal_charge == current.formal_charge
                    for state in atom_vocabulary
                ],
                dtype=torch.bool,
                device=predictions["nodes"].device,
            )
            if not valid.any():
                replacements = {}
                break
            replacements[node], _ = _sample_from_logits(
                predictions["nodes"][prediction_index, node],
                valid,
                generator,
            )
        if not replacements:
            continue
        aromatic_nodes.update(cycle_nodes)
        for node, replacement in replacements.items():
            node_states[node] = replacement
        for first, second in zip(
            cycle_nodes,
            cycle_nodes[1:] + cycle_nodes[:1],
            strict=True,
        ):
            pair = tuple(sorted((first, second)))
            aromatic_edges.add(pair)
            edges[first, second] = edges[second, first] = INDEX_TO_DENSE_BOND[3]
        repairs[f"aromatic_cycle_size_{ring_size}"] += 1

    for left, right in zip(*np.nonzero(np.triu(edges == INDEX_TO_DENSE_BOND[3], 1))):
        if (int(left), int(right)) not in aromatic_edges:
            edges[left, right] = edges[right, left] = INDEX_TO_DENSE_BOND[0]
            repairs["orphan_aromatic_bond_downgraded"] += 1
    for node, state_index in enumerate(node_states):
        if atom_vocabulary[int(state_index)].aromatic and node not in aromatic_nodes:
            current = atom_vocabulary[int(state_index)]
            valid = torch.tensor(
                [
                    not state.aromatic
                    and state.symbol == current.symbol
                    and state.formal_charge == current.formal_charge
                    for state in atom_vocabulary
                ],
                dtype=torch.bool,
                device=predictions["nodes"].device,
            )
            if not valid.any():
                valid = ~aromatic_state_mask
            replacement, _ = _sample_from_logits(
                predictions["nodes"][prediction_index, node],
                valid,
                generator,
            )
            node_states[node] = replacement
            repairs["orphan_aromatic_atom_resampled"] += 1


def _closure_pair_log_bias(
    parents: np.ndarray,
    valid_pairs: Any,
    ring_size_log_probabilities: Any,
    *,
    strength: float,
) -> Any:
    """Convert a target ring-size marginal into multiplicity-corrected pair bias."""

    if strength < 0.0:
        raise Phase1FlowError("closure ring-size prior strength cannot be negative")
    if valid_pairs.ndim != 2 or valid_pairs.shape[0] != valid_pairs.shape[1]:
        raise Phase1FlowError("closure-pair validity mask must be square")
    log_probabilities = torch.as_tensor(
        ring_size_log_probabilities,
        dtype=torch.float32,
        device=valid_pairs.device,
    )
    if log_probabilities.ndim != 1 or log_probabilities.numel() < 4:
        raise Phase1FlowError("closure ring-size prior has invalid support")
    ring_sizes = torch.as_tensor(
        tree_pair_ring_sizes(parents),
        dtype=torch.long,
        device=valid_pairs.device,
    ).clamp(max=log_probabilities.numel() - 1)
    bias = torch.zeros_like(ring_sizes, dtype=torch.float32)
    for bucket in torch.unique(ring_sizes[valid_pairs]).tolist():
        bucket_mask = valid_pairs & (ring_sizes == int(bucket))
        pair_count = int(bucket_mask.sum().item())
        target_log_probability = log_probabilities[int(bucket)]
        if not torch.isfinite(target_log_probability):
            raise Phase1FlowError("valid closure pairs must retain nonzero ring-size prior support")
        bias[bucket_mask] = strength * (target_log_probability - float(np.log(pair_count)))
    return bias


def _degree_continuation_log_bias(
    regions: np.ndarray,
    current_degrees: np.ndarray,
    log_probabilities: Any,
    *,
    strength: float,
    device: Any,
) -> Any:
    """Return a dynamic parent-choice bias from region-specific degree survival."""

    if strength < 0.0:
        raise Phase1FlowError("degree-continuation prior strength cannot be negative")
    if regions.shape != current_degrees.shape:
        raise Phase1FlowError("degree-continuation regions and degrees must align")
    table = torch.as_tensor(log_probabilities, dtype=torch.float32, device=device)
    if table.ndim != 2 or table.shape[0] <= int(regions.max(initial=0)):
        raise Phase1FlowError("degree-continuation prior has invalid region support")
    region_index = torch.as_tensor(regions, dtype=torch.long, device=device)
    degree_index = torch.as_tensor(current_degrees, dtype=torch.long, device=device)
    degree_index = degree_index.clamp(max=table.shape[1] - 1)
    bias = table[region_index, degree_index]
    if torch.isnan(bias).any() or (bias > 0.0).any():
        raise Phase1FlowError("degree-continuation prior contains invalid log probabilities")
    return strength * bias


def sample_product_endpoints(
    model: Any,
    atom_vocabulary: Sequence[AtomState],
    node_marginal: np.ndarray,
    bond_marginal: np.ndarray,
    *,
    sample_count: int,
    sample_steps: int,
    batch_size: int,
    seed: int,
    device: str,
    node_count_distribution: np.ndarray | None = None,
    closure_count_distribution: np.ndarray | None = None,
    closure_ring_size_prior_strength: float = 0.0,
    region_marginal: np.ndarray | None = None,
    region_atom_marginal: np.ndarray | None = None,
    aromatic_cycle_sizes: Sequence[int] = (5, 6),
    aromatic_cycle_probability_threshold: float = 0.5,
    degree_continuation_log_prior: np.ndarray | None = None,
    degree_continuation_prior_strength: float = 0.0,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Sample connected endpoints using the trained count distributions and R-star flow."""

    if torch is None:
        raise Phase1FlowError("Phase 1 product generation requires torch")
    if sample_count <= 0 or sample_steps < 2 or batch_size <= 0:
        raise Phase1FlowError("generation counts and batch size must be positive")
    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device).manual_seed(seed)
    node_p0 = torch.tensor(node_marginal, dtype=torch.float32, device=resolved_device)
    bond_p0 = torch.tensor(bond_marginal, dtype=torch.float32, device=resolved_device)
    maximum_closures = int(model.closure_count_logits.shape[0] - 1)
    maximum_heavy_atoms = int(model.node_count_logits.shape[0] - 1)
    closure_ring_size_log_probabilities = (
        model.closure_ring_size_log_probabilities
        if getattr(model, "closure_ring_size_buckets", 0)
        else None
    )
    region_classes = int(getattr(model, "region_classes", 0))
    if region_classes:
        if region_marginal is None:
            raise Phase1FlowError("region-aware sampling requires a region marginal")
        if (
            region_marginal.shape != (region_classes,)
            or not np.isfinite(region_marginal).all()
            or np.any(region_marginal <= 0)
            or not np.isclose(region_marginal.sum(), 1.0)
        ):
            raise Phase1FlowError("region marginal is not a valid probability distribution")
        region_p0 = torch.tensor(
            region_marginal,
            dtype=torch.float32,
            device=resolved_device,
        )
        if (
            region_atom_marginal is None
            or region_atom_marginal.shape != (region_classes, len(atom_vocabulary))
            or not np.isfinite(region_atom_marginal).all()
            or np.any(region_atom_marginal <= 0)
            or not np.allclose(region_atom_marginal.sum(axis=1), 1.0)
        ):
            raise Phase1FlowError(
                "region-conditioned atom marginal is not a valid probability matrix"
            )
        region_atom_p0 = torch.tensor(
            region_atom_marginal,
            dtype=torch.float32,
            device=resolved_device,
        )
    else:
        if region_marginal is not None or region_atom_marginal is not None:
            raise Phase1FlowError("region-free model cannot use region marginals")
        region_p0 = None
        region_atom_p0 = None
    if node_count_distribution is not None and node_count_distribution.shape != (
        maximum_heavy_atoms + 1,
    ):
        raise Phase1FlowError("node-count calibration has the wrong support")
    if closure_count_distribution is not None and closure_count_distribution.shape != (
        maximum_closures + 1,
    ):
        raise Phase1FlowError("closure-count calibration has the wrong support")
    for label, distribution in (
        ("node-count", node_count_distribution),
        ("closure-count", closure_count_distribution),
    ):
        if distribution is not None and (
            not np.isfinite(distribution).all()
            or np.any(distribution < 0)
            or not np.isclose(distribution.sum(), 1.0)
        ):
            raise Phase1FlowError(f"{label} calibration is not a probability distribution")
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    repair_totals: Counter[str] = Counter()
    sampled_node_counts: list[int] = []
    sampled_closure_counts: list[int] = []
    start = time.perf_counter()
    model.eval()

    with torch.no_grad():
        for offset in range(0, sample_count, batch_size):
            local_batch = min(batch_size, sample_count - offset)
            counts, requested = _sample_counts(
                model,
                sample_count=local_batch,
                generator=generator,
                node_count_distribution=node_count_distribution,
                closure_count_distribution=closure_count_distribution,
            )
            counts = counts.to(resolved_device)
            requested = requested.to(resolved_device)
            sampled_node_counts.extend(int(value) for value in counts.tolist())
            sampled_closure_counts.extend(int(value) for value in requested.tolist())
            n_max = int(counts.max())
            node_mask = torch.arange(n_max, device=resolved_device)[None, :] < counts[:, None]
            child_mask = node_mask.clone()
            child_mask[:, 0] = False
            closure_mask = (
                torch.arange(maximum_closures, device=resolved_device)[None, :] < requested[:, None]
            )
            parent_candidates = _parent_candidate_mask(node_mask)
            endpoint_candidates = _endpoint_candidate_mask(node_mask, maximum_closures)

            regions = None
            if region_p0 is not None:
                regions = torch.zeros(
                    (local_batch, n_max),
                    dtype=torch.long,
                    device=resolved_device,
                )
                regions[node_mask] = torch.multinomial(
                    region_p0,
                    int(node_mask.sum()),
                    replacement=True,
                    generator=generator,
                )
            nodes = torch.zeros((local_batch, n_max), dtype=torch.long, device=resolved_device)
            if regions is None:
                nodes[node_mask] = torch.multinomial(
                    node_p0,
                    int(node_mask.sum()),
                    replacement=True,
                    generator=generator,
                )
            else:
                nodes[node_mask] = torch.multinomial(
                    region_atom_p0[regions[node_mask]],
                    1,
                    generator=generator,
                ).squeeze(1)
            parents = torch.zeros_like(nodes)
            parents[child_mask] = torch.multinomial(
                (
                    parent_candidates.to(torch.float32)
                    / parent_candidates.sum(dim=-1, keepdim=True).clamp(min=1)
                )[child_mask],
                1,
                generator=generator,
            ).squeeze(1)
            parent_bonds = torch.zeros_like(nodes)
            parent_bonds[child_mask] = torch.multinomial(
                bond_p0,
                int(child_mask.sum()),
                replacement=True,
                generator=generator,
            )
            closure_left = torch.zeros(
                (local_batch, maximum_closures),
                dtype=torch.long,
                device=resolved_device,
            )
            closure_right = torch.zeros_like(closure_left)
            closure_bonds = torch.zeros_like(closure_left)
            if closure_mask.any():
                endpoint_p0 = endpoint_candidates.to(torch.float32)
                endpoint_p0 /= endpoint_p0.sum(dim=-1, keepdim=True).clamp(min=1)
                closure_left[closure_mask] = torch.multinomial(
                    endpoint_p0[closure_mask],
                    1,
                    generator=generator,
                ).squeeze(1)
                closure_right[closure_mask] = torch.multinomial(
                    endpoint_p0[closure_mask],
                    1,
                    generator=generator,
                ).squeeze(1)
                closure_bonds[closure_mask] = torch.multinomial(
                    bond_p0,
                    int(closure_mask.sum()),
                    replacement=True,
                    generator=generator,
                )

            predictions: Mapping[str, Any] | None = None
            for step in range(sample_steps):
                t_value = step / sample_steps
                t = torch.full((local_batch,), t_value, device=resolved_device)
                predictions = model(
                    nodes,
                    parents,
                    parent_bonds,
                    closure_left,
                    closure_right,
                    closure_bonds,
                    t,
                    node_mask,
                    child_mask,
                    closure_mask,
                    regions=regions,
                )
                if step == sample_steps - 1:
                    break
                dt = 1.0 / sample_steps
                nodes = _rstar_step(
                    nodes,
                    predictions["nodes"].softmax(dim=-1),
                    (node_p0 if regions is None else region_atom_p0[regions]),
                    t_value,
                    dt,
                    node_mask,
                    generator,
                )
                if regions is not None:
                    regions = _rstar_step(
                        regions,
                        predictions["regions"].softmax(dim=-1),
                        region_p0,
                        t_value,
                        dt,
                        node_mask,
                        generator,
                    )
                parents = pointer_rstar_step(
                    parents,
                    predictions["parents"],
                    parent_candidates,
                    child_mask,
                    t_value,
                    dt,
                    generator,
                )
                parent_bonds = _rstar_step(
                    parent_bonds,
                    predictions["parent_bonds"].softmax(dim=-1),
                    bond_p0,
                    t_value,
                    dt,
                    child_mask,
                    generator,
                )
                if closure_mask.any():
                    closure_left = pointer_rstar_step(
                        closure_left,
                        predictions["closure_left"],
                        endpoint_candidates,
                        closure_mask,
                        t_value,
                        dt,
                        generator,
                    )
                    closure_right = pointer_rstar_step(
                        closure_right,
                        predictions["closure_right"],
                        endpoint_candidates,
                        closure_mask,
                        t_value,
                        dt,
                        generator,
                    )
                    closure_bonds = _rstar_step(
                        closure_bonds,
                        predictions["closure_bonds"].softmax(dim=-1),
                        bond_p0,
                        t_value,
                        dt,
                        closure_mask,
                        generator,
                    )
            if predictions is None:  # pragma: no cover - guarded by sample_steps
                raise Phase1FlowError("product sampler produced no terminal prediction")
            terminal = {key: value.detach().cpu() for key, value in predictions.items()}
            cpu_generator = torch.Generator().manual_seed(seed + offset + 1)
            for index, count in enumerate(counts.tolist()):
                node_states, edges, repairs = _construct_product_terminal_graph(
                    terminal,
                    index,
                    int(count),
                    int(requested[index]),
                    atom_vocabulary,
                    maximum_closures,
                    cpu_generator,
                    closure_ring_size_log_probabilities=(
                        closure_ring_size_log_probabilities.detach().cpu()
                        if closure_ring_size_log_probabilities is not None
                        else None
                    ),
                    closure_ring_size_prior_strength=closure_ring_size_prior_strength,
                    aromatic_cycle_sizes=aromatic_cycle_sizes,
                    aromatic_cycle_probability_threshold=(aromatic_cycle_probability_threshold),
                    degree_continuation_log_prior=degree_continuation_log_prior,
                    degree_continuation_prior_strength=(degree_continuation_prior_strength),
                )
                samples.append((node_states, edges))
                repair_totals.update(repairs)

    elapsed = time.perf_counter() - start
    fallback_repairs = sum(
        value for key, value in repair_totals.items() if "fallback" in key or "omitted" in key
    )
    terminal_decisions = sum(
        len(nodes) + int(np.count_nonzero(np.triu(edges, 1))) for nodes, edges in samples
    )
    return samples, {
        "samples": sample_count,
        "sampling_steps": sample_steps,
        "wall_seconds": elapsed,
        "graph_steps_per_second": sample_count * sample_steps / elapsed,
        "node_count_source": (
            "frozen_training_split_categorical_MLE"
            if node_count_distribution is not None
            else "learned_global_phase1_distribution"
        ),
        "closure_count_source": (
            "frozen_training_split_categorical_MLE_with_graph_bound"
            if closure_count_distribution is not None
            else "learned_global_phase1_distribution_with_graph_bound"
        ),
        "sampled_node_count_mean": float(np.mean(sampled_node_counts)),
        "sampled_node_count_minimum": min(sampled_node_counts),
        "sampled_node_count_maximum": max(sampled_node_counts),
        "sampled_closure_count_mean": float(np.mean(sampled_closure_counts)),
        "closure_ring_size_prior_strength": closure_ring_size_prior_strength,
        "degree_continuation_prior_strength": degree_continuation_prior_strength,
        "terminal_constraint_events": dict(sorted(repair_totals.items())),
        "terminal_constraint_repair_fraction": fallback_repairs / max(1, terminal_decisions),
    }
