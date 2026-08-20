"""Deterministic label-free R0 pretraining for the M0-07 graph oracle.

The objective masks complete atom feature vectors and paired directed bond
feature vectors while preserving graph topology. It consumes only the frozen,
constitutionally deduplicated R0 ledger produced by the graph-corpus gate.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from rdkit import Chem, rdBase
from torch import Tensor, nn

from forge.core.hashing import sha256_file
from forge.core.io import atomic_write as _atomic_write
from forge.potency.oracle.oracle_graph import (
    DMPNNEncoder,
    GraphBatch,
    GraphFeatureVocabulary,
    GraphTensor,
    batch_graphs,
    sum_mean_pool,
    tensorize_smiles,
)

CONFIG_SCHEMA_VERSION = "m0_07_oracle_graph_pretraining_config.v1"
RESULT_SCHEMA_VERSION = "m0_07_oracle_graph_pretraining.v1"
ALLOWED_INPUTS = frozenset({"graph_corpus_result", "r0_pretraining_ledger"})
ALLOWED_LEDGER_FIELDS = frozenset(
    {
        "graph_id",
        "constitutional_smiles",
        "atom_count",
        "undirected_bond_count",
        "directed_edge_count",
    }
)


class OracleGraphPretrainingError(ValueError):
    """Raised when label-free graph pretraining violates its frozen contract."""


@dataclass(frozen=True)
class PretrainingRecord:
    """One label-free graph and its reconstruction targets."""

    graph_id: str
    connectivity_key: str
    graph: GraphTensor
    element_targets: Tensor
    charge_targets: Tensor
    bond_targets: Tensor


@dataclass(frozen=True)
class MaskedPretrainingBatch:
    """A sparse graph batch plus deterministic corruption indices."""

    graph: GraphBatch
    node_mask: Tensor
    directed_edge_mask: Tensor
    masked_bond_forward_edges: Tensor
    element_targets: Tensor
    charge_targets: Tensor
    bond_targets: Tensor
    graph_ids: tuple[str, ...]

    def to(self, device: torch.device | str) -> MaskedPretrainingBatch:
        return MaskedPretrainingBatch(
            graph=self.graph.to(device),
            node_mask=self.node_mask.to(device),
            directed_edge_mask=self.directed_edge_mask.to(device),
            masked_bond_forward_edges=self.masked_bond_forward_edges.to(device),
            element_targets=self.element_targets.to(device),
            charge_targets=self.charge_targets.to(device),
            bond_targets=self.bond_targets.to(device),
            graph_ids=self.graph_ids,
        )


@dataclass(frozen=True)
class PretrainingLogits:
    element: Tensor
    charge: Tensor
    bond: Tensor


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise OracleGraphPretrainingError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OracleGraphPretrainingError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OracleGraphPretrainingError(f"{label} must be a JSON object")
    return payload


def _require_number(
    value: Any,
    *,
    label: str,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise OracleGraphPretrainingError(f"{label} must be a finite number")
    number = float(value)
    if lower is not None and number <= lower:
        raise OracleGraphPretrainingError(f"{label} must be greater than {lower}")
    if upper is not None and number >= upper:
        raise OracleGraphPretrainingError(f"{label} must be less than {upper}")
    return number


def load_pretraining_config(path: Path) -> dict[str, Any]:
    config = _load_json(path, label="graph pretraining config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise OracleGraphPretrainingError("unsupported graph pretraining config schema")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != ALLOWED_INPUTS:
        raise OracleGraphPretrainingError(
            "pretraining inputs must contain only the graph-corpus result and R0 ledger"
        )
    policy = config.get("data_policy")
    required_policy = {
        "biological_labels_used": False,
        "component_annotations_used": False,
        "provenance_fields_used": False,
        "virtual_candidates_used": False,
        "stereochemistry_used": False,
        "graph_truncation_allowed": False,
    }
    if policy != required_policy:
        raise OracleGraphPretrainingError("label-free pretraining data policy changed")
    split = config.get("split")
    if not isinstance(split, dict) or split.get("method") != (
        "sha256_seeded_standard_inchi_connectivity_group"
    ):
        raise OracleGraphPretrainingError("unsupported pretraining split")
    _require_number(
        split.get("validation_fraction"),
        label="validation_fraction",
        lower=0.0,
        upper=1.0,
    )
    corruption = config.get("corruption")
    if (
        not isinstance(corruption, dict)
        or corruption.get("mask_both_directed_bond_copies") is not True
        or corruption.get("preserve_topology") is not True
    ):
        raise OracleGraphPretrainingError("paired-bond topology-preserving masking is required")
    for field in ("atom_mask_fraction", "undirected_bond_mask_fraction"):
        _require_number(corruption.get(field), label=field, lower=0.0, upper=1.0)
    objective = config.get("objective")
    if not isinstance(objective, dict) or objective.get("targets") != [
        "atomic_number",
        "formal_charge",
        "bond_type",
    ]:
        raise OracleGraphPretrainingError("pretraining target contract changed")
    return config


def _verified_input(
    repo_root: Path,
    spec: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    relative = spec.get("path")
    expected = spec.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise OracleGraphPretrainingError(f"{label} input specification is incomplete")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise OracleGraphPretrainingError(f"{label} path must remain repository-relative")
    path = (repo_root / relative_path).resolve()
    if not path.is_relative_to(repo_root.resolve()) or not path.is_file():
        raise OracleGraphPretrainingError(f"{label} not found inside repository: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise OracleGraphPretrainingError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return {"path": relative, "sha256": observed, "bytes": path.stat().st_size}


def _read_ledger(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            if fields != ALLOWED_LEDGER_FIELDS:
                raise OracleGraphPretrainingError(f"R0 ledger fields changed: {sorted(fields)}")
            return [dict(row) for row in reader]
    except FileNotFoundError as exc:
        raise OracleGraphPretrainingError(f"R0 pretraining ledger not found: {path}") from exc


def standard_inchi_connectivity_key(smiles: str) -> str:
    """Return the standard-InChI connectivity block used for grouped splitting."""

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise OracleGraphPretrainingError(f"invalid pretraining SMILES: {smiles!r}")
    key = Chem.MolToInchiKey(molecule)
    if not key or len(key) < 14:
        raise OracleGraphPretrainingError(f"could not derive standard InChIKey: {smiles!r}")
    return key[:14]


def connectivity_split(
    connectivity_key: str,
    *,
    seed: int,
    validation_fraction: float,
) -> str:
    """Assign a connectivity group deterministically without corpus-order dependence."""

    payload = f"{seed}|connectivity-split|{connectivity_key}".encode()
    unit = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
    return "validation" if unit < validation_fraction else "train"


def _stable_rank(token: str) -> int:
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")


def deterministic_mask_indices(
    count: int,
    fraction: float,
    *,
    seed: int,
    epoch: int,
    graph_id: str,
    entity: str,
) -> tuple[int, ...]:
    """Select at least one entity using stable hashes, independent of batch order."""

    if count < 0:
        raise OracleGraphPretrainingError("maskable entity count cannot be negative")
    if count == 0:
        return ()
    ranked = [
        (
            _stable_rank(f"{seed}|{epoch}|{graph_id}|{entity}|{index}"),
            index,
        )
        for index in range(count)
    ]
    threshold = int(fraction * 2**64)
    selected = sorted(index for value, index in ranked if value < threshold)
    if not selected:
        selected = [min(ranked)[1]]
    return tuple(selected)


def _targets_from_graph(
    graph: GraphTensor,
    vocabulary: GraphFeatureVocabulary,
) -> tuple[Tensor, Tensor, Tensor]:
    element_stop = len(vocabulary.elements)
    charge_stop = element_stop + len(vocabulary.formal_charges)
    element = graph.node_features[:, :element_stop].argmax(dim=1).to(torch.long)
    charge = graph.node_features[:, element_stop:charge_stop].argmax(dim=1).to(torch.long)
    if graph.num_directed_edges % 2:
        raise OracleGraphPretrainingError("directed edges are not paired")
    forward_edges = torch.arange(0, graph.num_directed_edges, 2, dtype=torch.long)
    if forward_edges.numel():
        reverse = graph.reverse_edge_index[forward_edges]
        if not torch.equal(reverse, forward_edges + 1):
            raise OracleGraphPretrainingError("bond directions are not adjacent reverse pairs")
        bond = graph.edge_features[forward_edges, : len(vocabulary.bond_types)].argmax(dim=1)
    else:
        bond = torch.empty((0,), dtype=torch.long)
    return element, charge, bond.to(torch.long)


def load_pretraining_records(
    ledger_path: Path,
    vocabulary: GraphFeatureVocabulary,
) -> list[PretrainingRecord]:
    rows = _read_ledger(ledger_path)
    seen: set[str] = set()
    records: list[PretrainingRecord] = []
    for row in rows:
        graph_id = row["graph_id"]
        if not graph_id or graph_id in seen:
            raise OracleGraphPretrainingError(f"duplicate or empty graph_id: {graph_id!r}")
        seen.add(graph_id)
        graph = tensorize_smiles(
            row["constitutional_smiles"],
            vocabulary,
            label=f"R0 pretraining graph {graph_id}",
        )
        if (
            graph.num_nodes != int(row["atom_count"])
            or graph.num_directed_edges != int(row["directed_edge_count"])
            or graph.num_directed_edges // 2 != int(row["undirected_bond_count"])
        ):
            raise OracleGraphPretrainingError(f"stored graph counts changed for {graph_id}")
        element, charge, bond = _targets_from_graph(graph, vocabulary)
        records.append(
            PretrainingRecord(
                graph_id=graph_id,
                connectivity_key=standard_inchi_connectivity_key(row["constitutional_smiles"]),
                graph=graph,
                element_targets=element,
                charge_targets=charge,
                bond_targets=bond,
            )
        )
    return records


def split_pretraining_records(
    records: Sequence[PretrainingRecord],
    *,
    seed: int,
    validation_fraction: float,
) -> tuple[list[PretrainingRecord], list[PretrainingRecord]]:
    train = []
    validation = []
    assignments: dict[str, str] = {}
    for record in records:
        split = connectivity_split(
            record.connectivity_key,
            seed=seed,
            validation_fraction=validation_fraction,
        )
        previous = assignments.setdefault(record.connectivity_key, split)
        if previous != split:
            raise OracleGraphPretrainingError("connectivity group crossed pretraining splits")
        (validation if split == "validation" else train).append(record)
    if not train or not validation:
        raise OracleGraphPretrainingError("pretraining split produced an empty partition")
    return train, validation


def make_masked_batch(
    records: Sequence[PretrainingRecord],
    vocabulary: GraphFeatureVocabulary,
    *,
    seed: int,
    epoch: int,
    atom_mask_fraction: float,
    bond_mask_fraction: float,
) -> MaskedPretrainingBatch:
    if not records:
        raise OracleGraphPretrainingError("cannot mask an empty record batch")
    graph = batch_graphs([record.graph for record in records])
    node_mask = torch.zeros(graph.node_features.shape[0], dtype=torch.bool)
    directed_edge_mask = torch.zeros(graph.edge_features.shape[0], dtype=torch.bool)
    masked_bond_forward_edges: list[int] = []
    elements: list[Tensor] = []
    charges: list[Tensor] = []
    bonds: list[Tensor] = []
    node_offset = 0
    edge_offset = 0
    for record in records:
        node_indices = deterministic_mask_indices(
            record.graph.num_nodes,
            atom_mask_fraction,
            seed=seed,
            epoch=epoch,
            graph_id=record.graph_id,
            entity="atom",
        )
        local_nodes = torch.tensor(node_indices, dtype=torch.long)
        node_mask[local_nodes + node_offset] = True
        elements.append(record.element_targets[local_nodes])
        charges.append(record.charge_targets[local_nodes])

        undirected_bonds = record.graph.num_directed_edges // 2
        bond_indices = deterministic_mask_indices(
            undirected_bonds,
            bond_mask_fraction,
            seed=seed,
            epoch=epoch,
            graph_id=record.graph_id,
            entity="bond",
        )
        local_bonds = torch.tensor(bond_indices, dtype=torch.long)
        if local_bonds.numel():
            forward = edge_offset + 2 * local_bonds
            reverse = forward + 1
            directed_edge_mask[forward] = True
            directed_edge_mask[reverse] = True
            masked_bond_forward_edges.extend(forward.tolist())
            bonds.append(record.bond_targets[local_bonds])
        node_offset += record.graph.num_nodes
        edge_offset += record.graph.num_directed_edges
    if not masked_bond_forward_edges:
        raise OracleGraphPretrainingError("record batch produced no masked bonds")
    return MaskedPretrainingBatch(
        graph=graph,
        node_mask=node_mask,
        directed_edge_mask=directed_edge_mask,
        masked_bond_forward_edges=torch.tensor(masked_bond_forward_edges, dtype=torch.long),
        element_targets=torch.cat(elements),
        charge_targets=torch.cat(charges),
        bond_targets=torch.cat(bonds),
        graph_ids=tuple(record.graph_id for record in records),
    )


def iter_record_batches(
    records: Sequence[PretrainingRecord],
    *,
    seed: int,
    epoch: int,
    maximum_atoms: int,
    maximum_directed_edges: int,
) -> Iterable[list[PretrainingRecord]]:
    """Yield deterministic, bounded sparse batches without dropping large graphs."""

    if maximum_atoms < 1 or maximum_directed_edges < 1:
        raise OracleGraphPretrainingError("batch budgets must be positive")
    ordered = sorted(
        records,
        key=lambda record: _stable_rank(f"{seed}|{epoch}|batch|{record.graph_id}"),
    )
    batch: list[PretrainingRecord] = []
    atoms = 0
    edges = 0
    for record in ordered:
        record_atoms = record.graph.num_nodes
        record_edges = record.graph.num_directed_edges
        if record_atoms > maximum_atoms or record_edges > maximum_directed_edges:
            raise OracleGraphPretrainingError(
                f"batch budget would truncate graph {record.graph_id}"
            )
        if batch and (
            atoms + record_atoms > maximum_atoms or edges + record_edges > maximum_directed_edges
        ):
            yield batch
            batch = []
            atoms = 0
            edges = 0
        batch.append(record)
        atoms += record_atoms
        edges += record_edges
    if batch:
        yield batch


class MaskedFeatureDMPNN(nn.Module):
    """D-MPNN with label-free atom, charge, and paired-bond reconstruction."""

    def __init__(
        self,
        vocabulary: GraphFeatureVocabulary,
        *,
        hidden_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.encoder = DMPNNEncoder(
            vocabulary.atom_feature_dim,
            vocabulary.bond_feature_dim,
            hidden_dim,
            depth,
            dropout,
        )
        self.atom_mask_token = nn.Parameter(torch.zeros(vocabulary.atom_feature_dim))
        self.bond_mask_token = nn.Parameter(torch.zeros(vocabulary.bond_feature_dim))
        self.element_head = nn.Linear(2 * hidden_dim, len(vocabulary.elements))
        self.charge_head = nn.Linear(2 * hidden_dim, len(vocabulary.formal_charges))
        self.bond_head = nn.Linear(2 * hidden_dim, len(vocabulary.bond_types))

    def _states(self, batch: GraphBatch) -> tuple[Tensor, Tensor, Tensor]:
        source, destination = batch.edge_index
        initial = torch.relu(
            self.encoder.edge_input(
                torch.cat((batch.node_features[source], batch.edge_features), dim=-1)
            )
        )
        messages = initial
        for _ in range(self.encoder.depth - 1):
            incoming = messages.new_zeros((batch.node_features.shape[0], self.encoder.hidden_dim))
            incoming.index_add_(0, destination, messages)
            nonbacktracking = incoming[source] - messages[batch.reverse_edge_index]
            messages = torch.relu(initial + self.encoder.message_update(nonbacktracking))
            messages = self.encoder.dropout(messages)
        atom_messages = messages.new_zeros((batch.node_features.shape[0], self.encoder.hidden_dim))
        atom_messages.index_add_(0, destination, messages)
        atom_states = torch.relu(
            self.encoder.atom_output(torch.cat((batch.node_features, atom_messages), dim=-1))
        )
        atom_states = self.encoder.dropout(atom_states)
        graph_states = self.encoder.readout(
            sum_mean_pool(atom_states, batch.graph_index, batch.num_graphs)
        )
        return atom_states, messages, graph_states

    def forward(self, batch: MaskedPretrainingBatch) -> PretrainingLogits:
        node_features = torch.where(
            batch.node_mask.unsqueeze(-1),
            self.atom_mask_token.unsqueeze(0),
            batch.graph.node_features,
        )
        edge_features = torch.where(
            batch.directed_edge_mask.unsqueeze(-1),
            self.bond_mask_token.unsqueeze(0),
            batch.graph.edge_features,
        )
        corrupted = GraphBatch(
            node_features=node_features,
            edge_index=batch.graph.edge_index,
            edge_features=edge_features,
            reverse_edge_index=batch.graph.reverse_edge_index,
            graph_index=batch.graph.graph_index,
            graph_ptr=batch.graph.graph_ptr,
        )
        atom_states, messages, graph_states = self._states(corrupted)
        masked_atoms = batch.node_mask.nonzero(as_tuple=False).flatten()
        atom_context = torch.cat(
            (
                atom_states[masked_atoms],
                graph_states[corrupted.graph_index[masked_atoms]],
            ),
            dim=-1,
        )
        forward = batch.masked_bond_forward_edges
        reverse = corrupted.reverse_edge_index[forward]
        symmetric_bonds = 0.5 * (messages[forward] + messages[reverse])
        bond_graphs = corrupted.graph_index[corrupted.edge_index[0, forward]]
        bond_context = torch.cat((symmetric_bonds, graph_states[bond_graphs]), dim=-1)
        return PretrainingLogits(
            element=self.element_head(atom_context),
            charge=self.charge_head(atom_context),
            bond=self.bond_head(bond_context),
        )


def class_weights(
    records: Sequence[PretrainingRecord],
    *,
    classes: int,
    target: str,
    maximum_weight: float,
) -> Tensor:
    if target == "element":
        values = torch.cat([record.element_targets for record in records])
    elif target == "charge":
        values = torch.cat([record.charge_targets for record in records])
    elif target == "bond":
        values = torch.cat([record.bond_targets for record in records])
    else:
        raise OracleGraphPretrainingError(f"unsupported target {target!r}")
    counts = torch.bincount(values, minlength=classes).to(torch.float64)
    if torch.any(counts == 0):
        raise OracleGraphPretrainingError(f"training split lacks a {target} class")
    weights = counts.rsqrt()
    weights /= weights.mean()
    return weights.clamp(max=maximum_weight).to(torch.float32)


def reconstruction_loss(
    logits: PretrainingLogits,
    batch: MaskedPretrainingBatch,
    weights: Mapping[str, Tensor],
) -> tuple[Tensor, dict[str, float]]:
    losses = {
        "element": functional.cross_entropy(
            logits.element,
            batch.element_targets,
            weight=weights["element"],
        ),
        "charge": functional.cross_entropy(
            logits.charge,
            batch.charge_targets,
            weight=weights["charge"],
        ),
        "bond": functional.cross_entropy(
            logits.bond,
            batch.bond_targets,
            weight=weights["bond"],
        ),
    }
    total = torch.stack(tuple(losses.values())).mean()
    return total, {name: float(loss.detach()) for name, loss in losses.items()}


def _balanced_accuracy(predictions: Tensor, targets: Tensor, classes: int) -> float:
    recalls = []
    for index in range(classes):
        selected = targets == index
        if selected.any():
            recalls.append(float((predictions[selected] == index).to(torch.float32).mean()))
    return sum(recalls) / len(recalls) if recalls else float("nan")


def _size_bin(record: PretrainingRecord) -> str:
    if record.graph.num_nodes <= 64:
        return "at_most_64"
    if record.graph.num_nodes <= 96:
        return "65_to_96"
    return "above_96"


def _evaluate(
    model: MaskedFeatureDMPNN,
    records: Sequence[PretrainingRecord],
    vocabulary: GraphFeatureVocabulary,
    config: Mapping[str, Any],
    weights: Mapping[str, Tensor],
    *,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    totals = Counter()
    loss_sums = Counter()
    prediction_rows: dict[str, list[Tensor]] = {
        "element_predictions": [],
        "element_targets": [],
        "charge_predictions": [],
        "charge_targets": [],
        "bond_predictions": [],
        "bond_targets": [],
    }
    optimization = config["optimization"]
    corruption = config["corruption"]
    with torch.no_grad():
        for record_batch in iter_record_batches(
            records,
            seed=seed,
            epoch=-1,
            maximum_atoms=optimization["maximum_atoms_per_batch"],
            maximum_directed_edges=optimization["maximum_directed_edges_per_batch"],
        ):
            masked = make_masked_batch(
                record_batch,
                vocabulary,
                seed=seed,
                epoch=-1,
                atom_mask_fraction=corruption["atom_mask_fraction"],
                bond_mask_fraction=corruption["undirected_bond_mask_fraction"],
            )
            logits = model(masked)
            total_loss, component_losses = reconstruction_loss(logits, masked, weights)
            totals["batches"] += 1
            totals["graphs"] += len(record_batch)
            totals["masked_atoms"] += masked.element_targets.numel()
            totals["masked_bonds"] += masked.bond_targets.numel()
            loss_sums["total"] += float(total_loss)
            for name, value in component_losses.items():
                loss_sums[name] += value
            prediction_rows["element_predictions"].append(logits.element.argmax(dim=1))
            prediction_rows["element_targets"].append(masked.element_targets)
            prediction_rows["charge_predictions"].append(logits.charge.argmax(dim=1))
            prediction_rows["charge_targets"].append(masked.charge_targets)
            prediction_rows["bond_predictions"].append(logits.bond.argmax(dim=1))
            prediction_rows["bond_targets"].append(masked.bond_targets)
    batches = totals["batches"]
    metrics = {
        "loss": {name: value / batches for name, value in sorted(loss_sums.items())},
        "counts": dict(sorted(totals.items())),
    }
    for target, classes in (
        ("element", len(vocabulary.elements)),
        ("charge", len(vocabulary.formal_charges)),
        ("bond", len(vocabulary.bond_types)),
    ):
        predictions = torch.cat(prediction_rows[f"{target}_predictions"])
        targets = torch.cat(prediction_rows[f"{target}_targets"])
        metrics[target] = {
            "accuracy": float((predictions == targets).to(torch.float32).mean()),
            "balanced_accuracy": _balanced_accuracy(predictions, targets, classes),
        }
    size_metrics = {}
    for size_name in ("at_most_64", "65_to_96", "above_96"):
        selected = [record for record in records if _size_bin(record) == size_name]
        if not selected:
            continue
        selected_loss = 0.0
        selected_batches = 0
        with torch.no_grad():
            for record_batch in iter_record_batches(
                selected,
                seed=seed,
                epoch=-1,
                maximum_atoms=optimization["maximum_atoms_per_batch"],
                maximum_directed_edges=optimization["maximum_directed_edges_per_batch"],
            ):
                masked = make_masked_batch(
                    record_batch,
                    vocabulary,
                    seed=seed,
                    epoch=-1,
                    atom_mask_fraction=corruption["atom_mask_fraction"],
                    bond_mask_fraction=corruption["undirected_bond_mask_fraction"],
                )
                logits = model(masked)
                loss, _ = reconstruction_loss(logits, masked, weights)
                selected_loss += float(loss)
                selected_batches += 1
        size_metrics[size_name] = {
            "graphs": len(selected),
            "mean_batch_loss": selected_loss / selected_batches,
        }
    metrics["size_stratified"] = size_metrics
    return metrics


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    def command(*parts: str) -> str:
        completed = subprocess.run(
            ("git", *parts),
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
        return completed.stdout

    commit = command("rev-parse", "HEAD").strip()
    status = command("status", "--porcelain=v1")
    diff = command("diff", "--binary", "HEAD")
    return {
        "commit": commit,
        "worktree_clean": not bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
    }


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run_graph_pretraining(
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
    *,
    profile_only: bool = True,
    profile_maximum_graphs: int | None = None,
) -> dict[str, Any]:
    """Run a bounded CPU profile or the explicitly requested full pretraining."""

    run_started = time.perf_counter()
    config = load_pretraining_config(config_path)
    config_relative = config_path.resolve().relative_to(repo_root.resolve()).as_posix()
    verified = {
        "config": {
            "path": config_relative,
            "sha256": sha256_file(config_path),
            "bytes": config_path.stat().st_size,
        }
    }
    for label, spec in sorted(config["inputs"].items()):
        verified[label] = _verified_input(repo_root, spec, label=label)
    corpus_result = _load_json(
        repo_root / verified["graph_corpus_result"]["path"],
        label="graph-corpus result",
    )
    expected = config["expected"]
    for field in (
        "post_exclusion_oracle_constitution_overlap",
        "post_exclusion_oracle_inchi_connectivity_overlap",
    ):
        if corpus_result.get("summary", {}).get(field) != expected[field]:
            raise OracleGraphPretrainingError(f"graph-corpus leakage gate failed: {field}")
    vocabulary = GraphFeatureVocabulary.from_corpus_result(
        repo_root / verified["graph_corpus_result"]["path"]
    )
    records = load_pretraining_records(
        repo_root / verified["r0_pretraining_ledger"]["path"],
        vocabulary,
    )
    if len(records) != expected["graphs"]:
        raise OracleGraphPretrainingError(
            f"expected {expected['graphs']} pretraining graphs, observed {len(records)}"
        )
    if sum(record.graph.num_nodes for record in records) != expected["atoms"]:
        raise OracleGraphPretrainingError("pretraining atom count changed")
    if sum(record.graph.num_directed_edges for record in records) != expected["directed_edges"]:
        raise OracleGraphPretrainingError("pretraining directed-edge count changed")
    if max(record.graph.num_nodes for record in records) != expected["maximum_atoms"]:
        raise OracleGraphPretrainingError("pretraining maximum graph size changed")

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config["optimization"]["cpu_threads"]))
    torch.use_deterministic_algorithms(True)
    train, validation = split_pretraining_records(
        records,
        seed=seed,
        validation_fraction=float(config["split"]["validation_fraction"]),
    )
    data_preparation_seconds = time.perf_counter() - run_started
    full_counts = {"train": len(train), "validation": len(validation)}
    weight_records = train
    if profile_only:
        maximum_train = profile_maximum_graphs or int(config["profile"]["maximum_train_graphs"])
        maximum_validation = min(
            profile_maximum_graphs or int(config["profile"]["maximum_validation_graphs"]),
            len(validation),
        )
        train = sorted(train, key=lambda record: record.graph_id)[:maximum_train]
        validation = sorted(validation, key=lambda record: record.graph_id)[:maximum_validation]
    architecture = config["architecture"]
    model = MaskedFeatureDMPNN(
        vocabulary,
        hidden_dim=int(architecture["hidden_dim"]),
        depth=int(architecture["depth"]),
        dropout=float(architecture["dropout"]),
    )
    maximum_weight = float(config["objective"]["maximum_class_weight"])
    weights = {
        "element": class_weights(
            weight_records,
            classes=len(vocabulary.elements),
            target="element",
            maximum_weight=maximum_weight,
        ),
        "charge": class_weights(
            weight_records,
            classes=len(vocabulary.formal_charges),
            target="charge",
            maximum_weight=maximum_weight,
        ),
        "bond": class_weights(
            weight_records,
            classes=len(vocabulary.bond_types),
            target="bond",
            maximum_weight=maximum_weight,
        ),
    }
    optimization = config["optimization"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    epochs = int(config["profile"]["epochs"] if profile_only else optimization["epochs"])
    patience = int(optimization["early_stopping_patience"])
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    training_history = []
    optimization_started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        graphs_seen = 0
        for record_batch in iter_record_batches(
            train,
            seed=seed,
            epoch=epoch,
            maximum_atoms=int(optimization["maximum_atoms_per_batch"]),
            maximum_directed_edges=int(optimization["maximum_directed_edges_per_batch"]),
        ):
            masked = make_masked_batch(
                record_batch,
                vocabulary,
                seed=seed,
                epoch=epoch,
                atom_mask_fraction=float(config["corruption"]["atom_mask_fraction"]),
                bond_mask_fraction=float(config["corruption"]["undirected_bond_mask_fraction"]),
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(masked)
            loss, _ = reconstruction_loss(logits, masked, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(optimization["gradient_clip_norm"]),
            )
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
            graphs_seen += len(record_batch)
        validation_metrics = _evaluate(
            model,
            validation,
            vocabulary,
            config,
            weights,
            seed=seed,
        )
        validation_loss = float(validation_metrics["loss"]["total"])
        training_history.append(
            {
                "epoch": epoch,
                "mean_training_loss": epoch_loss / batches,
                "training_batches": batches,
                "training_graphs_seen": graphs_seen,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
        elif not profile_only and epoch - best_epoch >= patience:
            break
    optimization_seconds = time.perf_counter() - optimization_started
    if best_state is None:
        raise OracleGraphPretrainingError("pretraining produced no checkpoint")
    model.load_state_dict(best_state)
    final_metrics = _evaluate(
        model,
        validation,
        vocabulary,
        config,
        weights,
        seed=seed,
    )

    mode = "profile" if profile_only else "full"
    checkpoint_name = f"oracle_graph_pretraining_{mode}_checkpoint.pt"
    checkpoint_path = output_dir / checkpoint_name
    checkpoint_payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "mode": mode,
        "inputs": {
            name: {"path": metadata["path"], "sha256": metadata["sha256"]}
            for name, metadata in sorted(verified.items())
        },
        "seed": seed,
        "encoder_state_dict": model.encoder.state_dict(),
        "pretraining_model_state_dict": model.state_dict(),
        "feature_vocabulary": vocabulary.to_dict(),
        "architecture": architecture,
        "corruption": config["corruption"],
        "objective": config["objective"],
        "data_policy": config["data_policy"],
        "best_epoch": best_epoch,
    }
    _atomic_torch_save(checkpoint_path, checkpoint_payload)
    checkpoint_metadata = {
        "path": checkpoint_name,
        "sha256": sha256_file(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "downstream_use_authorized": not profile_only,
    }
    source_files = (
        "src/forge/potency/oracle_graph.py",
        "src/forge/potency/oracle_graph_pretraining.py",
        "scripts/m0_07_oracle_graph_pretraining.py",
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": (
            "completed_cpu_profile" if profile_only else "completed_label_free_r0_pretraining"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "inputs": verified,
        "source_files": {
            relative: {
                "sha256": sha256_file(repo_root / relative),
                "bytes": (repo_root / relative).stat().st_size,
            }
            for relative in source_files
        },
        "git": _git_provenance(repo_root),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "randomness": {
            "global_seed": seed,
            "split": "sha256(global_seed, connectivity_key)",
            "training_masks": "sha256(global_seed, epoch, graph_id, entity_index)",
            "validation_masks": "sha256(global_seed, -1, graph_id, entity_index)",
            "deterministic_algorithms": True,
        },
        "data_policy": config["data_policy"],
        "feature_vocabulary": vocabulary.to_dict(),
        "architecture": architecture,
        "corruption": config["corruption"],
        "objective": config["objective"],
        "optimization": optimization,
        "partition_counts_before_profile_limit": full_counts,
        "trained_partition_counts": {
            "train": len(train),
            "validation": len(validation),
        },
        "best_epoch": best_epoch,
        "timing_seconds": {
            "data_preparation_and_tensorization": data_preparation_seconds,
            "optimization_and_validation": optimization_seconds,
            "total_before_artifact_serialization": time.perf_counter() - run_started,
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_history": training_history,
        "validation": final_metrics,
        "checkpoint": checkpoint_metadata,
        "decision": {
            "profile_checkpoint_is_not_a_frozen_encoder": profile_only,
            "full_checkpoint_selected_without_agile_labels": not profile_only,
            "all_source_graphs_loaded_without_truncation": (
                sum(full_counts.values()) == expected["graphs"]
            ),
            "profile_subset_is_intentional": profile_only,
        },
    }
    result_name = f"oracle_graph_pretraining_{mode}_result.json"
    _atomic_write(
        output_dir / result_name,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    return result
