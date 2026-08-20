"""Sparse, pure-PyTorch graph representations for the M0-07 oracle matrix.

The module deliberately keeps molecular identity and split metadata outside the
model tensors. Feature support is read from the frozen graph-corpus audit rather
than from a hand-written or stale manifest.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from rdkit import Chem, rdBase
from torch import Tensor, nn


class OracleGraphError(ValueError):
    """Raised when an oracle graph violates the frozen representation."""


def _one_hot(value: str, vocabulary: tuple[str, ...], label: str) -> list[float]:
    try:
        index = vocabulary.index(value)
    except ValueError as exc:
        raise OracleGraphError(
            f"unsupported {label} {value!r}; allowed values are {vocabulary}"
        ) from exc
    output = [0.0] * len(vocabulary)
    output[index] = 1.0
    return output


def _numeric_order(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(values, key=int))


@dataclass(frozen=True)
class GraphFeatureVocabulary:
    """Categorical feature support derived from the frozen R0 graph audit."""

    elements: tuple[str, ...]
    formal_charges: tuple[str, ...]
    total_degrees: tuple[str, ...]
    total_hydrogens: tuple[str, ...]
    hybridizations: tuple[str, ...]
    bond_types: tuple[str, ...]

    @classmethod
    def from_corpus_result(cls, path: Path) -> GraphFeatureVocabulary:
        try:
            result = json.loads(path.read_text())
        except FileNotFoundError as exc:
            raise OracleGraphError(f"graph-corpus result not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise OracleGraphError(f"invalid graph-corpus result: {path}") from exc
        profile = (
            result.get("graph_profiles", {}).get("r0_pretraining", {}).get("feature_vocabulary")
        )
        if not isinstance(profile, dict):
            raise OracleGraphError("graph-corpus result lacks the R0 feature vocabulary")
        expected_bond_order = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
        observed_bonds = set(profile.get("bond_types", {}))
        if observed_bonds != set(expected_bond_order):
            raise OracleGraphError(f"unexpected R0 bond vocabulary: {sorted(observed_bonds)}")
        vocabulary = cls(
            elements=tuple(sorted(profile.get("elements", {}))),
            formal_charges=_numeric_order(tuple(profile.get("formal_charges", {}))),
            total_degrees=_numeric_order(tuple(profile.get("total_degrees", {}))),
            total_hydrogens=_numeric_order(tuple(profile.get("total_hydrogens", {}))),
            hybridizations=tuple(sorted(profile.get("hybridizations", {}))),
            bond_types=expected_bond_order,
        )
        if not all(
            (
                vocabulary.elements,
                vocabulary.formal_charges,
                vocabulary.total_degrees,
                vocabulary.total_hydrogens,
                vocabulary.hybridizations,
                vocabulary.bond_types,
            )
        ):
            raise OracleGraphError("graph feature vocabulary contains an empty category")
        return vocabulary

    @property
    def atom_feature_dim(self) -> int:
        return (
            len(self.elements)
            + len(self.formal_charges)
            + len(self.total_degrees)
            + len(self.total_hydrogens)
            + len(self.hybridizations)
            + 2
        )

    @property
    def bond_feature_dim(self) -> int:
        return len(self.bond_types) + 2

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "elements": list(self.elements),
            "formal_charges": list(self.formal_charges),
            "total_degrees": list(self.total_degrees),
            "total_hydrogens": list(self.total_hydrogens),
            "hybridizations": list(self.hybridizations),
            "bond_types": list(self.bond_types),
        }


@dataclass(frozen=True)
class GraphTensor:
    """One unpadded molecular graph with directed edges."""

    node_features: Tensor
    edge_index: Tensor
    edge_features: Tensor
    reverse_edge_index: Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def num_directed_edges(self) -> int:
        return int(self.edge_features.shape[0])


@dataclass(frozen=True)
class GraphBatch:
    """A sparse concatenation of molecular graphs."""

    node_features: Tensor
    edge_index: Tensor
    edge_features: Tensor
    reverse_edge_index: Tensor
    graph_index: Tensor
    graph_ptr: Tensor

    @property
    def num_graphs(self) -> int:
        return int(self.graph_ptr.numel() - 1)

    def to(self, device: torch.device | str) -> GraphBatch:
        return GraphBatch(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            reverse_edge_index=self.reverse_edge_index.to(device),
            graph_index=self.graph_index.to(device),
            graph_ptr=self.graph_ptr.to(device),
        )


@dataclass(frozen=True)
class OracleGraphRecord:
    """Model-ready graphs with labels retained only for lookup and reporting."""

    label: str
    product: GraphTensor
    amine: GraphTensor
    aldehyde: GraphTensor
    isocyanide: GraphTensor
    targets: Tensor


@dataclass(frozen=True)
class GraphModelInput:
    """The complete tensor interface accepted by graph model forward calls."""

    product: GraphBatch
    amine: GraphBatch
    aldehyde: GraphBatch
    isocyanide: GraphBatch

    def to(self, device: torch.device | str) -> GraphModelInput:
        return GraphModelInput(
            product=self.product.to(device),
            amine=self.amine.to(device),
            aldehyde=self.aldehyde.to(device),
            isocyanide=self.isocyanide.to(device),
        )


@dataclass(frozen=True)
class SupervisedGraphBatch:
    """Model inputs and endpoint targets without audit identifiers."""

    inputs: GraphModelInput
    targets: Tensor

    def to(self, device: torch.device | str) -> SupervisedGraphBatch:
        return SupervisedGraphBatch(
            inputs=self.inputs.to(device),
            targets=self.targets.to(device),
        )


@dataclass(frozen=True)
class BatchMetadata:
    """Audit identifiers that never enter a model forward call."""

    labels: tuple[str, ...]


def _atom_features(
    atom: Chem.Atom,
    vocabulary: GraphFeatureVocabulary,
) -> list[float]:
    output = []
    output.extend(_one_hot(atom.GetSymbol(), vocabulary.elements, "element"))
    output.extend(_one_hot(str(atom.GetFormalCharge()), vocabulary.formal_charges, "formal charge"))
    output.extend(_one_hot(str(atom.GetTotalDegree()), vocabulary.total_degrees, "total degree"))
    output.extend(
        _one_hot(
            str(atom.GetTotalNumHs()),
            vocabulary.total_hydrogens,
            "total hydrogens",
        )
    )
    output.extend(
        _one_hot(str(atom.GetHybridization()), vocabulary.hybridizations, "hybridization")
    )
    output.extend((float(atom.GetIsAromatic()), float(atom.IsInRing())))
    return output


def _bond_features(
    bond: Chem.Bond,
    vocabulary: GraphFeatureVocabulary,
) -> list[float]:
    output = _one_hot(str(bond.GetBondType()), vocabulary.bond_types, "bond type")
    output.extend((float(bond.GetIsConjugated()), float(bond.IsInRing())))
    return output


def tensorize_molecule(
    molecule: Chem.Mol,
    vocabulary: GraphFeatureVocabulary,
) -> GraphTensor:
    """Convert a connected RDKit molecule without padding or truncation."""

    if molecule.GetNumAtoms() == 0:
        raise OracleGraphError("cannot tensorize an empty molecule")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise OracleGraphError("cannot tensorize a disconnected molecule")
    node_features = torch.tensor(
        [_atom_features(atom, vocabulary) for atom in molecule.GetAtoms()],
        dtype=torch.float32,
    )
    directed_sources: list[int] = []
    directed_destinations: list[int] = []
    edge_features: list[list[float]] = []
    reverse_indices: list[int] = []
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = _bond_features(bond, vocabulary)
        forward = len(edge_features)
        reverse = forward + 1
        directed_sources.extend((begin, end))
        directed_destinations.extend((end, begin))
        edge_features.extend((features, features))
        reverse_indices.extend((reverse, forward))
    if edge_features:
        edge_index = torch.tensor(
            [directed_sources, directed_destinations],
            dtype=torch.long,
        )
        edge_feature_tensor = torch.tensor(edge_features, dtype=torch.float32)
        reverse_edge_index = torch.tensor(reverse_indices, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_feature_tensor = torch.empty(
            (0, vocabulary.bond_feature_dim),
            dtype=torch.float32,
        )
        reverse_edge_index = torch.empty((0,), dtype=torch.long)
    return GraphTensor(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_feature_tensor,
        reverse_edge_index=reverse_edge_index,
    )


def tensorize_smiles(
    smiles: str,
    vocabulary: GraphFeatureVocabulary,
    *,
    label: str,
) -> GraphTensor:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise OracleGraphError(f"{label} contains invalid SMILES: {smiles!r}")
    return tensorize_molecule(molecule, vocabulary)


def batch_graphs(graphs: Sequence[GraphTensor]) -> GraphBatch:
    """Concatenate graphs while retaining reverse-edge and graph membership maps."""

    if not graphs:
        raise OracleGraphError("cannot batch zero graphs")
    nodes = []
    edges = []
    edge_features = []
    reverse_edges = []
    graph_indices = []
    graph_ptr = [0]
    node_offset = 0
    edge_offset = 0
    for graph_index, graph in enumerate(graphs):
        nodes.append(graph.node_features)
        edges.append(graph.edge_index + node_offset)
        edge_features.append(graph.edge_features)
        reverse_edges.append(graph.reverse_edge_index + edge_offset)
        graph_indices.append(torch.full((graph.num_nodes,), graph_index, dtype=torch.long))
        node_offset += graph.num_nodes
        edge_offset += graph.num_directed_edges
        graph_ptr.append(node_offset)
    return GraphBatch(
        node_features=torch.cat(nodes, dim=0),
        edge_index=torch.cat(edges, dim=1),
        edge_features=torch.cat(edge_features, dim=0),
        reverse_edge_index=torch.cat(reverse_edges, dim=0),
        graph_index=torch.cat(graph_indices, dim=0),
        graph_ptr=torch.tensor(graph_ptr, dtype=torch.long),
    )


def collate_oracle_records(
    records: Sequence[OracleGraphRecord],
) -> tuple[SupervisedGraphBatch, BatchMetadata]:
    if not records:
        raise OracleGraphError("cannot collate zero oracle records")
    return (
        SupervisedGraphBatch(
            inputs=GraphModelInput(
                product=batch_graphs([record.product for record in records]),
                amine=batch_graphs([record.amine for record in records]),
                aldehyde=batch_graphs([record.aldehyde for record in records]),
                isocyanide=batch_graphs([record.isocyanide for record in records]),
            ),
            targets=torch.stack([record.targets for record in records]),
        ),
        BatchMetadata(labels=tuple(record.label for record in records)),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    try:
        with opener(path, "rt", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except FileNotFoundError as exc:
        raise OracleGraphError(f"graph input not found: {path}") from exc


def load_oracle_graph_records(
    path: Path,
    vocabulary: GraphFeatureVocabulary,
    *,
    allowed_labels: set[str] | None = None,
) -> list[OracleGraphRecord]:
    """Load reconciled AGILE graphs and both endpoint targets."""

    rows = _read_csv(path)
    required = {
        "label",
        "model_smiles",
        "A_smiles",
        "B_smiles",
        "C_smiles",
        "expt_Hela",
        "expt_Raw",
    }
    if rows and not required <= set(rows[0]):
        raise OracleGraphError(
            f"curated oracle data is missing fields: {sorted(required - set(rows[0]))}"
        )
    labels = [row["label"] for row in rows]
    if len(set(labels)) != len(labels):
        raise OracleGraphError("curated oracle labels are not unique")
    output = []
    for row in rows:
        label = row["label"]
        if allowed_labels is not None and label not in allowed_labels:
            continue
        output.append(
            OracleGraphRecord(
                label=label,
                product=tensorize_smiles(row["model_smiles"], vocabulary, label=f"{label} product"),
                amine=tensorize_smiles(row["A_smiles"], vocabulary, label=f"{label} amine"),
                aldehyde=tensorize_smiles(row["B_smiles"], vocabulary, label=f"{label} aldehyde"),
                isocyanide=tensorize_smiles(
                    row["C_smiles"], vocabulary, label=f"{label} isocyanide"
                ),
                targets=torch.tensor(
                    [float(row["expt_Hela"]), float(row["expt_Raw"])],
                    dtype=torch.float32,
                ),
            )
        )
    if allowed_labels is not None and {record.label for record in output} != allowed_labels:
        missing = sorted(allowed_labels - {record.label for record in output})
        raise OracleGraphError(f"requested oracle labels are missing: {missing[:5]}")
    return output


def permute_graph_tensor(
    graph: GraphTensor,
    node_order: Tensor,
    edge_order: Tensor | None = None,
) -> GraphTensor:
    """Apply valid node and directed-edge permutations for invariance tests."""

    if (
        node_order.dtype != torch.long
        or node_order.shape != (graph.num_nodes,)
        or set(node_order.tolist()) != set(range(graph.num_nodes))
    ):
        raise OracleGraphError("node_order is not a complete node permutation")
    if edge_order is None:
        edge_order = torch.arange(graph.num_directed_edges, dtype=torch.long)
    if (
        edge_order.dtype != torch.long
        or edge_order.shape != (graph.num_directed_edges,)
        or set(edge_order.tolist()) != set(range(graph.num_directed_edges))
    ):
        raise OracleGraphError("edge_order is not a complete edge permutation")
    inverse_nodes = torch.empty_like(node_order)
    inverse_nodes[node_order] = torch.arange(graph.num_nodes, dtype=torch.long)
    inverse_edges = torch.empty_like(edge_order)
    inverse_edges[edge_order] = torch.arange(
        graph.num_directed_edges,
        dtype=torch.long,
    )
    remapped_edges = inverse_nodes[graph.edge_index[:, edge_order]]
    remapped_reverse = inverse_edges[graph.reverse_edge_index[edge_order]]
    return GraphTensor(
        node_features=graph.node_features[node_order],
        edge_index=remapped_edges,
        edge_features=graph.edge_features[edge_order],
        reverse_edge_index=remapped_reverse,
    )


def tensorize_structure_file(
    path: Path,
    *,
    smiles_field: str,
    vocabulary: GraphFeatureVocabulary,
    label_field: str,
) -> list[GraphTensor]:
    """Tensorize an unlabeled structure ledger for acceptance profiling."""

    rows = _read_csv(path)
    if rows and {smiles_field, label_field} - set(rows[0]):
        raise OracleGraphError(
            f"{path} is missing fields: {sorted({smiles_field, label_field} - set(rows[0]))}"
        )
    return [
        tensorize_smiles(
            row[smiles_field],
            vocabulary,
            label=f"{path.name} {row[label_field]}",
        )
        for row in rows
    ]


def sum_mean_pool(
    node_states: Tensor,
    graph_index: Tensor,
    num_graphs: int,
) -> Tensor:
    summed = node_states.new_zeros((num_graphs, node_states.shape[-1]))
    summed.index_add_(0, graph_index, node_states)
    counts = torch.bincount(graph_index, minlength=num_graphs).to(node_states.dtype)
    means = summed / counts.clamp_min(1).unsqueeze(-1)
    return torch.cat((summed, means), dim=-1)


class DMPNNEncoder(nn.Module):
    """Directed message-passing encoder with reverse-edge exclusion."""

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise OracleGraphError("D-MPNN depth must be positive")
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.edge_input = nn.Linear(atom_dim + bond_dim, hidden_dim)
        self.message_update = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.atom_output = nn.Linear(atom_dim + hidden_dim, hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch: GraphBatch) -> Tensor:
        source, destination = batch.edge_index
        initial = torch.relu(
            self.edge_input(
                torch.cat(
                    (batch.node_features[source], batch.edge_features),
                    dim=-1,
                )
            )
        )
        messages = initial
        for _ in range(self.depth - 1):
            incoming = messages.new_zeros((batch.node_features.shape[0], self.hidden_dim))
            incoming.index_add_(0, destination, messages)
            nonbacktracking = incoming[source] - messages[batch.reverse_edge_index]
            messages = torch.relu(initial + self.message_update(nonbacktracking))
            messages = self.dropout(messages)
        atom_messages = messages.new_zeros((batch.node_features.shape[0], self.hidden_dim))
        atom_messages.index_add_(0, destination, messages)
        atom_states = torch.relu(
            self.atom_output(torch.cat((batch.node_features, atom_messages), dim=-1))
        )
        atom_states = self.dropout(atom_states)
        return self.readout(sum_mean_pool(atom_states, batch.graph_index, batch.num_graphs))


class EdgeGINEncoder(nn.Module):
    """Edge-aware GIN control implemented with sparse index additions."""

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise OracleGraphError("GIN depth must be positive")
        self.node_input = nn.Linear(atom_dim, hidden_dim)
        self.edge_inputs = nn.ModuleList(nn.Linear(bond_dim, hidden_dim) for _ in range(depth))
        self.updates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(depth)
        )
        self.epsilons = nn.Parameter(torch.zeros(depth))
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch: GraphBatch) -> Tensor:
        source, destination = batch.edge_index
        states = torch.relu(self.node_input(batch.node_features))
        for index, (edge_input, update) in enumerate(
            zip(self.edge_inputs, self.updates, strict=True)
        ):
            messages = states[source] + edge_input(batch.edge_features)
            aggregated = states.new_zeros(states.shape)
            aggregated.index_add_(0, destination, messages)
            states = update((1.0 + self.epsilons[index]) * states + aggregated)
            states = self.dropout(states)
        return self.readout(sum_mean_pool(states, batch.graph_index, batch.num_graphs))


class WholeGraphRegressor(nn.Module):
    """Endpoint regressor over the complete product graph."""

    def __init__(self, encoder: nn.Module, hidden_dim: int, outputs: int = 2) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_dim, outputs)

    def forward(self, inputs: GraphModelInput) -> Tensor:
        return self.head(self.encoder(inputs.product))


class UgiRoleAwareRegressor(nn.Module):
    """Shared structural encoder over product and A/B/C role graphs."""

    def __init__(self, encoder: nn.Module, hidden_dim: int, outputs: int = 2) -> None:
        super().__init__()
        self.encoder = encoder
        self.role_embeddings = nn.Parameter(torch.zeros(4, hidden_dim))
        self.head = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, outputs),
        )

    def forward(self, inputs: GraphModelInput) -> Tensor:
        graphs = (inputs.product, inputs.amine, inputs.aldehyde, inputs.isocyanide)
        encoded = [
            self.encoder(graph) + self.role_embeddings[index] for index, graph in enumerate(graphs)
        ]
        return self.head(torch.cat(encoded, dim=-1))
