"""Masked edge-block models for TRAIN-only categorical dependency qualification."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional


def make_queries(graphs, fold, *, training, seed, updates=1024, per_stratum=16):
    rng = np.random.Generator(np.random.PCG64(seed))
    levels = (0.25, 0.5, 0.75)

    def query(index, edge, probability):
        graph = graphs[index]
        count = len(graph["tokens"]) + len(graph["bond_edges"])
        hidden = rng.random(count) < probability
        a, b, _ = graph["bond_edges"][edge]
        hidden[[a, b, len(graph["tokens"]) + edge]] = True
        return {
            "component_index": index,
            "edge": edge,
            "mask_probability": probability,
            "hidden": hidden.tolist(),
        }

    if training:
        roles = sorted({g["role"] for g in graphs})
        strata = [
            [
                i
                for i, g in enumerate(graphs)
                if g["fold"] != fold and g["role"] == role and g["measured"] == measured
            ]
            for role in roles
            for measured in (True, False)
        ]
        if any(not members for members in strata):
            raise ValueError("empty fitting stratum")
        return [
            [
                query(
                    int(i),
                    int(rng.integers(len(graphs[i]["bond_edges"]))),
                    float(rng.choice(levels)),
                )
                for members in strata
                for i in rng.choice(members, per_stratum, replace=True)
            ]
            for _ in range(updates)
        ]
    return [
        query(i, edge, level)
        for i, g in enumerate(graphs)
        if g["fold"] == fold
        for edge in range(len(g["bond_edges"]))
        for level in levels
    ]


def collate(graphs, queries, atom_classes, bond_classes):
    if not queries:
        raise ValueError("nonempty edge queries required")
    n = max(len(graphs[q["component_index"]]["tokens"]) for q in queries)
    tokens = torch.full((len(queries), n), atom_classes, dtype=torch.long)
    adjacency = torch.zeros(len(queries), bond_classes + 1, n, n)
    mask = torch.zeros(len(queries), n, dtype=torch.bool)
    roles, left, right, targets = [], [], [], []
    for i, query in enumerate(queries):
        graph = graphs[query["component_index"]]
        count, edges = len(graph["tokens"]), graph["bond_edges"]
        hidden = query["hidden"]
        a, b, bond = edges[query["edge"]]
        if len(hidden) != count + len(edges) or not all(
            hidden[j] for j in (a, b, count + query["edge"])
        ):
            raise ValueError("all focal endpoint/bond labels must be hidden")
        if any(type(value) is not bool for value in hidden):
            raise ValueError("hidden mask must contain booleans")
        mask[i, :count] = True
        for node, token in enumerate(graph["tokens"]):
            if not hidden[node]:
                tokens[i, node] = token
        for edge, (u, v, label) in enumerate(edges):
            channel = bond_classes if hidden[count + edge] else label
            adjacency[i, channel, u, v] = adjacency[i, channel, v, u] = 1
        roles.append(graph["role_index"])
        left.append(a)
        right.append(b)
        targets.append([graph["tokens"][a], graph["tokens"][b], bond])
    return {
        "tokens": tokens,
        "adjacency": adjacency,
        "node_mask": mask,
        "roles": torch.tensor(roles),
        "left": torch.tensor(left),
        "right": torch.tensor(right),
    }, torch.tensor(targets)


class EdgeBlockLaw(nn.Module):
    """Shared encoder architecture with factorized or symmetric full-table readout."""

    def __init__(self, atom_classes: int, bond_classes: int, *, joint: bool):
        super().__init__()
        self.atom_classes, self.bond_classes, self.joint = atom_classes, bond_classes, joint
        self.embedding = nn.Embedding(atom_classes + 1, 32)
        self.features = nn.Linear(
            2 + bond_classes + 1 + atom_classes + 1 + bond_classes + 1 + 2, 32
        )
        self.self_layers = nn.ModuleList(nn.Linear(32, 32) for _ in range(3))
        self.neighbor_layers = nn.ModuleList(
            nn.ModuleList(nn.Linear(32, 32, bias=False) for _ in range(bond_classes + 1))
            for _ in range(3)
        )
        if joint:
            self.output = nn.Linear(64, atom_classes * atom_classes * bond_classes)
            layers = [self.output]
        else:
            self.atom_output = nn.Linear(32, atom_classes)
            self.bond_output = nn.Linear(64, bond_classes)
            layers = [self.atom_output, self.bond_output]
        for layer in layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, tokens, adjacency, node_mask, roles, left, right):
        batch, n = tokens.shape
        rows = torch.arange(batch, device=tokens.device)
        if adjacency.shape != (batch, self.bond_classes + 1, n, n):
            raise ValueError("masked graph dimensions disagree")
        if (
            not (tokens[rows, left] == self.atom_classes).all()
            or not (tokens[rows, right] == self.atom_classes).all()
        ):
            raise ValueError("focal atom labels are visible")
        if not (adjacency[rows, self.bond_classes, left, right] == 1).all():
            raise ValueError("focal bond is not hidden")
        incident = adjacency.sum(-1).transpose(1, 2)
        nodes = node_mask.sum(-1).clamp_min(1)
        edges = adjacency.sum((1, 2, 3)) / 2
        fractions = (functional.one_hot(tokens, self.atom_classes + 1) * node_mask[..., None]).sum(
            1
        ) / nodes[:, None]
        bonds = adjacency.sum((2, 3)) / (2 * edges.clamp_min(1)[:, None])
        global_features = torch.cat(
            (
                functional.one_hot(roles, 2),
                fractions,
                bonds,
                torch.stack((nodes.log1p(), edges.log1p()), -1),
            ),
            -1,
        )
        features = torch.cat((incident, global_features[:, None, :].expand(-1, n, -1)), -1)
        hidden = torch.relu(self.embedding(tokens) + self.features(features)) * node_mask[..., None]
        degree = incident.sum(-1).clamp_min(1)[..., None]
        for layer, neighbors in zip(self.self_layers, self.neighbor_layers, strict=True):
            update = layer(hidden)
            for channel, edge_layer in enumerate(neighbors):
                update = update + torch.bmm(adjacency[:, channel], edge_layer(hidden)) / degree
            hidden = torch.relu(update) * node_mask[..., None]
        a, b = hidden[rows, left], hidden[rows, right]
        if not self.joint:
            return (
                self.atom_output(a),
                self.atom_output(b),
                self.bond_output(torch.cat((a + b, (a - b).abs()), -1)),
            )
        shape = (-1, self.atom_classes, self.atom_classes, self.bond_classes)
        ab = self.output(torch.cat((a, b), -1)).reshape(shape)
        ba = self.output(torch.cat((b, a), -1)).reshape(shape).transpose(1, 2)
        # Symmetric logits define one normalized full-support joint table.
        return (ab + ba) / 2


def train_loss(output, targets, *, joint):
    if joint:
        index = (targets[:, 0] * output.shape[2] + targets[:, 1]) * output.shape[3] + targets[:, 2]
        return functional.cross_entropy(output.flatten(1), index)
    return sum(functional.cross_entropy(logits, targets[:, i]) for i, logits in enumerate(output))


def summarize(graphs, rows):
    by_component = defaultdict(list)
    for row in rows:
        by_component[row["component_index"]].append(row)
    if set(by_component) != set(range(len(graphs))):
        raise ValueError("excluded-fold predictions do not cover all components")
    fields = (
        "independent_nll",
        "raw_joint_nll",
        "projected_nll",
        "independent_accuracy",
        "raw_joint_accuracy",
        "projected_accuracy",
        "independent_entropy",
        "projected_entropy",
        "left_atom_nll",
        "right_atom_nll",
        "bond_nll",
        "left_atom_accuracy",
        "right_atom_accuracy",
        "bond_accuracy",
    )
    per_component = []
    for index, graph in enumerate(graphs):
        selected = by_component[index]
        expected = {
            (edge, level) for edge in range(len(graph["bond_edges"])) for level in (0.25, 0.5, 0.75)
        }
        observed = [(r["edge"], r["mask_probability"]) for r in selected]
        if len(observed) != len(expected) or set(observed) != expected:
            raise ValueError("edge/mask prediction coverage differs")
        per_component.append(
            {
                "component_index": index,
                "queries": len(selected),
                **{key: math.fsum(r[key] for r in selected) / len(selected) for key in fields},
            }
        )
    strata = {}
    for role in sorted({g["role"] for g in graphs}):
        for measured in (True, False):
            selected = [
                r
                for r in per_component
                if graphs[r["component_index"]]["role"] == role
                and graphs[r["component_index"]]["measured"] == measured
            ]
            strata[f"{role}:{'measured' if measured else 'unmeasured'}"] = {
                "components": len(selected),
                "queries": sum(r["queries"] for r in selected),
                **{key: math.fsum(r[key] for r in selected) / len(selected) for key in fields},
            }
    return {"strata": strata, "per_component": per_component}
