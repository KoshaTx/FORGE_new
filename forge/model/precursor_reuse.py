"""Qualified precursor equality relations and a residual sparse-flow conditioning path.

Qualification may inspect TRAIN graph targets. The network receives only atom correspondence
groups; it receives no target atom, bond, precursor identity or target pointer through this path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from torch import nn

from forge.assembly.library_generation import check_generated_program
from forge.assembly.library_programs import LibraryProgramLimits
from forge.model.reaction_program_flow import collate_synthesis_program_layouts
from forge.model.synthesis_program_graph import SynthesisProgramGraphRecord


class PrecursorReuseError(ValueError):
    """A reuse relation or its binding to a batch is invalid."""


@dataclass(frozen=True)
class PrecursorReusePlan:
    record_id: str
    groups: tuple[int, ...]
    status: str
    exact_source_programs: int

    def __post_init__(self) -> None:
        if (
            not self.record_id
            or not self.groups
            or any(type(g) is not int or g < 0 for g in self.groups)
        ):
            raise PrecursorReuseError("invalid reuse group coordinates")
        labels = set(self.groups) - {0}
        if labels and (
            labels != set(range(1, max(labels) + 1))
            or any(self.groups.count(g) < 2 for g in labels)
        ):
            raise PrecursorReuseError(
                "reuse groups must be contiguous and have multiple occurrences"
            )
        if bool(labels) != (self.status == "qualified"):
            raise PrecursorReuseError("reuse status and coordinates disagree")


def qualify_reuse(
    record: SynthesisProgramGraphRecord, adapter: Any, policy: dict, limits: dict
) -> PrecursorReusePlan:
    """Require exact source reuse plus unambiguous product-fragment correspondences.

    This deliberately abstains on symmetric internal correspondences. Different serialization
    traversals are aligned only by their unique exact labelled graph isomorphism.
    It does not remove the record from training or generation. Source equality is a computed
    program property; disconnected role blocks alone never authorize a relation.
    """
    empty = (0,) * record.node_count

    def result(status: str, count: int = 0) -> PrecursorReusePlan:
        return PrecursorReusePlan(record.graph.structure_id, empty, status, count)

    accumulator = policy["accumulator_role"]
    if accumulator is None or record.program_depth == 1:
        return result("not_repeated")
    strict = check_generated_program(
        adapter,
        record.graph.canonical_smiles,
        depth=record.program_depth,
        accumulator_role=accumulator,
        limits=LibraryProgramLimits(policy["maximum_steps"], **limits),
    )
    if not strict.exact:
        return result("source_reuse_not_verified")
    count = len(strict.programs)
    roles = set(adapter.roles) - {accumulator}
    if len(roles) != 1:
        return result("unsupported_role_contract", count)
    role = next(iter(roles))
    blocks = [b for b in record.component_blocks if b.role == role]
    if len(blocks) != record.program_depth or len({b.atom_count for b in blocks}) != 1:
        return result("occurrence_partition_mismatch", count)
    molecule = Chem.MolFromSmiles(record.graph.canonical_smiles)
    fragments, signatures = [], []
    for block in blocks:
        atom_indices = list(map(int, record.canonical_atom_order[block.start : block.stop]))
        inverse = {old: new for new, old in enumerate(atom_indices)}
        fragment = Chem.RWMol()
        for offset, old in enumerate(atom_indices):
            atom = Chem.Atom(molecule.GetAtomWithIdx(old))
            # Core membership/position is admitted layout context. Do not number copies.
            atom.SetIsotope(int(record.core_position_states[block.start + offset]) + 1)
            fragment.AddAtom(atom)
        edges = []
        for bond in molecule.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in inverse and b in inverse:
                fragment.AddBond(inverse[a], inverse[b], bond.GetBondType())
                edges.append(
                    (
                        min(inverse[a], inverse[b]),
                        max(inverse[a], inverse[b]),
                        str(bond.GetBondType()),
                    )
                )
        fragment = fragment.GetMol()
        fragment.UpdatePropertyCache(strict=False)
        # Self automorphisms must uniquely locate every atom given the admitted core labels.
        if len(fragment.GetSubstructMatches(fragment, uniquify=False, maxMatches=2)) != 1:
            return result("ambiguous_atom_correspondence", count)
        fragments.append(fragment)
        signatures.append(
            (
                tuple(record.graph.node_states[block.start : block.stop].tolist()),
                tuple(record.core_position_states[block.start : block.stop].tolist()),
                tuple(sorted(edges)),
            )
        )
    groups = list(empty)
    for block, fragment, signature in zip(blocks, fragments, signatures, strict=True):
        matches = fragment.GetSubstructMatches(fragments[0], uniquify=False, maxMatches=2)
        if len(matches) != 1:
            return result("fragment_correspondence_mismatch", count)
        match = matches[0]
        # RDKit graph matching alone does not enforce every serialized atom-state coordinate.
        if any(tuple(signature[k][i] for i in match) != signatures[0][k] for k in (0, 1)):
            return result("fragment_correspondence_mismatch", count)
        inverse = {target: source for source, target in enumerate(match)}
        edges = tuple(
            sorted(
                (min(inverse[a], inverse[b]), max(inverse[a], inverse[b]), kind)
                for a, b, kind in signature[2]
            )
        )
        if edges != signatures[0][2]:
            return result("fragment_correspondence_mismatch", count)
        for group, target in enumerate(match, 1):
            groups[block.start + target] = group
    return PrecursorReusePlan(record.graph.structure_id, tuple(groups), "qualified", count)


def group_tensor(plans: list[PrecursorReusePlan], nodes: int, device: Any = "cpu") -> Any:
    if not plans or any(len(p.groups) > nodes for p in plans):
        raise PrecursorReuseError("reuse plans exceed batch support")
    array = np.zeros((len(plans), nodes), dtype=np.int64)
    for i, plan in enumerate(plans):
        array[i, : len(plan.groups)] = plan.groups
    return torch.as_tensor(array, device=device)


def pool_corresponding(hidden: Any, groups: Any) -> Any:
    """Mean over each within-record equality class, using bounded scatter operations."""
    if groups.shape != hidden.shape[:2] or groups.dtype != torch.long:
        raise PrecursorReuseError("reuse tensor shape/dtype mismatch")
    width = hidden.shape[1] + 1
    if bool((groups < 0).any()) or bool((groups >= width).any()):
        raise PrecursorReuseError("reuse group lies outside node support")
    indices = groups + torch.arange(len(groups), device=groups.device)[:, None] * width
    sums = hidden.new_zeros((len(groups) * width, hidden.shape[-1]))
    sums.index_add_(0, indices.flatten(), hidden.flatten(0, 1))
    counts = torch.bincount(indices.flatten(), minlength=len(groups) * width).clamp(min=1)
    return (sums / counts[:, None])[indices]


class PrecursorReuseFlow(nn.Module):
    """Use pooled noisy atom states only where precursor equality was separately qualified.

    The self-only control has identical parameters and eligible-node masks. Both start with an
    exactly zero residual. Equality is a soft condition, not a guarantee or acceptance gate.
    """

    def __init__(self, base: Any) -> None:
        super().__init__()
        self.base = base
        self.maximum_closures = base.maximum_closures
        self.maximum_heavy_atoms = base.maximum_heavy_atoms
        hidden = base.backbone.hidden_dim
        self.reuse_projection = nn.Linear(2 * hidden, hidden)
        nn.init.zeros_(self.reuse_projection.weight)
        nn.init.zeros_(self.reuse_projection.bias)

    def forward(self, *, reuse_groups: Any, share: bool, **inputs: Any) -> dict:
        if inputs.get("potency_condition") is not None:
            raise PrecursorReuseError("biological conditioning is outside this experiment")
        context = self.base.conditioning(
            **{
                key: inputs[key]
                for key in (
                    "program_states",
                    "role_states",
                    "core_position_states",
                    "program_depths",
                    "adapter_mask",
                )
            }
        )
        own = self.base.backbone.node_embedding(inputs["nodes"])
        pooled = pool_corresponding(own, reuse_groups)
        selected = pooled if share else own
        context = (
            context
            + self.reuse_projection(torch.cat((own, selected), dim=-1))
            * (reuse_groups > 0)[..., None]
        )
        return self.base.backbone(
            **{
                key: inputs[key]
                for key in (
                    "nodes",
                    "parents",
                    "parent_bonds",
                    "closure_left",
                    "closure_right",
                    "closure_bonds",
                    "t",
                    "node_mask",
                    "child_mask",
                    "closure_mask",
                )
            },
            node_context=context,
        )


_MEMORY_FIELDS = (
    "program_states",
    "role_states",
    "core_position_states",
    "program_depths",
    "adapter_mask",
    "repeat_group_states",
    "component_position_states",
    "role_morphology_states",
)


class ReuseSamplingView(nn.Module):
    """Bind immutable sidecars to the unchanged sampler's ordered semantic batches.

    Construction projects away graph targets. Each memory preparation validates batch semantics;
    forward calls accept only memory created by this view. A view is single-use.
    """

    def __init__(
        self,
        network: PrecursorReuseFlow,
        records: list,
        plans: list,
        *,
        batch_size: int,
        share: bool,
    ):
        super().__init__()
        if len(records) != len(plans) or batch_size < 1:
            raise PrecursorReuseError("sampling plan count/batch size mismatch")
        self.network, self.share = network, share
        self.maximum_closures = network.maximum_closures
        self.batches = []
        self.cursor = 0
        self.memories = []
        for offset in range(0, len(records), batch_size):
            local, local_plans = (
                records[offset : offset + batch_size],
                plans[offset : offset + batch_size],
            )
            if any(
                r.graph.structure_id != p.record_id or r.node_count != len(p.groups)
                for r, p in zip(local, local_plans, strict=True)
            ):
                raise PrecursorReuseError("reuse plan identity differs from layout")
            layout = collate_synthesis_program_layouts(
                local, maximum_closures=self.maximum_closures
            )
            self.batches.append(
                (
                    {k: layout[k] for k in _MEMORY_FIELDS},
                    group_tensor(local_plans, layout["nodes"].shape[1]),
                )
            )

    def prepare_program_memory(self, **inputs: Any) -> dict:
        if self.cursor >= len(self.batches):
            raise PrecursorReuseError("reuse sampling view exhausted")
        expected, groups = self.batches[self.cursor]
        if set(inputs) != set(expected) or any(
            not torch.equal(inputs[k].cpu(), expected[k]) for k in expected
        ):
            raise PrecursorReuseError("sampling semantics differ from bound reuse plan")
        memory = {"groups": groups.to(inputs["role_states"].device)}
        self.memories.append(memory)
        self.cursor += 1
        return memory

    def forward(self, *, program_memory: Any, **inputs: Any) -> dict:
        if not any(program_memory is m for m in self.memories):
            raise PrecursorReuseError("unbound reuse memory")
        return self.network(reuse_groups=program_memory["groups"], share=self.share, **inputs)

    def assert_consumed(self) -> None:
        if self.cursor != len(self.batches):
            raise PrecursorReuseError("sampler omitted bound reuse batches")
