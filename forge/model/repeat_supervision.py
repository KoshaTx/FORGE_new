"""Loss-only repeat correspondences for already-qualified source precursor occurrences.

Serialization positions need not identify the same atom in two repeated precursors.
Exact labelled fragment matching qualifies correspondences; ambiguous fragments abstain.
No target-derived correspondence from this module is passed to the generator at inference.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
from rdkit import Chem


def align_qualified_repeats(example: Any) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return atom/parent-edge equality groups without changing a graph or its conditioning.

    The caller must supply an admitted QualifiedProgramExample. Its exact source construction
    and quantities establish reuse; fragment similarity alone cannot establish source reuse.
    Symmetry or unequal retained fragments suppress this auxiliary supervision only.
    """
    record = example.record
    atoms = np.zeros(record.node_count, dtype=np.int64)
    bonds = np.zeros_like(atoms)
    counts: Counter[str] = Counter()
    roles = {role: q for role, q in example.source_quantities.items() if q > 1}
    if not roles:
        return atoms, bonds, dict(counts)
    mol = Chem.MolFromSmiles(record.graph.canonical_smiles)
    if mol is None:
        raise ValueError("Qualified repeated source product cannot be parsed")
    edges = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
        for bond in mol.GetBonds()
    ]
    atom_label, bond_label = 0, 0
    for role, quantity in sorted(roles.items()):
        blocks = [b for b in record.component_blocks if b.role == role]
        counts["repeated_roles"] += 1
        if len(blocks) != quantity:
            raise ValueError("Source repeat occurrences disagree with admitted quantity")
        if len({b.atom_count for b in blocks}) != 1:
            counts["unequal_fragments"] += 1
            continue
        fragments = []
        signatures = []
        for block in blocks:
            original = record.canonical_atom_order[block.start : block.stop].tolist()
            inverse = {old: new for new, old in enumerate(original)}
            fragment = Chem.RWMol()
            for offset, old in enumerate(original):
                atom = Chem.Atom(mol.GetAtomWithIdx(old))
                atom.SetIsotope(int(record.core_position_states[block.start + offset]) + 1)
                fragment.AddAtom(atom)
            for left, right, bond_type in edges:
                if left in inverse and right in inverse:
                    fragment.AddBond(inverse[left], inverse[right], bond_type)
            fragment = fragment.GetMol()
            fragment.UpdatePropertyCache(strict=False)
            fragments.append(fragment)
            signatures.append(
                (
                    record.graph.node_states[block.start : block.stop],
                    record.core_position_states[block.start : block.stop],
                )
            )
        if len(fragments[0].GetSubstructMatches(fragments[0], uniquify=False, maxMatches=2)) != 1:
            counts["ambiguous_correspondence"] += 1
            continue
        matches = [
            f.GetSubstructMatches(fragments[0], uniquify=False, maxMatches=2) for f in fragments
        ]
        if any(len(m) != 1 for m in matches):
            counts["fragment_mismatch_or_ambiguity"] += 1
            continue
        aligned = [np.asarray(m[0], dtype=np.int64) for m in matches]
        if any(
            not np.array_equal(signature[field][mapping], signatures[0][field])
            for signature, mapping in zip(signatures, aligned, strict=True)
            for field in (0, 1)
        ):
            counts["atom_state_mismatch"] += 1
            continue
        # Full-size induced graphs with equal edge counts and a full atom/bond match.
        if len({f.GetNumBonds() for f in fragments}) != 1:
            counts["edge_mismatch"] += 1
            continue
        counts["aligned_roles"] += 1
        correspondence = {}
        for block, mapping in zip(blocks, aligned, strict=True):
            for ref, local in enumerate(mapping):
                correspondence[block.start + int(local)] = ref
        for ref in range(blocks[0].atom_count):
            if signatures[0][1][ref] != 1:
                continue
            atom_label += 1
            for block, mapping in zip(blocks, aligned, strict=True):
                atoms[block.start + mapping[ref]] = atom_label
        # A parent-bond output denotes an edge, not an atom. Match its two endpoints;
        # different valid spanning-tree traversals need not use the same child position.
        edge_slots = defaultdict(list)
        for block in blocks:
            for child in range(block.start, block.stop):
                parent = int(record.graph.parents[child])
                if child == 0 or not block.start <= parent < block.stop:
                    continue
                if record.core_position_states[child] != 1:
                    continue
                edge = tuple(sorted((correspondence[child], correspondence[parent])))
                edge_slots[edge].append(child)
        for slots in edge_slots.values():
            if len(slots) < 2:
                continue
            if len(set(record.graph.parent_bonds[slots].tolist())) != 1:
                raise ValueError("Exact repeat edge correspondence changed bond targets")
            bond_label += 1
            bonds[slots] = bond_label
    return atoms, bonds, dict(counts)


def aligned_repeat_consistency(predictions: dict, atom_groups: Any, bond_groups: Any):
    """Mean classwise squared probability difference over admitted within-record pairs."""
    losses, counts = [], []
    for field, groups in (("nodes", atom_groups), ("parent_bonds", bond_groups)):
        logits = predictions[field]
        if groups.shape != logits.shape[:2] or groups.dtype != torch.long:
            raise ValueError("Repeat supervision must match batch/node dimensions as int64")
        if bool((groups < 0).any()):
            raise ValueError("Repeat supervision groups must be nonnegative")
        pairs = (groups[:, :, None] == groups[:, None, :]) & (groups[:, :, None] > 0)
        pairs = torch.triu(pairs, diagonal=1)
        batch, left, right = pairs.nonzero(as_tuple=True)
        probability = logits.softmax(-1)
        distance = (probability[batch, left] - probability[batch, right]).square().mean(-1)
        count = pairs.sum()
        losses.append(distance.sum() / count.clamp(min=1))
        counts.append(count)
    return losses[0] + losses[1], counts[0], counts[1]
