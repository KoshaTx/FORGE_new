"""Bounded TRAIN role-motif proposals on a fixed generated molecular topology.

Motif and element counts are an explicit reference prior, not source-validity or
lipid-quality rules. The caller supplies registry queries and independently
checks every proposed complete product against its original source executor.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from rdkit import Chem, rdBase

from forge.model.defog_feasibility import BOND_TYPE_TO_INDEX, graph_to_molecule
from forge.model.precursor_reuse_projection import fixed_graph_preserved, graph_smiles, state_graph


@dataclass(frozen=True)
class RoleMotifPrior:
    family: str
    role: str
    query: str
    minimum_matches: int
    element_counts: tuple[tuple[str, int], ...]
    degree_patterns: tuple[tuple[int, ...], ...]
    reference_components: int


def compile_role_motif_prior(
    family: str,
    role: str,
    registry_query: str,
    train_components: Sequence[str],
    removed_elements: Mapping[str, int],
) -> RoleMotifPrior:
    """Extract invariant motif/heteroatom counts, after source-derived atom loss.

    The query must describe concrete neutral atoms and single/double/triple
    bonds. It must be disjoint from the reaction site, qualified by the caller.
    Carbon count remains unconstrained so tail sizes are not whitelisted.
    """
    query = Chem.MolFromSmarts(registry_query)
    if query is None or any(
        a.GetAtomicNum() <= 0 or a.GetFormalCharge() or a.GetIsAromatic() for a in query.GetAtoms()
    ):
        raise ValueError("Role motif requires a concrete neutral registry query")
    if any(b.GetBondType() not in BOND_TYPE_TO_INDEX for b in query.GetBonds()):
        raise ValueError("Role motif query contains unsupported bond classes")
    if any(type(n) is not int or n < 0 for n in removed_elements.values()):
        raise ValueError("Source-derived element losses must be nonnegative integers")
    hetero = sorted({a.GetSymbol() for a in query.GetAtoms() if a.GetAtomicNum() != 6})
    canonical, counts, matches, degrees = set(), set(), set(), set()
    for smiles in sorted(set(train_components)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError("TRAIN role reference must be valid and connected")
        identity = Chem.MolToSmiles(mol, isomericSmiles=False)
        if identity in canonical:
            continue
        canonical.add(identity)
        found = mol.GetSubstructMatches(query, maxMatches=1025)
        if len(found) >= 1025:
            raise ValueError("TRAIN motif matches exceeded the reference enumeration bound")
        if len(found) == 2 and not set(found[0]).isdisjoint(found[1]):
            raise ValueError("TRAIN role motifs must occupy disjoint atoms")
        elements = Counter(a.GetSymbol() for a in mol.GetAtoms())
        local = tuple((s, elements[s] - removed_elements.get(s, 0)) for s in hetero)
        if any(n < 0 for _, n in local):
            raise ValueError("Source-derived loss exceeds TRAIN component element count")
        counts.add(local)
        matches.add(len(found))
        degrees.update(tuple(mol.GetAtomWithIdx(i).GetDegree() for i in m) for m in found)
    if not canonical or len(counts) != 1 or len(matches) != 1 or next(iter(matches)) not in (1, 2):
        raise ValueError("Role motif counts are not an invariant of the supplied TRAIN components")
    return RoleMotifPrior(
        family,
        role,
        registry_query,
        next(iter(matches)),
        next(iter(counts)),
        tuple(sorted(degrees)),
        len(canonical),
    )


def _measure(mol, slots, query, prior):
    all_matches = mol.GetSubstructMatches(query, maxMatches=1025)
    if len(all_matches) >= 1025:
        raise ValueError("Generated motif enumeration bound exceeded")
    matches = [m for m in all_matches if set(m) <= slots]
    # A carbonate can match an ester query twice at the same carbonyl. That is
    # not the two separate ester motifs present in the qualified TRAIN prior.
    if prior.minimum_matches not in (1, 2):
        raise ValueError("Only one or two disjoint TRAIN role motifs are qualified")
    disjoint = next(
        ([a, b] for a, b in combinations(matches, 2) if set(a).isdisjoint(b)),
        matches[:1],
    )
    observed = Counter(mol.GetAtomWithIdx(i).GetSymbol() for i in slots)
    deficit = max(0, prior.minimum_matches - len(disjoint)) + sum(
        abs(observed[s] - n) for s, n in prior.element_counts
    )
    return deficit, matches


def propose_role_motifs(
    layout,
    state,
    predictions,
    atoms,
    prior: RoleMotifPrior,
    *,
    maximum_proposals=2,
    beam_width=4,
    maximum_steps=2,
    maximum_placements=64,
    maximum_topology_matches=512,
):
    """Jointly materialize observed motifs without moving edges or ring atoms.

    Excess heteroatoms outside motifs may become a query-derived carbon state;
    this is explicit atom relabeling, never a count-only quality certification.
    Search truncation is recorded and original inputs are never overwritten.
    """
    budgets = (
        maximum_proposals,
        beam_width,
        maximum_steps,
        maximum_placements,
        maximum_topology_matches,
    )
    if any(type(v) is not int or v < 1 for v in budgets):
        raise ValueError("Role-motif budgets must be positive integers")
    if layout.family != prior.family:
        return {"status": "outside_policy_scope", "proposals": [], "candidates_evaluated": 0}
    raw_nodes = np.asarray(state["nodes"], dtype=object)
    if raw_nodes.shape != (layout.record.node_count,) or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not 0 <= value < len(atoms)
        for value in raw_nodes
    ):
        raise ValueError("Role-motif atom classes must be integer indices within vocabulary")
    nodes, edges = state_graph(state)
    if len(nodes) != layout.record.node_count or not fixed_graph_preserved(
        nodes, edges, layout.record
    ):
        raise ValueError("Role-motif input changes the fixed graph or atom budget")
    if not graph_smiles(nodes, edges, atoms):
        raise ValueError("Role-motif input must be a valid connected graph")
    logits = np.asarray(predictions["nodes"])
    if (
        logits.ndim != 2
        or logits.shape[0] < len(nodes)
        or logits.shape[1] != len(atoms)
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Role-motif atom predictions have invalid shape or nonfinite values")
    with rdBase.BlockLogs():
        original = graph_to_molecule(nodes, edges, atoms)
    blocks = [b for b in layout.record.component_blocks if b.role == prior.role]
    if len(blocks) > 1:
        return {
            "status": "repeated_policy_role_unqualified",
            "proposals": [],
            "candidates_evaluated": 0,
        }
    slots = {i for b in blocks for i in range(b.start, b.stop)}
    if not slots:
        return {"status": "missing_policy_role", "proposals": [], "candidates_evaluated": 0}
    movable = {
        i
        for i in slots
        if not layout.record.fixed_atom_mask[i]
        and layout.record.core_position_states[i] == 1
        and not original.GetAtomWithIdx(i).IsInRing()
    }
    query = Chem.MolFromSmarts(prior.query)
    initial, _ = _measure(original, slots, query, prior)
    if initial == 0:
        return {
            "status": "prior_already_satisfied",
            "proposals": [],
            "candidates_evaluated": 0,
            "initial_prior_deficit": 0,
        }
    options = Chem.AdjustQueryParameters()
    options.adjustDegree = False
    options.makeAtomsGeneric = True
    options.makeBondsGeneric = True
    topology_query = Chem.AdjustQueryProperties(query, options)
    matches = original.GetSubstructMatches(
        topology_query, uniquify=False, maxMatches=maximum_topology_matches + 1
    )
    truncated = len(matches) > maximum_topology_matches
    matches = matches[:maximum_topology_matches]
    states = {(a.symbol, a.formal_charge, a.aromatic): i for i, a in enumerate(atoms)}
    targets = [states.get((a.GetSymbol(), 0, False)) for a in query.GetAtoms()]
    if any(i is None for i in targets):
        return {
            "status": "query_states_outside_vocabulary",
            "proposals": [],
            "candidates_evaluated": 0,
        }
    carbon_states = [targets[a.GetIdx()] for a in query.GetAtoms() if a.GetAtomicNum() == 6]
    if not carbon_states:
        raise ValueError("Role-motif query has no carbon state for conservative count repair")
    carbon = carbon_states[0]
    placements = []
    for match in matches:
        if len(set(match)) != len(match) or not set(match) <= movable:
            continue
        if (
            tuple(original.GetAtomWithIdx(i).GetDegree() for i in match)
            not in prior.degree_patterns
        ):
            continue
        score = sum(
            float(logits[i, value] - logits[i, nodes[i]])
            for i, value in zip(match, targets, strict=True)
        )
        placements.append((-score, tuple(match)))
    placements = sorted(set(placements))[:maximum_placements]
    beam = [(initial, 0.0, graph_smiles(nodes, edges, atoms), nodes, edges, [])]
    evaluated, cleanup_evaluated, accepted = 0, 0, {}
    for _ in range(maximum_steps):
        next_beam = []
        for deficit, _, _, previous_nodes, previous_edges, history in beam:
            for _, match in placements:
                nn, ee = previous_nodes.copy(), previous_edges.copy()
                for index, target in zip(match, targets, strict=True):
                    nn[index] = target
                for bond in query.GetBonds():
                    a, b = match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()]
                    ee[a, b] = ee[b, a] = BOND_TYPE_TO_INDEX[bond.GetBondType()]
                evaluated += 1
                smiles = graph_smiles(nn, ee, atoms)
                if smiles is None:
                    continue
                with rdBase.BlockLogs():
                    mol = graph_to_molecule(nn, ee, atoms)
                for symbol, expected in prior.element_counts:
                    while sum(atoms[nn[i]].symbol == symbol for i in slots) > expected:
                        _, protected_matches = _measure(mol, slots, query, prior)
                        protected = {i for m in protected_matches for i in m}
                        choices = sorted(
                            (-(float(logits[i, carbon] - logits[i, nn[i]])), i)
                            for i in movable - protected
                            if atoms[nn[i]].symbol == symbol
                        )
                        changed = False
                        for _, index in choices:
                            trial = nn.copy()
                            trial[index] = carbon
                            cleanup_evaluated += 1
                            if graph_smiles(trial, ee, atoms):
                                nn = trial
                                with rdBase.BlockLogs():
                                    mol = graph_to_molecule(nn, ee, atoms)
                                changed = True
                                break
                        if not changed:
                            break
                cost, _ = _measure(mol, slots, query, prior)
                if cost >= deficit or not fixed_graph_preserved(nn, ee, layout.record):
                    continue
                assert np.array_equal(ee > 0, edges > 0)
                assert all(nn[i] == nodes[i] for i in range(len(nodes)) if i not in movable)
                score = sum(float(logits[i, nn[i]] - logits[i, nodes[i]]) for i in movable)
                smiles = graph_smiles(nn, ee, atoms)
                item = (cost, -score, smiles, nn, ee, history + [list(match)])
                next_beam.append(item)
                if cost == 0:
                    accepted[smiles] = item
        if not next_beam:
            break
        deduplicated = {}
        for item in sorted(next_beam, key=lambda x: x[:3]):
            deduplicated.setdefault(item[2], item)
        beam = list(deduplicated.values())[:beam_width]
        if all(item[0] == 0 for item in beam):
            break
    proposals = []
    for item in sorted(accepted.values(), key=lambda x: x[:3])[:maximum_proposals]:
        deficit, negative_score, smiles, nn, ee, history = item
        proposals.append(
            {
                "nodes": nn.tolist(),
                "edges": ee.tolist(),
                "smiles": smiles,
                "prior_deficit": deficit,
                "atom_logit_change": -negative_score,
                "motif_placements": history,
                "atom_changes": int(np.count_nonzero(nn != nodes)),
                "bond_changes": int(np.count_nonzero(np.triu(ee != edges, 1))),
            }
        )
    return {
        "status": "bounded_TRAIN_role_motif_prior",
        "proposals": proposals,
        "initial_prior_deficit": initial,
        "candidates_evaluated": evaluated,
        "cleanup_candidates_evaluated": cleanup_evaluated,
        "topology_search_truncated": truncated,
        "placements_retained": len(placements),
        "interpretation": "Not a chemical validity gate; full independent source assessment required",
    }
