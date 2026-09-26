"""Independent final-graph design checks, separate from source chemistry and realism."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forge.model.compose_lipid_head_survival import assess_head_survival, load_head_survival_policy
from forge.model.compose_lipid_structural_audit import (
    RegistryQuery,
    audit_layout_rings,
    audit_smiles,
)
from forge.model.defog_feasibility import AtomState, graph_to_molecule
from forge.model.precursor_reuse_projection import graph_smiles
from forge.model.sparse_topology_feasibility import SPARSE_BOND_TO_INDEX
from results.phase1.compose_lipid_quality_selection_v2.run import pin, read


@dataclass(frozen=True)
class GateContext:
    policy: dict
    atoms: tuple
    head: object
    queries: dict
    basic: tuple
    rubric: dict
    elements: set
    charges: set
    bonds: set


def load_gate_context(root: Path, policy_path: Path) -> GateContext:
    policy = read(policy_path)
    for name, expected in policy["inputs"].items():
        if pin(root / expected["path"]) != expected:
            raise ValueError("Frozen design-gate dependency changed: " + name)
    inputs = policy["inputs"]
    head_spec = read(root / inputs["head_qualification"]["path"])
    head = load_head_survival_policy(
        root,
        config_pin=head_spec["inputs"]["config"],
        retained_registry_pin=head_spec["inputs"]["retained_registry"],
        reference_pin=head_spec["inputs"]["reference"],
        motif_families=tuple(head_spec["motif_families"]),
    )
    if head.controls != head_spec["controls"]:
        raise ValueError("Qualified head controls no longer reproduce")
    definitions = read(root / inputs["queries"]["path"])
    queries = {}
    for family, items in definitions["by_family"].items():
        qs = [RegistryQuery(**item) for item in items]
        queries[family] = [(q, q.compile()) for q in qs]
    basic = RegistryQuery(**definitions["basic_query"])
    vocabulary = read(root / inputs["vocabulary"]["path"])
    atoms = tuple(
        AtomState(**{k: v for k, v in a.items() if k not in {"index", "train_atoms"}})
        for a in vocabulary["atom_vocabulary"]
    )
    return GateContext(
        policy,
        atoms,
        head,
        queries,
        (basic, basic.compile()),
        read(root / inputs["rubric"]["path"]),
        {a["symbol"] for a in vocabulary["atom_vocabulary"]},
        {(a["symbol"], a["formal_charge"]) for a in vocabulary["atom_vocabulary"]},
        {
            str(b)
            for b, i in SPARSE_BOND_TO_INDEX.items()
            if i in vocabulary["active_sparse_bond_indices"]
        },
    )


def assess_design(layout, nodes, edges, state, basis, smiles, source_exact, context):
    """Every status is evaluated on this final graph; intermediate passes do not carry."""
    if graph_smiles(nodes, edges, context.atoms) != smiles:
        raise ValueError("Design assessment graph differs from source-assessed product")
    chemical = audit_smiles(
        smiles,
        registry_queries=context.queries[layout.family],
        basic_query=context.basic,
        policy=context.rubric,
        allowed_elements=context.elements,
        allowed_atom_charges=context.charges,
        allowed_kekule_bonds=context.bonds,
    )
    chemical_failed = (
        not chemical["valid"]
        or not chemical["connected"]
        or any(
            f["tier"] in {"demonstrated_representation_issue", "justified_chemical_alert"}
            for f in chemical["flags"]
        )
    )
    ring = audit_layout_rings(edges, layout, tree_state=state, tree_basis=basis)
    no_variable_cycles = all(
        r["observed_variable_cycle_rank"] == r["declared_variable_closures"] == 0
        for r in ring["roles"].values()
    )
    comparable = basis in context.policy["comparable_ordered_tree_bases"]
    if not comparable and state is not None:
        molecule = graph_to_molecule(np.asarray(nodes), np.asarray(edges), context.atoms)
        rings = [set(r) for r in molecule.GetRingInfo().AtomRings()]
        # Disjoint or spiro simple cycles have no shared cyclic edge/path and their
        # individual fundamental cycle lengths are basis-invariant. SSSR sizes are
        # never substituted for the sampled fundamental-size condition.
        comparable = all(
            len(left & right) <= 1 for i, left in enumerate(rings) for right in rings[i + 1 :]
        )
    ring["comparison_to_sampled_sizes_qualified"] = comparable
    ring["variable_cycle_size_not_applicable"] = no_variable_cycles
    if ring["cycle_allocation_mismatch"] or ring["unexpected_cross_origin_edges"]:
        ring_status = "fail"
    elif no_variable_cycles:
        ring_status = "pass"
    elif not comparable or ring["fundamental_size_mismatch"] is None:
        ring_status = "abstain"
    else:
        ring_status = "fail" if ring["fundamental_size_mismatch"] else "pass"
    head = assess_head_survival(layout, nodes, edges, context.atoms, context.head)
    statuses = {
        "chemical": "fail" if chemical_failed else "pass",
        "ring": ring_status,
        "head": head["status"],
    }
    clean = bool(
        source_exact and all(status in {"pass", "not_applicable"} for status in statuses.values())
    )
    return {
        "source_exact": bool(source_exact),
        "qualified_design_pass": clean,
        "status_by_axis": statuses,
        "chemical": chemical,
        "ring": ring,
        "head": head,
        "interpretation": "New development design checks, not complete chemistry, synthesis-success, pKa or delivery certification",
    }
