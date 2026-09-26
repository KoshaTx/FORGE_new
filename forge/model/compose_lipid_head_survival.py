"""A scoped, opt-in head-survival design requirement, separate from source L1.

The retained-N query is borrowed from a pinned source registry; its transfer to
aldehyde Ugi4 is a new design policy, not newly admitted chemistry or a pKa model.
Repairs are local generated-atom proposals, never complete known-head replacement.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase

from forge.core.hashing import resolve_pin
from forge.model.defog_feasibility import graph_to_molecule
from forge.model.precursor_reuse_projection import fixed_graph_preserved, graph_smiles, state_graph
from forge.model.sparse_topology_feasibility import BOND_VALENCE_UNITS
from forge.model.synthesis_program_sampling import _atom_capacity_table


@dataclass(frozen=True)
class HeadSurvivalPolicy:
    family: str
    head_role: str
    query_smarts: str
    minimum_retained: int
    train_environments: tuple[str, ...]
    motif_families: tuple[str, ...]
    inputs: dict
    controls: dict


def _read_pin(root, value, label):
    return json.loads(resolve_pin(value, root, label=label).read_text())


def _query(smarts):
    query = Chem.MolFromSmarts(smarts)
    if query is None or query.GetNumAtoms() != 1:
        raise ValueError("Head-survival policy requires a single-atom registry query")
    return query


def _environment(molecule, index, reactive, members):
    """Local TRAIN support includes head attachment, charge, aromaticity and ring sizes."""
    atom = molecule.GetAtomWithIdx(index)
    neighbors = []
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetIdx() not in members:
            return None
        neighbors.append(
            (
                "reactive_head_site" if other.GetIdx() in reactive else other.GetSymbol(),
                other.GetFormalCharge(),
                other.GetIsAromatic(),
                str(bond.GetBondType()),
            )
        )
    return json.dumps(
        dict(
            symbol=atom.GetSymbol(),
            charge=atom.GetFormalCharge(),
            aromatic=atom.GetIsAromatic(),
            neighbors=sorted(neighbors),
            ring_sizes=sorted(
                len(ring) for ring in molecule.GetRingInfo().AtomRings() if index in ring
            ),
        ),
        sort_keys=True,
        separators=(",", ":"),
    )


def load_head_survival_policy(
    root: Path,
    *,
    config_pin: dict,
    retained_registry_pin: dict,
    reference_pin: dict,
    motif_families: tuple[str, ...] = ("aldehyde_ugi4",),
) -> HeadSurvivalPolicy:
    """Calibrate the new policy on pinned unique TRAIN heads before candidate scoring."""
    if (
        not motif_families
        or len(set(motif_families)) != len(motif_families)
        or set(motif_families) - {"aldehyde_ugi4", "ketone_ugi4"}
    ):
        raise ValueError("Local head motif transfer supports only the declared Ugi4 families")
    config = _read_pin(root, config_pin, "aldehyde Ugi4 source config")
    if config["family"] != "aldehyde_ugi4":
        raise ValueError("Head-survival transfer is qualified only for aldehyde Ugi4")
    registry = _read_pin(root, retained_registry_pin, "retained head-query registry")
    reaction = next(r for r in registry["reactions"] if r["reaction_id"] == "source_ketone_ugi4")
    retained = next(
        q for q in reaction["retained_queries"] if q["name"] == "distinct_head_basic_nitrogen"
    )
    basic = _query(retained["smarts"])
    adjudication = _read_pin(root, config["inputs"]["adjudication"], "Ugi4 source adjudication")
    contract = adjudication["source_contract"]
    head_role = contract["registry_to_source_roles"][retained["role"]]
    reference = _read_pin(root, reference_pin, "TRAIN role controls")
    if reference["heldout_structures_used"] is not False:
        raise ValueError("Head-survival reference must exclude heldout structures")
    query_ref = contract["query_references"][retained["role"]]
    functional = _read_pin(root, config["inputs"][query_ref["input"]], "reactive head query")
    role = next(
        role
        for r in functional["reactions"]
        if r["reaction_id"] == query_ref["reaction_id"]
        for role in r["reactant_roles"]
        if role["name"] == query_ref["role"]
    )
    reactive_query = _query(role["required_handle_smarts"])
    sites = [s for s in contract["site_contract"] if s["role"] == retained["role"]]
    if len(sites) != 1 or set(sites[0]["properties"]) - {
        "atomic_number",
        "formal_charge",
        "total_hydrogens",
    }:
        raise ValueError("Head-survival requires a single qualified reactive head site")
    controls, environments, environment_sources = [], set(), {}
    ketone_role = next(r for r in reaction["reactant_roles"] if r["name"] == retained["role"])
    for family in motif_families:
        local_role = head_role if family == config["family"] else retained["role"]
        local_reactive_query = (
            reactive_query
            if family == config["family"]
            else _query(ketone_role["required_handle_smarts"])
        )
        heads = sorted(
            {
                reference["component_smiles_by_identity"][identity]
                for identity in reference["component_ids_by_family_role"][family][local_role]
            }
        )
        for smiles in heads:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
                raise ValueError("TRAIN head is not a valid connected graph")
            reactive = set()
            for (index,) in molecule.GetSubstructMatches(local_reactive_query):
                atom = molecule.GetAtomWithIdx(index)
                properties = dict(
                    atomic_number=atom.GetAtomicNum(),
                    formal_charge=atom.GetFormalCharge(),
                    total_hydrogens=atom.GetTotalNumHs(),
                )
                if all(properties[k] == v for k, v in sites[0]["properties"].items()):
                    reactive.add(index)
            if not reactive:
                raise ValueError("TRAIN head has no source-qualified reacting site")
            neutral = all(atom.GetFormalCharge() == 0 for atom in molecule.GetAtoms())
            surviving = sorted({m[0] for m in molecule.GetSubstructMatches(basic)} - reactive)
            status = (
                "abstain"
                if not neutral
                else ("pass" if len(surviving) >= retained["minimum_matches"] else "fail")
            )
            controls.append(
                dict(
                    family=family,
                    role=local_role,
                    smiles=smiles,
                    reactive_atoms=sorted(reactive),
                    retained_atoms=surviving,
                    status=status,
                )
            )
            if status == "pass":
                for index in surviving:
                    atom = molecule.GetAtomWithIdx(index)
                    if atom.GetSymbol() == "N" and not atom.GetIsAromatic():
                        environment = _environment(
                            molecule, index, reactive, set(range(molecule.GetNumAtoms()))
                        )
                        if environment is not None:
                            environments.add(environment)
                            environment_sources.setdefault(environment, []).append(
                                dict(family=family, role=local_role, smiles=smiles, atom=index)
                            )
    product_controls = reference["product_controls"][config["family"]][:64]
    return HeadSurvivalPolicy(
        family=config["family"],
        head_role=head_role,
        query_smarts=retained["smarts"],
        minimum_retained=retained["minimum_matches"],
        train_environments=tuple(sorted(environments - {None})),
        motif_families=tuple(motif_families),
        inputs=dict(
            config=config_pin,
            retained_registry=retained_registry_pin,
            reference=reference_pin,
            source_adjudication=config["inputs"]["adjudication"],
            reactive_query_registry=config["inputs"][query_ref["input"]],
        ),
        controls=dict(
            unique_train_heads=controls,
            environment_sources=environment_sources,
            local_prior_scope="Cross-family TRAIN local retained-N motif support only; not exact whole-head identity or reaction execution transfer",
            by_family={
                f: dict(
                    heads=sum(c["family"] == f for c in controls),
                    failures=sum(c["family"] == f and c["status"] == "fail" for c in controls),
                    abstentions=sum(
                        c["family"] == f and c["status"] == "abstain" for c in controls
                    ),
                )
                for f in motif_families
            },
            head_policy_failures=sum(c["status"] == "fail" for c in controls),
            head_policy_abstentions=sum(c["status"] == "abstain" for c in controls),
            false_positive_interpretation="Source-control disagreements with new design policy, not proof that source lipids lack ionization",
            product_controls=len(product_controls),
            product_controls_without_any_query_match=sum(
                not Chem.MolFromSmiles(s).HasSubstructMatch(basic) for s in product_controls
            ),
            product_control_scope="Global query only; origin-specific positive controls are the unique TRAIN heads above",
        ),
    )


def assess_head_survival(layout, nodes, edges, atoms: Sequence, policy: HeadSurvivalPolicy) -> dict:
    """Assess only the qualified head-origin design requirement; never rewrite L1."""
    result = dict(
        policy="aldehyde_ugi4_neutral_head_survival_v1",
        status="not_applicable",
        applicable=False,
        reason="family_outside_qualified_scope",
        atom_index_basis="generated graph node order",
        retained_head_atoms=[],
        interpretation="New scoped design requirement; registry candidate predicate is not pKa, delivery or exact experimental evidence",
    )
    if layout.family != policy.family:
        return result
    nodes, edges = np.asarray(nodes), np.asarray(edges)
    if len(nodes) != layout.record.node_count or not fixed_graph_preserved(
        nodes, edges, layout.record
    ):
        raise ValueError("Head-survival assessment changed declared atom count or fixed graph")
    with rdBase.BlockLogs():
        molecule = graph_to_molecule(nodes, edges, atoms)
    blocks = [b for b in layout.record.component_blocks if b.role == policy.head_role]
    if len(blocks) != 1:
        result.update(status="abstain", reason="head_origin_not_uniquely_qualified")
        return result
    if any(atom.GetFormalCharge() for atom in molecule.GetAtoms()):
        result.update(status="abstain", reason="charged_product_outside_neutral_policy")
        return result
    members = set(range(blocks[0].start, blocks[0].stop))
    core = {i for i in members if layout.record.core_position_states[i] > 1}
    if len(core) != 1 or molecule.GetAtomWithIdx(next(iter(core))).GetSymbol() != "N":
        result.update(status="abstain", reason="reacting_head_origin_not_uniquely_qualified")
        return result
    surviving = sorted(
        {match[0] for match in molecule.GetSubstructMatches(_query(policy.query_smarts))}
        & (members - core)
    )
    result.update(
        status="pass" if len(surviving) >= policy.minimum_retained else "fail",
        applicable=True,
        reason="qualified_neutral_head_origin",
        head_atoms=sorted(members),
        consumed_head_atoms=sorted(core),
        retained_head_atoms=surviving,
    )
    return result


def propose_head_survival(
    layout,
    state: Mapping,
    predictions: Mapping,
    atoms: Sequence,
    policy: HeadSurvivalPolicy,
    *,
    source_assessor: Callable[[str], dict],
    maximum_candidates: int = 8,
) -> dict:
    """Try score-ranked single C-to-N edits with TRAIN local support and full L1 replay.

    The required callback must run the unchanged complete family source executor
    and return its normal assessment with ``exact``. No unchecked graph is selected.
    Every call and rejected assessed candidate is retained. The original survives
    if scope, local support, valence, the new design predicate or source L1 fails.
    """
    if type(maximum_candidates) is not int or not 1 <= maximum_candidates <= 8:
        raise ValueError("Head-survival proposals require a bound from one to eight")
    local = {k: [int(x) for x in values] for k, values in state.items()}
    nodes, edges = state_graph(local)
    before = assess_head_survival(layout, nodes, edges, atoms, policy)
    result = dict(
        state=local,
        smiles=graph_smiles(nodes, edges, atoms),
        before=before,
        proposals=[],
        changed=False,
        costs=dict(candidate_graphs=0, source_checks=0),
        search_censored=False,
    )
    if before["status"] != "fail":
        return result
    logits = np.asarray(predictions["nodes"])
    if (
        logits.ndim != 2
        or logits.shape[0] < len(nodes)
        or logits.shape[1] != len(atoms)
        or not np.isfinite(logits).all()
    ):
        raise ValueError(
            "Head-survival requires finite node logits covering the full atom vocabulary"
        )
    nitrogen = [
        i
        for i, a in enumerate(atoms)
        if a.symbol == "N" and a.formal_charge == 0 and not a.aromatic and a.explicit_hydrogens == 0
    ]
    if len(nitrogen) != 1:
        result["reason"] = "neutral_generated_nitrogen_state_not_unique"
        return result
    nitrogen = nitrogen[0]
    capacity = _atom_capacity_table(atoms)[nitrogen]
    units = np.asarray([0, *BOND_VALENCE_UNITS.tolist()])[edges].sum(1)
    core, members = set(before["consumed_head_atoms"]), set(before["head_atoms"])
    possible = []
    for index in sorted(members - core):
        atom = atoms[int(nodes[index])]
        if (
            layout.record.fixed_atom_mask[index]
            or atom.symbol != "C"
            or atom.formal_charge
            or atom.aromatic
            or units[index] > capacity
        ):
            continue
        trial = nodes.copy()
        trial[index] = nitrogen
        with rdBase.BlockLogs():
            smiles = graph_smiles(trial, edges, atoms)
        if smiles is None:
            continue
        molecule = graph_to_molecule(trial, edges, atoms)
        if _environment(molecule, index, core, members) not in policy.train_environments:
            continue
        after = assess_head_survival(layout, trial, edges, atoms, policy)
        if after["status"] != "pass":
            continue
        score = float(logits[index, nitrogen] - logits[index, int(nodes[index])])
        possible.append((score, index, smiles, trial, after))
    possible.sort(key=lambda item: (-item[0], item[1], item[2]))
    result["search_censored"] = len(possible) > maximum_candidates
    result["eligible_local_edits"] = len(possible)
    for score, index, smiles, trial, after in possible[:maximum_candidates]:
        check = source_assessor(smiles)
        result["costs"]["candidate_graphs"] += 1
        result["costs"]["source_checks"] += 1
        proposal = dict(
            node=index,
            before_state=int(nodes[index]),
            after_state=nitrogen,
            node_logit_change=score,
            smiles=smiles,
            nodes=trial.tolist(),
            edges=edges.tolist(),
            design_assessment=after,
            source_assessment=check,
            forms_nitrogen_nitrogen_bond=any(
                atoms[int(nodes[j])].symbol == "N" for j in np.flatnonzero(edges[index])
            ),
            local_environment_sources=policy.controls["environment_sources"][
                _environment(graph_to_molecule(trial, edges, atoms), index, core, members)
            ],
            accepted=check.get("exact") is True,
        )
        result["proposals"].append(proposal)
        if proposal["accepted"] and not result["changed"]:
            result["state"]["nodes"] = trial.tolist()
            result.update(smiles=smiles, changed=True)
    return result
