"""Auditable target sampling and mapped-net-reaction export for corpus v2.

These checks establish graph correspondence, NOT reaction yield, availability,
ionization, or delivery. Historical feature frequencies are never acceptance
criteria. The net-reaction annotation is explicitly not an elementary mechanism.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math

from rdkit import Chem

from compose_lipid.chemistry.assembly_condition import (
    AssemblyAnchor, AssemblyCondition, AssemblyEvent, AssemblyInput,
)
from compose_lipid.chemistry.assembly_source import source_report
from compose_lipid.chemistry.scaffold_holdout import ScaffoldExclusion
from compose_lipid.data.source_mcr_checks import composition


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def constitution(mol):
    mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else Chem.Mol(mol)
    if mol is None:
        raise ValueError("unparseable molecular graph")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def split_for(smiles, seed, validation_fraction):
    """A constitutional combination holdout, NOT a component/scaffold holdout."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must lie in (0,1)")
    u = int(digest(["product_split", seed, smiles])[:16], 16) / 2**64
    return "validation" if u < validation_fraction else "train"


def target_weights(records, source_mass=0.5):
    """Equal reaction-class mass, then evidence stratum, then unique target.

    The 50/50 default is a declared confidence mixture, not an optimum or an
    attempt to reproduce historical chemistry percentages. Empty strata receive
    no mass; the surviving stratum receives its family's entire allocation.
    """
    if not 0 < source_mass < 1:
        raise ValueError("both evidence strata must have positive mass")
    groups, seen = defaultdict(lambda: defaultdict(list)), set()
    for r in records:
        if r["split"] != "train" or r["target_id"] in seen:
            raise ValueError("sampling requires unique training-only targets")
        seen.add(r["target_id"])
        groups[r["family"]][r["evidence_lane"]].append(r["target_id"])
    if not groups:
        raise ValueError("empty training corpus")
    weights = {}
    for lanes in groups.values():
        if set(lanes) - {"source_reconstructed", "virtual"}:
            raise ValueError("unknown evidence lane")
        masses = {k: source_mass if k == "source_reconstructed" else 1-source_mass for k in lanes}
        denom = sum(masses.values())
        for lane, ids in lanes.items():
            for key in ids:
                weights[key] = masses[lane] / denom / len(groups) / len(ids)
    if not math.isclose(sum(weights.values()), 1, abs_tol=1e-10):
        raise AssertionError("target weights do not normalize")
    return weights


def ordered_precursors(raw, trace):
    """Preserve actual reagent COPIES; dict order is never an atom-map order."""
    regions = trace["precursor_regions"]
    if [r["precursor_index"] for r in regions] != list(range(len(regions))):
        raise ValueError("precursor region indices must be consecutive and ordered")
    if isinstance(raw, dict):
        return [{"role": r["role"], "smiles": raw[r["role"]]} for r in regions]
    if not isinstance(raw, list) or len(raw) != len(regions):
        raise ValueError("missing full ordered precursor instances")
    out = []
    for value, region in zip(raw, regions, strict=True):
        if isinstance(value, dict):
            if value["role"] != region["role"]:
                raise ValueError("precursor role disagrees with mapped witness")
            value = value["smiles"]
        out.append({"role": region["role"], "smiles": value})
    return out


def _bond(mol, indices, a, b):
    bond = mol.GetBondBetweenAtoms(indices[a], indices[b])
    return bond.GetBondTypeAsDouble() if bond else 0


def verify_net_trace(precursors, trace, expected_smiles):
    """Reverse recorded changes, recover each complete precursor, check balance.

    Uses map ranges rather than assuming canonical SMILES preserve atom order.
    Only constitutional replay is asserted: stereo may be destroyed by a
    reaction and cannot be reconstructed from an absent stereo edit record.
    """
    if not (trace.get("all_heavy_atoms_accounted_for") is True and
            trace.get("element_isotope_hydrogen_charge_balanced") is True):
        raise ValueError("missing qualified full net trace")
    inputs = [Chem.MolFromSmiles(p["smiles"]) for p in precursors]
    outputs = [Chem.MolFromSmiles(s) for s in
               [trace["mapped_product_smiles"], *trace["mapped_byproduct_smiles"]]]
    if any(m is None or len(Chem.GetMolFrags(m)) != 1 for m in inputs + outputs):
        raise ValueError("invalid precursor or mapped output")
    if constitution(outputs[0]) != constitution(expected_smiles):
        raise ValueError("mapped product differs from stored target")
    counts = composition(inputs)
    hydrogen_equivalents = trace.get("net_reduction_hydrogen_equivalents", 0)
    if type(hydrogen_equivalents) is not int or hydrogen_equivalents < 0 or hydrogen_equivalents % 2:
        raise ValueError("invalid formal reduction equivalents")
    counts[(1, 0)] += hydrogen_equivalents
    if +counts != composition(outputs) or sum(Chem.GetFormalCharge(m) for m in inputs) != sum(
            Chem.GetFormalCharge(m) for m in outputs):
        raise ValueError("net element/isotope/hydrogen/charge balance failed")
    combined = Chem.Mol(outputs[0])
    for mol in outputs[1:]:
        combined = Chem.CombineMols(combined, mol)
    keys = [a.GetAtomMapNum() for a in combined.GetAtoms()]
    if sorted(keys) != list(range(1, sum(m.GetNumAtoms() for m in inputs)+1)):
        raise ValueError("lost, duplicate, or unknown atom map")
    indices = {a.GetAtomMapNum(): a.GetIdx() for a in combined.GetAtoms()}
    reverse = Chem.RWMol(combined)
    changed, pairs = set(), set()
    types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
             3: Chem.BondType.TRIPLE, 1.5: Chem.BondType.AROMATIC}
    for change in trace["bond_changes"]:
        a, b = change["atom_maps"]
        pair = tuple(sorted((a,b)))
        if pair in pairs or a == b or a not in indices or b not in indices:
            raise ValueError("invalid/repeated changed bond")
        pairs.add(pair)
        if change["before"] == change["after"] or _bond(reverse, indices, a, b) != change["after"]:
            raise ValueError("recorded successor bond differs")
        if change["after"]:
            reverse.RemoveBond(indices[a], indices[b])
        if change["before"]:
            reverse.AddBond(indices[a], indices[b], types[change["before"]])
        changed.update((a,b))
    attributes = set()
    for change in trace["atom_attribute_changes"]:
        key = change["atom_map"]
        if key in attributes or key not in indices:
            raise ValueError("invalid/repeated atom attribute change")
        attributes.add(key)
        atom = reverse.GetAtomWithIdx(indices[key])
        old = combined.GetAtomWithIdx(indices[key])
        if (old.GetFormalCharge(), old.GetTotalNumHs()) != (
                change["formal_charge"][1], change["hydrogens"][1]):
            raise ValueError("recorded successor atom attributes differ")
        atom.SetFormalCharge(change["formal_charge"][0])
        atom.SetNumExplicitHs(change["hydrogens"][0])
        atom.SetNoImplicit(True)
        changed.add(key)
    Chem.SanitizeMol(reverse)
    product_maps = {a.GetAtomMapNum() for a in outputs[0].GetAtoms()}
    core = changed & product_maps
    if not core or core != set(trace["reaction_center_product_atom_maps"]):
        raise ValueError("reaction core differs from net changes")
    offset, ownership = 0, {}
    for i, (mol, region) in enumerate(zip(inputs, trace["precursor_regions"], strict=True)):
        keys = set(range(offset+1, offset+mol.GetNumAtoms()+1))
        selected = [indices[k] for k in sorted(keys)]
        fragment = Chem.MolFragmentToSmiles(reverse, selected, canonical=True, isomericSmiles=True)
        if constitution(fragment) != constitution(mol):
            raise ValueError("reverse net trace does not recover complete precursor")
        # No residual inter-precursor edge may be hidden by fragment extraction.
        if any(n.GetAtomMapNum() not in keys for idx in selected
               for n in reverse.GetAtomWithIdx(idx).GetNeighbors()):
            raise ValueError("reverse trace leaves a cross-precursor bond")
        if (region["precursor_index"] != i or region["role"] != precursors[i]["role"] or
                set(region["product_atom_maps"]) != keys & product_maps or
                set(region["noncore_product_atom_maps"]) != (keys & product_maps)-core):
            raise ValueError("precursor-origin partition mismatch")
        ownership.update({k: i for k in keys})
        offset += mol.GetNumAtoms()
    return outputs[0], core, ownership


def reduction_trace(raw):
    """Normalize the saved Xue net-reduction record; no new chemistry is proposed."""
    if raw.get("all_atoms_accounted_for") is not True:
        raise ValueError("incomplete source reduction trace")
    inputs = [Chem.MolFromSmiles(s) for s in raw["precursors"]]
    product = Chem.MolFromSmiles(raw["mapped_product_smiles"])
    outputs = [product, *(Chem.MolFromSmiles(s) for s in raw["byproduct_mapped_smiles"])]
    before, after, offset = {}, {}, 0
    atoms = {}
    for i,m in enumerate(inputs):
        for a in m.GetAtoms():
            atoms[offset+a.GetIdx()+1] = (i,a)
        for b in m.GetBonds():
            before[tuple(sorted((offset+b.GetBeginAtomIdx()+1,offset+b.GetEndAtomIdx()+1)))] = b.GetBondTypeAsDouble()
        offset += m.GetNumAtoms()
    attrs = []
    for m in outputs:
        for a in m.GetAtoms():
            k = a.GetAtomMapNum()
            previous = atoms[k][1]
            if (previous.GetAtomicNum(),previous.GetIsotope()) != (a.GetAtomicNum(),a.GetIsotope()):
                raise ValueError("reduction atom correspondence changed")
            if (previous.GetTotalNumHs(),previous.GetFormalCharge()) != (a.GetTotalNumHs(),a.GetFormalCharge()):
                attrs.append({"atom_map": k,"hydrogens": [previous.GetTotalNumHs(),a.GetTotalNumHs()],
                              "formal_charge": [previous.GetFormalCharge(),a.GetFormalCharge()]})
        for b in m.GetBonds():
            after[tuple(sorted((b.GetBeginAtom().GetAtomMapNum(),b.GetEndAtom().GetAtomMapNum())))] = b.GetBondTypeAsDouble()
    changes = [{"atom_maps": list(k),"before": before.get(k,0),"after": after.get(k,0)}
               for k in sorted(before.keys() | after.keys()) if before.get(k,0) != after.get(k,0)]
    recorded = {tuple(sorted(c["maps"])): (c["before"],c["after"]) for c in raw["forward_edits"]}
    if recorded != {tuple(c["atom_maps"]): (c["before"],c["after"]) for c in changes}:
        raise ValueError("source reduction edits differ from mapped graph")
    pm = {a.GetAtomMapNum() for a in product.GetAtoms()}
    core = ({k for c in changes for k in c["atom_maps"]} | {c["atom_map"] for c in attrs}) & pm
    roles = ["amine"] + ["preassembled_aryl_ester_aldehyde"]*(len(inputs)-1)
    return {"mapped_product_smiles": raw["mapped_product_smiles"],
            "mapped_byproduct_smiles": raw["byproduct_mapped_smiles"],
            "bond_changes": changes, "atom_attribute_changes": attrs,
            "reaction_center_product_atom_maps": sorted(core),
            "precursor_regions": [{"precursor_index": i,"role": role,
                "product_atom_maps": sorted(k for k in pm if atoms[k][0] == i),
                "noncore_product_atom_maps": sorted(k for k in pm-core if atoms[k][0] == i)}
                for i,role in enumerate(roles)],
            "all_heavy_atoms_accounted_for": True, "element_isotope_hydrogen_charge_balanced": True,
            "net_reduction_hydrogen_equivalents": raw["hydrogen_equivalents_for_net_reduction"],
            "accounting_note": raw["accounting_note"]}


@lru_cache(maxsize=2048)
def restore_agile_precursor(stub, role):
    """Restore the typed handle to an AUTHOR product exterior; inference only."""
    from compose_lipid.data.agile_release_checks import precursor_substituents
    mol = Chem.MolFromSmiles(stub)
    if mol is None or role not in {"amine_side","aldehyde_side","isocyanide_side"}:
        raise ValueError("unsupported AGILE exterior")
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum()==0]
    if len(dummies)!=1 or dummies[0].GetDegree()!=1:
        raise ValueError("AGILE exterior requires one attachment")
    index = dummies[0].GetIdx()
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(index)
    atom.SetAtomicNum(6 if role=="aldehyde_side" else 7)
    atom.SetIsotope(0)
    atom.SetAtomMapNum(0)
    atom.SetFormalCharge(1 if role=="isocyanide_side" else 0)
    atom.SetNumExplicitHs(0)
    atom.SetNoImplicit(role=="isocyanide_side")
    if role!="amine_side":
        other = Chem.Atom(8 if role=="aldehyde_side" else 6)
        if role=="isocyanide_side":
            other.SetFormalCharge(-1)
            other.SetNoImplicit(True)
        rw.AddBond(index,rw.AddAtom(other),Chem.BondType.DOUBLE if role=="aldehyde_side" else Chem.BondType.TRIPLE)
    Chem.SanitizeMol(rw)
    result = constitution(rw)
    projection = precursor_substituents(result,role)
    if projection["status"]!="complete" or constitution(stub) not in projection["substituents"]:
        raise ValueError("restored AGILE handle fails exterior round trip")
    return result


@lru_cache(maxsize=1)
def exclusion():
    return ScaffoldExclusion.load()


def export_record(candidate, max_heavy=80):
    """Geometry for a *net recipe* supplied source, without hidden exteriors.

    One aggregate recipe event consumes all explicitly stored reagent copies.
    Upstream synthesis of a supplied complex precursor is not fabricated. This
    annotation granularity must be bound identically in training and inference.
    """
    trace = candidate["trace"]
    precursors = ordered_precursors(candidate["precursors"], trace)
    mapped, core_maps, ownership = verify_net_trace(precursors, trace, candidate["smiles"])
    mol = Chem.Mol(mapped)
    maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(mol)
    # completion_task serializes its target canonically. Its correspondence is
    # therefore defined on a freshly parsed canonical target, not raw map order.
    target_smiles = constitution(mol)
    target = Chem.MolFromSmiles(target_smiles)
    match = mol.GetSubstructMatch(target)
    if len(match) != mol.GetNumAtoms():
        raise ValueError("target index correspondence failed")
    target_maps = [maps[i] for i in match]
    if not 1 <= target.GetNumHeavyAtoms() <= max_heavy:
        raise ValueError("outside_declared_80_heavy_atom_target_support")
    if any(a.GetSymbol() not in {"C", "N", "O", "P", "S"} or a.GetIsotope()
           or a.GetNumRadicalElectrons() for a in target.GetAtoms()):
        raise ValueError("outside_current_element_isotope_radical_support")
    verdict = exclusion().verdict(target, family=candidate["family"])
    if verdict.excluded:
        raise ValueError("mechanistic_holdout:" + ",".join(verdict.criteria))
    origins = [{"target_atom": i, "precursor_role": precursors[ownership[k]]["role"],
                "reagent_instance": ownership[k], "atom_map": k,
                "core_events": [0] if k in core_maps else []}
               for i,k in enumerate(target_maps)]
    witness = {"target_smiles": target_smiles, "reaction_core_union": [
        i for i,k in enumerate(target_maps) if k in core_maps], "atom_origins": origins}
    source = source_report(witness)
    if source["status"] != "connected_capped_source":
        raise ValueError("source_geometry:" + source.get("reason", source["status"]))
    mapping = source["task"]["source_to_target"]
    if len(mapping) == target.GetNumAtoms():
        raise ValueError("no_exterior_to_generate")
    roles = tuple(p["role"] for p in precursors)
    core_slots = tuple(i for i,t in enumerate(mapping) if target_maps[t] in core_maps)
    condition = AssemblyCondition(roles, (AssemblyEvent(tuple(
        AssemblyInput("reagent", i, role) for i,role in enumerate(roles)), core_slots),),
        tuple(AssemblyAnchor(i, origins[t]["precursor_role"], origins[t]["reagent_instance"],
                             None, target_maps[t] in core_maps) for i,t in enumerate(mapping)))
    deployment = {"source_condition": source["task"]["condition"], "assembly": condition.as_record(),
                  "event_reaction_ids": [candidate["family"]]}
    return {"target_id": candidate["target_id"], "family": candidate["family"],
            "split": candidate["split"], "evidence_lane": candidate["evidence_lane"],
            "constitutional_smiles": constitution(target),
            "source_isomeric_smiles": candidate["smiles"],
            "target_mass_before_sampling": 1, "source_reference": candidate["reference"],
            "evidence_tier": candidate["tier"], "precursors": precursors,
            "net_trace": trace, "program": witness, "source": source,
            "deployment": deployment, "reaction_annotation": "aggregate_net_recipe_not_elementary_steps",
            "precursor_roles_are_biological_regions": False,
            "source_hull_may_include_precursor_connector_and_ring_atoms": True,
            "constitutional_net_replay": True, "stereochemical_replay_claimed": False,
            "synthetic_feasibility_certified": False, "training_ready": False}


def weighted_prior(records, weights):
    """Joint supplied-core/program prior under the exact declared target measure."""
    entries, masses, counts = {}, Counter(), Counter()
    seen = set()
    for r in records:
        if r["split"] != "train" or r["target_id"] in seen:
            raise ValueError("prior requires unique training-only targets")
        seen.add(r["target_id"])
        item = {"family": r["family"], **r["deployment"]}
        AssemblyCondition.from_record(item["assembly"])
        key = digest(item)
        entries[key] = item
        masses[key] += weights[r["target_id"]]
        counts[key] += 1
    if seen != set(weights) or not math.isclose(sum(masses.values()),1,abs_tol=1e-10):
        raise ValueError("prior and target measure disagree")
    return {"schema": "weighted_net_recipe_prior_v1", "validation_targets_used": 0,
            "training_target_count": len(seen), "entries": [
                {"identity": k, "probability": masses[k], "target_count": counts[k], **entries[k]}
                for k in sorted(entries)]}
