"""Check source-defined precursor cores and coupled arms without changing the molecule."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rdkit import Chem

from forge.assembly.families import LibraryAssemblyError, constitutional_molecule


def assess_precursor_scaffold(
    smiles: str, specification: Mapping[str, Any], *, maximum_matches: int = 256
) -> dict[str, Any]:
    """Retain every scaffold witness; ambiguity or bounded search never qualifies.

    SMARTS, attachment bonds, allowed elements and coupling requirements come from the
    source-qualified registry. Fragmentation only checks an internal precursor invariant;
    it does not replace the complete precursor with smaller assembly components.
    """
    if type(maximum_matches) is not int or maximum_matches < 2:
        raise LibraryAssemblyError("scaffold match bound must be an integer of at least two")
    canonical = constitutional_molecule(smiles)[0]
    molecule = Chem.MolFromSmiles(canonical)
    rules = specification["precursor_scaffolds"]["scaffolds"]
    if not rules or len({r["id"] for r in rules}) != len(rules):
        raise LibraryAssemblyError("scaffold rules must be nonempty and uniquely named")
    requirement = specification["additional_required_query"]
    query = Chem.MolFromSmarts(requirement["smarts"])
    if query is None or type(requirement["exact_count"]) is not int:
        raise LibraryAssemblyError("invalid additional precursor query")
    matches = molecule.GetSubstructMatches(query, maxMatches=maximum_matches)
    saturated = len(matches) >= maximum_matches
    handle_count_pass = len(matches) == requirement["exact_count"] and not saturated
    witnesses = {}
    for rule in rules:
        core = Chem.MolFromSmarts(rule["core_smarts"])
        if core is None:
            raise LibraryAssemblyError("invalid source scaffold SMARTS")
        mapped = {a.GetAtomMapNum(): a.GetIdx() for a in core.GetAtoms() if a.GetAtomMapNum()}
        if len(mapped) != sum(bool(a.GetAtomMapNum()) for a in core.GetAtoms()):
            raise LibraryAssemblyError("source scaffold atom maps are duplicated")
        ports = rule["cut_bond_maps"]
        if (
            rule["anchor_map"] not in mapped
            or not ports
            or any(len(p) != 2 or any(n not in mapped for n in p) for p in ports)
            or len({tuple(sorted(p)) for p in ports}) != len(ports)
        ):
            raise LibraryAssemblyError("source scaffold anchor or attachment bonds are invalid")
        matches = molecule.GetSubstructMatches(core, uniquify=False, maxMatches=maximum_matches)
        saturated |= len(matches) >= maximum_matches
        for match in matches:
            indices = {number: match[index] for number, index in mapped.items()}
            bonds = [molecule.GetBondBetweenAtoms(indices[a], indices[b]) for a, b in ports]
            if any(bond is None for bond in bonds):
                raise LibraryAssemblyError("declared scaffold port is not a bond")
            fragmented = Chem.FragmentOnBonds(
                molecule,
                [bond.GetIdx() for bond in bonds],
                addDummies=True,
                dummyLabels=[(0, 0)] * len(bonds),
            )
            original_atoms: list[tuple[int, ...]] = []
            fragments = Chem.GetMolFrags(
                fragmented, asMols=True, sanitizeFrags=True, fragsMolAtomMapping=original_atoms
            )
            core_fragments, arm_fragments = [], []
            for fragment, atoms in zip(fragments, original_atoms, strict=True):
                if indices[rule["anchor_map"]] in atoms:
                    core_fragments.append(fragment)
                else:
                    arm_fragments.append(fragment)
            core_smiles = sorted(
                Chem.MolToSmiles(m, canonical=True, isomericSmiles=False) for m in core_fragments
            )
            arms = sorted(
                Chem.MolToSmiles(m, canonical=True, isomericSmiles=False) for m in arm_fragments
            )
            checks = {
                "one_core_and_all_arms_detached": len(core_fragments) == 1
                and len(arms) == len(ports),
                "identical_arms": not rule["identical_detached_arms"]
                or (bool(arms) and len(set(arms)) == 1),
                "arm_elements": bool(arms)
                and all(
                    atom.GetAtomicNum() in rule["arm_atomic_numbers"]
                    for fragment in arm_fragments
                    for atom in fragment.GetAtoms()
                ),
                "arm_aromaticity": rule["arm_aromatic_atoms_allowed"]
                or not any(
                    atom.GetIsAromatic()
                    for fragment in arm_fragments
                    for atom in fragment.GetAtoms()
                ),
            }
            witness = {
                "scaffold_id": rule["id"],
                "architecture_subfamily": rule["architecture_subfamily"],
                "core_fragments": core_smiles,
                "detached_arm_fragments": arms,
                "checks": checks,
            }
            witnesses[(rule["id"], tuple(core_smiles), tuple(arms))] = witness
    ordered = [witnesses[key] for key in sorted(witnesses)]
    return {
        "canonical_precursor": canonical,
        "complete_search": not saturated,
        "additional_handle_count_pass": handle_count_pass,
        "witnesses": ordered,
        "computed_scaffold_pass": not saturated
        and handle_count_pass
        and len(ordered) == 1
        and all(ordered[0]["checks"].values()),
        "experimental_source_bank_membership": False,
        "complete_precursor_retained": True,
    }
