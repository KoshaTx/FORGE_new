"""Derive an opt-in source-audited atom-origin variant without changing graph chemistry."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from forge.assembly.families import LibraryAssemblyError


def derive_product_atom_map_variant(entry: dict, specification: Mapping) -> dict:
    """Permute only product map labels among equivalent element/charge/isotope states.

    The permutation is supplied by a separate pinned source adjudication, never inferred from
    target products. Reactant templates, product queries, role policies and all examples survive.
    This utility establishes map conservation, not scientific correctness of an adjudication.
    """
    if set(specification) != {
        "reaction_id",
        "reaction_version",
        "product_map_permutation",
        "source",
    }:
        raise LibraryAssemblyError("atom-map variant fields are not exact")
    name, version = specification["reaction_id"], specification["reaction_version"]
    if (
        not isinstance(name, str)
        or not name
        or name == entry["reaction_id"]
        or type(version) is not int
        or version <= entry["reaction_version"]
    ):
        raise LibraryAssemblyError("atom-map variant requires a new identity and later version")
    source = specification["source"]
    if not isinstance(source, dict) or not all(
        isinstance(source.get(key), str) and source[key]
        for key in ("kind", "identifier", "locator")
    ):
        raise LibraryAssemblyError("atom-map variant requires explicit source attribution")
    raw = specification["product_map_permutation"]
    if not isinstance(raw, dict) or not raw:
        raise LibraryAssemblyError("product map permutation is missing")
    if any(
        not isinstance(k, str)
        or not k.isdecimal()
        or str(int(k)) != k
        or type(v) is not int
        or v < 1
        for k, v in raw.items()
    ):
        raise LibraryAssemblyError("product map permutation must use positive integer labels")
    permutation = {int(k): v for k, v in raw.items()}
    if set(permutation) != set(permutation.values()) or all(k == v for k, v in permutation.items()):
        raise LibraryAssemblyError("product map permutation must be a nontrivial bijection")
    reaction = rdChemReactions.ReactionFromSmarts(entry["atom_mapped_reaction_smarts"])
    if reaction is None or reaction.GetNumProductTemplates() != 1:
        raise LibraryAssemblyError("atom-map variant needs a single-product transform")

    def atoms(templates):
        found = {}
        for template in templates:
            for atom in template.GetAtoms():
                label = atom.GetAtomMapNum()
                if label:
                    if label in found:
                        raise LibraryAssemblyError("duplicate atom map in reaction side")
                    found[label] = atom
        return found

    reactants = atoms(reaction.GetReactants())
    products = atoms(reaction.GetProducts())
    if set(permutation) - (set(reactants) & set(products)):
        raise LibraryAssemblyError("permutation refers to absent reactant/product atom maps")

    def state(atom):
        return atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsotope()

    for old, new in permutation.items():
        if state(reactants[old]) != state(reactants[new]) or state(products[old]) != state(
            products[new]
        ):
            raise LibraryAssemblyError("permutation changes elemental, charge or isotope identity")
    product = Chem.Mol(reaction.GetProductTemplate(0))
    for atom in product.GetAtoms():
        label = atom.GetAtomMapNum()
        atom.SetAtomMapNum(permutation.get(label, label))
    left, _ = entry["atom_mapped_reaction_smarts"].split(">>")
    variant = deepcopy(entry)
    variant.update(
        reaction_id=name,
        reaction_version=version,
        atom_mapped_reaction_smarts=left + ">>" + Chem.MolToSmarts(product),
    )
    variant["sources"] = [*variant.get("sources", []), deepcopy(source)]
    return variant
