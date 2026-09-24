"""Derive the source-drawn four-component graph from pinned registry fragments."""

import copy
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from forge.core.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def pin(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": str(sha256_file(path))}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reaction(path, identity):
    return next(
        r for r in json.loads(path.read_text())["reactions"] if r["reaction_id"] == identity
    )


def build():
    ugi_path = ROOT / "data/vendor/qualified_reactions_v1.json"
    family_path = ROOT / "data/vendor/qualified_reaction_families_v1.json"
    functional_path = ROOT / "data/vendor/qualified_a3_source_program_v2.json"
    ugi = reaction(ugi_path, "ugi_3cr_agile")
    acid_parent = reaction(family_path, "passerini_3cr")
    functional = reaction(functional_path, "a3_amine_aldehyde_terminal_alkyne")
    ugi_rxn = rdChemReactions.ReactionFromSmarts(ugi["atom_mapped_reaction_smarts"])
    acid_rxn = rdChemReactions.ReactionFromSmarts(acid_parent["atom_mapped_reaction_smarts"])

    # Read all atom/handle query text from existing registries. Only the source-drawn
    # graph change is new: an acid-derived acyl carbon bonds to the amine N (Fig. 1).
    reactants = ugi["atom_mapped_reaction_smarts"].split(">>")[0].split(".")
    reactants[0] = functional["atom_mapped_reaction_smarts"].split(">>")[0].split(".")[0]
    acid = Chem.Mol(acid_rxn.GetReactantTemplate(0))
    for atom in acid.GetAtoms():
        atom.SetAtomMapNum(atom.GetAtomMapNum() + 4)
    reactants.append(Chem.MolToSmarts(acid))
    product = Chem.RWMol(ugi_rxn.GetProductTemplate(0))
    for atom in product.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(7)
    acid_product = acid_rxn.GetProductTemplate(0)
    indexes = {}
    for atom in acid_product.GetAtoms():
        if atom.GetAtomMapNum() in (1, 2):
            cloned = copy.copy(atom)
            cloned.SetAtomMapNum(atom.GetAtomMapNum() + 4)
            indexes[atom.GetAtomMapNum()] = product.AddAtom(cloned)
    product.AddBond(indexes[1], indexes[2], acid_product.GetBondBetweenAtoms(0, 1).GetBondType())
    nitrogen = next(a.GetIdx() for a in product.GetAtoms() if a.GetAtomMapNum() == 1)
    product.AddBond(nitrogen, indexes[1], acid_product.GetBondBetweenAtoms(0, 2).GetBondType())

    roles = copy.deepcopy(ugi["reactant_roles"])
    roles[0]["required_handle_smarts"] = functional["reactant_roles"][0]["required_handle_smarts"]
    roles.append(copy.deepcopy(acid_parent["reactant_roles"][0]))
    roles[-1]["mapped_reactive_atoms"] = [5, 6, 7]
    # The v8 key explicitly specifies one complete monocarboxylic acid.
    roles[-1]["allowed_site_multiplicity"] = [1]
    entry = {
        "reaction_id": "source_aldehyde_ugi4",
        "reaction_version": 1,
        "status": "qualified_for_enumeration",
        "architecture": "one-event primary-amine aldehyde isocyanide carboxylic-acid Ugi 4CR",
        "atom_mapped_reaction_smarts": ".".join(reactants) + ">>" + Chem.MolToSmarts(product),
        "reactant_roles": roles,
        "net_byproducts": {"H": 2, "O": 1, "formal_charge": 0},
        "selectivity_policy": "Enumerate unfiltered sites. Require a primary neutral non-acyl NH2 witness, an aldehyde CH witness, one complete inverse tuple and one exact forward product.",
        "stereochemistry_policy": "Constitutional stereo-free model; no stereoselectivity claim.",
        "protonation_and_salt_policy": ugi["protonation_and_salt_policy"],
        "conditions": {
            "solvent": ["dichloromethane", "methanol"],
            "temperature_c": None,
            "time_h": None,
            "catalyst_or_reagent": [],
            "reported_yield_range": None,
        },
        "sources": [
            {
                "kind": "doi",
                "identifier": "10.1038/s41563-024-01867-3",
                "locator": "Main Fig. 1; SI printed pp. 7-9 and 15 (PDF pp. 8-10 and 16), 119-23",
                "notes": "Computed constitutional assembly only; corpus rows do not inherit source execution or potency.",
            }
        ],
        "implementation": {
            "kind": "source_drawing_overlay",
            "parent_registries": {
                "ugi": pin(ugi_path),
                "acid": pin(family_path),
                "functional_groups": pin(functional_path),
            },
            "derivation": pin(Path(__file__).resolve()),
        },
        "known_positive_examples": [],
        "known_negative_examples": [],
    }
    assets = {
        "supplement": pin(HERE / "si.pdf"),
        "main_scheme": pin(HERE / "fig1.png"),
        "decomposition_key": pin(
            ROOT / "data/source_cache/compose_lipid_drive_2026-09-18/family_decomposition_key.json"
        ),
    }
    output = ROOT / "data/vendor/qualified_ugi4_source_program_v1.json"
    dump(
        output,
        {
            "schema_version": "forge.source_reaction_registry.v1",
            "source_assets": assets,
            "curation": {
                "evidence_basis": "computed_transform_consistency",
                "original_registries_modified": False,
                "graph_derivation": "Reuse Ugi N1/C2/C3/N4 product scaffold and Passerini acid C/O query fragments. Add acid C5=O6, bond N1-C5 and map the product amide O7 to the acid hydroxy O. This is a Mumm atom-origin convention, not isotope evidence. Aldehyde oxygen leaves in the net water inventory. No physical sequence of isolated intermediates is asserted.",
            },
            "reactions": [entry],
        },
    )
    return output, assets, functional_path


if __name__ == "__main__":
    build()
