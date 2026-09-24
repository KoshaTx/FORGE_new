"""Derive source-drawn maleate and O-ester graphs from frozen registry fragments."""

from __future__ import annotations

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


def entry(path, identity):
    return next(
        r for r in json.loads(path.read_text())["reactions"] if r["reaction_id"] == identity
    )


def fragment(mol, maps, remap=None):
    result = Chem.RWMol(mol)
    for i in reversed(range(result.GetNumAtoms())):
        if result.GetAtomWithIdx(i).GetAtomMapNum() not in maps:
            result.RemoveAtom(i)
    for atom in result.GetAtoms():
        atom.SetAtomMapNum((remap or {}).get(atom.GetAtomMapNum(), atom.GetAtomMapNum()))
    return result.GetMol()


def clear_maps(mol):
    value = Chem.Mol(mol)
    for atom in value.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmarts(value)


def role(name, handle, maximum, forbidden):
    return {
        "name": name,
        "required_handle_smarts": handle,
        "allowed_site_multiplicity": list(range(1, maximum + 1)),
        "forbidden_smarts": forbidden,
        "count": 1,
    }


def build():
    families = ROOT / "data/vendor/qualified_reaction_families_v1.json"
    michael = ROOT / "data/vendor/qualified_michael_source_program_v1.json"
    parent = entry(michael, "source_aza_michael_acrylate")
    acid = entry(families, "amide_coupling_acid_amine")
    acetal = entry(families, "acetal_aldehyde_diol")
    amine_handle = parent["reactant_roles"][0]["required_handle_smarts"]
    acid_handle = acid["reactant_roles"][0]["required_handle_smarts"]
    p = rdChemReactions.ReactionFromSmarts(parent["atom_mapped_reaction_smarts"])
    maleate = Chem.RWMol(p.GetReactantTemplate(1))
    # SI Figs. 1 and 6: replace terminal beta CH2 with CH and attach a second ester.
    beta = copy.copy(maleate.GetAtomWithIdx(1))
    beta.SetAtomMapNum(2)
    maleate.ReplaceAtom(0, beta)
    ester = fragment(p.GetReactantTemplate(1), {4, 5, 6}, {4: 7, 5: 8, 6: 9})
    offset = maleate.GetNumAtoms()
    maleate = Chem.RWMol(Chem.CombineMols(maleate, ester))
    maleate.AddBond(0, offset, Chem.BondType.SINGLE)
    product = Chem.RWMol(p.GetProductTemplate(0))
    beta_product = copy.copy(beta)
    product.ReplaceAtom(1, beta_product)
    offset = product.GetNumAtoms()
    product = Chem.RWMol(Chem.CombineMols(product, ester))
    product.AddBond(1, offset, Chem.BondType.SINGLE)
    # The drawn product and main text support one addition per N, including NH2.
    # Exclude N already attached to this succinate motif at the reactive atom.
    adduct = clear_maps(product)
    donor = parent["atom_mapped_reaction_smarts"].split(">>")[0].split(".")[0]
    donor = donor.removesuffix(":1]") + ";!$(" + adduct + "):1]"
    guarded_handle = donor.replace(":1]", "]")
    maleate_entry = {
        "reaction_id": "source_maleate_single_addition_per_nitrogen",
        "atom_mapped_reaction_smarts": donor
        + "."
        + Chem.MolToSmarts(maleate)
        + ">>"
        + Chem.MolToSmarts(product),
        "reactant_roles": [
            role("amine_head", guarded_handle, 7, [clear_maps(maleate)]),
            role("maleate", clear_maps(maleate), 1, [amine_handle]),
        ],
        "source_program": {
            "accumulator_role": "amine_head",
            "repeated_roles": ["maleate"],
            "event_count_field": "occupancy",
            "maximum_events": 7,
            "net_byproducts_per_event": {"formal_charge": 0},
        },
        "sources": [
            {
                "kind": "doi",
                "identifier": "10.1101/2025.02.25.640222",
                "publication_status": "preprint",
                "locator": "Main combinatorial synthesis; SI Fig. 1 and Figs. 6-7 (PDF pp. 1,7-8), 5D8",
            }
        ],
        "conditions": {
            "temperature_c": 80,
            "time_h": 24,
            "catalyst_or_reagent": [],
            "solvent": [],
            "exceptions": "Main procedure allows TEA for salts and isopropanol for insoluble amines.",
        },
        "selectivity_policy": "One addition per neutral non-acyl primary/secondary nitrogen, with all eligible sites and maleate orientations enumerated. Asymmetric or partial-occupancy constitutional ambiguity abstains. Thiol mechanism is outside this entry. Existing succinate-bound N is excluded.",
    }
    acid_rxn = rdChemReactions.ReactionFromSmarts(acid["atom_mapped_reaction_smarts"])
    acetal_rxn = rdChemReactions.ReactionFromSmarts(acetal["atom_mapped_reaction_smarts"])
    alcohol = fragment(acetal_rxn.GetReactantTemplate(1), {3, 4}, {3: 4, 4: 5})
    alcohol_product = fragment(acetal_rxn.GetProductTemplate(0), {3, 4}, {3: 4, 4: 5})
    acyl_product = fragment(acid_rxn.GetProductTemplate(0), {1, 2})
    ester_product = Chem.RWMol(Chem.CombineMols(acyl_product, alcohol_product))
    ester_product.AddBond(0, acyl_product.GetNumAtoms(), Chem.BondType.SINGLE)
    ester_entry = {
        "reaction_id": "source_aminoalcohol_o_esterification",
        "atom_mapped_reaction_smarts": Chem.MolToSmarts(acid_rxn.GetReactantTemplate(0))
        + "."
        + Chem.MolToSmarts(alcohol)
        + ">>"
        + Chem.MolToSmarts(ester_product),
        "reactant_roles": [
            role("acid_tail", acid_handle, 1, [amine_handle, clear_maps(alcohol)]),
            role("aminoalcohol_head", clear_maps(alcohol), 3, [amine_handle, acid_handle]),
        ],
        "source_program": {
            "accumulator_role": "aminoalcohol_head",
            "repeated_roles": ["acid_tail"],
            "event_count_field": "occupancy",
            "maximum_events": 3,
            "net_byproducts_per_event": {"H": 2, "O": 1, "formal_charge": 0},
        },
        "sources": [
            {
                "kind": "doi",
                "identifier": "10.1021/acsnano.2c07822",
                "locator": "SI Figs. S1-S3 (PDF pp. 3-5), AA3-DLin",
            }
        ],
        "conditions": {
            "temperature_c": None,
            "time_h": None,
            "catalyst_or_reagent": ["Candida antarctica lipase B"],
            "solvent": None,
            "procedure_status": "Full main-text synthesis procedure not recovered; no exact-execution admission.",
        },
        "selectivity_policy": "Aliphatic alcohol O esterification only, monocarboxylic acid, no competing free amine on either component. Enumerate all OH sites and require a unique final constitutional product. Net water per ester. OH oxygen retained, a transform convention rather than isotope evidence.",
    }
    for value in (maleate_entry, ester_entry):
        value.update(
            {
                "reaction_version": 1,
                "status": "qualified_for_enumeration",
                "known_positive_examples": [],
                "known_negative_examples": [],
                "stereochemistry_policy": "Constitutional stereo-free only.",
                "protonation_and_salt_policy": parent["protonation_and_salt_policy"],
                "implementation": {
                    "kind": "source_drawing_overlay",
                    "parent_registries": {"family": pin(families), "michael": pin(michael)},
                    "derivation": pin(Path(__file__).resolve()),
                },
            }
        )
    assets = {
        name: pin(HERE / filename)
        for name, filename in {
            "maleate_main": "maleate_article.xml",
            "maleate_si": "media-1.pdf",
            "esterification_si": "nn2c07822_si_001.pdf",
        }.items()
    }
    output = ROOT / "data/vendor/qualified_maleate_ester_source_program_v1.json"
    dump(
        output,
        {
            "schema_version": "forge.source_reaction_registry.v1",
            "source_assets": assets,
            "curation": {
                "evidence_basis": "computed_transform_consistency",
                "source_execution_admitted": False,
                "original_registries_modified": False,
            },
            "reactions": [maleate_entry, ester_entry],
        },
    )
    return output, assets


if __name__ == "__main__":
    build()
