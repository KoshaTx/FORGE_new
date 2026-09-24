"""Derive Han's epoxide opening and acyl-chloride O-acylation from pinned fragments."""

import copy
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdMolDescriptors

from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def build():
    parents = {
        name: ROOT / "data/vendor" / file
        for name, file in {
            "family": "qualified_reaction_families_v1.json",
            "ester": "qualified_maleate_ester_source_program_v1.json",
            "non_acyl_amine": "qualified_a3_source_program_v2.json",
            "head_constraint": "qualified_staar_source_program_v1.json",
        }.items()
    }
    docs = {name: json.loads(path.read_text()) for name, path in parents.items()}
    source_path = HERE / "control-transcriptions.json"
    source = json.loads(source_path.read_text())
    epoxide = copy.deepcopy(
        next(r for r in docs["family"]["reactions"] if r["reaction_id"] == "epoxide_opening_amine")
    )
    amine = docs["non_acyl_amine"]["reactions"][0]
    ester = copy.deepcopy(
        next(
            r
            for r in docs["ester"]["reactions"]
            if r["reaction_id"] == "source_aminoalcohol_o_esterification"
        )
    )
    chloroformate = next(
        r
        for r in docs["family"]["reactions"]
        if r["reaction_id"] == "carbamate_amine_chloroformate"
    )
    epoxide["reaction_id"] = "source_han_db_amine_epoxide"
    epoxide["atom_mapped_reaction_smarts"] = (
        amine["atom_mapped_reaction_smarts"].split(">>")[0].split(".")[0]
        + "."
        + epoxide["atom_mapped_reaction_smarts"].split(">>")[0].split(".")[1]
        + ">>"
        + epoxide["atom_mapped_reaction_smarts"].split(">>")[1]
    )
    epoxide["reactant_roles"][0]["required_handle_smarts"] = amine["reactant_roles"][0][
        "required_handle_smarts"
    ]
    epoxide["reactant_roles"][1]["name"] = "epoxide_tail"
    epoxide["reactant_roles"][1]["allowed_site_multiplicity"] = [1]
    ester_rxn = rdChemReactions.ReactionFromSmarts(ester["atom_mapped_reaction_smarts"])
    chloro_rxn = rdChemReactions.ReactionFromSmarts(chloroformate["atom_mapped_reaction_smarts"])
    chlorine = next(
        a for a in chloro_rxn.GetReactantTemplate(1).GetAtoms() if a.GetSymbol() == "Cl"
    )
    acid = Chem.RWMol(ester_rxn.GetReactantTemplate(0))
    chlorine = Chem.AtomFromSmarts(chlorine.GetSmarts())
    chlorine.SetAtomMapNum(3)
    leaving = next(a.GetIdx() for a in acid.GetAtoms() if a.GetAtomMapNum() == 3)
    acid.ReplaceAtom(leaving, chlorine)
    product = Chem.RWMol(ester_rxn.GetProductTemplate(0))
    # A source acyl chloride has a C substituent on the carbonyl carbon;
    # preserve that complete substituent and exclude chloroformates/carbamoyl chlorides.
    alpha = next(a for a in product.GetAtoms() if a.GetAtomMapNum() == 5)
    for molecule in (acid, product):
        carbon = Chem.AtomFromSmarts(alpha.GetSmarts())
        carbon.SetAtomMapNum(6)
        index = molecule.AddAtom(carbon)
        carbonyl = next(a.GetIdx() for a in molecule.GetAtoms() if a.GetAtomMapNum() == 1)
        molecule.AddBond(carbonyl, index, Chem.BondType.SINGLE)
    handle = Chem.Mol(acid)
    for atom in handle.GetAtoms():
        atom.SetAtomMapNum(0)
    ester["reaction_id"] = "source_han_db_acyl_chloride_o_acylation"
    ester["atom_mapped_reaction_smarts"] = (
        Chem.MolToSmarts(acid)
        + "."
        + Chem.MolToSmarts(ester_rxn.GetReactantTemplate(1))
        + ">>"
        + Chem.MolToSmarts(product)
    )
    ester["reactant_roles"][0]["name"] = "acyl_tail"
    ester["reactant_roles"][0]["required_handle_smarts"] = Chem.MolToSmarts(handle)
    ester["reactant_roles"][0]["mapped_reactive_atoms"] = [1, 2, 3, 6]
    ester["reactant_roles"][1]["allowed_site_multiplicity"] = [1, 2, 3]
    for reaction in (epoxide, ester):
        reaction["sources"] = [
            {
                "kind": "doi",
                "identifier": source["doi"],
                "locator": "Main Methods, General method for the synthesis of DB-lipidoids; SI PDF p. 3 Fig. S2; p. 23 Table S1.",
                "notes": "Two complete incorporated epoxide copies followed by two complete incorporated acyl chlorides. Computed source-stage consistency, not individual experimental execution.",
            }
        ]
        reaction["implementation"] = {
            "kind": "source_drawing_overlay",
            "parent_registries": {name: pin(ROOT, path) for name, path in parents.items()},
            "derivation": pin(ROOT, Path(__file__).resolve()),
        }
        reaction["known_positive_examples"] = []
        reaction["known_negative_examples"] = []
        reaction.pop("source_program", None)
        reaction["stereochemistry_policy"] = (
            "Constitutional stereo-free model; retain source stereo in the primary drawings."
        )
        reaction["conditions"] = copy.deepcopy(
            source["source_procedure"]["stage_1" if reaction is epoxide else "stage_2"]
        )
        reaction["architecture"] = (
            "One of two source-ordered grouped stages; no separate source execution claim for a mono-adduct."
        )
    epoxide["selectivity_policy"] = (
        "Source-drawn terminal epoxide carbon attacked by a neutral non-acyl N-H amine; all N attachment outcomes retained through both declared events."
    )
    ester["selectivity_policy"] = (
        "Source acyl chloride and aliphatic hydroxyl only; unprotected N-H competitors excluded; all O attachment outcomes retained through both declared events."
    )
    program = {
        "program_id": "source_han_db_two_grouped_stages",
        "initial_role": "amine_head",
        "architecture_subfamily": "han_db_two_stage_four_arm_rocket",
        "scope": "Net source-stage consistency with two identical complete epoxide copies followed by two identical complete acyl chloride copies. Generated component scope: neutral C/H/N/O amines and C/H/O body/acyl hydrophobes; no experimental execution claim.",
        "stages": [
            {
                "reaction_id": epoxide["reaction_id"],
                "accumulator_role": "amine_head",
                "added_roles": ["epoxide_tail"],
                "events": 2,
                "net_byproducts_per_event": {"formal_charge": 0},
                "source_step": "80 C, 48 h; two incorporated epoxide copies",
            },
            {
                "reaction_id": ester["reaction_id"],
                "accumulator_role": "aminoalcohol_head",
                "added_roles": ["acyl_tail"],
                "events": 2,
                "net_byproducts_per_event": {"H": 1, "Cl": 1, "formal_charge": 0},
                "source_step": "DCM/TEA, room temperature, 12 h; two O-acylations",
            },
        ],
        "terminal_constraints": {
            "amine_head": {"allowed_atomic_numbers": [1, 6, 7, 8], "formal_charge": 0},
            "epoxide_tail": {
                "allowed_atomic_numbers": [1, 6, 8],
                "formal_charge": 0,
                "allow_aromatic_atoms": False,
            },
            "acyl_tail": {
                "allowed_atomic_numbers": [1, 6, 8, 17],
                "formal_charge": 0,
                "allow_aromatic_atoms": False,
                "exact_element_counts": {"Cl": 1},
            },
        },
        "product_constraints": copy.deepcopy(
            docs["head_constraint"]["programs"][0]["product_constraints"]
        ),
    }
    assets = {
        **source["assets"],
        "control_transcriptions": pin(ROOT, source_path),
        "family_definitions": pin(
            ROOT,
            ROOT
            / "data/source_cache/compose_lipid_supplement_2026-09-19/family_reaction_definitions.json",
        ),
    }
    target = ROOT / "data/vendor/qualified_han_db_grouped_source_program_v1.json"
    dump(
        target,
        {
            "schema_version": "forge.source_grouped_registry.v1",
            "source_assets": assets,
            "reactions": [epoxide, ester],
            "grouped_programs": [program],
        },
    )
    family = source["family"]
    controls = []
    for c in source["controls"]:
        controls.append(
            {
                "label": c["label"],
                "family": family,
                "source_asset": "supplement",
                "components": c["complete_components"],
                "expected_product": c["expected_final_product"],
                "expected_stage_products": [
                    c["expected_stage_1_product"],
                    c["expected_final_product"],
                ],
                "expected_formula": c["neutral_formula"],
                "expected_computed_consistency_pass": True,
                "expected_stage_widths": [1, 1, 1],
            }
        )
    adjudication = {
        "schema_version": "forge.grouped_source_adjudication.v1",
        "registry": pin(ROOT, target),
        "assets": assets,
        "families": {
            family: {
                "program_id": program["program_id"],
                "registry_to_source_roles": {
                    role: role for role in program["terminal_constraints"]
                },
            }
        },
        "source_controls": controls,
        "regression_controls": [],
        "training_admitted": False,
        "experimental_execution_admitted": False,
        "evidence_basis": "computed_transform_consistency",
        "limitations": source["limitations"][:3]
        + [
            "Uniqueness is required at source-stage endpoints; all within-stage event paths are preserved without a claim of experimental event order."
        ],
    }
    # A remaining initial NH competitor cannot be silently consumed by O-acylation.
    negative = copy.deepcopy(controls[0])
    negative.update(
        label="unprotected_head_NH_competitor",
        expected_computed_consistency_pass=False,
        expected_stage_widths=[1, 2, 0],
        evidence_kind="synthetic_regression_fixture_not_experimental_evidence",
        required_failed_checks=["unique_each_completed_source_stage", "unique_forward_exact"],
    )
    negative.pop("source_asset")
    negative.pop("expected_stage_products")
    negative["components"]["amine_head"] = "NCCNC"
    negative["expected_product"] = "CNCCN(CC(OC(=O)CCCCC)CCCC)CC(OC(=O)CCCCC)CCCC"
    negative["expected_formula"] = rdMolDescriptors.CalcMolFormula(
        Chem.MolFromSmiles(negative["expected_product"])
    )
    adjudication["regression_controls"] = [negative]
    dump(HERE / "adjudication.json", adjudication)
    return target, adjudication


if __name__ == "__main__":
    build()
