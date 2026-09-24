"""Xu SI Fig. S1: acid opens terminal epoxide, then amino-acid O-esterification."""

import copy
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.corpus.compose_lipid_source_view import dump, pin
from forge.corpus.compose_lipid_staged_replay import CONFIG_SCHEMA, POLICY

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FAMILY = "acid_epoxide_diester_multistep"


def main():
    parents = {
        "family": ROOT / "data/vendor/qualified_reaction_families_v1.json",
        "ester": ROOT / "data/vendor/qualified_maleate_ester_source_program_v1.json",
        "basic_head": ROOT / "data/vendor/qualified_staar_source_program_v1.json",
    }
    documents = {k: json.loads(v.read_text()) for k, v in parents.items()}
    epoxide = next(
        r for r in documents["family"]["reactions"] if r["reaction_id"] == "epoxide_opening_amine"
    )
    ester = copy.deepcopy(
        next(
            r
            for r in documents["ester"]["reactions"]
            if r["reaction_id"] == "source_aminoalcohol_o_esterification"
        )
    )
    acid_fragment = ester["atom_mapped_reaction_smarts"].split(">>")[0].split(".")[0]
    epoxide_fragment = epoxide["atom_mapped_reaction_smarts"].split(">>")[0].split(".")[1]
    molecule = Chem.MolFromSmarts(epoxide_fragment)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(atom.GetAtomMapNum() + 2)
    epoxide_fragment = Chem.MolToSmarts(molecule)
    opening = copy.deepcopy(ester)
    opening.update(
        reaction_id="source_xu_acid_terminal_epoxide_opening",
        # New net graph from SI Fig. S1 step 1. All six mapped atoms derive from
        # the pinned acid and terminal-epoxide interfaces; no oxygen is introduced.
        atom_mapped_reaction_smarts=acid_fragment
        + "."
        + epoxide_fragment
        + ">>[C:1](=[O:2])[O:3][CH2:4][CH1:5][OH1:6]",
        architecture="Source-drawn primary hydroxy-ester from hydrophobic acid and terminal epoxide",
        selectivity_policy="Source Fig. S1 ester attaches to the terminal epoxide carbon and OH remains on the substituted carbon. Enumerate every eligible interface; no target-selected site.",
        conditions={
            "catalyst_or_reagent": ["FeCl3", "pyridine"],
            "temperature": "room temperature",
            "time_h": 12,
            "solvent": None,
            "quantities": None,
            "source_locator": "SI PDF p. 2, Fig. S1 step 1",
        },
    )
    acid_role = copy.deepcopy(ester["reactant_roles"][0])
    acid_role["name"] = "hydrophobic_acid"
    acid_role["mapped_reactive_atoms"] = [1, 2, 3]
    ring_role = copy.deepcopy(epoxide["reactant_roles"][1])
    ring_role.update(name="epoxide", allowed_site_multiplicity=[1], mapped_reactive_atoms=[4, 5, 6])
    ring_role["forbidden_smarts"] = [
        *ester["reactant_roles"][1]["forbidden_smarts"],
        ester["reactant_roles"][1]["required_handle_smarts"],
    ]
    opening["reactant_roles"] = [acid_role, ring_role]
    ester.update(
        reaction_id="source_xu_aminoacid_esterification",
        architecture="Source-drawn amino-acid ester on the retained secondary alcohol",
        conditions={
            "catalyst_or_reagent": ["EDC hydrochloride", "DMAP", "DIPEA"],
            "solvent": ["dichloromethane"],
            "temperature": "room temperature",
            "time_h": 12,
            "quantities": None,
            "source_locator": "SI PDF p. 2, Fig. S1 step 2",
        },
    )
    ester["reactant_roles"][0]["name"] = "amine_acid_head"
    ester["reactant_roles"][1]["name"] = "hydroxy_ester"
    ester["reactant_roles"][1]["allowed_site_multiplicity"] = [1]
    for reaction in (opening, ester):
        reaction.pop("source_program", None)
        reaction["sources"] = [
            {
                "kind": "doi",
                "identifier": "10.1002/adhm.202302691",
                "locator": "SI PDF p. 2 Fig. S1; p. 3 E12CA1A3 drawing, NMR and HRMS; p. 21 Fig. S23 step f corroborates amino-acid identity.",
                "notes": "Drawn 4-(dimethylamino)butanoyl graph agrees with reported formula/mass; printed propanoyl name conflicts and is preserved separately. Computed consistency only.",
            }
        ]
        reaction["implementation"] = {
            "kind": "source_drawing_overlay",
            "parent_registries": {k: pin(ROOT, v) for k, v in parents.items()},
            "derivation": pin(ROOT, Path(__file__).resolve()),
        }
        reaction["known_positive_examples"] = []
        reaction["known_negative_examples"] = []
        reaction["stereochemistry_policy"] = (
            "Constitutional stereo-free; source drawings and spectra preserved separately."
        )
    basic = copy.deepcopy(documents["basic_head"]["programs"][0]["product_constraints"])
    program = {
        "program_id": "source_xu_acid_epoxide_then_aminoacid_ester",
        "initial_role": "epoxide",
        "architecture_subfamily": "acid_epoxide_site_ordered_12_diester",
        "scope": "Source-ordered 1:1:1 net construction with complete monocarboxylic acids and a complete terminal epoxide. Neutral aliphatic CHO hydrophobes and CHNO ionizable acid head; no source execution/yield claim for generated components.",
        "equal_component_groups": [],
        "stages": [
            {
                "reaction_id": opening["reaction_id"],
                "accumulator_role": "epoxide",
                "added_roles": ["hydrophobic_acid"],
                "net_byproducts": {"formal_charge": 0},
                "source_step": "SI Fig. S1 step 1: terminal acid ring opening",
            },
            {
                "reaction_id": ester["reaction_id"],
                "accumulator_role": "hydroxy_ester",
                "added_roles": ["amine_acid_head"],
                "net_byproducts": {"H": 2, "O": 1, "formal_charge": 0},
                "source_step": "SI Fig. S1 step 2: esterification of remaining OH",
            },
        ],
        "terminal_constraints": {
            "epoxide": {
                "allowed_atomic_numbers": [1, 6, 8],
                "formal_charge": 0,
                "allow_aromatic_atoms": False,
            },
            "hydrophobic_acid": {
                "allowed_atomic_numbers": [1, 6, 8],
                "formal_charge": 0,
                "allow_aromatic_atoms": False,
            },
            "amine_acid_head": {
                "allowed_atomic_numbers": [1, 6, 7, 8],
                "allow_aromatic_atoms": False,
                **basic,
            },
        },
        "product_constraints": basic,
    }
    assets = {
        "xu_si": pin(
            ROOT,
            ROOT
            / "data/source_cache/compose_lipid_user_papers_20260922/adhm202302691-sup-0001-suppmat.pdf",
        ),
        "intake_review": pin(
            ROOT, ROOT / "results/phase1/compose_lipid_user_papers_v1/review.json"
        ),
    }
    target = ROOT / "data/vendor/qualified_acid_epoxide_staged_source_program_v1.json"
    if target.exists():
        raise FileExistsError(target)
    dump(
        target,
        {
            "schema_version": "forge.source_staged_registry.v1",
            "source_assets": assets,
            "reactions": [opening, ester],
            "programs": [program],
        },
    )
    product = "CN(C)CCCC(=O)OC(CCCCCCCCCC)COC(=O)C(CCCCCC)CCCCCCCC"
    components = {
        "epoxide": "CCCCCCCCCCC1CO1",
        "hydrophobic_acid": "O=C(O)C(CCCCCC)CCCCCCCC",
        "amine_acid_head": "CN(C)CCCC(=O)O",
    }
    positive = {
        "label": "Xu_E12CA1A3_drawn_structure",
        "family": FAMILY,
        "source_asset": "xu_si",
        "source_locator": "SI pp. 2-3, Fig. S1 and E12CA1A3 characterization; p. 21 step f and p. 22 analogue corroboration",
        "components": components,
        "expected_product": product,
        "expected_formula": "C34H67NO4",
        "expected_stage_products": ["CCCCCCCCCCC(O)COC(=O)C(CCCCCC)CCCCCCCC", product],
        "expected_computed_consistency_pass": True,
        "expected_stage_widths": [1, 1, 1],
        "evidence_basis": "computed_transform_consistency",
        "disposition": "admit_transform_consistency",
    }
    regressions = []
    for label, wrong, failed in [
        (
            "printed_shorter_head_name",
            "CN(C)CCC(=O)OC(CCCCCCCCCC)COC(=O)C(CCCCCC)CCCCCCCC",
            ["unique_forward_exact", "full_element_hydrogen_charge_balance"],
        ),
        (
            "head_and_hydrophobe_site_swap",
            "CN(C)CCCC(=O)OCC(CCCCCCCCCC)OC(=O)C(CCCCCC)CCCCCCCC",
            ["unique_forward_exact"],
        ),
    ]:
        regressions.append(
            {
                "label": label,
                "family": FAMILY,
                "components": components,
                "expected_product": wrong,
                "expected_formula": rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(wrong)),
                "expected_computed_consistency_pass": False,
                "expected_stage_widths": [1, 1, 1],
                "required_failed_checks": failed,
                "evidence_basis": (
                    "source_conflict"
                    if label.startswith("printed")
                    else "computed_transform_consistency"
                ),
                "disposition": "reject_claim",
                "scope": "Rejection of equivalence to the source-drawn construction; not a claimed experimental failure.",
            }
        )
    binding = {
        FAMILY: {
            "program_id": program["program_id"],
            "source_role_occurrences": {r: [r] for r in components},
        }
    }
    adjudication_path = HERE / "adjudication.json"
    dump(
        adjudication_path,
        {
            "schema_version": "forge.staged_source_adjudication.v1",
            "registry": pin(ROOT, target),
            "assets": assets,
            "families": binding,
            "source_controls": [positive],
            "regression_controls": regressions,
            "training_admitted": False,
            "experimental_execution_admitted": False,
            "reported_analytical_evidence": {
                "source_locator": "SI p. 3",
                "ion_formula": "C34H68NO4+",
                "ion": "[M+H]+",
                "calculated_mz": 554.5143,
                "found_mz": 554.5173,
                "appearance": "colorless oil",
                "spectra": ["1H NMR", "13C NMR", "ESI-HRMS"],
            },
            "source_discrepancy": {
                "printed_name_head": "3-(dimethylamino)propanoyl",
                "drawn_head": "4-(dimethylamino)butanoyl",
                "resolution_for_computed_control": "Use independently drawn structure corroborated by reported formula/mass and Fig. S23 step f; retain conflicting printed name and admit no exact-execution label.",
            },
            "limitations": [
                "Reagent amounts, full workup, isolated yield and numerical purity are not supplied by Fig. S1; no exact-execution admission.",
                "Alcohol/acid oxygen mapping is a net-transform convention, not isotope tracing.",
                "C3/C4/C5 analogues use a different diol route and do not expand the terminal-epoxide ring-opening claim.",
            ],
        },
    )
    dump(
        HERE / "replay-config.json",
        {
            "schema_version": CONFIG_SCHEMA,
            "policy": POLICY,
            "families": binding,
            "inputs": {
                "preparation": pin(
                    ROOT, ROOT / "results/phase1/compose_lipid_full_preparation_v1/result.json"
                ),
                "registry": pin(ROOT, target),
                "transform_controls": pin(ROOT, adjudication_path),
                "family_definitions": pin(
                    ROOT,
                    ROOT
                    / "data/source_cache/compose_lipid_supplement_2026-09-19/family_reaction_definitions.json",
                ),
            },
            "search_bounds": {
                "maximum_events": 2,
                "maximum_outcomes": 256,
                "maximum_states": 4096,
                "maximum_transitions": 16384,
            },
        },
    )


if __name__ == "__main__":
    main()
