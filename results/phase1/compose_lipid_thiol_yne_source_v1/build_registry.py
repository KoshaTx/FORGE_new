"""Source Fig. 1 net double thiol addition, followed by its documented amidation."""

import copy
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.core.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": str(sha256_file(path))}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build():
    family_path = ROOT / "data/vendor/qualified_reaction_families_v1.json"
    a3_path = ROOT / "data/vendor/qualified_a3_source_program_v2.json"
    thiol_path = ROOT / "data/vendor/qualified_thiol_maleate_a3_program_v1.json"
    family = json.loads(family_path.read_text())["reactions"]
    amide = copy.deepcopy(
        next(r for r in family if r["reaction_id"] == "amide_coupling_acid_amine")
    )
    a3 = json.loads(a3_path.read_text())["reactions"][0]
    thiol = json.loads(thiol_path.read_text())["reactions"][0]
    alkyne = a3["atom_mapped_reaction_smarts"].split(">>")[0].split(".")[-1]
    alkyne = alkyne.replace(":4]", ":1]").replace(":5]", ":2]")
    sh = thiol["atom_mapped_reaction_smarts"].split(">>")[0].split(".")[0]
    # Only the mapped net graph is new: source Fig. 1B puts one S on each
    # formerly alkyne carbon. No vinyl intermediate or regioselection is claimed.
    template = (
        ".".join([alkyne, sh.replace(":1]", ":3]"), sh.replace(":1]", ":4]")])
        + ">>[CH2:1]([S:3])[CH1:2][S:4]"
    )
    source = [
        {
            "kind": "doi",
            "identifier": "10.1016/j.biomaterials.2012.07.044",
            "locator": "Author-hosted accepted article PDF p. 3, Fig. 1A-B; p. 2, section 2.2.",
            "notes": "Source net connectivity and two-stage order. Computed consistency only; no individual analytical or biological labels transferred.",
        }
    ]
    implementation = {
        "kind": "source_drawing_overlay",
        "parent_registries": {
            "family": pin(family_path),
            "a3": pin(a3_path),
            "thiol": pin(thiol_path),
        },
        "derivation": pin(Path(__file__).resolve()),
    }
    shared = {
        k: copy.deepcopy(amide[k])
        for k in [
            "reaction_version",
            "status",
            "stereochemistry_policy",
            "protonation_and_salt_policy",
        ]
    }
    double = {
        **shared,
        "reaction_id": "source_li_double_thiol_yne",
        "atom_mapped_reaction_smarts": template,
        "architecture": "Two copies of one monothiol add across a terminal alkyne to form the source vicinal bis(thioether).",
        "reactant_roles": [],
        "sources": source,
        "implementation": implementation,
        "conditions": {
            "solvent": ["THF"],
            "temperature_c": None,
            "time_h": 1,
            "catalyst_or_reagent": ["DMPA", "UV 365 nm", "argon"],
        },
        "selectivity_policy": "All unfiltered net stage outcomes; equal source thiol identities required by program; no isolated vinyl intermediate claimed.",
        "known_positive_examples": [],
        "known_negative_examples": [],
    }
    alkyne_handle = Chem.MolFromSmarts(alkyne)
    for atom in alkyne_handle.GetAtoms():
        atom.SetAtomMapNum(0)
    for name, handle, maps in [
        ("alkynoic_linker", Chem.MolToSmarts(alkyne_handle), [1, 2]),
        ("thiol_first", thiol["reactant_roles"][0]["required_handle_smarts"], [3]),
        ("thiol_second", thiol["reactant_roles"][0]["required_handle_smarts"], [4]),
    ]:
        double["reactant_roles"].append(
            {
                "name": name,
                "count": 1,
                "required_handle_smarts": handle,
                "forbidden_smarts": [],
                "allowed_site_multiplicity": [1],
                "mapped_reactive_atoms": maps,
            }
        )
    # Retain the parent's N-H query syntax required by inverse hydrogen repair,
    # while adding its independently qualified non-acylated-amine scope.
    nh = a3["atom_mapped_reaction_smarts"].split(">>")[0].split(".")[0].replace(":1]", ":4]")
    amide["atom_mapped_reaction_smarts"] = (
        amide["atom_mapped_reaction_smarts"].split(">>")[0].split(".")[0]
        + "."
        + nh
        + ">>"
        + amide["atom_mapped_reaction_smarts"].split(">>")[1]
    )
    amide.update(
        reaction_id="source_li_tail_amidation",
        sources=source,
        implementation=implementation,
        known_positive_examples=[],
        known_negative_examples=[],
        selectivity_policy="Complete source preassembled acid and unprotected N-H head. Retain all forward and inverse attachment sites.",
        conditions={
            "solvent": ["dichloromethane", "DMF reagent solution"],
            "temperature_c": None,
            "time_h": 16,
            "catalyst_or_reagent": ["DIC", "HOBt", "argon"],
        },
    )
    amide["reactant_roles"][1]["required_handle_smarts"] = a3["reactant_roles"][0][
        "required_handle_smarts"
    ]
    program = {
        "program_id": "source_li_thiol_yne_then_amidation",
        "initial_role": "alkynoic_linker",
        "architecture_subfamily": "source_two_identical_thiol_tails",
        "scope": "Source two-stage net connectivity with generated component supervision restricted to neutral aliphatic monothiols, C/H/O alkynoic acid and C/H/N/O head. No claim of individual experimental execution for generated precursors.",
        "equal_component_groups": [["thiol_first", "thiol_second"]],
        "stages": [
            {
                "reaction_id": double["reaction_id"],
                "accumulator_role": "alkynoic_linker",
                "added_roles": ["thiol_first", "thiol_second"],
                "net_byproducts": {"formal_charge": 0},
                "source_step": "Fig. 1B step a; two thiol equivalents, one net stage",
            },
            {
                "reaction_id": amide["reaction_id"],
                "accumulator_role": "carboxylic_acid_tail",
                "added_roles": ["amine_head"],
                "net_byproducts": {"H": 2, "O": 1, "formal_charge": 0},
                "source_step": "Fig. 1B step b; documented subsequent amide coupling",
            },
        ],
        "terminal_constraints": {},
        "product_constraints": {},
    }
    basic_path = ROOT / "data/vendor/qualified_staar_source_program_v1.json"
    if not basic_path.exists():
        # The frozen source-contract registry is discovered by its existing name.
        basic_path = next((ROOT / "data/vendor").glob("*staar*registry*.json"))
    staar = json.loads(basic_path.read_text())
    # This query comes from the previously qualified source registry, not memory.
    basic = staar["programs"][0]["product_constraints"]["required_queries"]
    program["product_constraints"] = {"formal_charge": 0, "required_queries": copy.deepcopy(basic)}
    program["terminal_constraints"] = {
        "alkynoic_linker": {
            "allowed_atomic_numbers": [1, 6, 8],
            "formal_charge": 0,
            "allow_aromatic_atoms": False,
            "exact_element_counts": {"O": 2},
        },
        "amine_head": {"allowed_atomic_numbers": [1, 6, 7, 8], "formal_charge": 0},
        **{
            role: {
                "allowed_atomic_numbers": [1, 6, 16],
                "exact_element_counts": {"S": 1},
                "formal_charge": 0,
                "allow_aromatic_atoms": False,
            }
            for role in ["thiol_first", "thiol_second"]
        },
    }
    for rxn in (double, amide):
        rxn["implementation"]["parent_registries"]["head_constraint"] = pin(basic_path)
    assets = {
        "article": pin(
            ROOT
            / "results/phase1/compose_lipid_all_family_sources_v1/remaining/22902058/li-thiol-yne-main.pdf"
        ),
        "family_definitions": pin(
            ROOT
            / "data/source_cache/compose_lipid_supplement_2026-09-19/family_reaction_definitions.json"
        ),
    }
    registry = {
        "schema_version": "forge.source_staged_registry.v1",
        "source_assets": assets,
        "reactions": [double, amide],
        "programs": [program],
    }
    output = ROOT / "data/vendor/qualified_thiol_yne_staged_source_program_v1.json"
    dump(output, registry)
    source_controls = []
    for label, linker, head, product in [
        ("Fig1B_A1C11", "C#CCCC(=O)O", "CN(C)CCN", "CN(C)CCNC(=O)CCC(SCCCCCCCCCCC)CSCCCCCCCCCCC"),
        (
            "Fig1A_library_B7C6",
            "C#CCCCC(=O)O",
            "CN(C)CCN1CCNCC1",
            "CN(C)CCN1CCN(C(=O)CCCC(SCCCCCC)CSCCCCCC)CC1",
        ),
    ]:
        tail = "CCCCCCCCCCCS" if label.startswith("Fig1B") else "CCCCCCS"
        source_controls.append(
            {
                "label": label,
                "source_asset": "article",
                "family": "preassembled_thiol_yne_tail_amidation",
                "source_locator": (
                    "Fig. 1B fully drawn typical structure"
                    if label.startswith("Fig1B")
                    else "Fig. 1A drawn B, amine 7 and C6, combined through the Fig. 1B net graph; no individually characterized product claim"
                ),
                "components": {
                    "alkynoic_linker": linker,
                    "amine_head": head,
                    "thiol_first": tail,
                    "thiol_second": tail,
                },
                "expected_product": product,
                "expected_formula": rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(product)),
                "expected_computed_consistency_pass": True,
                "expected_stage_widths": [1, 1, 1],
            }
        )
    controls = {
        "schema_version": "forge.staged_source_adjudication.v1",
        "registry": pin(output),
        "assets": assets,
        "families": {
            "preassembled_thiol_yne_tail_amidation": {
                "program_id": program["program_id"],
                "source_role_occurrences": {
                    "alkynoic_linker": ["alkynoic_linker"],
                    "amine_head": ["amine_head"],
                    "thiol_tail": ["thiol_first", "thiol_second"],
                },
            }
        },
        "source_controls": source_controls,
        "regression_controls": [],
        "training_admitted": False,
        "experimental_execution_admitted": False,
        "limitations": [
            "The source 80% average yield and 70-80% estimated purity are library summaries, not labels for individual records.",
            "Fig. 1 states 5% DMPA; the procedure states 5 mg / 0.02 mmol for 0.5 mmol acid. Preserve both descriptions.",
            "Thiol-yne is a net stage; vinyl intermediates are not characterized by this program.",
            "Generated precursors outside this article are computational support only.",
        ],
    }
    negative = copy.deepcopy(source_controls[0])
    negative.update(
        label="non_equivalent_head_amines",
        expected_product="NCCN(C)CCCNC(=O)CCC(SCCCCCCCCCCC)CSCCCCCCCCCCC",
        expected_computed_consistency_pass=False,
        expected_stage_widths=[1, 1, 2],
        required_failed_checks=["unique_each_forward_stage", "unique_forward_exact"],
        evidence_kind="synthetic_regression_fixture_not_experimental_evidence",
    )
    negative.pop("source_asset")
    negative.pop("source_locator")
    negative["components"]["amine_head"] = "NCCN(C)CCCN"
    negative["expected_formula"] = rdMolDescriptors.CalcMolFormula(
        Chem.MolFromSmiles(negative["expected_product"])
    )
    controls["regression_controls"] = [negative]
    controls["procedure"] = {
        "source_locator": "Article PDF p. 2, section 2.2",
        "first_stage": {
            "alkynoic_acid_mmol": 0.5,
            "thiol_equivalents": 2,
            "THF_ml": [0.5, 0.5, 0.2],
            "DMPA_printed": "5 mg, 0.02 mmol",
            "argon_ultrasound_degas_min": 3,
            "UV_wavelength_nm": 365,
            "UV_intensity_mw_cm2": 1.87,
            "time_h": 1,
        },
        "second_stage": {
            "workup": "Evaporate THF, dissolve in 8 ml DCM, split into seven 1 ml batches",
            "DIC_ul_mmol": [12, 0.075],
            "amine_mmol": 0.063,
            "HOBt_solution_ul": 20,
            "HOBt_DMF_g_ml": 0.5,
            "time_h": 16,
            "atmosphere": "argon",
            "extraction": "Evaporate, extract with 2 ml hexane, centrifuge 10000g for 5 min, evaporate supernatant",
        },
    }
    dump(HERE / "adjudication.json", controls)
    from forge.assembly.repeated_components import RepeatBounds
    from forge.corpus.compose_lipid_staged_replay import CONFIG_SCHEMA, POLICY
    from dataclasses import asdict

    dump(
        HERE / "replay-config.json",
        {
            "schema_version": CONFIG_SCHEMA,
            "policy": POLICY,
            "inputs": {
                "preparation": pin(
                    ROOT / "results/phase1/compose_lipid_full_preparation_v1/result.json"
                ),
                "registry": pin(output),
                "family_definitions": assets["family_definitions"],
                "transform_controls": pin(HERE / "adjudication.json"),
            },
            "families": controls["families"],
            "search_bounds": asdict(RepeatBounds(maximum_events=2)),
        },
    )


if __name__ == "__main__":
    build()
