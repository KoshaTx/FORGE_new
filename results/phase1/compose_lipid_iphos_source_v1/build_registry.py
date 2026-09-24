"""Source-drawn iPhos charge forms, with one ring opening per original nitrogen."""

import copy
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdMolDescriptors, rdqueries

from forge.core.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SI = ROOT / "results/phase1/compose_lipid_all_family_sources_v1/remaining/33542471"


def pin(p):
    return {"path": p.relative_to(ROOT).as_posix(), "sha256": str(sha256_file(p))}


def dump(p, d):
    p.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")


def build():
    paths = {
        k: ROOT / v
        for k, v in {
            "ring_connectivity": "data/vendor/qualified_reaction_families_v1.json",
            "functional_amine": "data/vendor/qualified_a3_source_program_v2.json",
            "basic_nitrogen": "data/vendor/qualified_staar_source_program_v1.json",
        }.items()
    }
    parent = next(
        r
        for r in json.loads(paths["ring_connectivity"].read_text())["reactions"]
        if r["reaction_id"] == "iphos_amine_dioxaphospholane"
    )
    functional = json.loads(paths["functional_amine"].read_text())["reactions"][0]
    basic = json.loads(paths["basic_nitrogen"].read_text())["programs"][0]["product_constraints"][
        "required_queries"
    ][0]["smarts"]
    # Extract the existing neutral aliphatic non-acyl N alternative, excluding pyridine.
    all_nitrogen = basic[len("[$(") :].split("]),$(")[0] + "]"
    assert Chem.MolFromSmarts(all_nitrogen).GetNumAtoms() == 1
    rxn = rdChemReactions.ReactionFromSmarts(parent["atom_mapped_reaction_smarts"])
    ring = Chem.RWMol(rxn.GetReactantTemplate(1))
    product = Chem.RWMol(rxn.GetProductTemplate(0))
    # The source Pm has an alkoxy substituent. Add this intact O-C branch to the
    # pinned ring fragment; do not silently allow a P-C or P-N substitute.
    source = Chem.MolFromSmiles("COP1(=O)OCCO1")
    for mol in (ring, product):
        p_index = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 5)
        oxygen = Chem.AtomFromSmarts(source.GetAtomWithIdx(1).GetSmarts())
        oxygen.ExpandQuery(rdqueries.FormalChargeEqualsQueryAtom(0))
        oxygen.SetAtomMapNum(8)
        carbon = Chem.AtomFromSmarts(source.GetAtomWithIdx(0).GetSmarts())
        carbon.ExpandQuery(rdqueries.HybridizationEqualsQueryAtom(int(Chem.HybridizationType.SP3)))
        carbon.SetAtomMapNum(9)
        oi, ci = mol.AddAtom(oxygen), mol.AddAtom(carbon)
        mol.AddBond(p_index, oi, Chem.BondType.SINGLE)
        mol.AddBond(oi, ci, Chem.BondType.SINGLE)
    for atom in ring.GetAtoms():
        if atom.GetAtomMapNum() == 7:
            atom.ExpandQuery(rdqueries.FormalChargeEqualsQueryAtom(0))
    ring_handle = Chem.Mol(ring)
    for atom in ring_handle.GetAtoms():
        atom.SetAtomMapNum(0)
    installed = Chem.Mol(product)
    for atom in installed.GetAtoms():
        atom.SetAtomMapNum(0)
    installed = Chem.MolToSmarts(installed)
    nh = functional["atom_mapped_reaction_smarts"].split(">>")[0].split(".")[0].replace(":1]", "]")
    nh = nh[:-1] + ";!$(" + installed + ")]"
    assets = {
        "supplement": pin(SI / "iphos-si.pdf"),
        "main_figure": pin(HERE / "fig1.png"),
        "family_definitions": pin(
            ROOT
            / "data/source_cache/compose_lipid_supplement_2026-09-19/family_reaction_definitions.json"
        ),
    }
    reactions = []
    for form, handle in [("neutral_proton_transfer", nh), ("zwitterion", all_nitrogen)]:
        r = copy.deepcopy(parent)
        out = Chem.Mol(product)
        if form == "zwitterion":
            rw = Chem.RWMol(out)
            for atom in list(out.GetAtoms()):
                if atom.GetAtomMapNum() in (1, 7):
                    charge = 1 if atom.GetAtomMapNum() == 1 else -1
                    replacement = Chem.AtomFromSmarts(
                        atom.GetSmarts().replace(":", f'{"+" if charge==1 else "-"}:', 1)
                    )
                    rw.ReplaceAtom(atom.GetIdx(), replacement)
            out = rw.GetMol()
        r["reaction_id"] = "source_iphos_" + form
        r["atom_mapped_reaction_smarts"] = (
            handle[:-1] + ":1]." + Chem.MolToSmarts(ring) + ">>" + Chem.MolToSmarts(out)
        )
        r["reactant_roles"][0].update(
            required_handle_smarts=handle, allowed_site_multiplicity=list(range(1, 6))
        )
        r["reactant_roles"][1].update(
            name="phosphate_tail",
            required_handle_smarts=Chem.MolToSmarts(ring_handle),
            allowed_site_multiplicity=[1],
            mapped_reactive_atoms=list(range(2, 10)),
        )
        r["source_program"] = {
            "accumulator_role": "amine_head",
            "event_count_field": "event_count",
            "maximum_events": 5,
            "net_byproducts_per_event": {"formal_charge": 0},
            "repeated_roles": ["phosphate_tail"],
        }
        r["selectivity_policy"] = (
            "One phosphate unit per original neutral aliphatic non-acyl N. All sites and intermediate states enumerated. Partial occupancy without unique constitution abstains."
        )
        r["protonation_and_salt_policy"] = (
            "Keep the declared "
            + form
            + " representation exactly. No neutralization, salt stripping, or transfer between representations in corpus records."
        )
        r["conditions"] = {
            "solvent": ["anhydrous dimethyl sulfoxide"],
            "temperature_c": 70,
            "time_h": 72,
            "catalyst_or_reagent": [],
            "reported_yield_range": None,
        }
        r["sources"] = [
            {
                "kind": "doi",
                "identifier": "10.1038/s41563-020-00886-0",
                "locator": "Main Fig. 1c-d; SI PDF pp. 6-7 and 9-10, Figs. S2-S3 and S5-S6; Methods, General synthesis of ionizable phospholipids library",
                "notes": "Both neutral OH/tertiary-N and zwitterionic NH+/O- forms drawn for 9A1P9 and 10A1P10. Exact corpus execution and biological outcomes are not inherited.",
            }
        ]
        r["implementation"] = {
            "kind": "source_drawing_overlay",
            "parent_registries": {k: pin(v) for k, v in paths.items()},
            "derivation": pin(Path(__file__).resolve()),
        }
        r["known_positive_examples"] = []
        r["known_negative_examples"] = []
        reactions.append(r)
    output = ROOT / "data/vendor/qualified_iphos_source_program_v1.json"
    dump(
        output,
        {
            "schema_version": "forge.source_reaction_registry.v1",
            "source_assets": assets,
            "curation": {
                "evidence_basis": "computed_transform_consistency",
                "original_registries_modified": False,
            },
            "reactions": reactions,
        },
    )
    for form in ("neutral_proton_transfer", "zwitterion"):
        reaction_id = "source_iphos_" + form
        controls = []
        for label, head, tail, neutral, charged in [
            (
                "9A1P9",
                "CCCCCCCCNCCCCCCCC",
                "CCCCCCCCCOP1(=O)OCCO1",
                "CCCCCCCCN(CCCCCCCC)CCOP(=O)(O)OCCCCCCCCC",
                "CCCCCCCC[NH+](CCCCCCCC)CCOP(=O)([O-])OCCCCCCCCC",
            ),
            (
                "10A1P10",
                "CCCCCCCCCCNCCCCCCCCCC",
                "CCCCCCCCCCOP1(=O)OCCO1",
                "CCCCCCCCCCN(CCCCCCCCCC)CCOP(=O)(O)OCCCCCCCCCC",
                "CCCCCCCCCC[NH+](CCCCCCCCCC)CCOP(=O)([O-])OCCCCCCCCCC",
            ),
        ]:
            smi = neutral if form == "neutral_proton_transfer" else charged
            controls.append(
                {
                    "label": label + "_" + form,
                    "family": "iphos_ring_opening",
                    "reaction_id": reaction_id,
                    "components": {"amine_head": head, "phosphate_tail": tail},
                    "events": 1,
                    "expected_product": smi,
                    "expected_formula": rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smi)),
                    "expected_forward_product_count": 1,
                    "expected_computed_consistency_pass": True,
                    "source_asset": "supplement",
                    "pdf_pages_one_based": [9 if label == "9A1P9" else 10],
                    "source_reported_neutral_mass": 491.4 if label == "9A1P9" else 561.5,
                    "source_reported_mh": 492.4 if label == "9A1P9" else 563.6,
                }
            )
        target = (
            "CN(CCOP(=O)(O)OC)CCCN"
            if form == "neutral_proton_transfer"
            else "C[NH+](CCOP(=O)([O-])OC)CCCN"
        )
        ambiguity = {
            "label": "non_equivalent_nitrogens_" + form,
            "family": "iphos_ring_opening",
            "reaction_id": reaction_id,
            "components": {"amine_head": "CNCCCN", "phosphate_tail": "COP1(=O)OCCO1"},
            "events": 1,
            "expected_product": target,
            "expected_formula": rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(target)),
            "expected_forward_product_count": 2,
            "expected_computed_consistency_pass": False,
            "evidence_kind": "synthetic_regression_fixture_not_experimental_evidence",
        }
        dump(
            HERE / ("adjudication-" + form + ".json"),
            {
                "schema_version": "forge.repeated_source_adjudication.v1",
                "registry": pin(output),
                "assets": assets,
                "source_controls": controls,
                "ambiguity_controls": [ambiguity],
                "disposition": "admit_transform_consistency",
                "evidence_basis": "computed_transform_consistency",
                "source_procedure": {
                    "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8188687/",
                    "section": "General synthesis of ionizable phospholipids (iPhos) library",
                    "solvent": "anhydrous DMSO",
                    "starting_material_concentration_g_ml": 0.3,
                    "temperature_c": 70,
                    "time_h": 72,
                    "Pm_reagent_equivalents_per_intended_N": 1.1,
                    "incorporated_Pm_per_modified_N": 1,
                    "workup": "Vacuum dry to remove DMSO. Crude library screened; selected examples flash purified with reported chloroform/methanol gradient and vacuum dried 24 h.",
                },
                "limitations": [
                    "The source draws both charge forms; each is a separate exact computational representation. No corpus neutralization occurs.",
                    "Source SI 10A1P10 calculated neutral mass 561.5 and reported [M+H]+ 563.6 are inconsistent by approximately 1 Da; retain the printed values, do not promote an exact analytical claim.",
                    "General source execution and spectra do not establish exact execution of generated precursor variants.",
                ],
                "training_admitted": False,
                "experimental_execution_admitted": False,
            },
        )
    return output


if __name__ == "__main__":
    build()
