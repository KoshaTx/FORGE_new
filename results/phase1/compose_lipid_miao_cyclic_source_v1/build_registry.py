"""Transcribe the independently drawn Miao dihydroimidazole, using pinned query fragments."""

import copy
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdMolDescriptors, rdqueries

from forge.core.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "results/phase1/compose_lipid_all_family_sources_v1/remaining/31570898"


def pin(p):
    return {"path": str(p.relative_to(ROOT)), "sha256": str(sha256_file(p))}


def dump(p, d):
    p.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")


def build():
    parent = ROOT / "data/vendor/qualified_ketone_ugi4_source_program_v1.json"
    old = json.loads(parent.read_text())["reactions"][0]
    rxn = rdChemReactions.ReactionFromSmarts(old["atom_mapped_reaction_smarts"])
    # Source Iso5 is the complete ethyl isocyanoacetate, not a generic isocyanide.
    # Preserve every atom and constrain the drawn H count at each site. Other
    # ester substituents need their own source review before admission.
    iso_source = Chem.MolFromSmiles("[C-]#[N+]CC(=O)OCC")
    iso = Chem.MolFromSmarts(Chem.MolToSmarts(iso_source))
    for number, (query, atom) in enumerate(
        zip(iso.GetAtoms(), iso_source.GetAtoms(), strict=True), 3
    ):
        query.SetAtomMapNum(number)
        query.ExpandQuery(rdqueries.HCountEqualsQueryAtom(atom.GetTotalNumHs()))
        query.ExpandQuery(rdqueries.FormalChargeEqualsQueryAtom(atom.GetFormalCharge()))
    core = {a.GetAtomMapNum(): a for a in rxn.GetProductTemplate(0).GetAtoms()}
    product = Chem.RWMol()
    indexes = {}
    for number in (1, 2):
        indexes[number] = product.AddAtom(core[number])
    for atom in iso.GetAtoms():
        number = atom.GetAtomMapNum()
        if number == 3:
            cloned = Chem.AtomFromSmarts(core[3].GetSmarts())
            cloned.ExpandQuery(rdqueries.HCountEqualsQueryAtom(1))
            cloned.SetNumExplicitHs(1)
        elif number == 4:
            cloned = Chem.AtomFromSmarts(core[4].GetSmarts().replace("H1", "H0"))
        elif number == 5:
            cloned = Chem.AtomFromSmarts(atom.GetSmarts().replace("H2", "H1"))
        else:
            cloned = copy.copy(atom)
        indexes[number] = product.AddAtom(cloned)
    for bond in iso.GetBonds():
        a, b = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()
        kind = Chem.BondType.DOUBLE if {a, b} == {3, 4} else bond.GetBondType()
        product.AddBond(indexes[a], indexes[b], kind)
    for a, b in ((1, 2), (2, 3), (1, 5)):
        product.AddBond(indexes[a], indexes[b], Chem.BondType.SINGLE)
    reactants = old["atom_mapped_reaction_smarts"].split(">>")[0].split(".")[:2]
    reactants.append(Chem.MolToSmarts(iso))
    handle = Chem.Mol(iso)
    for atom in handle.GetAtoms():
        atom.SetAtomMapNum(0)
    roles = copy.deepcopy(old["reactant_roles"][:3])
    roles[2].update(
        required_handle_smarts=Chem.MolToSmarts(handle), mapped_reactive_atoms=list(range(3, 11))
    )
    reaction = {
        **copy.deepcopy(old),
        "reaction_id": "source_miao_ethyl_isocyanoacetate_cyclization",
        "architecture": "One primary amine and complete ketone plus source Iso5 ethyl isocyanoacetate give the drawn dihydroimidazole ring; no acyclic branch equivalence.",
        "atom_mapped_reaction_smarts": ".".join(reactants) + ">>" + Chem.MolToSmarts(product),
        "reactant_roles": roles,
        "net_byproducts": {"H": 2, "O": 1, "formal_charge": 0},
        "conditions": {
            "solvent": ["anhydrous dichloromethane", "ethanol"],
            "temperature_c": None,
            "temperature_text": "room temperature",
            "time_h": None,
            "time_text": "overnight",
            "catalyst_or_reagent": [],
            "reported_yield_range": [28.4, 34.0],
        },
        "sources": [
            {
                "kind": "doi",
                "identifier": "10.1038/s41587-019-0247-3",
                "locator": "Main Figs. 1b-c and 3a; SI PDF pp. 2-3 (printed 1-2), A12 and A2 procedures and characterization; PDF p. 5 (printed 4), structure interpretation.",
                "notes": "Computed net-transform consistency only. Proposed intermediate sequence is not treated as observed. Distinct head nitrogen is a deliberately narrower computational scope than the generic schematic.",
            }
        ],
        "implementation": {
            "kind": "source_drawing_overlay",
            "parent_registries": {"ketone_and_amine_queries": pin(parent)},
            "derivation": pin(Path(__file__).resolve()),
        },
    }
    reaction["selectivity_policy"] = (
        "Primary reacting NH2; exact source Iso5; one complete ketone; all unfiltered site outcomes must be unique. Preserve distinct unconsumed head basic N in this bounded program. Other source isocyanides and head scopes require separate programs."
    )
    assets = {
        "supplement": pin(SOURCE / "miao-si.pdf"),
        "main_figure_1": pin(SOURCE / "fig1.png"),
        "main_figure_3": pin(SOURCE / "fig3.png"),
        "family_definitions": pin(
            ROOT
            / "data/source_cache/compose_lipid_supplement_2026-09-19/family_reaction_definitions.json"
        ),
    }
    output = ROOT / "data/vendor/qualified_miao_cyclic_source_program_v1.json"
    dump(
        output,
        {
            "schema_version": "forge.source_reaction_registry.v1",
            "source_assets": assets,
            "curation": {
                "original_registries_modified": False,
                "evidence_basis": "computed_transform_consistency",
                "atom_origin_policy": "All two-carbonyl-arm atoms remain in one ketone precursor. Iso5 C3/N4/alpha-C5 and amine N1 form the source-drawn five-membered ring. The source drawing supports the net graph, not isolated mechanistic intermediates.",
            },
            "reactions": [reaction],
        },
    )
    ketone = "CCCCCCCCC=CCCCCCCCC(=O)CCCCCCCC=CCCCCCCCC"
    source_controls = []
    for label, head, product, mass in [
        (
            "A12_Iso5_2DC18",
            "CN(C)CCCN",
            "CCOC(=O)C1N=CC(CCCCCCCC=CCCCCCCCC)(CCCCCCCC=CCCCCCCCC)N1CCCN(C)C",
            700.7,
        ),
        (
            "A2_Iso5_2DC18",
            "NCCCN1CCCC1",
            "CCOC(=O)C1N=CC(CCCCCCCC=CCCCCCCCC)(CCCCCCCC=CCCCCCCCC)N1CCCN1CCCC1",
            726.7,
        ),
    ]:
        source_controls.append(
            {
                "label": label,
                "family": "alpha_isocyanoester_dihydroimidazole",
                "source_asset": "main_figure_3",
                "components": {
                    "amine_head": head,
                    "coupled_ketone": ketone,
                    "isocyanide": "CCOC(=O)C[N+]#[C-]",
                },
                "expected_product": product,
                "expected_formula": rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(product)),
                "reported_mh": mass,
                "expected_forward_product_count": 1,
                "expected_computed_consistency_pass": True,
            }
        )
    prior = json.loads(
        (ROOT / "results/phase1/compose_lipid_ketone_ugi4_source_v1/adjudication.json").read_text()
    )
    sites = [
        s
        for s in prior["families"]["ketone_ugi4"]["site_contract"]
        if s["role"] != "carboxylic_acid"
    ]
    ambiguous = "CCOC(=O)C1N=CC(C)(CC)N1CCN(C)CCCN"
    controls = {
        "schema_version": "forge.fixed_source_event_adjudication.v1",
        "registry": pin(output),
        "assets": assets,
        "disposition": "admit_transform_consistency",
        "evidence_basis": "computed_transform_consistency",
        "families": {
            "alpha_isocyanoester_dihydroimidazole": {
                "reaction_id": reaction["reaction_id"],
                "registry_to_source_roles": {r["name"]: r["name"] for r in roles},
                "site_contract": sites,
            }
        },
        "source_controls": source_controls,
        "regression_controls": [
            {
                "label": "non_equivalent_primary_amines",
                "family": "alpha_isocyanoester_dihydroimidazole",
                "components": {
                    "amine_head": "NCCN(C)CCCN",
                    "coupled_ketone": "CC(=O)CC",
                    "isocyanide": "CCOC(=O)C[N+]#[C-]",
                },
                "expected_product": ambiguous,
                "expected_formula": rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(ambiguous)),
                "expected_forward_product_count": 2,
                "expected_computed_consistency_pass": False,
                "required_failed_checks": ["unique_unfiltered_forward_exact"],
                "evidence_kind": "synthetic_regression_fixture_not_experimental_evidence",
            }
        ],
        "source_procedure": {
            "A12": {
                "each_precursor_mmol": 0.2,
                "DCM_ul": 80,
                "ethanol_ul": 120,
                "temperature": "room temperature",
                "time": "overnight",
                "vessel": "capped glass vial",
                "purification": "flash chromatography",
                "product": "yellow oil",
                "mass_mg": 39.8,
                "amount_mmol": 0.057,
                "yield_percent": 28.4,
            },
            "A2": {
                "each_precursor_mmol": 0.3,
                "DCM_ul": 120,
                "ethanol_ul": 180,
                "procedure_ref": "A12",
                "printed_mass": "74.1 g",
                "printed_amount_mmol": 0.1,
                "printed_yield_percent": 34,
            },
        },
        "limitations": [
            "A2 printed 74.1 g conflicts with 0.1 mmol and reported molecular mass; preserve the source typo, do not use as a quantitative yield/mass label.",
            "Only source Iso5 ethyl isocyanoacetate is qualified here; other isocyanoester substituents are not silently generalized.",
            "The retained separate head N policy is a narrower support scope; it does not assert that every primary amine in the family needs a second head N.",
            "All source stereochemical annotations are retained in the source figures but removed only for the declared constitutional model. No source potency is transferred.",
        ],
        "training_admitted": False,
        "experimental_execution_admitted": False,
    }
    dump(HERE / "adjudication.json", controls)
    return output, controls, reaction


if __name__ == "__main__":
    build()
