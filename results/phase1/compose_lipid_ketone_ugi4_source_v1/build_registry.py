"""Derive the FO-32 ketone Ugi graph from pinned Ugi and functional-group fragments."""

import copy
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdMolDescriptors

from forge.core.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "results/phase1/compose_lipid_all_family_sources_v1/remaining/39658727"


def pin(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": str(sha256_file(path))}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build():
    parents = {
        name: ROOT / path
        for name, path in {
            "aldehyde_ugi4": "data/vendor/qualified_ugi4_source_program_v1.json",
            "functional_groups": "data/vendor/qualified_a3_source_program_v2.json",
            "basic_nitrogen": "data/vendor/qualified_staar_source_program_v1.json",
        }.items()
    }
    reaction = copy.deepcopy(json.loads(parents["aldehyde_ugi4"].read_text())["reactions"][0])
    functional = json.loads(parents["functional_groups"].read_text())["reactions"][0]
    retained = copy.deepcopy(
        json.loads(parents["basic_nitrogen"].read_text())["programs"][0]["product_constraints"][
            "required_queries"
        ][0]
    )
    retained.update(role="amine_head", name="distinct_head_basic_nitrogen")
    left, right = reaction["atom_mapped_reaction_smarts"].split(">>")
    reactants = left.split(".")
    ketone = functional["atom_mapped_reaction_smarts"].split(">>")[0].split(".")[1]
    # FO-32 has a carbonyl carbon bearing two carbon substituents and no H.
    # Preserve the parent exclusion of single-bonded heteroatoms (esters/amides).
    ketone = ketone.replace("$([C;H1,H2])", "H0").replace(":3]", "]")
    reactants[1] = ketone
    rxn = rdChemReactions.ReactionFromSmarts(reaction["atom_mapped_reaction_smarts"])
    product = Chem.Mol(rxn.GetProductTemplate(0))
    center = next(a for a in product.GetAtoms() if a.GetAtomMapNum() == 2)
    # Retain the original carbon query and change only its drawn hydrogen count.
    replacement = Chem.AtomFromSmarts(center.GetSmarts().replace("H1", "H0"))
    product = Chem.RWMol(product)
    product.ReplaceAtom(center.GetIdx(), replacement)
    names = {
        "oxoester_aldehyde_body_tail": "coupled_ketone",
        "isocyanide_tail": "isocyanide",
        "carboxylic_acid_tail": "carboxylic_acid",
    }
    for role in reaction["reactant_roles"]:
        role["name"] = names.get(role["name"], role["name"])
        if role["name"] == "coupled_ketone":
            role["required_handle_smarts"] = functional["reactant_roles"][1][
                "required_handle_smarts"
            ].replace("H1,H2", "H0")
    reaction.update(
        reaction_id="source_ketone_ugi4",
        architecture="One complete coupled ketone, primary amine, acid and isocyanide; retain a distinct basic head N",
        atom_mapped_reaction_smarts=".".join(reactants) + ">>" + Chem.MolToSmarts(product),
        retained_queries=[retained],
        selectivity_policy="All unfiltered products and inverse tuples must be unique. The reacting head N is NH2. Retention must be witnessed on an unconsumed atom originating in the complete amine head; another role cannot rescue it.",
        conditions={
            "solvent": ["dichloromethane", "methanol"],
            "temperature_c": None,
            "temperature_text": "room temperature",
            "time_h": 72,
            "precondensation_time_h": 2,
            "catalyst_or_reagent": [],
            "reported_yield_range": [8.5, 8.5],
        },
        sources=[
            {
                "kind": "doi",
                "identifier": "10.1038/s41587-024-02490-y",
                "locator": "Supplementary information PDF page 2, FO-32 complete graph and procedure",
                "notes": "Constitutional computed transform support only; no corpus execution, potency, L2 or L3 admission.",
            }
        ],
        implementation={
            "kind": "source_drawing_overlay",
            "parent_registries": {k: pin(p) for k, p in parents.items()},
            "derivation": pin(Path(__file__).resolve()),
        },
    )
    assets = {
        "supplement": pin(SOURCE / "ketone-ugi4-si.pdf"),
        "source_transcription": pin(SOURCE / "fo32-source-extraction.json"),
        "family_definitions": pin(
            ROOT
            / "data/source_cache/compose_lipid_supplement_2026-09-19/family_reaction_definitions.json"
        ),
    }
    output = ROOT / "data/vendor/qualified_ketone_ugi4_source_program_v1.json"
    dump(
        output,
        {
            "schema_version": "forge.source_reaction_registry.v1",
            "source_assets": assets,
            "curation": {
                "original_registries_modified": False,
                "evidence_basis": "computed_transform_consistency",
                "atom_origin_policy": "Inherited Ugi Mumm oxygen convention, not isotope evidence. Ketone arms remain attached to their single complete precursor.",
            },
            "reactions": [reaction],
        },
    )
    old = json.loads(
        (ROOT / "results/phase1/compose_lipid_v8_ugi4_source_v1/adjudication.json").read_text()
    )
    sites = copy.deepcopy(old["source_contract"]["site_contract"])
    for site in sites:
        site["role"] = names.get(site["role"], site["role"])
        if site["role"] == "coupled_ketone":
            site["properties"]["total_hydrogens"] = 0
    source = json.loads((SOURCE / "fo32-source-extraction.json").read_text())
    product = source["source_drawn_product_stereo_free"]
    control = {
        "label": "FO_32",
        "family": "ketone_ugi4",
        "source_asset": "supplement",
        "components": source["components"],
        "expected_product": product,
        "expected_formula": source["computed_neutral_formula"],
        "reported_mz_calculated": 1026.84494,
        "reported_mz_found": 1026.8444,
        "expected_computed_consistency_pass": True,
        "expected_forward_product_count": 1,
    }
    document = {
        "schema_version": "forge.fixed_source_event_adjudication.v1",
        "registry": pin(output),
        "assets": assets,
        "disposition": "admit_transform_consistency",
        "evidence_basis": "computed_transform_consistency",
        "source_controls": [control],
        "regression_controls": [],
        "families": {
            "ketone_ugi4": {
                "reaction_id": reaction["reaction_id"],
                "registry_to_source_roles": {
                    r["name"]: r["name"] for r in reaction["reactant_roles"]
                },
                "site_contract": sites,
            }
        },
        "source_procedure": source["procedure"],
        "limitations": [
            "FO-32 supports the net ketone Ugi connection, not every corpus precursor or architecture as experimentally executed.",
            "Cyclic, asymmetric and other component variants receive only a computed exactness determination. Architecture and source-bank claims remain separate.",
            "The amine-role basic-N predicate is structural and is not a pKa or protonation measurement.",
        ],
        "visual_inspection": {
            "supplement_pdf_pages_one_based": [2],
            "graph": "complete four-component FO-32 scheme and drawn product",
        },
        "training_admitted": False,
        "experimental_execution_admitted": False,
    }
    for label, head, acid, target, count, failures in (
        (
            "asymmetric_two_primary_amines",
            "NCCN(C)CCCN",
            "CCC(=O)O",
            "CCC(=O)N(CCN(C)CCCN)C(C)(CC)C(=O)NCC",
            2,
            ["unique_unfiltered_forward_exact"],
        ),
        (
            "cross_role_basic_nitrogen_cannot_rescue_head",
            "CCN",
            "CN(C)CCC(=O)O",
            "CN(C)CCC(=O)N(CC)C(C)(CC)C(=O)NCC",
            1,
            ["distinct_source_role_handles_retained"],
        ),
    ):
        document["regression_controls"].append(
            {
                "label": label,
                "family": "ketone_ugi4",
                "components": {
                    "amine_head": head,
                    "coupled_ketone": "CC(=O)CC",
                    "carboxylic_acid": acid,
                    "isocyanide": "CC[N+]#[C-]",
                },
                "expected_product": target,
                "expected_formula": rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(target)),
                "expected_forward_product_count": count,
                "expected_computed_consistency_pass": False,
                "required_failed_checks": failures,
                "evidence_kind": "synthetic_regression_fixture_not_experimental_evidence",
            }
        )
    dump(HERE / "adjudication.json", document)
    return output


if __name__ == "__main__":
    build()
