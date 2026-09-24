"""Add the source-drawn deprotonated iPhos form with explicit proton bookkeeping."""

import copy
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdMolDescriptors

from forge.core.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def pin(p):
    return {"path": p.relative_to(ROOT).as_posix(), "sha256": str(sha256_file(p))}


def dump(p, d):
    p.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")


def build():
    parent = ROOT / "data/vendor/qualified_iphos_source_program_v1.json"
    document = json.loads(parent.read_text())
    neutral, charged = document["reactions"]
    reaction = copy.deepcopy(neutral)
    nr = rdChemReactions.ReactionFromSmarts(neutral["atom_mapped_reaction_smarts"])
    cr = rdChemReactions.ReactionFromSmarts(charged["atom_mapped_reaction_smarts"])
    product = Chem.RWMol(nr.GetProductTemplate(0))
    negative = next(a for a in cr.GetProductTemplate(0).GetAtoms() if a.GetAtomMapNum() == 7)
    destination = next(a.GetIdx() for a in product.GetAtoms() if a.GetAtomMapNum() == 7)
    product.ReplaceAtom(destination, negative)
    reaction["reaction_id"] = "source_iphos_deprotonated"
    reaction["atom_mapped_reaction_smarts"] = (
        neutral["atom_mapped_reaction_smarts"].split(">>")[0] + ">>" + Chem.MolToSmarts(product)
    )
    reaction["source_program"]["net_byproducts_per_event"] = {"H": 1, "formal_charge": 1}
    reaction["protonation_and_salt_policy"] = (
        "Source-drawn neutral nitrogen/anionic phosphate form (main Fig. 1a, physiological panel). Record one released proton per event explicitly; do not neutralize or rewrite the corpus target. Tertiary starting amines remain exclusive to the zwitterion program."
    )
    reaction["implementation"] = {
        "kind": "source_drawing_overlay",
        "parent_registries": {"source_iphos_charge_forms": pin(parent)},
        "derivation": pin(Path(__file__).resolve()),
    }
    reaction["sources"][0][
        "locator"
    ] = "Main Fig. 1a physiological versus acidic form, and Fig. 1c-d; SI Figs. S2 and S5-S6"
    output = ROOT / "data/vendor/qualified_iphos_deprotonated_program_v1.json"
    dump(
        output,
        {
            "schema_version": "forge.source_reaction_registry.v1",
            "source_assets": document["source_assets"],
            "curation": {
                "evidence_basis": "computed_transform_consistency",
                "original_registries_modified": False,
                "proton_inventory": "Neutral precursor sum equals anionic product plus one H+ per incorporated phosphate. This specifies a chemical form; no pKa, buffer composition or exact reaction-solution speciation is inferred.",
            },
            "reactions": [reaction],
        },
    )
    old = json.loads((HERE / "adjudication-neutral_proton_transfer.json").read_text())
    control = copy.deepcopy(old)
    control["registry"] = pin(output)
    for kind in ("source_controls", "ambiguity_controls"):
        for c in control[kind]:
            c["label"] = c["label"].replace("neutral_proton_transfer", "deprotonated")
            c["reaction_id"] = reaction["reaction_id"]
            # The source figure transfers an explicit proton away from the neutral
            # phosphate OH form; no connectivity or attachment-site changes occur.
            mol = Chem.MolFromSmiles(c["expected_product"])
            for atom in mol.GetAtoms():
                if (
                    atom.GetAtomicNum() == 8
                    and atom.GetTotalNumHs() == 1
                    and any(n.GetAtomicNum() == 15 for n in atom.GetNeighbors())
                ):
                    atom.SetFormalCharge(-1)
                    atom.SetNumExplicitHs(0)
                    atom.SetNoImplicit(True)
            Chem.SanitizeMol(mol)
            c["expected_product"] = Chem.MolToSmiles(mol)
            c["expected_formula"] = rdMolDescriptors.CalcMolFormula(mol)
            if kind == "source_controls":
                c["source_asset"] = "main_figure"
                c["chemical_form_locator"] = (
                    "Main Fig. 1a physiological N-neutral/phosphate-anion depiction; full substituents from the source 9A1P9 or 10A1P10 graph in SI S5/S6"
                )
                c["source_analytical_form_is_not_this_anion"] = True
    control["limitations"].append(
        "The anionic source form is supported by the main figure equilibrium drawing; the SI positive-ion mass is not an analytical measurement of this anion. The released H+ is explicit bookkeeping, not an observed reaction byproduct assay."
    )
    dump(HERE / "adjudication-deprotonated.json", control)
    return output


if __name__ == "__main__":
    build()
