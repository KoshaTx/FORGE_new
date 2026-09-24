"""Qualify only the three alpha-isocyanoacetates drawn in Miao Fig. 1c."""

import copy
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from forge.core.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "compose_lipid_miao_cyclic_source_v1"


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": str(sha256_file(path))}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build():
    parent = ROOT / "data/vendor/qualified_miao_cyclic_source_program_v1.json"
    registry = json.loads(parent.read_text())
    reaction = registry["reactions"][0]
    smarts = reaction["atom_mapped_reaction_smarts"]
    parsed = rdChemReactions.ReactionFromSmarts(smarts)

    # Maps 9 and 10 belong to the spectator ethyl substituent. Leave the ester
    # O-alkyl bond outside the net reaction template so the complete original
    # spectator graph is retained in both directions. Scope remains enumerated
    # by the three full source precursor queries below, never by a target match.
    def core(molecule):
        result = Chem.RWMol(molecule)
        for atom in reversed(list(result.GetAtoms())):
            if atom.GetAtomMapNum() in {9, 10}:
                result.RemoveAtom(atom.GetIdx())
        return Chem.MolToSmarts(result)

    reactants = smarts.split(">>")[0].split(".")[:2]
    reactants.append(core(parsed.GetReactantTemplate(2)))
    reaction["atom_mapped_reaction_smarts"] = (
        ".".join(reactants) + ">>" + core(parsed.GetProductTemplate(0))
    )
    source_isocyanides = {
        "Iso4": "[C-]#[N+]CC(=O)OC",
        "Iso5": "[C-]#[N+]CC(=O)OCC",
        "Iso6": "[C-]#[N+]CC(=O)OC(C)(C)C",
    }
    # Whole structures plus H counts rule out homologs and alpha substitution.
    queries = []
    for smiles in source_isocyanides.values():
        molecule = Chem.MolFromSmiles(smiles)
        from rdkit.Chem import rdqueries

        query = Chem.MolFromSmarts(Chem.MolToSmarts(molecule))
        for q, a in zip(query.GetAtoms(), molecule.GetAtoms(), strict=True):
            q.ExpandQuery(rdqueries.HCountEqualsQueryAtom(a.GetTotalNumHs()))
        queries.append("$(" + Chem.MolToSmarts(query) + ")")
    role = reaction["reactant_roles"][2]
    role["required_handle_smarts"] = "[" + ",".join(queries) + "]"
    role["mapped_reactive_atoms"] = list(range(3, 9))
    reaction["reaction_id"] = "source_miao_drawn_alpha_isocyanoacetates_cyclization"
    reaction["architecture"] = (
        "Source-drawn dihydroimidazole net connection for Fig. 1c Iso4, Iso5, or Iso6, retaining the whole ester alkyl group."
    )
    reaction["selectivity_policy"] = (
        "Primary reacting NH2 and distinct retained head basic N; exact full source Iso4/5/6 only; complete ketone; all unfiltered forward and inverse outcomes must be unique."
    )
    reaction["implementation"] = {
        "kind": "source_drawing_overlay",
        "parent_registries": {"exact_ethoxy_cyclic_source_program": pin(parent)},
        "derivation": pin(Path(__file__).resolve()),
    }
    reaction["sources"][0][
        "notes"
    ] = "Fig. 1b draws the cyclic ester generically and Fig. 1c explicitly lists Iso4 (methyl), Iso5 (ethyl), Iso6 (tert-butyl). Fig. 3a and SI characterize Iso5 cyclic controls. This supports computed net-transform consistency for the three drawn precursors, not individual experimental execution or selectivity for Iso4/Iso6."
    registry["curation"][
        "atom_origin_policy"
    ] = "Net ring atom origins follow the pinned Iso5 control; the complete source spectator ester substituent is unchanged. No alpha-substituted Iso7 or beta-isocyanoester Iso8 admission."
    registry["source_isocyanides"] = source_isocyanides
    target = ROOT / "data/vendor/qualified_miao_cyclic_source_program_v2.json"
    dump(target, registry)
    adjudication = json.loads((OLD / "adjudication.json").read_text())
    adjudication["registry"] = pin(target)
    family = "alpha_isocyanoester_dihydroimidazole"
    adjudication["families"][family]["reaction_id"] = reaction["reaction_id"]
    adjudication["limitations"][
        1
    ] = "Only the three full alpha-isocyanoacetates Iso4/5/6 explicitly drawn in Fig. 1c are in scope; individual cyclic products of Iso4 and Iso6 have no isolated characterization admitted here."
    adjudication["source_library_scope"] = {
        "asset": "main_figure_1",
        "locator": "Fig. 1b generic cyclic ester and Fig. 1c Iso4/5/6 drawings",
        "precursors": source_isocyanides,
        "evidence_basis": "computed_transform_consistency",
        "individual_execution_admitted": False,
    }
    dump(HERE / "adjudication.json", adjudication)
    cfg = json.loads((OLD / "replay-config.json").read_text())
    cfg["families"][family]["reaction_id"] = reaction["reaction_id"]
    cfg["inputs"]["registry"] = pin(target)
    cfg["inputs"]["transform_controls"] = pin(HERE / "adjudication.json")
    dump(HERE / "replay-config.json", cfg)


if __name__ == "__main__":
    build()
