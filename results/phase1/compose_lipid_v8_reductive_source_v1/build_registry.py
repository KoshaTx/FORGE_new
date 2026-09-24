"""Derive a new source overlay; never edit the frozen parent reaction registries."""

import copy
import hashlib
import json
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent


def pin(path):
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parent_path = ROOT / "data/vendor/qualified_reaction_families_v1.json"
    group_path = ROOT / "data/vendor/qualified_a3_source_program_v2.json"
    parent = json.loads(parent_path.read_text())
    functional = json.loads(group_path.read_text())["reactions"][0]
    original = next(
        r for r in parent["reactions"] if r["reaction_id"] == "reductive_amination_amine_aldehyde"
    )
    amine = next(r for r in functional["reactant_roles"] if r["name"] == "amine_head")
    # Keep recursive H spelling: serializing the full reaction can flatten its alternative H query.
    nitrogen_template = functional["atom_mapped_reaction_smarts"].split(".", 1)[0]
    left, right = original["atom_mapped_reaction_smarts"].split(">>")
    left = nitrogen_template + "." + left.split(".", 1)[1]
    scaffolds = json.loads((SOURCE / "scaffolds.json").read_text())
    controls = json.loads((SOURCE / "controls.json").read_text())["controls"]
    reactions = []
    for family, contract in scaffolds["families"].items():
        reaction = copy.deepcopy(original)
        reaction["reaction_id"] = "source_" + family
        reaction["atom_mapped_reaction_smarts"] = left + ">>" + right
        reaction["reactant_roles"][0]["required_handle_smarts"] = amine["required_handle_smarts"]
        reaction["reactant_roles"][0]["forbidden_smarts"] = [
            original["reactant_roles"][1]["required_handle_smarts"]
        ]
        queries = []
        for shape in contract["scaffolds"]:
            query = Chem.MolFromSmarts(shape["core_smarts"])
            assert query is not None
            for atom in query.GetAtoms():
                atom.SetAtomMapNum(0)
            queries.append("$(" + Chem.MolToSmarts(query) + ")")
        reaction["reactant_roles"][1].update(
            name=contract["precursor_role"],
            required_handle_smarts="[" + ",".join(queries) + "]",
            allowed_site_multiplicity=[1],
            forbidden_smarts=[amine["required_handle_smarts"]],
        )
        reaction["precursor_scaffolds"] = contract
        reaction["additional_required_query"] = {
            "role": contract["precursor_role"],
            "smarts": original["reactant_roles"][1]["required_handle_smarts"],
            "exact_count": 1,
        }
        reaction["implementation"] = {
            "parent_registry": pin(parent_path),
            "functional_group_registry": pin(group_path),
            "builder": pin(Path(__file__)),
        }
        reaction["known_positive_examples"] = [
            {
                "reactants": [c["components"][r["name"]] for r in reaction["reactant_roles"]],
                "expected": c["expected_product"],
                "reason": c["label"] + "; independent source drawing and formula",
            }
            for c in controls
            if c["family"] == family
        ]
        reaction["sources"] = [
            {
                "kind": "doi",
                "identifier": (
                    "10.1038/s41565-023-01548-3"
                    if family == "reductive_amination"
                    else "10.1038/s41467-024-45422-9"
                ),
                "locator": "Pinned source scaffolds and controls specify exact figure/procedure locators",
            }
        ]
        reaction["curation"] = {
            "evidence_basis": "computed_transform_consistency",
            "disposition": "admit_transform_consistency",
            "experimental_execution_admitted": False,
        }
        reaction["stereochemistry_policy"] = "Constitutional stereo-free Phase 1 identity."
        reactions.append(reaction)
    assets = {
        name: pin(SOURCE / name)
        for name in (
            "scaffolds.json",
            "controls.json",
            "jiang-article.html",
            "jiang-si.pdf",
            "xue-article.html",
            "xue-si.pdf",
        )
    }
    assets["decomposition_key"] = pin(
        ROOT / "data/source_cache/compose_lipid_drive_2026-09-18/family_decomposition_key.json"
    )
    output = ROOT / "data/vendor/qualified_reductive_source_program_v1.json"
    output.write_text(
        json.dumps(
            {
                "schema_version": "forge.source_reductive_program_registry.v1",
                "source_assets": assets,
                "reactions": reactions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(pin(output))


if __name__ == "__main__":
    main()
