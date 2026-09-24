"""Build a source-qualified STAAR overlay without changing any parent registry."""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent


def pin(path):
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parent_path = ROOT / "data/vendor/qualified_reaction_families_v1.json"
    functional_path = ROOT / "data/vendor/qualified_a3_source_program_v2.json"
    parent = json.loads(parent_path.read_text())
    functional = json.loads(functional_path.read_text())["reactions"][0]
    contract = json.loads((SOURCE / "stage_contract.json").read_text())
    amine = next(r for r in functional["reactant_roles"] if r["name"] == "amine_head")
    amine_template = (
        functional["atom_mapped_reaction_smarts"].split(".", 1)[0].replace(":1]", ":11]")
    )
    first, second = contract["stages"]
    thiol = copy.deepcopy(
        next(r for r in parent["reactions"] if r["reaction_id"] == second["parent_reaction"])
    )
    # Restrict the inherited N/O interface to the source's acrylate ester.
    restriction = second["parent_interface_restriction"]
    assert thiol["atom_mapped_reaction_smarts"].count(restriction["from"]) == 1
    thiol["atom_mapped_reaction_smarts"] = thiol["atom_mapped_reaction_smarts"].replace(
        restriction["from"], restriction["to"]
    )
    thiol["reaction_id"] = second["reaction_id"]
    thiol["reactant_roles"][0].update(
        name=second["accumulator_role"], allowed_site_multiplicity=[1]
    )
    thiol["reactant_roles"][1].update(
        name=second["added_role"],
        required_handle_smarts=second["acrylate_handle"],
        allowed_site_multiplicity=[1],
        forbidden_smarts=[
            amine["required_handle_smarts"],
            thiol["reactant_roles"][0]["required_handle_smarts"],
        ],
    )
    aminolysis = {
        "reaction_id": first["reaction_id"],
        "reaction_version": 1,
        "status": "qualified_for_enumeration",
        "atom_mapped_reaction_smarts": first["reaction_template"].replace(
            "{{amine}}", amine_template
        ),
        "reactant_roles": [
            {
                "name": first["accumulator_role"],
                "count": 1,
                "required_handle_smarts": first["thiolactone_handle"],
                "allowed_site_multiplicity": [1],
                "mapped_reactive_atoms": [1, 3],
                "forbidden_smarts": [
                    amine["required_handle_smarts"],
                    thiol["reactant_roles"][0]["required_handle_smarts"],
                    second["acrylate_handle"],
                ],
            },
            {
                "name": first["added_role"],
                "count": 1,
                "required_handle_smarts": amine["required_handle_smarts"],
                "allowed_site_multiplicity": list(
                    range(1, contract["maximum_reactive_sites_within_declared_size_support"] + 1)
                ),
                "mapped_reactive_atoms": [11],
                "forbidden_smarts": [
                    first["thiolactone_handle"],
                    second["acrylate_handle"],
                    thiol["reactant_roles"][0]["required_handle_smarts"],
                ],
            },
        ],
    }
    for reaction in (aminolysis, thiol):
        reaction["implementation"] = {
            "parent_registry": pin(parent_path),
            "functional_group_registry": pin(functional_path),
            "builder": pin(Path(__file__)),
        }
        reaction["sources"] = [
            {
                "kind": "doi",
                "identifier": "10.1038/s42004-025-01516-z",
                "locator": contract["source_locator"],
            }
        ]
        reaction["curation"] = {
            "evidence_basis": "computed_transform_consistency",
            "experimental_execution_admitted": False,
        }
        reaction["stereochemistry_policy"] = "Constitutional stereo-free identity."
        reaction.pop("known_positive_examples", None)
        reaction.pop("known_negative_examples", None)
    assets = {
        name: pin(SOURCE / name)
        for name in (
            "article.html",
            "si.pdf",
            "fig1.jpg",
            "fig2.jpg",
            "fig3.jpg",
            "control_transcriptions.json",
            "stage_contract.json",
            "transcription_check.json",
        )
    }
    assets["decomposition_key"] = pin(
        ROOT / "data/source_cache/compose_lipid_drive_2026-09-18/family_decomposition_key.json"
    )
    program = {
        k: contract[k]
        for k in (
            "program_id",
            "initial_role",
            "terminal_constraints",
            "product_constraints",
            "architecture_subfamily",
            "scope",
        )
    }
    program["stages"] = [
        {
            k: s[k]
            for k in (
                "reaction_id",
                "accumulator_role",
                "added_role",
                "net_byproducts",
                "source_step",
            )
        }
        for s in contract["stages"]
    ]
    output = ROOT / "data/vendor/qualified_staar_source_program_v1.json"
    output.write_text(
        json.dumps(
            {
                "schema_version": "forge.source_sequential_registry.v1",
                "source_assets": assets,
                "reactions": [aminolysis, thiol],
                "programs": [program],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(pin(output))


if __name__ == "__main__":
    main()
