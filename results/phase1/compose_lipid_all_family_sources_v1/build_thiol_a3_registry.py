"""Add a separate source maleate-thiol mechanism and supplied repeated A3 program."""

import copy
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def read(name):
    path = ROOT / name
    return path, json.loads(path.read_text())


def main():
    family_path, family = read("data/vendor/qualified_reaction_families_v1.json")
    maleate_path, maleate = read("data/vendor/qualified_maleate_ester_source_program_v1.json")
    a3_path, a3 = read("data/vendor/qualified_a3_source_program_v2.json")
    prior_path, prior = read("results/phase1/compose_lipid_v8_a3_source_v1/adjudication-v2.json")
    thiol = next(r for r in family["reactions"] if r["reaction_id"] == "thiol_michael_thioether")
    maleate = copy.deepcopy(maleate["reactions"][0])
    old = rdChemReactions.ReactionFromSmarts(maleate["atom_mapped_reaction_smarts"])
    sulfur = rdChemReactions.ReactionFromSmarts(thiol["atom_mapped_reaction_smarts"])
    product = Chem.RWMol(old.GetProductTemplate(0))
    product.ReplaceAtom(0, copy.copy(sulfur.GetProductTemplate(0).GetAtomWithIdx(0)))
    amine_handle = next(
        r
        for r in json.loads(
            (ROOT / "data/vendor/qualified_michael_source_program_v1.json").read_text()
        )["reactions"]
        if r["reaction_id"] == "source_aza_michael_acrylate"
    )["reactant_roles"][0]["required_handle_smarts"]
    maleate.update(
        reaction_id="source_thiol_maleate_addition",
        atom_mapped_reaction_smarts=thiol["atom_mapped_reaction_smarts"]
        .split(">>")[0]
        .split(".")[0]
        + "."
        + Chem.MolToSmarts(old.GetReactantTemplate(1))
        + ">>"
        + Chem.MolToSmarts(product),
    )
    maleate["reactant_roles"][0] = copy.deepcopy(thiol["reactant_roles"][0])
    maleate["reactant_roles"][0].update(
        name="thiol_head", allowed_site_multiplicity=[1], forbidden_smarts=[amine_handle]
    )
    maleate["reactant_roles"][1]["forbidden_smarts"].append(
        thiol["reactant_roles"][0]["required_handle_smarts"]
    )
    maleate["source_program"].update(accumulator_role="thiol_head", maximum_events=1)
    maleate["selectivity_policy"] = (
        "Separate monothiol mechanism; tertiary amine topology retained. Free NH competitors excluded. All complete maleate orientations enumerated. No pooling with the amine mechanism."
    )
    maleate["sources"][0][
        "locator"
    ] = "Main Fig. 1a-b reaction and library; SI Fig. 1 (PDF p. 1), thiol 49 and D8. Product is the explicit library combination of these source-drawn ingredients and generic drawn S-addition graph; no individual 49D8 spectrum claimed."
    a3 = copy.deepcopy(a3["reactions"][0])
    a3["reaction_id"] = "source_fixed_components_repeated_a3"
    for role in a3["reactant_roles"]:
        role["allowed_site_multiplicity"] = (
            list(range(1, 7)) if role["name"] == "amine_head" else [1]
        )
    a3["source_program"] = {
        "accumulator_role": "amine_head",
        "repeated_roles": ["aldehyde", "alkyne"],
        "event_count_field": "events",
        "maximum_events": 6,
        "net_byproducts_per_event": {"H": 2, "O": 1, "formal_charge": 0},
    }
    a3["selectivity_policy"] = (
        "One complete fixed aldehyde and alkyne identity repeated at all declared N-H events. No ordered heterogeneous program or target-directed site selection. Virtual computed consistency only; the source-key claim that Han asymmetric libraries require two ordered events is rejected, and is never admitted as a label."
    )
    a3["sources"] = [
        {
            "kind": "doi",
            "identifier": prior["primary_doi"],
            "locator": "Main Methods p.1421; SI PDF pp.5-7, control 11(aB)2 and 31hP-M; prior source adjudication preserved.",
        }
    ]
    a3["architecture_qualified"] = False
    a3["architecture_conflicts"] = prior["conflicts"]
    a3["experimental_execution_admitted"] = False
    for entry, parents in [
        (maleate, {"family": family_path, "maleate": maleate_path}),
        (a3, {"a3": a3_path, "source_adjudication": prior_path}),
    ]:
        entry["implementation"] = {
            "kind": "source_qualified_program_overlay",
            "parent_registries": {k: pin(ROOT, v) for k, v in parents.items()},
            "derivation": pin(ROOT, Path(__file__).resolve()),
        }
    assets = {
        **json.loads(maleate_path.read_text())["source_assets"],
        "maleate_main_fig1": pin(ROOT, HERE / "maleate-main-fig1.jpg"),
        **{"a3_" + k: v for k, v in prior["assets"].items()},
    }
    output = ROOT / "data/vendor/qualified_thiol_maleate_a3_program_v1.json"
    dump(
        output,
        {
            "schema_version": "forge.source_reaction_registry.v1",
            "source_assets": assets,
            "curation": {
                "evidence_basis": "computed_transform_consistency",
                "experimental_execution_admitted": False,
                "training_admitted": False,
            },
            "reactions": [maleate, a3],
        },
    )
    positive = [
        {
            "label": "maleate_library_49D8",
            "family": "maleate_addition",
            "reaction_id": maleate["reaction_id"],
            "components": {"thiol_head": "CN(C)CCS", "maleate": "CCCCCCCCOC(=O)C=CC(=O)OCCCCCCCC"},
            "events": 1,
            "expected_product": "CN(C)CCSC(CC(=O)OCCCCCCCC)C(=O)OCCCCCCCC",
            "expected_formula": "C24H47NO4S",
            "expected_computed_consistency_pass": True,
            "expected_forward_product_count": 1,
            "locator": maleate["sources"][0]["locator"],
            "evidence_basis": "computed_transform_consistency",
        }
    ]
    for c in prior["source_controls"]:
        c = copy.deepcopy(c)
        c.update(
            family="a3_amine_aldehyde_alkyne",
            reaction_id=a3["reaction_id"],
            expected_computed_consistency_pass=True,
            expected_forward_product_count=1,
        )
        positive.append(c)
    ambiguity = [
        {
            "label": "asymmetric_maleate_thiol_ambiguous_orientation",
            "family": "maleate_addition",
            "reaction_id": maleate["reaction_id"],
            "components": {"thiol_head": "CN(C)CCS", "maleate": "COC(=O)C=CC(=O)OCC"},
            "events": 1,
            "expected_product": "CN(C)CCSC(C(=O)OC)CC(=O)OCC",
            "expected_formula": "C11H21NO4S",
            "expected_computed_consistency_pass": False,
            "expected_forward_product_count": 2,
            "interpretation": "Synthetic attribution control; no experimental failure label.",
        },
        {
            "label": "a3_unresolved_amine_site",
            "family": "a3_amine_aldehyde_alkyne",
            "reaction_id": a3["reaction_id"],
            "components": {"amine_head": "NCCNC", "aldehyde": "C=O", "alkyne": "C#CC"},
            "events": 1,
            "expected_product": "CNCCNCC#CC",
            "expected_formula": "C7H14N2",
            "expected_computed_consistency_pass": False,
            "expected_forward_product_count": 2,
            "interpretation": "Synthetic attribution control; no experimental failure label.",
        },
    ]
    control_path = HERE / "thiol-a3-adjudication.json"
    dump(
        control_path,
        {
            "schema_version": "forge.repeated_source_adjudication.v1",
            "registry": pin(ROOT, output),
            "assets": assets,
            "source_controls": positive,
            "ambiguity_controls": ambiguity,
            "disposition": "admit_transform_consistency",
            "evidence_basis": "computed_transform_consistency",
            "training_admitted": False,
            "experimental_execution_admitted": False,
            "a3_architecture_qualified": False,
            "a3_architecture_conflicts": prior["conflicts"],
            "limitations": [
                "A3 fixed-component virtual replay does not validate conflicting source architecture annotations.",
                "49D8 control follows the reported library combination and generic drawn S-addition scheme; no individual isolated yield or spectrum is available.",
                "Thiol and amine mechanisms have separate registry identities and controls.",
            ],
        },
    )
    from forge.corpus.compose_lipid_current_replay import CONFIG_SCHEMA, POLICY

    prep = ROOT / "results/phase1/compose_lipid_full_preparation_v1/result.json"
    cfg = {
        "schema_version": CONFIG_SCHEMA,
        "policy": POLICY,
        "inputs": {
            "preparation": pin(ROOT, prep),
            "registry": pin(ROOT, output),
            "family_definitions": json.loads(prep.read_text())["inputs"]["family_definitions"],
            "transform_controls": pin(ROOT, control_path),
        },
        "families": {
            "maleate_addition": {
                "reaction_id": maleate["reaction_id"],
                "registry_to_source_roles": {"thiol_head": "amine_head", "maleate": "maleate"},
            },
            "a3_amine_aldehyde_alkyne": {
                "reaction_id": a3["reaction_id"],
                "registry_to_source_roles": {
                    "amine_head": "amine_head",
                    "aldehyde": "aldehyde",
                    "alkyne": "alkyne",
                },
            },
        },
        "search_bounds": {
            "maximum_events": 6,
            "maximum_outcomes": 256,
            "maximum_states": 4096,
            "maximum_transitions": 16384,
        },
    }
    dump(ROOT / "configs/multireaction/compose_lipid_supplied_thiol_a3_v1.json", cfg)


if __name__ == "__main__":
    main()
