"""Draft explicit B5 site programs from source drawings and frozen coupling primitives.

This is source-control qualification only. No corpus replay or training admission follows
from this draft; source profile and complete-terminal scope qualification remain required.
"""

import copy
import json
from pathlib import Path

from rdkit import Chem

from forge.assembly.families import constitutional_molecule
from forge.assembly.staged_program import RegistryStagedProgram
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def clear_maps(text):
    m = Chem.MolFromSmarts(text)
    if m is None:
        raise ValueError("Invalid source query")
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmarts(m)


def main():
    controls_path = HERE / "control-transcriptions.json"
    evidence = json.loads(controls_path.read_text())
    ester_path = ROOT / "data/vendor/qualified_maleate_ester_source_program_v1.json"
    family_path = ROOT / "data/vendor/qualified_reaction_families_v1.json"
    a3_path = ROOT / "data/vendor/qualified_a3_source_program_v2.json"
    basic_path = ROOT / "data/vendor/qualified_staar_source_program_v1.json"
    ester = next(
        r
        for r in json.loads(ester_path.read_text())["reactions"]
        if r["reaction_id"] == "source_aminoalcohol_o_esterification"
    )
    amide = next(
        r
        for r in json.loads(family_path.read_text())["reactions"]
        if r["reaction_id"] == "amide_coupling_acid_amine"
    )
    nh = (
        json.loads(a3_path.read_text())["reactions"][0]["atom_mapped_reaction_smarts"]
        .split(">>")[0]
        .split(".")[0]
        .replace(":1]", ":4]")
    )
    acid, alcohol = ester["atom_mapped_reaction_smarts"].split(">>")[0].split(".")
    nh_handle = clear_maps(nh)
    oh_handle = clear_maps(alcohol)
    parents = {
        "ester": pin(ROOT, ester_path),
        "amide": pin(ROOT, family_path),
        "nonacyl_nh": pin(ROOT, a3_path),
        "basic_head": pin(ROOT, basic_path),
    }
    # Explicit source core (SI Figs.S1-S3) gets complete local atom maps. No corpus
    # target or current generated precursor is consulted to derive a site query.
    core_source = evidence["controls"][0]["components"]["vitamin_b5_core"]
    core_mol = Chem.MolFromSmiles(core_source)
    Chem.RemoveStereochemistry(core_mol)
    for i, a in enumerate(core_mol.GetAtoms()):
        a.SetAtomMapNum(100 + i)
    core = Chem.MolToSmiles(core_mol, canonical=False)
    expected = "[CH3:100][C:101]([CH3:102])([CH2:103][OH:104])[CH:105]([OH:106])[C:107](=[O:108])[NH:109][CH2:110][CH2:111][C:112](=[O:113])[OH:114]"
    if core != expected:
        raise ValueError("Source core atom-index transcription changed")
    # The full source scaffold fixes primary O104, secondary O106 and acid C112.
    # Local maps are reaction-local; they do not redefine global component IDs.
    after_tail = core.replace("[OH:114]", "[O:114][C&X4:115]")
    after_head = after_tail.replace("[OH:104]", "[O:104][C&X3:116](=[O:117])")
    dual_tail = core.replace("[OH:104]", "[O:104][C&X3:116](=[O:117])").replace(
        "[OH:106]", "[O:106][C&X3:118](=[O:119])"
    )
    source = [
        {
            "kind": "doi",
            "identifier": evidence["doi"],
            "locator": "SI Figs.S1-S3, PDF pp.7,16,25; exact control-specific pages in control-transcriptions.json",
            "notes": "Net composite stage controls only. Source discrepancies retained; no generated-product execution or direct unprotected chemoselectivity claim.",
        }
    ]
    reactions = []

    def reaction(name, reactants, product, forbidden, condition):
        roles = []
        for (role, fragment), blocked in zip(reactants, forbidden, strict=True):
            roles.append(
                {
                    "name": role,
                    "count": 1,
                    "required_handle_smarts": clear_maps(fragment),
                    "allowed_site_multiplicity": [1],
                    "forbidden_smarts": blocked,
                }
            )
        r = {
            k: copy.deepcopy(ester[k])
            for k in (
                "reaction_version",
                "status",
                "stereochemistry_policy",
                "protonation_and_salt_policy",
            )
        }
        r.update(
            reaction_id=name,
            atom_mapped_reaction_smarts=".".join(s for _, s in reactants) + ">>" + product,
            reactant_roles=roles,
            implementation={
                "kind": "source_drawing_overlay",
                "parent_registries": parents,
                "derivation": pin(ROOT, Path(__file__).resolve()),
            },
            sources=source,
            conditions=condition,
            selectivity_policy="All sites satisfying the complete independently transcribed B5 core and source attachment assignment; every outcome/inverse retained. Net composite source stages do not assert one-pot selectivity.",
            known_positive_examples=[],
            known_negative_examples=[],
        )
        reactions.append(r)
        return name

    first_i7 = reaction(
        "source_b5_i7_carboxyl_tail_net",
        [("vitamin_b5_core", core), ("tail6_alcohol", alcohol)],
        core.replace("[OH:114]", "[O:4][C:5]"),
        [[], [nh_handle]],
        {
            "source_steps": "Acetal protection, tail6 coupling, acetal removal; SI pp.7-11",
            "solvent": ["DCM"],
            "temperature_c": None,
            "time_h": None,
            "catalyst_or_reagent": ["DMP", "p-TsOH", "EDCI.HCl", "DMAP", "DTT"],
            "net_composite": True,
        },
    )
    first_i8 = reaction(
        "source_b5_i8_carboxyl_tail_net",
        [("vitamin_b5_core", core), ("r2_tail_alcohol", alcohol)],
        core.replace("[OH:114]", "[O:4][C:5]"),
        [[], [nh_handle]],
        {
            "source_steps": "Acetal protection, R2 tail coupling, acetal removal; SI pp.16-18",
            "solvent": ["DCM"],
            "temperature_c": None,
            "time_h": None,
            "catalyst_or_reagent": ["DMP", "p-TsOH", "EDCI.HCl", "DMAP", "DTT"],
            "net_composite": True,
        },
    )
    head = reaction(
        "source_b5_primary_oh_head",
        [("b5_tail_diol", after_tail), ("head_acid", acid)],
        after_tail.replace("[OH:104]", "[O:104][C:1](=[O:2])"),
        [[nh_handle], [nh_handle, oh_handle]],
        {
            "source_steps": "Source primary RA head ester; SI Fig.S1 or S2",
            "solvent": ["DCM"],
            "temperature_c": None,
            "time_h": None,
            "catalyst_or_reagent": ["EDCI.HCl", "DMAP"],
            "net_composite": False,
        },
    )
    secondary = reaction(
        "source_b5_secondary_oh_tail",
        [("b5_head_secondary_alcohol", after_head), ("tail_acid", acid)],
        after_head.replace("[OH:106]", "[O:106][C:1](=[O:2])"),
        [[nh_handle], [nh_handle, oh_handle]],
        {
            "source_steps": "Source secondary RB tail ester; SI Fig.S2",
            "solvent": ["DCM"],
            "temperature_c": None,
            "time_h": None,
            "catalyst_or_reagent": ["EDCI.HCl", "DMAP"],
            "net_composite": False,
        },
    )
    second_acid = acid.replace(":1]", ":21]").replace(":2]", ":22]").replace(":3]", ":23]")
    dual = reaction(
        "source_b5_site_ordered_dual_tail_net",
        [("vitamin_b5_core", core), ("tail_acid_1", acid), ("tail_acid_2", second_acid)],
        core.replace("[OH:104]", "[O:104][C:1](=[O:2])").replace(
            "[OH:106]", "[O:106][C:21](=[O:22])"
        ),
        [[], [nh_handle, oh_handle], [nh_handle, oh_handle]],
        {
            "source_steps": "Fm protection, primary then secondary O-acylation for heteropair / simultaneous identical pair, Fm removal to isolated23/27; SI pp.25-33",
            "solvent": ["DCM", "DMF"],
            "temperature_c": None,
            "time_h": None,
            "catalyst_or_reagent": ["Fm protecting group", "EDCI.HCl", "DMAP", "piperidine"],
            "net_composite": True,
            "condition_conflicts_retained": True,
        },
    )
    ends = {}
    for kind, fragment, productlink in [
        ("alcohol", alcohol, "[O:4][C:5]"),
        (
            "amine",
            nh,
            next(
                a.GetSmarts()
                for a in Chem.MolFromSmarts(
                    amide["atom_mapped_reaction_smarts"].split(">>")[1]
                ).GetAtoms()
                if a.GetAtomMapNum() == 4
            ),
        ),
    ]:
        ends[kind] = reaction(
            "source_b5_carboxyl_head_" + kind,
            [("b5_dual_tail_acid", dual_tail), ("head_nucleophile", fragment)],
            dual_tail.replace("[OH:114]", productlink),
            [[nh_handle], [nh_handle] if kind == "alcohol" else [oh_handle]],
            {
                "source_steps": "Source terminal head ester or amide at RC; SI pp.29-36",
                "solvent": ["DCM"],
                "temperature_c": None,
                "time_h": None,
                "catalyst_or_reagent": (
                    ["EDCI.HCl", "DMAP"] if kind == "alcohol" else ["HATU", "DIPEA"]
                ),
                "net_composite": False,
            },
        )

    def stage(rxn, added, water, source_step):
        r = next(r for r in reactions if r["reaction_id"] == rxn)
        return {
            "reaction_id": rxn,
            "accumulator_role": r["reactant_roles"][0]["name"],
            "added_roles": added,
            "net_byproducts": {"H": 2 * water, "O": water, "formal_charge": 0},
            "source_step": source_step,
        }

    basic = json.loads(basic_path.read_text())["programs"][0]["product_constraints"][
        "required_queries"
    ]
    core_constraint = {
        "allowed_atomic_numbers": [1, 6, 7, 8],
        "formal_charge": 0,
        "allow_aromatic_atoms": False,
        "exact_element_counts": dict(
            element for element in [("C", 9), ("H", 17), ("N", 1), ("O", 5)]
        ),
        "required_queries": [
            {"name": "source_b5_complete_core", "smarts": clear_maps(core), "minimum_matches": 1}
        ],
    }
    programs = []
    for profile in [
        "I7",
        "I8",
        "I9_hetero_amine",
        "I9_hetero_alcohol",
        "I9_homo_amine",
        "I9_homo_alcohol",
    ]:
        if profile in ("I7", "I8"):
            tailrole = "tail6_alcohol" if profile == "I7" else "r2_tail_alcohol"
            stages = [
                stage(
                    first_i7 if profile == "I7" else first_i8,
                    [tailrole],
                    1,
                    "Source acetal protection/coupling/deprotection composite to8/14",
                ),
                stage(head, ["head_acid"], 1, "Source RA head acylation toI7/15/16"),
            ]
            if profile == "I8":
                stages.append(stage(secondary, ["tail_acid"], 1, "Source RB tail acylation toI8"))
        else:
            kind = profile.split("_")[-1]
            stages = [
                stage(
                    dual,
                    ["tail_acid_1", "tail_acid_2"],
                    2,
                    "Source Fm cycle with exact RA/RB assignment to23 or27",
                ),
                stage(ends[kind], ["head_nucleophile"], 1, "Source RC head " + kind + " coupling"),
            ]
        roles = {"vitamin_b5_core", *(r for s in stages for r in s["added_roles"])}
        constraints = {
            r: {
                "allowed_atomic_numbers": [1, 6, 7, 8] if r.startswith("head") else [1, 6, 8],
                "formal_charge": 0,
                "allow_aromatic_atoms": False,
            }
            for r in roles
        }
        constraints["vitamin_b5_core"] = core_constraint
        for role, n in [
            ("tail6_alcohol", 5),
            ("r2_tail_alcohol", 1),
            ("tail_acid", 4),
            ("tail_acid_1", 4),
            ("tail_acid_2", 4),
        ]:
            if role in constraints:
                constraints[role]["exact_element_counts"] = {"O": n}
        for role in roles:
            if role.startswith("head"):
                constraints[role]["required_queries"] = copy.deepcopy(basic)
        programs.append(
            {
                "program_id": "source_b5_" + profile,
                "initial_role": "vitamin_b5_core",
                "stages": stages,
                "terminal_constraints": constraints,
                "product_constraints": {
                    "formal_charge": 0,
                    "required_queries": copy.deepcopy(basic),
                },
                "architecture_subfamily": profile,
                "scope": "DRAFT source-control validation. Net composite stages contract documented protecting-group cycles while preserving exact source sites and independently transcribed isolated stage endpoints. Full terminal grammar and corpus role binding NOT yet qualified. Computed consistency only.",
                "equal_component_groups": (
                    [["tail_acid_1", "tail_acid_2"]] if "_homo_" in profile else []
                ),
            }
        )
    registry = {
        "schema_version": "forge.source_staged_registry.v1",
        "curation": {
            "status": "draft_source_controls_only",
            "training_admitted": False,
            "corpus_replay_qualified": False,
        },
        "source_assets": {**evidence["assets"], "control_transcriptions": pin(ROOT, controls_path)},
        "reactions": reactions,
        "programs": programs,
    }
    path = HERE / "draft-registry.json"
    dump(path, registry)
    checks = {}
    for control in evidence["controls"]:
        program = RegistryStagedProgram.from_registry(
            path,
            program_id="source_b5_" + control["profile"],
            expected_sha256=pin(ROOT, path)["sha256"],
        )
        r = program.replay(control["components"], control["expected_product"])
        endpoints = [
            sorted([constitutional_molecule(s)[0]]) for s in control["expected_stage_products"]
        ]
        checks[control["label"]] = {
            "computed_consistency_pass": r["computed_consistency_pass"],
            "independent_stage_endpoints_match": r["forward_layers"][1:] == endpoints,
            "replay": r,
        }
        print(
            control["label"],
            r["computed_consistency_pass"],
            [len(x) for x in r["forward_layers"]],
            r["checks"],
            flush=True,
        )
    dump(
        HERE / "draft-control-replay.json",
        {
            "schema_version": "forge.b5_draft_source_control_replay.v1",
            "seed": 0,
            "inputs": {"controls": pin(ROOT, controls_path), "draft_registry": pin(ROOT, path)},
            "implementation": {
                str(p.relative_to(ROOT)): pin(ROOT, p)
                for p in [
                    Path(__file__).resolve(),
                    ROOT / "forge/assembly/staged_program.py",
                    ROOT / "forge/assembly/families.py",
                    ROOT / "forge/assembly/repeated_components.py",
                    ROOT / "forge/assembly/program.py",
                    ROOT / "forge/assembly/registry.py",
                ]
            },
            "controls": checks,
            "training_admitted": False,
            "corpus_replay_qualified": False,
        },
    )


if __name__ == "__main__":
    main()
