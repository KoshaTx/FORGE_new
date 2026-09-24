"""Overlay complete source-derived B5 precursor grammars on frozen draft programs.

Hydrocarbon and spacer variations qualify computed consistency, never source execution.
The input-only scopes cannot inspect a target or select a reaction outcome.
"""

import copy
import json
from pathlib import Path

from forge.assembly.component_scope import RegistryComponentScopes
from forge.corpus.compose_lipid_source_view import dump, pin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "compose_lipid_b5_source_v1"


def main():
    draft = PRIOR / "draft-registry.json"
    controls_path = PRIOR / "control-transcriptions.json"
    proposal_path = PRIOR / "complete-terminal-scope-proposal.json"
    registry = json.loads(draft.read_text())
    controls = {c["label"]: c for c in json.loads(controls_path.read_text())["controls"]}
    proposal = json.loads(proposal_path.read_text())
    scopes = {
        "source_core": {"kind": "exact", "smiles": controls["I71"]["components"]["vitamin_b5_core"]}
    }
    for name, grammar in proposal["source_head_grammars"].items():
        for suffix, counts in (("spacer", None), ("reported", [2])):
            spec = {
                "kind": "template",
                "templates": grammar["complete_smiles_templates"],
                "repeat_smiles": "C",
                "minimum_repeats": 1,
            }
            if counts is not None:
                spec["repeat_counts"] = counts
            scopes[name + "_" + suffix] = spec
    body = {
        "allowed_atomic_numbers": [0, 6],
        "allowed_bond_types": ["SINGLE", "DOUBLE"],
        "maximum_double_bonds": 1,
        "maximum_branch_carbons": 1,
        "branch_root_distances": None,
    }
    # Queries identify bonds in the independently transcribed source precursors;
    # complete retained fragments and body grammars enforce the rest of the graph.
    scopes["primary_tail_alcohol"] = {
        "kind": "cut",
        "query": "[OH1:1]-[CH2:2]-[C:3]",
        "cut_bonds": [[1, 2]],
        "central_anchor_map": 1,
        "central": {"smiles": "O[*:1]"},
        "body": body,
        "equal_bodies": False,
    }
    scopes["half_ester_tail_acid"] = {
        "kind": "cut",
        "query": "[CX3:1](=[OX1:2])-[OX2:3]-[CH2:4]-[C:5]",
        "cut_bonds": [[3, 4]],
        "central_anchor_map": 1,
        "central": {
            "templates": ["O=C(O){spacer}C(=O)O[*:1]"],
            "repeat_smiles": "C",
            "minimum_repeats": 1,
        },
        "body": copy.deepcopy(body),
        "equal_bodies": False,
    }
    scopes["paired_tail_alcohol"] = {
        "kind": "cut",
        "query": "[OH1:1]-[CH2:2]-[CH:3](-[O:4]-[CX3:5](=[OX1:6])-[C:7])-[CH2:8]-[O:9]-[CX3:10](=[OX1:11])-[C:12]",
        "cut_bonds": [[5, 7], [10, 12]],
        "central_anchor_map": 3,
        "central": {"smiles": "OCC(OC(=O)[*:1])COC(=O)[*:1]"},
        "body": copy.deepcopy(body),
        "equal_bodies": True,
    }
    registry["component_scopes"] = scopes
    profile_scopes = {}
    for control in controls.values():
        profile = control["profile"]
        mapping = {"vitamin_b5_core": "source_core"}
        if profile == "I7":
            mapping.update(head_acid="I7_head_acid_spacer", tail6_alcohol="paired_tail_alcohol")
        elif profile == "I8":
            mapping.update(
                head_acid="I8_head_acid_spacer",
                r2_tail_alcohol="primary_tail_alcohol",
                tail_acid="half_ester_tail_acid",
            )
        else:
            kind = profile.split("_")[-1]
            mapping.update(
                head_nucleophile=f"I9_head_{kind}_spacer",
                tail_acid_1="half_ester_tail_acid",
                tail_acid_2="half_ester_tail_acid",
            )
        if set(mapping) != set(control["components"]):
            raise ValueError("Incomplete source-control role scope")
        profile_scopes[profile] = mapping
    registry["profile_component_scopes"] = profile_scopes
    # The task format binds each input by its position AND original generic role.
    # Neither a product match nor a guessed alias determines these bindings.
    generated = {
        "linear_length": {"kind": "body", "branches": 0, "double_bonds": 0},
        "conservative_branch_partition": {"kind": "body", "branches": 1, "double_bonds": 0},
        "positional_mono_unsaturation": {"kind": "body", "branches": 0, "double_bonds": 1},
    }

    def tail_rules(prefix, reported_name, reported_smiles):
        return {
            **{prefix + k: copy.deepcopy(v) for k, v in generated.items()},
            reported_name: {"kind": "reported", "smiles": reported_smiles},
        }

    acid11 = controls["I91"]["components"]["tail_acid_2"]
    acid18 = controls["I91"]["components"]["tail_acid_1"]
    r2 = [controls["I81"]["components"]["r2_tail_alcohol"], "OCC(CCCCCC)CCCCCCCC"]
    ester_fresh = {"nucleophile": "fresh", "operation": "ester", "site": "unique"}
    ester_primary = {"nucleophile": "current", "operation": "ester", "site": "primary"}
    ester_secondary = {"nucleophile": "current", "operation": "ester", "site": "secondary"}
    profiles = {}
    for lane, profile in [
        ("I7_head_x_paired_tail", "I7"),
        ("acid_one_component", "I8"),
        ("r2_one_component", "I8"),
        ("matched_body", "I8"),
        ("one_tail_acid_knob_amine", "I9_hetero_amine"),
        ("one_tail_acid_knob_alcohol", "I9_hetero_alcohol"),
        ("matched_identical_tail_acids_amine", "I9_homo_amine"),
        ("matched_identical_tail_acids_alcohol", "I9_homo_alcohol"),
    ]:
        mapping = profile_scopes[profile]
        series = profile.split("_")[0]
        head_kind = "acid" if series != "I9" else profile.split("_")[-1]
        head_role = "head_acid" if series != "I9" else "head_nucleophile"
        entry = {
            "selector": {"series": series, "design_lane": lane},
            "profile": profile,
            "program_id": "source_b5_" + profile,
            "component_scopes": mapping,
            "registry_to_source_roles": {
                r: "paired_tail_alcohol" if r == "tail6_alcohol" else r for r in mapping
            },
            "head_role": head_role,
            "head_kind": head_kind,
            "reported_head_scope": f"{series}_head_{head_kind}_reported",
            "equal_body_roles": [],
            "equal_component_roles": [],
            "minimum_reported_tails": 0,
        }
        if series == "I7":
            entry["reagent_order"] = [
                ["vitamin_b5_core", "source_carboxylic_acid"],
                ["tail6_alcohol", "source_alcohol"],
                ["head_acid", "source_carboxylic_acid"],
            ]
            entry["events"] = [ester_fresh, ester_primary]
            entry["tail_roles"] = ["tail6_alcohol"]
            entry["tail_rules"] = {
                "tail6_alcohol": tail_rules(
                    "paired_identical_arms_",
                    "reported_tail6",
                    [controls["I71"]["components"]["tail6_alcohol"]],
                )
            }
        elif series == "I8":
            entry["reagent_order"] = [
                ["vitamin_b5_core", "source_carboxylic_acid"],
                ["r2_tail_alcohol", "source_alcohol"],
                ["head_acid", "source_carboxylic_acid"],
                ["tail_acid", "source_carboxylic_acid"],
            ]
            entry["events"] = [ester_fresh, ester_primary, ester_secondary]
            entry["tail_roles"] = ["r2_tail_alcohol", "tail_acid"]
            entry["tail_rules"] = {
                "r2_tail_alcohol": tail_rules("", "reported_R2_alcohol", r2),
                "tail_acid": tail_rules(
                    "internal_ester_spacer_plus_", "reported_tail_acid", [acid11]
                ),
            }
            if lane == "acid_one_component":
                entry["tail_rules"]["r2_tail_alcohol"] = {
                    "reported_R2_alcohol": {"kind": "reported", "smiles": r2}
                }
            elif lane == "r2_one_component":
                entry["tail_rules"]["tail_acid"] = {
                    "reported_tail_acid": {"kind": "reported", "smiles": [acid11]}
                }
            else:
                entry["equal_body_roles"] = ["r2_tail_alcohol", "tail_acid"]
        else:
            source_head_role = "source_primary_amine" if head_kind == "amine" else "source_alcohol"
            entry["reagent_order"] = [
                ["vitamin_b5_core", "source_alcohol"],
                ["tail_acid_1", "source_carboxylic_acid"],
                ["tail_acid_2", "source_carboxylic_acid"],
                ["head_nucleophile", source_head_role],
            ]
            entry["events"] = [
                ester_primary,
                ester_secondary,
                {**ester_fresh, "operation": "amide" if head_kind == "amine" else "ester"},
            ]
            entry["tail_roles"] = ["tail_acid_1", "tail_acid_2"]
            first = acid11 if "_homo_" in profile else acid18
            entry["tail_rules"] = {
                "tail_acid_1": tail_rules(
                    "internal_ester_spacer_plus_", "reported_tail_acid", [first]
                ),
                "tail_acid_2": tail_rules(
                    "internal_ester_spacer_plus_", "reported_tail_acid", [acid11]
                ),
            }
            if "_homo_" in profile:
                entry["equal_component_roles"] = ["tail_acid_1", "tail_acid_2"]
            else:
                entry["minimum_reported_tails"] = 1
        profiles[lane] = entry
    registry["original_task_contract"] = {
        "schema": "family_23_vitamin_b5_task_v8",
        "family": "vitamin_b5_multistep",
        "metadata_fields": ["series", "design_lane", "head_axis", "tail_axis", "head_id"],
        "core_role": "vitamin_b5_core",
        "source_core_smiles": controls["I71"]["components"]["vitamin_b5_core"],
        "source_head_alias": {"field": "head_id", "roles": ["series_head"]},
        "profiles": profiles,
        "reported_r2_locator": "SI Fig.S2 PDF p.16 and source alcohol procedure p.17",
        "reported_label_policy": "Reported identity must agree with the series- and site-specific source precursor. Incorrect reported labels abstain, even if a generic transform would match.",
    }
    registry["complete_scope_curation"] = {
        "status": "component_scope_qualification_candidate_not_corpus_admission",
        "derivation": pin(ROOT, Path(__file__).resolve()),
        "inputs": {
            "draft_programs": pin(ROOT, draft),
            "source_controls": pin(ROOT, controls_path),
            "source_scope_proposal": pin(ROOT, proposal_path),
            "family_definitions": pin(
                ROOT,
                ROOT
                / "data/source_cache/compose_lipid_supplement_2026-09-19/family_reaction_definitions.json",
            ),
        },
        "body_policy": "Acyclic neutral hydrocarbon, at most one C=C and one degree-three branch. Branch position may vary under the supplied family definition's conservative hydrophobe branch knob; source attachment sites and complete functional skeletons remain fixed. Root labels preserve attachment identity. This grammar supports computed consistency only.",
    }
    RegistryComponentScopes(scopes)
    output = ROOT / "data/vendor/qualified_b5_staged_source_program_v1.json"
    dump(output, registry)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
