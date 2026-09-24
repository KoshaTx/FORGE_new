"""Source-qualified Ren/Love stages with original incorporated quantities."""

from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.grouped_program import RegistryGroupedProgram
from forge.assembly.repeated_components import RepeatBounds
from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_grouped_replay import replay_record as grouped_replay

FAMILIES = {"amine_alkylation": "ren", "amine_epoxide_opening": "love"}
POLICY = {
    "seed": 0,
    "population": "current_full_partition_eligible_pending_only",
    "component_join": "global_component_id",
    "occupancy": "original_task_incorporated_quantities_not_feed_equivalents",
    "search": "complete_unfiltered_forward_and_inverse_at_declared_stage_boundary",
    "evidence_basis": "computed_transform_consistency",
    "training_admitted": False,
    "experimental_execution_admitted": False,
    "training_calls": 0,
}


def load_contract(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if (
        config.get("schema_version") != "forge.ren_love_replay_config.v1"
        or config.get("policy") != POLICY
    ):
        raise ComposeLipidError("Ren/Love replay contract differs")
    if set(config.get("inputs", {})) != {"registry", "source_control", "family_definitions"}:
        raise ComposeLipidError("Ren/Love replay inputs differ")
    paths = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    registry = json.loads(paths["registry"].read_text())
    sources = json.loads(paths["source_control"].read_text())
    definitions = json.loads(paths["family_definitions"].read_text())
    if (
        sources.get("schema_version") != "forge.ren_love_source_transcriptions.v1"
        or registry["source_assets"]["control_transcriptions"] != config["inputs"]["source_control"]
        or registry.get("training_admitted") is not False
        or registry.get("experimental_execution_admitted") is not False
        or definitions["amine_alkylation"]["roles"] != {"amine_head": 1, "bromoester_arm": 2}
        or definitions["amine_alkylation"]["variable"]
        or definitions["amine_epoxide_opening"]["roles"] != {"amine_head": 1, "epoxide_tail": 6}
        or definitions["amine_epoxide_opening"]["variable"] != ["occupancy"]
    ):
        raise ComposeLipidError("Ren/Love source binding differs")
    for name, value in registry["source_assets"].items():
        resolve_pin(value, repo, label=name)
    for reaction in registry["reactions"]:
        for name, value in reaction["implementation"]["parent_registries"].items():
            resolve_pin(value, repo, label=name)
        resolve_pin(reaction["implementation"]["derivation"], repo, label="derivation")
    executors, checked = {}, {}
    for family, key in FAMILIES.items():
        source = sources["controls"][key]
        side = "bromoester_arm" if key == "ren" else "epoxide_tail"
        programs = {}
        for events in ([2] if key == "ren" else range(1, 7)):
            program = RegistryGroupedProgram.from_registry(
                paths["registry"],
                program_id=f"source_{key}_{events}_incorporated_arms",
                expected_sha256=config["inputs"]["registry"]["sha256"],
                bounds=RepeatBounds(**config["search_bounds"]),
            )
            if program.quantities != {"amine_head": 1, side: events}:
                raise ComposeLipidError("Source program multiplicity differs")
            programs[events] = {
                "program": program,
                "mapping": {role: role for role in program.roles},
            }
        mol = Chem.MolFromSmiles(source["expected_product"])
        if (
            source["family"] != family
            or source["asset"] != registry["source_assets"][key]
            or mol is None
            or rdMolDescriptors.CalcMolFormula(mol) != source["expected_formula"]
        ):
            raise ComposeLipidError("Independent source control differs")
        program = programs[source["quantities"][side]]["program"]
        control = program.replay(source["components"], source["expected_product"])
        if (
            source["quantities"] != program.quantities
            or not control["computed_consistency_pass"]
            or list(map(len, control["forward_layers"])) != [1, 1]
        ):
            raise ComposeLipidError("Independent source control failed")
        executors[family] = programs
        checked[source["label"]] = control
    return config, paths, executors, checked


def replay_record(item: dict, structures: dict[str, str], executors: dict) -> dict:
    prepared = item["preparation"]
    if prepared.get("eligible_for_program_preparation") is not True:
        raise ComposeLipidError("Protected or unassigned row reached Ren/Love replay")
    family = prepared.get("family")
    if family not in FAMILIES:
        raise ComposeLipidError("Wrong family reached Ren/Love replay")

    def pending(reason):
        return {"computed_consistency_pass": False, "disposition": reason}

    if prepared.get("construction_basis") != "v8_exact_family_task":
        return pending("missing_original_task_occupancy")
    supplied = {}
    for role, identity, quantity in prepared["component_instances"]:
        if identity not in structures:
            raise ComposeLipidError(f"Unresolved precursor: {identity}")
        if type(quantity) is not int or quantity < 1:
            raise ComposeLipidError("Invalid source incorporated quantity")
        if role in supplied:
            return pending("unsupported_source_role_tuple")
        supplied[role] = (identity, quantity)
    side = "bromoester_arm" if family == "amine_alkylation" else "epoxide_tail"
    if set(supplied) != {"amine_head", side}:
        return pending("unsupported_source_role_tuple")
    events = supplied[side][1]
    if events not in executors[family] or supplied["amine_head"][1] != 1:
        return pending("outside_qualified_program_multiplicity")
    executor = executors[family][events]
    result = grouped_replay(item, structures, executor)
    result.update(
        program_id=executor["program"].specification["program_id"],
        source_occupancy=events,
        evidence_basis="computed_transform_consistency",
        experimental_execution_admitted=False,
        training_admitted=False,
    )
    return result
