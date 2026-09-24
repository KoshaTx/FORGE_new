"""Source-ordered AEMA replay using incorporated counts from original generator tasks."""

from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import constitutional_molecule
from forge.assembly.grouped_program import RegistryGroupedProgram
from forge.assembly.repeated_components import RepeatBounds
from forge.assembly.staged_program import _constraints
from forge.core.hashing import resolve_pin
from forge.corpus.compose_lipid_grouped_replay import replay_record as grouped_replay

FAMILY = "aema_aza_thiol_addition"
SCHEMA = "forge.compose_lipid_aema_replay_config.v1"
POLICY = {
    "seed": 0,
    "population": "current_full_partition_eligible_only",
    "component_join": "global_component_id",
    "occupancy_source": "v8_exact_family_task_incorporated_quantities",
    "fixed_scaffold": "whole_AEMA_identity_from_pinned_registry",
    "stages": "acrylate_aza_addition_then_methacrylate_thiol_addition",
    "search": "complete_forward_and_inverse_without_target_pruning",
    "domain": "neutral_saturated_aliphatic_CHNO_core_and_alkyl_or_alkenyl_CHS_monothiol",
    "training_admitted": False,
    "experimental_execution_admitted": False,
    "training_calls": 0,
}


def load_contract(repo: Path, config_path: Path) -> tuple[dict, dict, dict, dict]:
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != SCHEMA or config.get("policy") != POLICY:
        raise ComposeLipidError("AEMA replay schema or policy differs")
    if set(config.get("inputs", {})) != {"registry", "source_control", "family_definitions"}:
        raise ComposeLipidError("AEMA replay inputs differ")
    paths = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    registry = json.loads(paths["registry"].read_text())
    source = json.loads(paths["source_control"].read_text())
    definition = json.loads(paths["family_definitions"].read_text())[FAMILY]
    if (
        registry.get("training_admitted") is not False
        or registry.get("experimental_execution_admitted") is not False
        or source.get("schema_version") != "forge.aema_source_control.v1"
        or source.get("registry") != config["inputs"]["registry"]
        or source.get("source_asset") != registry["source_assets"]["lee_si"]
        or source.get("experimental_execution_admitted") is not False
        or definition["roles"] != {"amine_core": 1, "thiol_periphery": 1}
        or definition["variable"]
        or set(registry.get("fixed_components", {})) != {"AEMA"}
    ):
        raise ComposeLipidError("AEMA source or fixed-scaffold contract differs")
    for name, value in registry["source_assets"].items():
        resolve_pin(value, repo, label=name)
    for reaction in registry["reactions"]:
        for name, value in reaction["implementation"]["parent_registries"].items():
            resolve_pin(value, repo, label=name)
        resolve_pin(reaction["implementation"]["derivation"], repo, label="AEMA derivation")
    executors = {}
    bond_types = registry["component_bond_types"]
    if set(bond_types) != {"amine_core", "thiol_periphery"} or any(
        not values or any(value not in Chem.BondType.names for value in values)
        for values in bond_types.values()
    ):
        raise ComposeLipidError("AEMA source component bond domain differs")
    for spec in registry["grouped_programs"]:
        program = RegistryGroupedProgram.from_registry(
            paths["registry"],
            program_id=spec["program_id"],
            expected_sha256=config["inputs"]["registry"]["sha256"],
            bounds=RepeatBounds(**config["search_bounds"]),
        )
        events = spec["stages"][0]["events"]
        if (
            events in executors
            or len(spec["stages"]) != 2
            or program.quantities != {"amine_core": 1, "AEMA": events, "thiol_periphery": events}
        ):
            raise ComposeLipidError("AEMA stage multiplicities or roles differ")
        executors[events] = {
            "program": program,
            "mapping": {role: role for role in program.roles},
            "fixed_AEMA": constitutional_molecule(registry["fixed_components"]["AEMA"])[0],
            "component_bond_types": bond_types,
        }
    if set(executors) != set(range(1, 7)):
        raise ComposeLipidError("AEMA source occupancy domain differs")
    control_mol = Chem.MolFromSmiles(source["expected_product"])
    if (
        control_mol is None
        or rdMolDescriptors.CalcMolFormula(control_mol) != source["expected_formula"]
        or source["declared_quantities"] != executors[4]["program"].quantities
        or constitutional_molecule(source["components"]["AEMA"])[0] != executors[4]["fixed_AEMA"]
    ):
        raise ComposeLipidError("AEMA independent source control identity differs")
    replay = executors[4]["program"].replay(source["components"], source["expected_product"])
    if not replay["computed_consistency_pass"] or replay["forward_layers"][1:] != [
        [constitutional_molecule(s)[0]] for s in source["expected_stage_products"]
    ]:
        raise ComposeLipidError("AEMA independent source control failed")
    return config, paths, executors, {source["label"]: replay}


def component_domain(core: str, thiol: str, executor: dict) -> bool:
    """Conservative transferred domain; ester/aminothiol peripheries stay pending."""
    program = executor["program"]
    for role, smiles in (("amine_core", core), ("thiol_periphery", thiol)):
        if not _constraints(
            smiles,
            program.specification["terminal_constraints"][role],
            program.bounds.maximum_outcomes,
        )["pass"]:
            return False
        _, molecule = constitutional_molecule(smiles)
        allowed = {Chem.BondType.names[name] for name in executor["component_bond_types"][role]}
        if any(bond.GetBondType() not in allowed for bond in molecule.GetBonds()):
            return False
    return True


def replay_record(item: dict, structures: dict[str, str], executors: dict) -> dict:
    prepared = item["preparation"]
    if prepared.get("eligible_for_program_preparation") is not True:
        raise ComposeLipidError("Protected or unassigned target reached AEMA replay")
    if prepared.get("family") != FAMILY:
        raise ComposeLipidError("Wrong family reached AEMA replay")

    def pending(reason):
        return {"computed_consistency_pass": False, "disposition": reason}

    if prepared.get("construction_basis") != "v8_exact_family_task":
        return pending("missing_original_task_occupancy")
    supplied = {}
    for role, identity, quantity in prepared["component_instances"]:
        if identity not in structures:
            raise ComposeLipidError(f"Unresolved AEMA precursor: {identity}")
        if type(quantity) is not int or quantity < 1:
            raise ComposeLipidError("Invalid incorporated AEMA quantity")
        if role in supplied:
            return pending("unsupported_source_role_tuple")
        supplied[role] = (identity, quantity)
    if set(supplied) != {"AEMA", "amine_core", "thiol_periphery"}:
        return pending("unsupported_source_role_tuple")
    events = supplied["AEMA"][1]
    if (
        events not in executors
        or supplied["amine_core"][1] != 1
        or supplied["thiol_periphery"][1] != events
    ):
        return pending("outside_qualified_program_multiplicity")
    executor = executors[events]
    parts = {role: structures[identity] for role, (identity, _) in supplied.items()}
    if constitutional_molecule(parts["AEMA"])[0] != executor["fixed_AEMA"]:
        return pending("wrong_complete_AEMA_scaffold")
    if not component_domain(parts["amine_core"], parts["thiol_periphery"], executor):
        return pending("outside_qualified_source_component_domain")
    result = grouped_replay(item, structures, executor)
    result.update(
        program_id=executor["program"].specification["program_id"],
        source_occupancy=events,
        fixed_scaffold_identity_verified=True,
        evidence_basis="computed_transform_consistency",
        experimental_execution_admitted=False,
        training_admitted=False,
    )
    return result
