"""Zhou SI identity-specific extensions to the qualified ordered AEMA program."""

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
from forge.corpus import compose_lipid_aema_replay as parent
from forge.corpus.compose_lipid_grouped_replay import replay_record as grouped_replay


def load_contract(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != "forge.zhou_aema_replay_config.v1":
        raise ComposeLipidError("Zhou source schema differs")
    paths = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    if set(paths) != {"registry", "source_control", "parent_contract"}:
        raise ComposeLipidError("Zhou source inputs differ")
    base, _, prior, _ = parent.load_contract(repo, paths["parent_contract"])
    registry = json.loads(paths["registry"].read_text())
    source = json.loads(paths["source_control"].read_text())
    if (
        registry["source_assets"]["zhou_control_transcription"]
        != config["inputs"]["source_control"]
        or registry["source_component_extensions"] != source["additional_exact_components"]
        or registry["training_admitted"] is not False
        or registry["experimental_execution_admitted"] is not False
        or source["expected_formula"]
        != rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(source["expected_product"]))
    ):
        raise ComposeLipidError("Zhou independent source binding differs")
    for name, value in registry["source_assets"].items():
        resolve_pin(value, repo, label=name)
    for reaction in registry["reactions"]:
        for name, value in reaction["implementation"]["parent_registries"].items():
            resolve_pin(value, repo, label=name)
        resolve_pin(reaction["implementation"]["derivation"], repo, label="Zhou derivation")
    allowlists = {
        role: {constitutional_molecule(s)[0] for s in values.values()}
        for role, values in registry["source_component_extensions"].items()
    }
    if set(allowlists) != {"amine_core", "thiol_periphery"}:
        raise ComposeLipidError("Zhou source extension role domain differs")
    executors = {}
    for events, original in prior.items():
        program = RegistryGroupedProgram.from_registry(
            paths["registry"],
            program_id=f"source_zhou_aema_ordered_{events}_arms",
            expected_sha256=config["inputs"]["registry"]["sha256"],
            bounds=RepeatBounds(**base["search_bounds"]),
        )
        if program.quantities != original["program"].quantities:
            raise ComposeLipidError("Zhou source event count differs")
        executors[events] = {
            **original,
            "program": program,
            "parent": original,
            "allowlists": allowlists,
        }
    e = executors[6]
    if source["declared_quantities"] != e["program"].quantities or not component_domain(
        source["components"], e
    ):
        raise ComposeLipidError("Zhou control is outside its source component domain")
    replay = e["program"].replay(source["components"], source["expected_product"])
    if not replay["computed_consistency_pass"]:
        raise ComposeLipidError("Zhou independent aminothiol control failed")
    return config, paths, executors, {source["label"]: replay}


def component_domain(parts, executor):
    for role in ("amine_core", "thiol_periphery"):
        identity, molecule = constitutional_molecule(parts[role])
        if identity in executor["allowlists"][role]:
            continue
        original = executor["parent"]
        if not _constraints(
            identity,
            original["program"].specification["terminal_constraints"][role],
            original["program"].bounds.maximum_outcomes,
        )["pass"]:
            return False
        allowed = {Chem.BondType.names[s] for s in original["component_bond_types"][role]}
        if any(b.GetBondType() not in allowed for b in molecule.GetBonds()):
            return False
    return True


def replay_record(item, structures, executors):
    p = item["preparation"]
    if p.get("eligible_for_program_preparation") is not True:
        raise ComposeLipidError("Protected row reached Zhou source replay")
    if p.get("family") != parent.FAMILY:
        raise ComposeLipidError("Wrong source family reached Zhou replay")

    def pending(reason):
        return {"computed_consistency_pass": False, "disposition": reason}

    if p.get("construction_basis") != "v8_exact_family_task":
        return pending("missing_original_task_occupancy")
    supplied = {}
    for role, identity, quantity in p["component_instances"]:
        if identity not in structures or type(quantity) is not int or quantity < 1:
            raise ComposeLipidError("Invalid source component or quantity")
        if role in supplied:
            return pending("unsupported_source_role_tuple")
        supplied[role] = (identity, quantity)
    if set(supplied) != {"amine_core", "AEMA", "thiol_periphery"}:
        return pending("unsupported_source_role_tuple")
    events = supplied["AEMA"][1]
    if (
        events not in executors
        or supplied["amine_core"][1] != 1
        or supplied["thiol_periphery"][1] != events
    ):
        return pending("outside_qualified_program_multiplicity")
    e = executors[events]
    parts = {r: structures[i] for r, (i, _) in supplied.items()}
    if constitutional_molecule(parts["AEMA"])[0] != e["fixed_AEMA"]:
        return pending("wrong_complete_AEMA_scaffold")
    if not component_domain(parts, e):
        return pending("outside_qualified_source_component_domain")
    result = grouped_replay(item, structures, e)
    result.update(
        program_id=e["program"].specification["program_id"],
        source_occupancy=events,
        evidence_basis="computed_transform_consistency",
        training_admitted=False,
        experimental_execution_admitted=False,
    )
    return result
