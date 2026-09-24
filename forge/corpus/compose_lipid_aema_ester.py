"""Source-bounded ester-thiol extension of the ordered AEMA transform objective."""

from __future__ import annotations

import json
from pathlib import Path

from rdkit.Chem import rdMolDescriptors

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import RegistryAssemblyAdapter, constitutional_molecule
from forge.assembly.grouped_program import RegistryGroupedProgram
from forge.assembly.repeated_components import RepeatBounds, element_inventory
from forge.core.hashing import resolve_pin
from forge.corpus import compose_lipid_aema_replay as base
from forge.corpus.compose_lipid_ester_thiol import component_domain

FAMILY = base.FAMILY


def load_contract(repo: Path, config_path: Path) -> tuple[dict, dict, dict, dict]:
    config = json.loads(config_path.read_text())
    if (
        config.get("schema_version") != "forge.aema_ester_thiol_config.v1"
        or config.get("training_admitted") is not False
        or config.get("experimental_execution_admitted") is not False
        or config.get("training_calls") != 0
        or set(config.get("inputs", {})) != {"base_config", "registry", "source_control"}
    ):
        raise ComposeLipidError("AEMA ester-thiol contract differs")
    paths = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    parent_config, _, parent_executors, parent_controls = base.load_contract(
        repo, paths["base_config"]
    )
    registry = json.loads(paths["registry"].read_text())
    source = json.loads(paths["source_control"].read_text())
    for name, asset in {**registry["source_assets"], **registry["derivation"]}.items():
        resolve_pin(asset, repo, label=name)
    if (
        registry["derivation"]["parent"] != parent_config["inputs"]["registry"]
        or registry["source_assets"]["bowman_control"] != config["inputs"]["source_control"]
        or config["search_bounds"] != parent_config["search_bounds"]
        or registry["training_admitted"] is not False
        or registry["experimental_execution_admitted"] is not False
        or source["generated_disposition"] != "admit_transform_consistency"
    ):
        raise ComposeLipidError("AEMA extension changed the source evidence contract")
    domain = registry["ester_thiol_domain"]
    domain_registry = json.loads(
        resolve_pin(registry["derivation"]["domain_definition"], repo, label="domain").read_text()
    )
    if domain != domain_registry["ester_thiol_domain"]:
        raise ComposeLipidError("AEMA ester-thiol transfer domain differs")
    executors = {}
    for spec in registry["grouped_programs"]:
        events = spec["stages"][0]["events"]
        program = RegistryGroupedProgram.from_registry(
            paths["registry"],
            program_id=spec["program_id"],
            expected_sha256=config["inputs"]["registry"]["sha256"],
            bounds=RepeatBounds(**config["search_bounds"]),
        )
        parent = parent_executors[events]
        if events in executors or program.quantities != parent["program"].quantities:
            raise ComposeLipidError("AEMA extension changed source quantities")
        executors[events] = {
            **parent,
            "program": program,
            "component_bond_types": registry["component_bond_types"],
            "ester_thiol_domain": domain,
        }
    if set(executors) != set(parent_executors):
        raise ComposeLipidError("AEMA extension changed source occupancies")
    adapter = RegistryAssemblyAdapter.from_registry(
        paths["registry"],
        reaction_id=registry["grouped_programs"][0]["stages"][1]["reaction_id"],
        expected_sha256=config["inputs"]["registry"]["sha256"],
    )
    expected, mol = constitutional_molecule(source["expected_product"])
    forward = adapter.forward_products(source["components"])
    inventory = {}
    for smiles in source["components"].values():
        for element, number in element_inventory(smiles).items():
            inventory[element] = inventory.get(element, 0) + number
    if (
        forward.saturated
        or sorted(forward.products) != [expected]
        or rdMolDescriptors.CalcMolFormula(mol) != source["expected_formula"]
        or inventory != dict(element_inventory(expected))
        or not component_domain(source["components"]["thiol_periphery"], domain)
    ):
        raise ComposeLipidError("Independent Bowman methacrylate/ester-thiol control failed")
    return (
        config,
        paths,
        executors,
        {
            "Lee_ordered_AEMA": parent_controls,
            "Bowman_ester_thiol_methacrylate": {
                "products": [expected],
                "inventory": inventory,
                "source_control_pass": True,
                "generated_evidence_basis": "computed_transform_consistency",
            },
        },
    )


def replay_record(item: dict, structures: dict[str, str], executors: dict) -> dict:
    # The original guards reject protected records before reading any structures.
    prepared = item["preparation"]
    if prepared.get("eligible_for_program_preparation") is not True:
        raise ComposeLipidError("Protected or unassigned target reached AEMA replay")
    if prepared.get("family") != FAMILY:
        raise ComposeLipidError("Wrong family reached AEMA replay")
    domain = executors[1]["ester_thiol_domain"]
    for role, identity, _ in prepared["component_instances"]:
        if role == "thiol_periphery":
            if identity not in structures:
                raise ComposeLipidError(f"Unresolved AEMA precursor: {identity}")
            if not component_domain(structures[identity], domain):
                return {
                    "computed_consistency_pass": False,
                    "disposition": "outside_bounded_ester_thiol_domain",
                }
    result = base.replay_record(item, structures, executors)
    if "checks" in result:
        result["checks"]["bounded_ester_thiol_domain"] = True
    return result
