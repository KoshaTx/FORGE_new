"""Bounded ester-thiol source domain; generated recipes retain computed-only labels."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.assembly.families import RegistryAssemblyAdapter, constitutional_molecule
from forge.assembly.repeated_components import RepeatBounds, element_inventory
from forge.assembly.staged_program import RegistryStagedProgram, _constraints
from forge.core.hashing import resolve_pin


def component_domain(smiles: str, specification: dict) -> bool:
    """Check the declared domain without a target, component ID, or source frequency."""
    _, mol = constitutional_molecule(smiles)
    if not _constraints(smiles, specification["constraints"], 256)["pass"]:
        return False
    if specification["acyclic"] and mol.GetRingInfo().NumRings():
        return False
    query = Chem.MolFromSmarts(specification["ester_query"])
    matches = mol.GetSubstructMatches(query)
    if len(matches) != 1:
        return False
    ester_bonds = {
        mol.GetBondBetweenAtoms(match[b.GetBeginAtomIdx()], match[b.GetEndAtomIdx()]).GetIdx()
        for match in matches
        for b in query.GetBonds()
    }
    single = Chem.BondType.names[specification["other_bond_type"]]
    return all(b.GetBondType() == single or b.GetIdx() in ester_bonds for b in mol.GetBonds())


def load_contract(repo: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    if config["schema_version"] != "forge.ester_thiol_source_config.v1":
        raise ValueError("Ester-thiol source schema differs")
    paths = {k: resolve_pin(v, repo, label=k) for k, v in config["inputs"].items()}
    registry = json.loads(paths["registry"].read_text())
    source = json.loads(paths["source_control"].read_text())
    for name, asset in registry["source_assets"].items():
        resolve_pin(asset, repo, label=name)
    if registry["source_assets"]["control"] != config["inputs"]["source_control"]:
        raise ValueError("Ester-thiol source transcription differs")
    program = RegistryStagedProgram.from_registry(
        paths["registry"],
        program_id=config["program_id"],
        expected_sha256=config["inputs"]["registry"]["sha256"],
        bounds=RepeatBounds(**config["search_bounds"]),
    )
    adapter = RegistryAssemblyAdapter.from_registry(
        paths["registry"],
        reaction_id=program.specification["stages"][0]["reaction_id"],
        expected_sha256=config["inputs"]["registry"]["sha256"],
    )
    control = source["source_control"]
    expected, mol = constitutional_molecule(control["expected_product"])
    forward = adapter.forward_products(control["components"])
    # The source 1-octyne is a reaction control, not the full acid/amidation program.
    products = sorted(forward.products)
    inventory = {}
    for s in control["components"].values():
        for k, n in element_inventory(s).items():
            inventory[k] = inventory.get(k, 0) + n
    if (
        forward.saturated
        or products != [expected]
        or rdMolDescriptors.CalcMolFormula(mol) != control["expected_formula"]
        or inventory != dict(element_inventory(expected))
        or not component_domain(
            control["components"]["thiol_first"], registry["ester_thiol_domain"]
        )
    ):
        raise ValueError("Independent Minozzi ester-thiol control failed")
    return (
        config,
        program,
        registry["ester_thiol_domain"],
        {"products": products, "inventory": inventory},
    )


def qualify_saved_replay(program, domain, replay, components):
    """Reassess terminal scope only; preserve complete authenticated reaction searches."""
    if set(components) != set(program.roles):
        raise ValueError("Saved ester-thiol roles differ")
    if sorted(k for k, v in replay["checks"].items() if not v) != ["terminal_constraints"]:
        raise ValueError("Saved replay has unresolved checks beyond terminal scope")
    result = copy.deepcopy(replay)
    result["previous_terminal_constraints"] = result["terminal_constraints"]
    result["terminal_constraints"] = {
        role: _constraints(components[role], spec, program.bounds.maximum_outcomes)
        for role, spec in program.specification["terminal_constraints"].items()
    }
    result["checks"]["terminal_constraints"] = all(
        c["pass"] for c in result["terminal_constraints"].values()
    )
    result["checks"]["bounded_ester_thiol_domain"] = all(
        component_domain(components[role], domain) for role in ("thiol_first", "thiol_second")
    )
    result["computed_consistency_pass"] = all(v is True for v in result["checks"].values())
    result["disposition"] = (
        "exact_computed_reconstruction"
        if result["computed_consistency_pass"]
        else "unresolved_computed_reconstruction"
    )
    result["evidence_basis"] = "computed_transform_consistency"
    result["experimental_execution_admitted"] = False
    return result
