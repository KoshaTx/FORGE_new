"""Source-bounded primary-amine acyclic Miao branch; no secondary-amine inference."""

from __future__ import annotations

import json
from functools import partial

from rdkit import Chem

from forge.corpus.compose_lipid_fixed_replay import load_contract as load_fixed_contract
from forge.corpus.compose_lipid_fixed_replay import qualify_controls


def component_domain(components, specification):
    if set(components) != set(specification):
        return False
    for role, smiles in components.items():
        molecule = Chem.MolFromSmiles(smiles)
        domain = specification[role]
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            return False
        if any(
            a.GetAtomicNum() not in domain["allowed_atomic_numbers"] for a in molecule.GetAtoms()
        ):
            return False
        if sum(a.GetFormalCharge() for a in molecule.GetAtoms()) != domain["formal_charge"]:
            return False
        if domain.get("acyclic") and molecule.GetRingInfo().NumRings():
            return False
        for element, count in domain.get("element_counts", {}).items():
            if sum(a.GetAtomicNum() == int(element) for a in molecule.GetAtoms()) != count:
                return False
    return True


def scoped_replay(base, domain, components, target):
    result = base(components, target)
    result["checks"]["source_component_domain"] = component_domain(components, domain)
    result["computed_consistency_pass"] = all(v is True for v in result["checks"].values())
    result["evidence_basis"] = "computed_transform_consistency"
    result["experimental_execution_admitted"] = False
    return result


def load_contract(repo, path):
    config, paths, executors, _ = load_fixed_contract(repo, path)
    registry = json.loads(paths["registry"].read_text())
    domain = registry["component_domain"]
    for executor in executors.values():
        executor["run"] = partial(scoped_replay, executor["run"], domain)
        executor["domain"] = domain
    controls = qualify_controls(json.loads(paths["transform_controls"].read_text()), executors)
    return config, paths, executors, controls
