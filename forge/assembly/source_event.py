"""Qualify one source-declared event using a frozen transform and explicit site witnesses."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from rdkit import Chem, rdBase

from forge.assembly.families import (
    LibraryAssemblyError,
    RegistryAssemblyAdapter,
    constitutional_molecule,
)
from forge.assembly.repeated_components import element_inventory


def _properties(atom: Chem.Atom) -> dict[str, Any]:
    return {
        "atomic_number": atom.GetAtomicNum(),
        "formal_charge": atom.GetFormalCharge(),
        "total_hydrogens": atom.GetTotalNumHs(),
    }


def check_source_event(
    adapter: RegistryAssemblyAdapter,
    components: Mapping[str, str],
    target: str,
    *,
    site_contract: list[dict[str, Any]],
    role_queries: Mapping[str, Chem.Mol],
    maximum_outcomes: int = 256,
) -> dict[str, Any]:
    """Require a unique frozen-transform replay and a source-permitted attachment witness.

    Additional source predicates can only restrict the original registry. They do not remove
    competing products to manufacture uniqueness or relax its reactive-site multiplicities.
    """
    if (
        not site_contract
        or set(components) != set(adapter.roles)
        or set(role_queries) != set(adapter.roles)
        or any(query is None for query in role_queries.values())
        or {item["role"] for item in site_contract} != set(adapter.roles)
        or len({item["atom_map"] for item in site_contract}) != len(site_contract)
    ):
        raise LibraryAssemblyError("source event requires explicit role/site contracts")
    canonical = {role: constitutional_molecule(smi)[0] for role, smi in components.items()}
    molecules = {role: Chem.MolFromSmiles(smi) for role, smi in canonical.items()}
    target = constitutional_molecule(target)[0]
    for item in site_contract:
        if (
            item["role"] not in adapter.roles
            or not item["properties"]
            or set(item["properties"]) - {"atomic_number", "formal_charge", "total_hydrogens"}
        ):
            raise LibraryAssemblyError("unsupported source reactive-atom properties")
        template = adapter.reaction.forward.GetReactantTemplate(adapter.roles.index(item["role"]))
        if sum(a.GetAtomMapNum() == item["atom_map"] for a in template.GetAtoms()) != 1:
            raise LibraryAssemblyError("source map must resolve once in its declared reactant role")
    forward = adapter.forward_products(canonical, maximum_outcomes=maximum_outcomes)
    if forward.saturated:
        raise LibraryAssemblyError("source event forward enumeration saturated")
    inverse = adapter.decompose(target, maximum_outcomes=maximum_outcomes)
    allowed = {
        role: {match[0] for match in molecule.GetSubstructMatches(role_queries[role])}
        for role, molecule in molecules.items()
        if role in role_queries
    }
    witnesses = set()
    if all(a.qualified for a in adapter.assess_roles(canonical)):
        with rdBase.BlockLogs():
            raw = adapter.reaction.forward.RunReactants(
                tuple(molecules[r] for r in adapter.roles), maxProducts=maximum_outcomes
            )
        if len(raw) >= maximum_outcomes:
            raise LibraryAssemblyError("source event witness enumeration saturated")
        for outcome in raw:
            if len(outcome) != 1:
                continue
            product = Chem.Mol(outcome[0])
            try:
                with rdBase.BlockLogs():
                    Chem.SanitizeMol(product)
                identity = constitutional_molecule(Chem.MolToSmiles(product))[0]
            except (ValueError, RuntimeError):
                continue
            if identity != target:
                continue
            witness = []
            for item in site_contract:
                mapped = [
                    a
                    for a in product.GetAtoms()
                    if a.HasProp("old_mapno") and a.GetIntProp("old_mapno") == item["atom_map"]
                ]
                if len(mapped) != 1 or not mapped[0].HasProp("react_atom_idx"):
                    break
                atom_index = mapped[0].GetIntProp("react_atom_idx")
                role = item["role"]
                if mapped[0].HasProp("react_idx") and mapped[0].GetIntProp(
                    "react_idx"
                ) != adapter.roles.index(role):
                    raise LibraryAssemblyError("reactive-atom role provenance changed")
                atom = molecules[role].GetAtomWithIdx(atom_index)
                if any(
                    _properties(atom)[name] != value for name, value in item["properties"].items()
                ):
                    break
                if role in allowed and atom_index not in allowed[role]:
                    break
                witness.append((role, item["atom_map"], atom_index))
            else:
                witnesses.add(tuple(witness))
    left: Counter[str] = Counter()
    for smiles in canonical.values():
        # Counter addition drops nonpositive totals, which would corrupt charge accounting.
        for key, count in element_inventory(smiles).items():
            left[key] += count
    right = element_inventory(target)
    balanced = all(left[key] == right[key] for key in left.keys() | right.keys())
    checks = {
        "unique_unfiltered_forward_exact": forward.products == (target,),
        "unique_unfiltered_inverse_exact": len(inverse) == 1
        and dict(inverse[0].components) == canonical,
        "source_reactive_site_witness": bool(witnesses),
        "full_element_hydrogen_charge_balance": balanced,
    }
    return {
        "checks": checks,
        "computed_consistency_pass": all(checks.values()),
        "site_witnesses": [
            [
                {"role": role, "atom_map": number, "canonical_reactant_atom_index": index}
                for role, number, index in witness
            ]
            for witness in sorted(witnesses)
        ],
        "forward_products": list(forward.products),
        "inverse_candidate_count": len(inverse),
        "experimental_execution_admitted": False,
    }
