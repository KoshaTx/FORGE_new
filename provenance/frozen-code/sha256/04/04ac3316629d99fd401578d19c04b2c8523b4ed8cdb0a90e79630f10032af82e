"""Source-role retention predicates for a complete, unfiltered condensation event."""

from __future__ import annotations

from rdkit import Chem, rdBase

from forge.assembly.condensation_event import check_condensation_event
from forge.assembly.families import LibraryAssemblyError, constitutional_molecule


def check_retained_condensation(
    adapter, components, target, *, retained_queries, maximum_outcomes=256, **kwargs
):
    """Require surviving source-role handles without using them to prune outcomes.

    The predicate is evaluated on atom provenance in the raw product, so a basic
    atom supplied by another role cannot rescue the consumed headgroup nitrogen.
    """
    result = check_condensation_event(
        adapter, components, target, maximum_outcomes=maximum_outcomes, **kwargs
    )
    constraints = []
    for item in retained_queries:
        query = Chem.MolFromSmarts(item["smarts"])
        if (
            item["role"] not in adapter.roles
            or query is None
            or type(item["minimum_matches"]) is not int
            or item["minimum_matches"] < 1
            or any(item["name"] == prior[0]["name"] for prior in constraints)
        ):
            raise LibraryAssemblyError("Invalid source-role retention predicate")
        constraints.append((item, query))
    if not constraints:
        raise LibraryAssemblyError("Source-role retention predicates are required")
    canonical = {r: constitutional_molecule(s)[0] for r, s in components.items()}
    molecules = {r: Chem.MolFromSmiles(s) for r, s in canonical.items()}
    target = constitutional_molecule(target)[0]
    retained = []
    if all(a.qualified for a in adapter.assess_roles(canonical)):
        with rdBase.BlockLogs():
            outcomes = adapter.reaction.forward.RunReactants(
                tuple(molecules[r] for r in adapter.roles), maxProducts=maximum_outcomes
            )
        if len(outcomes) >= maximum_outcomes:
            raise LibraryAssemblyError("Source-role retention enumeration saturated")
        for outcome in outcomes:
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
            evidence = {}
            for item, query in constraints:
                role = item["role"]
                reactive_maps = {
                    site["atom_map"] for site in kwargs["site_contract"] if site["role"] == role
                }
                consumed = {
                    atom.GetIntProp("react_atom_idx")
                    for atom in product.GetAtoms()
                    if atom.HasProp("old_mapno")
                    and atom.GetIntProp("old_mapno") in reactive_maps
                    and atom.HasProp("react_atom_idx")
                }
                before = {m[0] for m in molecules[role].GetSubstructMatches(query)} - consumed
                origins = set()
                for match in product.GetSubstructMatches(query):
                    atom = product.GetAtomWithIdx(match[0])
                    if (
                        atom.HasProp("react_idx")
                        and atom.GetIntProp("react_idx") == adapter.roles.index(role)
                        and atom.HasProp("react_atom_idx")
                        and atom.GetIntProp("react_atom_idx") in before
                    ):
                        origins.add(atom.GetIntProp("react_atom_idx"))
                evidence[item["name"]] = sorted(origins)
            if all(
                len(evidence[item["name"]]) >= item["minimum_matches"] for item, _ in constraints
            ):
                retained.append(evidence)
    # Keep the original uniqueness checks: a target-matching retention witness
    # cannot turn a nonunique reaction into an exact reconstruction.
    result["checks"]["distinct_source_role_handles_retained"] = bool(retained)
    result["retained_source_role_witnesses"] = retained
    result["computed_consistency_pass"] = all(result["checks"].values())
    return result
