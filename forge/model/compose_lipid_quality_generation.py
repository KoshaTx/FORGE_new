"""Project retained source-component domains into bounded construction predicates."""

import copy

from rdkit import Chem


def with_retained_component_domain(executor, policies):
    """Add an executor's ester-thiol domain when mapped heavy atoms are retained.

    Element counts and required-query definitions come from the executor's pinned
    registry. The final independent source-domain check remains authoritative:
    this projection does not claim to encode every domain restriction.
    """
    domain = executor.get("ester_thiol_domain")
    roles = None
    if domain is None and executor.get("domain_kind") == "ester_thiol":
        domain, roles = executor["domain"], executor["domain_roles"]
    if domain is None:
        return copy.deepcopy(policies)
    specification = executor["program"].specification
    constraints = domain["constraints"]
    if roles is None:
        # Resolve the declared source role by its registry element contract,
        # rather than retyping a role name or assuming a program-specific index.
        roles = [
            role
            for role, value in specification["terminal_constraints"].items()
            if value.get("allowed_atomic_numbers") == constraints["allowed_atomic_numbers"]
            and value.get("exact_element_counts")
            and all(
                constraints["exact_element_counts"].get(symbol) == count
                for symbol, count in value["exact_element_counts"].items()
            )
        ]
        if len(roles) != 1:
            raise ValueError("Source component domain has an ambiguous terminal role")
    output = copy.deepcopy(policies)
    for role in roles:
        reactive_elements = set()
        used = False
        for stage, adapter in zip(
            specification["stages"], executor["program"].adapters, strict=True
        ):
            if role not in stage.get("added_roles", ()):
                continue
            used = True
            template = adapter.reaction.forward.GetReactantTemplate(adapter.roles.index(role))
            products = {
                a.GetAtomMapNum(): a.GetAtomicNum()
                for mol in adapter.reaction.forward.GetProducts()
                for a in mol.GetAtoms()
                if a.GetAtomMapNum()
            }
            for atom in template.GetAtoms():
                if atom.GetAtomicNum() <= 1:
                    raise ValueError("Domain projection requires concrete mapped heavy atoms")
                if products.get(atom.GetAtomMapNum()) != atom.GetAtomicNum():
                    raise ValueError("Domain projection would assume changed heavy atoms survive")
                reactive_elements.add(atom.GetAtomicNum())
        if not used:
            raise ValueError("Domain role is not an explicitly added terminal component")
        queries = []
        for query in constraints.get("required_queries", []):
            molecule = Chem.MolFromSmarts(query["smarts"])
            if molecule is None or any(
                atom.GetAtomicNum() <= 0 or atom.GetAtomicNum() in reactive_elements
                for atom in molecule.GetAtoms()
            ):
                raise ValueError("Required query is not disjoint from the reacted component atoms")
            queries.append({**query, "whole_component": True})
        source_role = executor["mapping"][role]
        policy = next((p for p in output if p["role"] == source_role), None)
        if policy is None:
            policy = {"role": source_role}
            output.append(policy)
        policy["product_element_counts"] = dict(constraints["exact_element_counts"])
        for key in ("allowed_atomic_numbers", "allow_aromatic_atoms"):
            if key in constraints:
                policy[key] = copy.deepcopy(constraints[key])
        existing = policy.setdefault("required_queries", [])
        existing.extend(query for query in queries if query not in existing)
    return output
