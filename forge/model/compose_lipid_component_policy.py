"""Compile conservative proposal predicates from complete qualified executors.

These predicates only rank construction candidates. Complete inverse and unfiltered
forward replay, including component domains, remain the admission authority.
"""

from collections import Counter


def compile_component_policies(executor):
    if executor.get("full_source_contract") is not True:
        raise ValueError("Component policies require a complete source contract")
    mapping = executor["mapping"]
    result = {}

    def role_policy(role):
        source = mapping[role]
        return result.setdefault(source, {"role": source})

    program = executor.get("program")
    if hasattr(program, "specification"):
        spec = program.specification
        product_queries = spec.get("product_constraints", {}).get("required_queries", [])
        product_names = {q["name"] for q in product_queries}
        for role, constraints in spec.get("terminal_constraints", {}).items():
            p = role_policy(role)
            for key in ("allowed_atomic_numbers", "allow_aromatic_atoms"):
                if key in constraints:
                    p[key] = constraints[key]
            # Only queries explicitly required to survive into the product are
            # projected. Consumed SH, OH, epoxide and amine handles are not.
            retained = [
                q for q in constraints.get("required_queries", []) if q["name"] in product_names
            ]
            if retained:
                p["required_queries"] = retained
            # A single-stage program has an unambiguous per-precursor element
            # loss. Multi-stage accumulated-origin losses are deliberately omitted.
            if len(spec["stages"]) == 1 and "exact_element_counts" in constraints:
                adapter = program.adapters[0]
                reaction = adapter.reaction.forward
                template = reaction.GetReactantTemplate(adapter.roles.index(role))
                maps = {
                    a.GetAtomMapNum()
                    for a in reaction.GetProductTemplate(0).GetAtoms()
                    if a.GetAtomMapNum()
                }
                lost = Counter(
                    a.GetSymbol()
                    for a in template.GetAtoms()
                    if not a.GetAtomMapNum() or a.GetAtomMapNum() not in maps
                )
                counts = {
                    symbol: count - lost[symbol]
                    for symbol, count in constraints["exact_element_counts"].items()
                }
                if any(value < 0 for value in counts.values()):
                    raise ValueError("Source element projection is inconsistent")
                if "product_element_counts" in p and p["product_element_counts"] != counts:
                    raise ValueError("Repeated source roles have incompatible element policies")
                p["product_element_counts"] = counts
        # These source programs consume a head amine to an amide and explicitly
        # require a surviving basic candidate. Constrain its origin to the head
        # during proposal construction; full replay still decides admission.
        head_roles = {
            "source_staar_two_stage": "amine_head",
            "source_li_thiol_yne_then_amidation": "amine_head",
            "source_ester_thiol_yne_then_amidation": "amine_head",
        }
        if spec["program_id"] in head_roles:
            role_policy(head_roles[spec["program_id"]])["required_queries"] = product_queries
        # Forbid source-disallowed functionality wholly outside reaction cores.
        # Accumulator-only roles are not terminals and cannot be projected here.
        for adapter in getattr(program, "adapters", ()):
            for role in adapter.reaction.definition.reactant_roles:
                if role.name not in mapping:
                    continue
                for smarts in role.forbidden_smarts:
                    query = dict(smarts=smarts, maximum_matches=0, exclude_core=True)
                    queries = role_policy(role.name).setdefault("remaining_handles", [])
                    if query not in queries:
                        queries.append(query)
    reaction = executor.get("reaction", {})
    for query in reaction.get("retained_queries", []):
        role_policy(query["role"]).setdefault("required_queries", []).append(
            {**query, "exclude_core": True}
        )
    contract = executor.get("source_contract", {})
    if "retained_amine_policy" in contract:
        role_policy(contract["retained_amine_role"])["retained_amine_policy"] = contract[
            "retained_amine_policy"
        ]
    if reaction.get("reaction_id") == "source_miao_drawn_alpha_isocyanoacetates_cyclization":
        role = next(r for r in reaction["reactant_roles"] if r["name"] == "coupled_ketone")
        if role["allowed_site_multiplicity"] != [1]:
            raise ValueError("Miao consumed-ketone projection requires exactly one source site")
        role_policy(role["name"])["remaining_handles"] = [
            {"smarts": role["required_handle_smarts"], "maximum_matches": 0}
        ]
    for role, queries in executor.get("proposal_retained_queries", {}).items():
        role_policy(role).setdefault("required_queries", []).extend(queries)
    return list(result.values())
