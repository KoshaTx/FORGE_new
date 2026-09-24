"""Exact single-event checks with an explicit registry-owned byproduct inventory."""

from collections import Counter
from collections.abc import Mapping

from forge.assembly.families import LibraryAssemblyError
from forge.assembly.repeated_components import element_inventory
from forge.assembly.source_event import check_source_event


def check_condensation_event(adapter, components, target, *, net_byproducts, **kwargs) -> dict:
    """Keep every source/site/uniqueness check; account explicitly for eliminated atoms."""
    if (
        not isinstance(net_byproducts, Mapping)
        or not net_byproducts
        or any(not isinstance(key, str) or not key for key in net_byproducts)
        or any(type(count) is not int for count in net_byproducts.values())
        or any(count < 0 for key, count in net_byproducts.items() if key != "formal_charge")
    ):
        raise LibraryAssemblyError("condensation requires an explicit valid byproduct inventory")
    result = check_source_event(adapter, components, target, **kwargs)
    left = Counter()
    for smiles in components.values():
        for element, count in element_inventory(smiles).items():
            left[element] += count
    right = Counter(element_inventory(target))
    for element, count in net_byproducts.items():
        right[element] += count
    # Never use Counter subtraction: it drops negative values, including charge deficits.
    result["checks"]["full_element_hydrogen_charge_balance"] = all(
        left[key] == right[key] for key in left.keys() | right.keys()
    )
    result["net_byproducts"] = dict(net_byproducts)
    result["computed_consistency_pass"] = all(result["checks"].values())
    return result
