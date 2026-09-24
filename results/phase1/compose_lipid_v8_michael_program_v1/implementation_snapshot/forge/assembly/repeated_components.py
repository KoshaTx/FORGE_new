"""Exhaustive bounded replay of a declared fixed-component repeated assembly.

This checks an ID-conditioned constitutional program. It does not infer experimental
selectivity, the order of commuting events, or uniqueness among other precursor tuples.
Chemistry and role policies belong to the pinned registry, not this executor.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from forge.assembly.families import (
    LibraryAssemblyError,
    RegistryAssemblyAdapter,
    constitutional_molecule,
)


@dataclass(frozen=True)
class RepeatBounds:
    maximum_events: int = 32
    maximum_outcomes: int = 256
    maximum_states: int = 4096
    maximum_transitions: int = 16384

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value < 1:
                raise LibraryAssemblyError(f"{name} must be a positive integer")


def element_inventory(smiles: str) -> Counter:
    """Include implicit/explicit hydrogen and net formal charge in the balance."""
    _, mol = constitutional_molecule(smiles)
    counts = Counter()
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] += 1
        counts["H"] += atom.GetTotalNumHs()
        counts["formal_charge"] += atom.GetFormalCharge()
    return counts


def replay_repeated_components(
    adapter: RegistryAssemblyAdapter,
    components: Mapping[str, str],
    product: str,
    *,
    accumulator_role: str,
    events: int,
    byproducts_per_event: Mapping[str, int],
    bounds: RepeatBounds = RepeatBounds(),
) -> dict:
    """Enumerate every forward site; condition reverse replay on declared side reagents.

    No target-directed forward pruning is permitted. Reaching any enumeration bound
    invalidates the entire check, including a target found before that bound.
    """
    if accumulator_role not in adapter.roles or set(components) != set(adapter.roles):
        raise LibraryAssemblyError("complete registry roles and an accumulator are required")
    if type(events) is not int or not 1 <= events <= bounds.maximum_events:
        raise LibraryAssemblyError("declared event count is outside the explicit search bound")
    if not byproducts_per_event or any(
        not isinstance(key, str) or type(value) is not int or value < 0
        for key, value in byproducts_per_event.items()
    ):
        raise LibraryAssemblyError("explicit nonnegative byproduct inventory is required")
    canonical = {role: constitutional_molecule(smi)[0] for role, smi in components.items()}
    target = constitutional_molecule(product)[0]
    start = canonical[accumulator_role]
    side = {role: value for role, value in canonical.items() if role != accumulator_role}
    reasons: list[str] = []
    forward_layers = [[start]]
    forward_edges: list[list[str | int]] = []
    reverse_layers = [[target]]
    reverse_edges: list[list[str | int]] = []

    def over_budget(layers, edges):
        return (
            sum(map(len, layers)) > bounds.maximum_states or len(edges) > bounds.maximum_transitions
        )

    for depth in range(events):
        next_layer = set()
        for state in forward_layers[-1]:
            result = adapter.forward_products(
                {**side, accumulator_role: state}, maximum_outcomes=bounds.maximum_outcomes
            )
            if result.saturated:
                reasons.append("forward_outcome_bound")
                break
            for child in result.products:
                next_layer.add(child)
                forward_edges.append([depth + 1, state, child])
            if over_budget(forward_layers + [next_layer], forward_edges):
                reasons.append("forward_state_or_transition_bound")
                break
        forward_layers.append(sorted(next_layer))
        if reasons:
            break

    if not reasons:
        for depth in range(events):
            next_layer = set()
            for state in reverse_layers[-1]:
                try:
                    inverse = adapter.decompose(state, maximum_outcomes=bounds.maximum_outcomes)
                except LibraryAssemblyError as exc:
                    if "saturated" not in str(exc):
                        raise
                    reasons.append("reverse_outcome_bound")
                    break
                for candidate in inverse:
                    values = dict(candidate.components)
                    if any(values[role] != value for role, value in side.items()):
                        continue
                    parent = values[accumulator_role]
                    next_layer.add(parent)
                    reverse_edges.append([depth + 1, state, parent])
                if over_budget(reverse_layers + [next_layer], reverse_edges):
                    reasons.append("reverse_state_or_transition_bound")
                    break
            reverse_layers.append(sorted(next_layer))
            if reasons:
                break

    left = element_inventory(start)
    for value in side.values():
        for atom, count in element_inventory(value).items():
            left[atom] += count * events
    right = element_inventory(target)
    for atom, count in byproducts_per_event.items():
        right[atom] += count * events
    balanced = all(left[atom] == right[atom] for atom in left.keys() | right.keys())
    checks = {
        "complete_search": not reasons,
        "declared_event_count_replayed": (
            len(forward_layers) == events + 1 and all(forward_layers) and not reasons
        ),
        "unique_forward_exact": forward_layers[-1] == [target] and not reasons,
        "unique_declared_side_inverse": reverse_layers[-1] == [start] and not reasons,
        "full_element_hydrogen_charge_balance": balanced,
    }
    return {
        "checks": checks,
        "computed_consistency_pass": all(checks.values()),
        "bound_reasons": reasons,
        "forward_layers": forward_layers,
        "reverse_layers": reverse_layers,
        "forward_edges": forward_edges,
        "reverse_edges": reverse_edges,
        "balance": {
            "reactants": dict(sorted(left.items())),
            "product_and_byproducts": dict(sorted(right.items())),
        },
        "inverse_scope": "conditioned_on_declared_fixed_side_components",
        "experimental_selectivity_qualified": False,
        "physical_event_order_qualified": False,
    }
