"""Bounded registry programs over recovered generated precursors for context proposals."""

from dataclasses import dataclass

from forge.assembly.families import LibraryAssemblyError, constitutional_molecule


@dataclass(frozen=True)
class ProgramExpansion:
    status: str
    paths: tuple[tuple[str, ...], ...]
    expansions: int
    reason: str | None = None


def expand_generated_program(adapter, components, *, depth, accumulator_role, limits):
    """Retain every exact path at requested depth; any incomplete layer abstains in full."""
    if type(depth) is not int or not 1 <= depth <= limits.maximum_steps:
        raise LibraryAssemblyError("expansion depth exceeds the declared program")
    if set(components) != set(adapter.roles):
        raise LibraryAssemblyError("expansion components differ from registry roles")
    if accumulator_role is None:
        if depth != 1:
            raise LibraryAssemblyError("fixed-arity program requires one step")
    elif len(adapter.roles) != 2 or accumulator_role not in adapter.roles:
        raise LibraryAssemblyError("invalid expansion accumulator role")
    canonical = {r: constitutional_molecule(s)[0] for r, s in components.items()}
    paths, expansions = {()}, 0
    for _ in range(depth):
        following = set()
        for path in sorted(paths):
            if expansions >= limits.maximum_expansions:
                return ProgramExpansion("abstain", (), expansions, "expansion_limit")
            reactants = dict(canonical)
            if path:
                reactants[accumulator_role] = path[-1]
            outcome = adapter.forward_products(reactants, maximum_outcomes=limits.maximum_outcomes)
            expansions += 1
            if outcome.saturated:
                return ProgramExpansion("abstain", (), expansions, "outcome_limit")
            following.update((*path, product) for product in outcome.products)
            if len(following) > limits.maximum_states:
                return ProgramExpansion("abstain", (), expansions, "state_limit")
        if not following:
            return ProgramExpansion("no_qualified_program", (), expansions)
        paths = following
    return ProgramExpansion("complete", tuple(sorted(paths)), expansions)
