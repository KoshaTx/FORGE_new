"""Bounded inverse/forward witnesses for generated products, without source components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from forge.assembly.families import (
    LibraryAssemblyError,
    RegistryAssemblyAdapter,
    constitutional_molecule,
)
from forge.assembly.library_programs import LibraryProgramLimits, replay_library_program


@dataclass(frozen=True)
class GeneratedProgramCheck:
    status: str
    programs: tuple[dict[str, Any], ...]
    expansions: int
    reason: str | None = None

    @property
    def exact(self) -> bool:
        return self.status == "exact_computed_program"


def check_generated_program(
    adapter: RegistryAssemblyAdapter,
    product_smiles: str,
    *,
    depth: int,
    accumulator_role: str | None,
    limits: LibraryProgramLimits,
) -> GeneratedProgramCheck:
    """Exhaust the requested depth and forward-replay every returned witness.

    Repeated programs use one unchanged co-reactant identity at every step, matching the
    current library dataset contract. Distinct witnesses remain distinct; none is asserted to
    be the experimental route. Any incomplete search abstains, including after finding a hit.
    """
    if type(depth) is not int or not 1 <= depth <= limits.maximum_steps:
        raise LibraryAssemblyError("requested depth is outside the declared program support")
    if accumulator_role is None:
        if depth != 1:
            raise LibraryAssemblyError("fixed-arity assembly requires depth one")
    elif len(adapter.roles) != 2 or accumulator_role not in adapter.roles:
        raise LibraryAssemblyError("invalid repeated-program accumulator role")
    target, _ = constitutional_molecule(product_smiles)
    # Each frontier entry retains its full path, so convergent alternatives are not erased.
    frontier: set[tuple[str, tuple[tuple[str, str], ...], tuple[str, ...]]] = {(target, (), ())}
    expansions = 0
    witnesses: dict[tuple[Any, ...], dict[str, Any]] = {}
    try:
        for level in range(depth):
            following = set()
            for current, fixed_reagents, reverse_path in sorted(frontier):
                if expansions >= limits.maximum_expansions:
                    return GeneratedProgramCheck("abstain", (), expansions, "expansion_limit")
                expansions += 1
                decompositions = adapter.decompose(
                    current, maximum_outcomes=limits.maximum_outcomes
                )
                for decomposition in decompositions:
                    components = dict(decomposition.components)
                    reagents = tuple(
                        (role, components[role])
                        for role in adapter.roles
                        if role != accumulator_role
                    )
                    if fixed_reagents and reagents != fixed_reagents:
                        continue
                    path = (*reverse_path, current)
                    if level + 1 == depth:
                        intermediates = tuple(reversed(path))
                        if not replay_library_program(
                            adapter,
                            components,
                            intermediates,
                            accumulator_role=accumulator_role,
                            maximum_outcomes=limits.maximum_outcomes,
                        ):
                            raise LibraryAssemblyError("inverse witness failed independent replay")
                        key = (tuple(sorted(components.items())), intermediates)
                        witnesses[key] = {
                            "components": components,
                            "intermediate_products": list(intermediates),
                            "exact_forward_replay": True,
                        }
                        if len(witnesses) > limits.maximum_states:
                            return GeneratedProgramCheck("abstain", (), expansions, "witness_limit")
                    else:
                        assert accumulator_role is not None  # Depth > 1 was validated above.
                        following.add((components[accumulator_role], reagents, path))
                        if len(following) > limits.maximum_states:
                            return GeneratedProgramCheck("abstain", (), expansions, "state_limit")
            frontier = following
    except LibraryAssemblyError as error:
        if "saturated" not in str(error):
            raise
        return GeneratedProgramCheck("abstain", (), expansions, str(error))
    return GeneratedProgramCheck(
        "exact_computed_program" if witnesses else "no_exact_program",
        tuple(witnesses[key] for key in sorted(witnesses)),
        expansions,
    )
