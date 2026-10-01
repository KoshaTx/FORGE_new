"""Infer complete repeated-component tuples at an explicitly declared event count.

Only one identity per side role is allowed across events. Every inverse branch is
retained; a later forward replay must independently establish product uniqueness.
"""

from __future__ import annotations

from forge.assembly.families import (
    LibraryAssemblyError,
    RegistryAssemblyAdapter,
    constitutional_molecule,
)
from forge.assembly.repeated_components import RepeatBounds


def infer_repeated_components(
    adapter: RegistryAssemblyAdapter,
    product: str,
    *,
    accumulator_role: str,
    events: int,
    bounds: RepeatBounds = RepeatBounds(),
) -> dict:
    """Return all complete tuples, or no admissible result when the search is truncated.

    Counts and search bounds describe computation only. Neither a deterministic path
    nor a unique constitutional tuple establishes experimental order or selectivity.
    """
    if accumulator_role not in adapter.roles or len(adapter.roles) < 2:
        raise LibraryAssemblyError("repeated inverse requires an accumulator and side roles")
    if type(events) is not int or not 1 <= events <= bounds.maximum_events:
        raise LibraryAssemblyError("declared event count is outside the explicit search bound")
    target = constitutional_molecule(product)[0]
    side_roles = tuple(role for role in adapter.roles if role != accumulator_role)
    # A state binds a residual accumulator to the same complete side reagents at every event.
    states = {(target, ())}
    state_counts = [1]
    transition_count = 0
    reasons = []
    for _ in range(events):
        following = set()
        for state, assigned in sorted(states):
            try:
                inverse = adapter.decompose(state, maximum_outcomes=bounds.maximum_outcomes)
            except LibraryAssemblyError as exc:
                if "saturated" not in str(exc):
                    raise
                reasons.append("inverse_outcome_bound")
                break
            for candidate in inverse:
                components = dict(candidate.components)
                side = tuple((role, components[role]) for role in side_roles)
                if assigned and side != assigned:
                    continue
                following.add((components[accumulator_role], side))
                transition_count += 1
                if (
                    sum(state_counts) + len(following) > bounds.maximum_states
                    or transition_count > bounds.maximum_transitions
                ):
                    reasons.append("inverse_state_or_transition_bound")
                    break
            if reasons:
                break
        state_counts.append(len(following))
        states = following
        if reasons:
            break
    candidates = []
    if not reasons:
        for head, side in sorted(states):
            candidates.append(dict(sorted(((accumulator_role, head), *side))))
    return {
        "complete_search": not reasons,
        "declared_events": events,
        "candidate_components": candidates,
        "states_by_depth": state_counts,
        "transitions": transition_count,
        "bound_reasons": reasons,
        "inverse_scope": "all_tuples_with_one_repeated_identity_per_side_role_at_declared_count",
        "experimental_selectivity_qualified": False,
        "physical_event_order_qualified": False,
    }
