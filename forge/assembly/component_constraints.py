"""TRAIN-label consistency of exact inverse candidates; never source-chemistry admission.

A label is scoped by family and role. Constraint propagation computes necessary conditions
under the explicit assumption that one such label denotes one constitution. Arc consistency
is not proof of a global solution or of the source's intended disconnection.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentConstraint:
    target_id: str
    variables: tuple[tuple[str, str, str], ...]
    candidates: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if (
            not self.target_id
            or not self.variables
            or len(set(self.variables)) != len(self.variables)
        ):
            raise ValueError("constraint requires a target and distinct scoped labels")
        if any(
            len(v) != 3 or any(not isinstance(s, str) or not s for s in v) for v in self.variables
        ):
            raise ValueError("variables require nonempty family, role, and source label")
        if not self.candidates or any(
            len(c) != len(self.variables) or any(not isinstance(s, str) or not s for s in c)
            for c in self.candidates
        ):
            raise ValueError("candidate arity/identity differs from scoped labels")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("duplicate constraint candidates")


@dataclass(frozen=True)
class ComponentConsistency:
    surviving_candidates: dict[str, tuple[tuple[str, ...], ...]]
    conflicting_targets: tuple[str, ...]
    singleton_component_witness_targets: tuple[str, ...]


def propagate_component_labels(constraints: list[ComponentConstraint]) -> ComponentConsistency:
    """Intersect row supports to a fixed point; quarantine whole contradictory components.

    Only a connected component whose every row has one remaining candidate has a demonstrated
    global assignment. Other components may contain arc-consistent but unsatisfiable cycles.
    No arbitrary branch, source label ordering, or held-out information breaks a tie.
    """
    rows = {row.target_id: row for row in constraints}
    if len(rows) != len(constraints):
        raise ValueError("duplicate target in component constraints")
    neighbours = defaultdict(set)
    for row in constraints:
        for variable in row.variables:
            neighbours[variable].add(row.target_id)
    active = {key: set(row.candidates) for key, row in rows.items()}
    domains: dict[tuple[str, str, str], set[str]] = {}
    queue = deque(sorted(rows))
    queued = set(queue)
    while queue:
        key = queue.popleft()
        queued.remove(key)
        row = rows[key]
        active[key] = {
            candidate
            for candidate in active[key]
            if all(
                variable not in domains or candidate[i] in domains[variable]
                for i, variable in enumerate(row.variables)
            )
        }
        for i, variable in enumerate(row.variables):
            support = {candidate[i] for candidate in active[key]}
            previous = domains.get(variable)
            narrowed = support if previous is None else previous & support
            if previous is not None and narrowed == previous:
                continue
            domains[variable] = narrowed
            for other in sorted(neighbours[variable] - queued):
                queue.append(other)
                queued.add(other)
    conflicting, witnessed, visited = set(), set(), set()
    for start in sorted(rows):
        if start in visited:
            continue
        component, expanded_variables, todo = set(), set(), [start]
        while todo:
            key = todo.pop()
            if key in component:
                continue
            component.add(key)
            for variable in rows[key].variables:
                if variable not in expanded_variables:
                    expanded_variables.add(variable)
                    todo.extend(neighbours[variable] - component)
        visited.update(component)
        if any(not active[key] for key in component):
            conflicting.update(component)
        elif all(len(active[key]) == 1 for key in component):
            witnessed.update(component)
    return ComponentConsistency(
        {key: tuple(sorted(active[key])) for key in sorted(rows)},
        tuple(sorted(conflicting)),
        tuple(sorted(witnessed)),
    )
