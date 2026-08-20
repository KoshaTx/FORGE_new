"""Exact global morphology programs for the Phase 1 V5 lipid flow.

The neural model supplies local scores.  These routines condition those scores
on a small, generated global program so that local choices form one connected
tree with exact regional atom counts and exact regional junction budgets.
No fragment identity or building-block catalog appears in this state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

from forge.design.flow.lipid_context import HEAD_REGION, INTERFACE_REGION, TAIL_REGION
from forge.design.flow.phase1_tree_topology_flow import (
    TreeTopologyFlowError,
    preorder_offspring_to_parents,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None


class MorphologyProgramError(RuntimeError):
    """Raised when a global morphology program cannot be decoded exactly."""


@dataclass(frozen=True)
class V5GlobalMorphologyProgram:
    """Minimal generated global quantities for one sparse lipid graph."""

    n_head: int
    n_interface: int
    n_tail: int
    junction_budget_head_interface: int
    junction_budget_tail: int
    cycle_rank: int

    @property
    def node_count(self) -> int:
        return self.n_head + self.n_interface + self.n_tail

    @property
    def junction_budget_total(self) -> int:
        return self.junction_budget_head_interface + self.junction_budget_tail


def tree_junction_contributions(offspring: np.ndarray) -> np.ndarray:
    """Return per-node tree degree excess, excluding all closure edges."""

    if (
        offspring.ndim != 1
        or offspring.size < 1
        or not np.issubdtype(offspring.dtype, np.integer)
        or np.any(offspring < 0)
    ):
        raise MorphologyProgramError("offspring must be a nonempty nonnegative integer array")
    tree_degrees = offspring.astype(np.int64, copy=True)
    if tree_degrees.size > 1:
        tree_degrees[1:] += 1
    return np.maximum(tree_degrees - 2, 0)


def program_from_sparse_record(record: Any) -> V5GlobalMorphologyProgram:
    """Derive the exact global morphology target from one V5 sparse record."""

    if record.region_states is None:
        raise MorphologyProgramError("V5 morphology supervision requires region states")
    regions = np.asarray(record.region_states)
    if regions.shape != (record.node_count,) or np.any(
        (regions < HEAD_REGION) | (regions > TAIL_REGION)
    ):
        raise MorphologyProgramError("invalid V5 region supervision")
    contributions = tree_junction_contributions(record.offspring)
    head_interface = regions != TAIL_REGION
    return V5GlobalMorphologyProgram(
        n_head=int(np.count_nonzero(regions == HEAD_REGION)),
        n_interface=int(np.count_nonzero(regions == INTERFACE_REGION)),
        n_tail=int(np.count_nonzero(regions == TAIL_REGION)),
        junction_budget_head_interface=int(contributions[head_interface].sum()),
        junction_budget_tail=int(contributions[~head_interface].sum()),
        cycle_rank=int(record.closure_count),
    )


def morphology_program_valid(
    program: V5GlobalMorphologyProgram,
    *,
    maximum_nodes: int,
    maximum_cycle_rank: int,
) -> bool:
    """Validate declared support without claiming chemical feasibility."""

    values = (
        program.n_head,
        program.n_interface,
        program.n_tail,
        program.junction_budget_head_interface,
        program.junction_budget_tail,
        program.cycle_rank,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        return False
    return bool(
        1 <= program.n_head
        and program.n_interface >= 0
        and program.n_tail >= 0
        and 1 <= program.node_count <= maximum_nodes
        and program.junction_budget_head_interface >= 0
        and program.junction_budget_tail >= 0
        and 0 <= program.cycle_rank <= maximum_cycle_rank
    )


def offspring_matches_program(
    offspring: np.ndarray,
    program: V5GlobalMorphologyProgram,
) -> bool:
    """Check tree size, exact decoding, and total junction budget."""

    if offspring.shape != (program.node_count,):
        return False
    try:
        preorder_offspring_to_parents(offspring)
        contributions = tree_junction_contributions(offspring)
    except (TreeTopologyFlowError, MorphologyProgramError):
        return False
    return int(contributions.sum()) == program.junction_budget_total


def regions_match_program(
    regions: np.ndarray,
    offspring: np.ndarray,
    program: V5GlobalMorphologyProgram,
) -> bool:
    """Check exact regional counts and regional tree-junction allocation."""

    if regions.shape != (program.node_count,) or int(regions[0]) != HEAD_REGION:
        return False
    if np.any((regions < HEAD_REGION) | (regions > TAIL_REGION)):
        return False
    contributions = tree_junction_contributions(offspring)
    return bool(
        np.count_nonzero(regions == HEAD_REGION) == program.n_head
        and np.count_nonzero(regions == INTERFACE_REGION) == program.n_interface
        and np.count_nonzero(regions == TAIL_REGION) == program.n_tail
        and int(contributions[regions != TAIL_REGION].sum())
        == program.junction_budget_head_interface
        and int(contributions[regions == TAIL_REGION].sum()) == program.junction_budget_tail
    )


def _tree_choice_valid(
    *,
    position: int,
    pending: int,
    remaining_budget: int,
    children: int,
    node_count: int,
) -> tuple[int, int] | None:
    next_pending = pending - 1 + children
    positions_after = node_count - position - 1
    contribution = max(0, children - (2 if position == 0 else 1))
    next_budget = remaining_budget - contribution
    if (
        next_budget < 0
        or next_pending < 0
        or next_pending > positions_after
        or (positions_after > 0 and next_pending == 0)
        or (positions_after == 0 and next_pending != 0)
    ):
        return None
    return next_pending, next_budget


def sample_offspring_with_exact_junction_budget(
    logits: Any,
    *,
    junction_budget: int,
    generator: Any,
) -> np.ndarray:
    """Sample local child-count scores conditioned on one exact tree program."""

    if torch is None:
        raise MorphologyProgramError("offspring sampling requires torch")
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2:
        raise MorphologyProgramError("offspring logits must be [nodes, child classes]")
    if junction_budget < 0:
        raise MorphologyProgramError("junction budget must be nonnegative")
    node_count, child_classes = logits.shape
    maximum_children = child_classes - 1
    log_probabilities = logits.to(torch.float64).log_softmax(dim=-1)
    suffix: list[dict[tuple[int, int], Any]] = [dict() for _ in range(node_count + 1)]
    suffix[node_count][(0, 0)] = logits.new_tensor(0.0, dtype=torch.float64)
    for position in range(node_count - 1, -1, -1):
        positions_including_current = node_count - position
        for pending in range(1, positions_including_current + 1):
            for budget in range(junction_budget + 1):
                terms = []
                for children in range(maximum_children + 1):
                    next_state = _tree_choice_valid(
                        position=position,
                        pending=pending,
                        remaining_budget=budget,
                        children=children,
                        node_count=node_count,
                    )
                    if next_state is None or next_state not in suffix[position + 1]:
                        continue
                    terms.append(
                        log_probabilities[position, children] + suffix[position + 1][next_state]
                    )
                if terms:
                    suffix[position][(pending, budget)] = torch.logsumexp(torch.stack(terms), dim=0)
    if (1, junction_budget) not in suffix[0]:
        raise MorphologyProgramError(
            "declared child-count support cannot realize the junction budget"
        )

    offspring = np.zeros(node_count, dtype=np.int64)
    pending = 1
    remaining_budget = junction_budget
    for position in range(node_count):
        choices: list[int] = []
        weights = []
        states: list[tuple[int, int]] = []
        for children in range(maximum_children + 1):
            next_state = _tree_choice_valid(
                position=position,
                pending=pending,
                remaining_budget=remaining_budget,
                children=children,
                node_count=node_count,
            )
            if next_state is None or next_state not in suffix[position + 1]:
                continue
            choices.append(children)
            states.append(next_state)
            weights.append(log_probabilities[position, children] + suffix[position + 1][next_state])
        selected = int(
            torch.multinomial(
                torch.stack(weights).softmax(dim=0),
                1,
                generator=generator,
            )
        )
        offspring[position] = choices[selected]
        pending, remaining_budget = states[selected]
    preorder_offspring_to_parents(offspring)
    if int(tree_junction_contributions(offspring).sum()) != junction_budget:
        raise MorphologyProgramError("junction-conditioned tree decoder violated its contract")
    return offspring


def sample_regions_with_exact_program(
    logits: Any,
    offspring: np.ndarray,
    program: V5GlobalMorphologyProgram,
    *,
    generator: Any,
    transition_log_probabilities: Any | None = None,
) -> np.ndarray:
    """Sample region scores under exact counts and regional junction budgets."""

    if torch is None:
        raise MorphologyProgramError("region sampling requires torch")
    if logits.shape != (program.node_count, 3):
        raise MorphologyProgramError("region logits must be [program.node_count, 3]")
    if not offspring_matches_program(offspring, program):
        raise MorphologyProgramError("offspring does not match the global program")
    if transition_log_probabilities is not None and transition_log_probabilities.shape != (3, 3):
        raise MorphologyProgramError("region transition scores must be [3, 3]")
    parents = preorder_offspring_to_parents(offspring)
    contributions = tree_junction_contributions(offspring)

    @cache
    def can_complete(
        position: int,
        heads: int,
        interfaces: int,
        tails: int,
        tail_budget: int,
    ) -> bool:
        if min(heads, interfaces, tails, tail_budget) < 0:
            return False
        if heads + interfaces + tails != program.node_count - position:
            return False
        if position == program.node_count:
            return heads == interfaces == tails == tail_budget == 0
        contribution = int(contributions[position])
        return (
            (heads > 0 and can_complete(position + 1, heads - 1, interfaces, tails, tail_budget))
            or (
                interfaces > 0
                and can_complete(position + 1, heads, interfaces - 1, tails, tail_budget)
            )
            or (
                tails > 0
                and tail_budget >= contribution
                and can_complete(
                    position + 1,
                    heads,
                    interfaces,
                    tails - 1,
                    tail_budget - contribution,
                )
            )
        )

    remaining = [program.n_head - 1, program.n_interface, program.n_tail]
    if not can_complete(1, *remaining, program.junction_budget_tail):
        raise MorphologyProgramError("tree cannot realize the regional morphology program")
    regions = np.full(program.node_count, HEAD_REGION, dtype=np.int64)
    remaining_tail_budget = program.junction_budget_tail
    for position in range(1, program.node_count):
        viable: list[int] = []
        weights = []
        for region in (HEAD_REGION, INTERFACE_REGION, TAIL_REGION):
            if remaining[region] <= 0:
                continue
            next_remaining = remaining.copy()
            next_remaining[region] -= 1
            next_tail_budget = remaining_tail_budget
            if region == TAIL_REGION:
                next_tail_budget -= int(contributions[position])
            if not can_complete(position + 1, *next_remaining, next_tail_budget):
                continue
            weight = logits[position, region]
            if transition_log_probabilities is not None:
                weight = (
                    weight
                    + transition_log_probabilities[int(regions[int(parents[position])]), region]
                )
            viable.append(region)
            weights.append(weight)
        if not viable:
            raise MorphologyProgramError("region prefix lost all exact completions")
        selected = int(
            torch.multinomial(
                torch.stack(weights).softmax(dim=0),
                1,
                generator=generator,
            )
        )
        region = viable[selected]
        regions[position] = region
        remaining[region] -= 1
        if region == TAIL_REGION:
            remaining_tail_budget -= int(contributions[position])
    if not regions_match_program(regions, offspring, program):
        raise MorphologyProgramError("region decoder violated the global program")
    return regions


def sample_morphology_with_exact_program(
    offspring_logits: Any,
    region_logits: Any,
    program: V5GlobalMorphologyProgram,
    *,
    generator: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Jointly decode a tree and tail allocation under the complete program.

    A tree with the correct *total* branch budget can still be incompatible
    with the requested tail branch budget.  This decoder therefore chooses
    every offspring count together with a head/interface-versus-tail group.
    Head versus interface labels are then sampled under their exact remaining
    counts.  This is a constrained terminal decoder, not graph repair.
    """

    if torch is None:
        raise MorphologyProgramError("joint morphology sampling requires torch")
    if offspring_logits.ndim != 2 or offspring_logits.shape[0] != program.node_count:
        raise MorphologyProgramError("offspring logits do not match the program")
    if region_logits.shape != (program.node_count, 3):
        raise MorphologyProgramError("region logits do not match the program")
    if offspring_logits.shape[1] < 2:
        raise MorphologyProgramError("offspring logits require at least two child classes")
    child_log_probabilities = offspring_logits.to(torch.float64).log_softmax(dim=-1)
    region_log_probabilities = region_logits.to(torch.float64).log_softmax(dim=-1)
    maximum_children = offspring_logits.shape[1] - 1

    @cache
    def suffix(
        position: int,
        pending: int,
        total_budget: int,
        tails: int,
        tail_budget: int,
    ) -> Any | None:
        if min(pending, total_budget, tails, tail_budget) < 0:
            return None
        positions_left = program.node_count - position
        if pending > positions_left or tails > positions_left:
            return None
        if position == program.node_count:
            if pending == total_budget == tails == tail_budget == 0:
                return offspring_logits.new_tensor(0.0, dtype=torch.float64)
            return None
        terms = []
        for children in range(maximum_children + 1):
            next_state = _tree_choice_valid(
                position=position,
                pending=pending,
                remaining_budget=total_budget,
                children=children,
                node_count=program.node_count,
            )
            if next_state is None:
                continue
            next_pending, next_total_budget = next_state
            contribution = max(0, children - (2 if position == 0 else 1))
            groups = (False,) if position == 0 else (False, True)
            for is_tail in groups:
                next_tails = tails - int(is_tail)
                next_tail_budget = tail_budget - (contribution if is_tail else 0)
                tail = suffix(
                    position + 1,
                    next_pending,
                    next_total_budget,
                    next_tails,
                    next_tail_budget,
                )
                if tail is None:
                    continue
                if position == 0:
                    group_score = offspring_logits.new_tensor(0.0, dtype=torch.float64)
                elif is_tail:
                    group_score = region_log_probabilities[position, TAIL_REGION]
                else:
                    group_score = torch.logsumexp(
                        region_log_probabilities[
                            position,
                            [HEAD_REGION, INTERFACE_REGION],
                        ],
                        dim=0,
                    )
                terms.append(child_log_probabilities[position, children] + group_score + tail)
        return torch.logsumexp(torch.stack(terms), dim=0) if terms else None

    initial = suffix(
        0,
        1,
        program.junction_budget_total,
        program.n_tail,
        program.junction_budget_tail,
    )
    if initial is None:
        raise MorphologyProgramError("local support cannot realize the complete morphology program")

    offspring = np.zeros(program.node_count, dtype=np.int64)
    tail_mask = np.zeros(program.node_count, dtype=np.bool_)
    pending = 1
    total_budget = program.junction_budget_total
    tails = program.n_tail
    tail_budget = program.junction_budget_tail
    for position in range(program.node_count):
        choices: list[tuple[int, bool, int, int, int, int]] = []
        weights = []
        for children in range(maximum_children + 1):
            next_state = _tree_choice_valid(
                position=position,
                pending=pending,
                remaining_budget=total_budget,
                children=children,
                node_count=program.node_count,
            )
            if next_state is None:
                continue
            next_pending, next_total_budget = next_state
            contribution = max(0, children - (2 if position == 0 else 1))
            groups = (False,) if position == 0 else (False, True)
            for is_tail in groups:
                next_tails = tails - int(is_tail)
                next_tail_budget = tail_budget - (contribution if is_tail else 0)
                tail = suffix(
                    position + 1,
                    next_pending,
                    next_total_budget,
                    next_tails,
                    next_tail_budget,
                )
                if tail is None:
                    continue
                if position == 0:
                    group_score = offspring_logits.new_tensor(0.0, dtype=torch.float64)
                elif is_tail:
                    group_score = region_log_probabilities[position, TAIL_REGION]
                else:
                    group_score = torch.logsumexp(
                        region_log_probabilities[
                            position,
                            [HEAD_REGION, INTERFACE_REGION],
                        ],
                        dim=0,
                    )
                choices.append(
                    (
                        children,
                        is_tail,
                        next_pending,
                        next_total_budget,
                        next_tails,
                        next_tail_budget,
                    )
                )
                weights.append(child_log_probabilities[position, children] + group_score + tail)
        if not choices:
            raise MorphologyProgramError("morphology prefix lost all exact completions")
        selected = int(
            torch.multinomial(
                torch.stack(weights).softmax(dim=0),
                1,
                generator=generator,
            )
        )
        (
            children,
            is_tail,
            pending,
            total_budget,
            tails,
            tail_budget,
        ) = choices[selected]
        offspring[position] = children
        tail_mask[position] = is_tail

    head_interface_positions = [
        position for position in range(1, program.node_count) if not bool(tail_mask[position])
    ]

    @cache
    def head_suffix(offset: int, heads: int) -> Any | None:
        positions_left = len(head_interface_positions) - offset
        if heads < 0 or heads > positions_left:
            return None
        if offset == len(head_interface_positions):
            return offspring_logits.new_tensor(0.0, dtype=torch.float64) if heads == 0 else None
        position = head_interface_positions[offset]
        terms = []
        for region in (HEAD_REGION, INTERFACE_REGION):
            next_heads = heads - int(region == HEAD_REGION)
            tail = head_suffix(offset + 1, next_heads)
            if tail is not None:
                terms.append(region_log_probabilities[position, region] + tail)
        return torch.logsumexp(torch.stack(terms), dim=0) if terms else None

    remaining_heads = program.n_head - 1
    if head_suffix(0, remaining_heads) is None:
        raise MorphologyProgramError("head/interface counts cannot be allocated")
    regions = np.full(program.node_count, TAIL_REGION, dtype=np.int64)
    regions[0] = HEAD_REGION
    for offset, position in enumerate(head_interface_positions):
        choices = []
        weights = []
        for region in (HEAD_REGION, INTERFACE_REGION):
            next_heads = remaining_heads - int(region == HEAD_REGION)
            tail = head_suffix(offset + 1, next_heads)
            if tail is None:
                continue
            choices.append(region)
            weights.append(region_log_probabilities[position, region] + tail)
        selected = int(
            torch.multinomial(
                torch.stack(weights).softmax(dim=0),
                1,
                generator=generator,
            )
        )
        region = choices[selected]
        regions[position] = region
        remaining_heads -= int(region == HEAD_REGION)
    if not offspring_matches_program(offspring, program) or not regions_match_program(
        regions, offspring, program
    ):
        raise MorphologyProgramError("joint morphology decoder violated its contract")
    return offspring, regions


def programs_from_records(records: Sequence[Any]) -> tuple[V5GlobalMorphologyProgram, ...]:
    """Derive deterministic global-program targets for a record collection."""

    if not records:
        raise MorphologyProgramError("at least one record is required")
    return tuple(program_from_sparse_record(record) for record in records)
