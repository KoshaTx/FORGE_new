"""Training-fold program prior for unconditional Ugi whole-graph generation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forge.model.ugi_morphology_program import UgiMorphologyProgram
from forge.potency.annotations import ROLE_NAMES


class UgiProgramPriorError(RuntimeError):
    """Raised when the coarse morphology-program prior violates its contract."""


ProgramState = tuple[int, int, int, int]


@dataclass(frozen=True)
class UgiProgramPrior:
    """Role-wise categorical prior over count, junction, cycle and attachment states."""

    support_by_role: tuple[tuple[ProgramState, ...], ...]
    probabilities_by_role: tuple[np.ndarray, ...]


def load_program_prior(path: Path) -> tuple[UgiProgramPrior, dict[str, int]]:
    """Load one hash-addressed program-prior artifact for generation."""

    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != "phase1_ugi_program_prior.v1"
        or value.get("status") != "pass"
    ):
        raise UgiProgramPriorError("unsupported program-prior artifact")
    support_by_role = []
    probabilities_by_role = []
    for role in ROLE_NAMES:
        records = value.get("role_priors", {}).get(role)
        if not isinstance(records, list) or not records:
            raise UgiProgramPriorError(f"program-prior artifact lacks role: {role}")
        support_by_role.append(
            tuple(
                (
                    int(record["node_count"]),
                    int(record["junction_budget"]),
                    int(record["cycle_rank"]),
                    int(record["attachment_count"]),
                )
                for record in records
            )
        )
        probabilities_by_role.append(
            np.asarray([float(record["probability"]) for record in records])
        )
    bounds = {str(key): int(item) for key, item in value.get("bounds", {}).items()}
    prior = UgiProgramPrior(
        support_by_role=tuple(support_by_role),
        probabilities_by_role=tuple(probabilities_by_role),
    )
    if validate_program_prior_support(prior, bounds)["status"] != "pass":
        raise UgiProgramPriorError("loaded program prior exceeds its frozen support")
    return prior, bounds


def build_weighted_program_prior(
    programs: Sequence[UgiMorphologyProgram],
    weights: np.ndarray,
) -> UgiProgramPrior:
    """Aggregate product weights into role-specific coarse-program marginals."""

    if (
        not programs
        or weights.shape != (len(programs),)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0)
        or float(weights.sum()) <= 0
    ):
        raise UgiProgramPriorError("invalid weighted program-prior request")
    normalized = weights.astype(np.float64, copy=True)
    normalized /= normalized.sum()
    support_by_role = []
    probabilities_by_role = []
    for role_index in range(len(ROLE_NAMES)):
        mass: dict[ProgramState, float] = {}
        for program, weight in zip(programs, normalized, strict=True):
            state = (
                int(program.node_counts[role_index]),
                int(program.junction_budgets[role_index]),
                int(program.cycle_ranks[role_index]),
                int(program.attachment_counts[role_index]),
            )
            mass[state] = mass.get(state, 0.0) + float(weight)
        support = tuple(sorted(mass))
        probabilities = np.asarray([mass[state] for state in support], dtype=np.float64)
        probabilities /= probabilities.sum()
        support_by_role.append(support)
        probabilities_by_role.append(probabilities)
    return UgiProgramPrior(
        support_by_role=tuple(support_by_role),
        probabilities_by_role=tuple(probabilities_by_role),
    )


def build_component_family_balanced_program_prior(
    programs: Sequence[UgiMorphologyProgram],
    component_keys: Sequence[Mapping[str, str]],
    component_families: Sequence[Mapping[str, str]],
) -> tuple[UgiProgramPrior, dict[str, int]]:
    """Give each role family equal mass and each component equal mass within family.

    Component and family identifiers are estimator metadata only. They do not
    enter the returned program state or any neural tensor.
    """

    if (
        not programs
        or len(programs) != len(component_keys)
        or len(programs) != len(component_families)
    ):
        raise UgiProgramPriorError("component-balanced prior inputs are misaligned")
    support_by_role = []
    probabilities_by_role = []
    component_counts: dict[str, int] = {}
    for role_index, role in enumerate(ROLE_NAMES):
        by_component: dict[str, tuple[str, ProgramState]] = {}
        for program, keys, families in zip(
            programs, component_keys, component_families, strict=True
        ):
            if set(keys) != set(ROLE_NAMES) or set(families) != set(ROLE_NAMES):
                raise UgiProgramPriorError("component-balanced prior lacks a role")
            component = str(keys[role])
            family = str(families[role])
            if not component or not family:
                raise UgiProgramPriorError("empty component or family key")
            state = (
                int(program.node_counts[role_index]),
                int(program.junction_budgets[role_index]),
                int(program.cycle_ranks[role_index]),
                int(program.attachment_counts[role_index]),
            )
            previous = by_component.get(component)
            if previous is not None and previous != (family, state):
                raise UgiProgramPriorError("component program or family is inconsistent")
            by_component[component] = (family, state)
        by_family: dict[str, list[tuple[str, ProgramState]]] = {}
        for component, (family, state) in by_component.items():
            by_family.setdefault(family, []).append((component, state))
        state_mass: dict[ProgramState, float] = {}
        family_mass = 1.0 / len(by_family)
        for values in by_family.values():
            component_mass = family_mass / len(values)
            for _, state in values:
                state_mass[state] = state_mass.get(state, 0.0) + component_mass
        support = tuple(sorted(state_mass))
        probabilities = np.asarray([state_mass[state] for state in support], dtype=np.float64)
        probabilities /= probabilities.sum()
        support_by_role.append(support)
        probabilities_by_role.append(probabilities)
        component_counts[role] = len(by_component)
    return (
        UgiProgramPrior(
            support_by_role=tuple(support_by_role),
            probabilities_by_role=tuple(probabilities_by_role),
        ),
        component_counts,
    )


def blend_program_priors(
    realism_prior: UgiProgramPrior,
    exploration_prior: UgiProgramPrior,
    *,
    exploration_mass: float,
) -> UgiProgramPrior:
    """Blend role-wise priors on the union of their exact supports."""

    if (
        len(realism_prior.support_by_role) != len(ROLE_NAMES)
        or len(exploration_prior.support_by_role) != len(ROLE_NAMES)
        or not 0.0 < exploration_mass < 1.0
    ):
        raise UgiProgramPriorError("invalid program-prior blend")
    support_by_role = []
    probabilities_by_role = []
    for (
        realism_support,
        realism_probabilities,
        exploration_support,
        exploration_probabilities,
    ) in zip(
        realism_prior.support_by_role,
        realism_prior.probabilities_by_role,
        exploration_prior.support_by_role,
        exploration_prior.probabilities_by_role,
        strict=True,
    ):
        mass: dict[ProgramState, float] = {}
        for state, probability in zip(realism_support, realism_probabilities, strict=True):
            mass[state] = mass.get(state, 0.0) + (1.0 - exploration_mass) * float(probability)
        for state, probability in zip(exploration_support, exploration_probabilities, strict=True):
            mass[state] = mass.get(state, 0.0) + exploration_mass * float(probability)
        support = tuple(sorted(mass))
        probabilities = np.asarray([mass[state] for state in support], dtype=np.float64)
        probabilities /= probabilities.sum()
        support_by_role.append(support)
        probabilities_by_role.append(probabilities)
    return UgiProgramPrior(
        support_by_role=tuple(support_by_role),
        probabilities_by_role=tuple(probabilities_by_role),
    )


def validate_program_prior_support(
    prior: UgiProgramPrior,
    bounds: Mapping[str, int],
) -> dict[str, Any]:
    """Validate every marginal state and the largest independent role combination."""

    if (
        len(prior.support_by_role) != len(ROLE_NAMES)
        or len(prior.probabilities_by_role) != len(ROLE_NAMES)
        or any(not support for support in prior.support_by_role)
    ):
        raise UgiProgramPriorError("program prior lacks a precursor role")
    required = {
        "maximum_component_atoms",
        "maximum_total_atoms",
        "maximum_junction_budget",
        "maximum_cycle_rank",
        "maximum_attachment_count",
    }
    if not required.issubset(bounds):
        raise UgiProgramPriorError("program-prior bounds are incomplete")
    maxima = {
        "component_atoms": 0,
        "total_atoms": 0,
        "junction_budget": 0,
        "cycle_rank": 0,
        "attachment_count": 0,
    }
    support_counts = {}
    role_maximum_counts = []
    for role, support, probabilities in zip(
        ROLE_NAMES,
        prior.support_by_role,
        prior.probabilities_by_role,
        strict=True,
    ):
        if (
            probabilities.shape != (len(support),)
            or np.any(~np.isfinite(probabilities))
            or np.any(probabilities <= 0)
            or not np.isclose(probabilities.sum(), 1.0)
        ):
            raise UgiProgramPriorError("invalid program-prior probabilities")
        support_counts[role] = len(support)
        role_maximum_counts.append(max(state[0] for state in support))
        maxima["component_atoms"] = max(
            maxima["component_atoms"], max(state[0] for state in support)
        )
        maxima["junction_budget"] = max(
            maxima["junction_budget"], max(state[1] for state in support)
        )
        maxima["cycle_rank"] = max(maxima["cycle_rank"], max(state[2] for state in support))
        maxima["attachment_count"] = max(
            maxima["attachment_count"], max(state[3] for state in support)
        )
    maxima["total_atoms"] = sum(role_maximum_counts)
    failures = []
    for metric, bound_name in (
        ("component_atoms", "maximum_component_atoms"),
        ("total_atoms", "maximum_total_atoms"),
        ("junction_budget", "maximum_junction_budget"),
        ("cycle_rank", "maximum_cycle_rank"),
        ("attachment_count", "maximum_attachment_count"),
    ):
        if maxima[metric] > int(bounds[bound_name]):
            failures.append(metric)
    return {
        "status": "pass" if not failures else "fail",
        "support_counts_by_role": support_counts,
        "independent_role_maxima": maxima,
        "failure_reasons": failures,
    }


def sample_weighted_program_prior(
    prior: UgiProgramPrior,
    *,
    count: int,
    rng: np.random.Generator,
    bounds: Mapping[str, int],
) -> tuple[UgiMorphologyProgram, ...]:
    """Sample coarse role programs without selecting molecular components."""

    if count < 1 or validate_program_prior_support(prior, bounds)["status"] != "pass":
        raise UgiProgramPriorError("program prior cannot be sampled within support")
    output = []
    for _ in range(count):
        selected = [
            support[int(rng.choice(len(support), p=probabilities))]
            for support, probabilities in zip(
                prior.support_by_role,
                prior.probabilities_by_role,
                strict=True,
            )
        ]
        program = UgiMorphologyProgram(
            node_counts=tuple(state[0] for state in selected),
            junction_budgets=tuple(state[1] for state in selected),
            cycle_ranks=tuple(state[2] for state in selected),
            attachment_counts=tuple(state[3] for state in selected),
        )
        if program.node_count > int(bounds["maximum_total_atoms"]):
            raise UgiProgramPriorError("sampled program exceeds total support")
        output.append(program)
    return tuple(output)
