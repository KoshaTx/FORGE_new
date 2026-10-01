"""Diagnostic repeated-program search; never changes the identical-reactant admission gate."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from rdkit import rdBase

from forge.assembly.families import (
    LibraryAssemblyError,
    RegistryAssemblyAdapter,
    constitutional_molecule,
)
from forge.assembly.library_programs import LibraryProgramLimits, _layer
from forge.assembly.program import repair_template_hydrogens


@dataclass(frozen=True, order=True)
class DiagnosticStep:
    components: tuple[tuple[str, str], ...]
    product: str


def replay_diagnostic_steps(
    adapter: RegistryAssemblyAdapter,
    steps: tuple[DiagnosticStep, ...],
    *,
    accumulator_role: str,
    maximum_outcomes: int,
) -> bool:
    """Check every step and accumulator handoff with the unchanged strict registry adapter."""
    if not steps or len(adapter.roles) != 2 or accumulator_role not in adapter.roles:
        raise LibraryAssemblyError("a repeated diagnostic needs steps and a valid accumulator")
    previous = None
    for step in steps:
        components = dict(step.components)
        if previous is not None and components.get(accumulator_role) != previous:
            return False
        check = adapter.check_forward(components, step.product, maximum_outcomes=maximum_outcomes)
        if not check.exact or check.saturated:
            return False
        previous = step.product
    return True


def _dead_end(adapter: RegistryAssemblyAdapter, smiles: str, maximum_outcomes: int) -> dict:
    """Inspect exact algebraic inverse/forward candidates solely to explain a strict dead end.

    This returns no accepted step or substructure-hit metric. A raw exact transform outside the
    role policy remains rejected. If fragments cannot be repaired or the transform cannot replay,
    no claim about chemical impossibility or the location of a model error follows.
    """
    target, molecule = constitutional_molecule(smiles)
    with rdBase.BlockLogs():
        outcomes = adapter.reaction.reverse.RunReactants((molecule,), maxProducts=maximum_outcomes)
    if len(outcomes) >= maximum_outcomes:
        raise LibraryAssemblyError("dead-end reverse enumeration saturated")
    candidates = set()
    for outcome in outcomes:
        if len(outcome) != len(adapter.roles):
            continue
        repaired = tuple(repair_template_hydrogens(fragment) for fragment in outcome)
        if any(item is None for item in repaired):
            continue
        candidates.add(
            tuple((role, item[0]) for role, item in zip(adapter.roles, repaired, strict=True))
        )
    for candidate in sorted(candidates):
        components = dict(candidate)
        products, saturated, qualified = _layer(adapter, components, maximum_outcomes)
        if saturated:
            raise LibraryAssemblyError("dead-end forward enumeration saturated")
        if target not in products:
            continue
        if qualified:
            raise LibraryAssemblyError("strict decomposition missed a policy-qualified exact step")
        return {
            "reason": "exact_transform_rejected_by_role_policy",
            "product": target,
            "components": components,
            "role_assessments": [asdict(a) for a in adapter.assess_roles(components)],
            "accepted": False,
        }
    return {"reason": "no_verified_inverse_forward_step", "product": target, "accepted": False}


def _program(steps: tuple[DiagnosticStep, ...], accumulator_role: str) -> dict:
    identities = [
        tuple((r, s) for r, s in step.components if r != accumulator_role) for step in steps
    ]
    return {
        "steps": [asdict(step) for step in steps],
        "identical_coreactants": len(set(identities)) == 1,
        "exact_forward_replay": True,
        "admission_status": "diagnostic_only",
    }


def audit_repeated_program(
    adapter: RegistryAssemblyAdapter,
    product_smiles: str,
    *,
    depth: int,
    accumulator_role: str,
    limits: LibraryProgramLimits,
) -> dict[str, Any]:
    """Search up to the requested depth with distinct co-reactants permitted diagnostically.

    All forward chemistry and role constraints are unchanged. Full-depth witnesses are never
    promoted to the original identical-repeat gate. The complete layer must finish before its
    witnesses count; a state, outcome or expansion cap makes the diagnostic inconclusive.
    """
    if type(depth) is not int or not 1 <= depth <= limits.maximum_steps:
        raise LibraryAssemblyError("diagnostic depth is outside the declared support")
    if len(adapter.roles) != 2 or accumulator_role not in adapter.roles:
        raise LibraryAssemblyError("diagnostic requires a two-role repeated program")
    target, _ = constitutional_molecule(product_smiles)
    frontier: set[tuple[str, tuple[DiagnosticStep, ...]]] = {(target, ())}
    expansions, completed_depth = 0, 0
    path_counts: dict[str, int] = {}
    full_programs, shorter_example = [], None
    dead_ends: Counter[str] = Counter()
    dead_end_examples: dict[str, dict] = {}
    reason = None
    try:
        for level in range(1, depth + 1):
            following = set()
            for current, reverse_steps in sorted(frontier):
                if expansions >= limits.maximum_expansions:
                    reason = "expansion_limit"
                    break
                expansions += 1
                decompositions = adapter.decompose(
                    current, maximum_outcomes=limits.maximum_outcomes
                )
                if not decompositions:
                    explanation = _dead_end(adapter, current, limits.maximum_outcomes)
                    key = f"reverse_depth_{level}:{explanation['reason']}"
                    dead_ends[key] += 1
                    dead_end_examples.setdefault(key, explanation)
                for decomposition in decompositions:
                    step = DiagnosticStep(decomposition.components, current)
                    next_steps = (*reverse_steps, step)
                    initial = dict(decomposition.components)[accumulator_role]
                    following.add((initial, next_steps))
                    if len(following) > limits.maximum_states:
                        reason = "state_limit"
                        break
                if reason:
                    break
            if reason:
                break
            completed_depth = level
            path_counts[str(level)] = len(following)
            # Independently replay all paths before treating even a shorter path as verified.
            for _, reverse_steps in sorted(following):
                steps = tuple(reversed(reverse_steps))
                if (
                    not replay_diagnostic_steps(
                        adapter,
                        steps,
                        accumulator_role=accumulator_role,
                        maximum_outcomes=limits.maximum_outcomes,
                    )
                    or steps[-1].product != target
                ):
                    raise LibraryAssemblyError("diagnostic path failed strict forward replay")
                if level == depth:
                    full_programs.append(_program(steps, accumulator_role))
            if following and level < depth:
                shorter_example = _program(tuple(reversed(min(following)[1])), accumulator_role)
            frontier = following
    except LibraryAssemblyError as error:
        if "saturated" not in str(error):
            raise
        reason = str(error)
    return {
        "status": "abstain" if reason else "complete",
        "reason": reason,
        "requested_depth": depth,
        "completed_depth": completed_depth,
        "expansions": expansions,
        "verified_path_counts_by_depth": path_counts,
        "requested_depth_programs": [] if reason else full_programs,
        "shorter_program_example": shorter_example,
        "dead_end_counts": dict(sorted(dead_ends.items())),
        "dead_end_examples": dead_end_examples,
        "original_gate_changed": False,
    }


def classify_repeat_attempt(original_status: str, audit: dict | None) -> str:
    if original_status == "invalid_graph":
        return "invalid_graph"
    if original_status == "abstain":
        return "original_search_abstained"
    if audit is None:
        raise LibraryAssemblyError("valid repeated attempt is missing its diagnostic")
    if audit["status"] == "abstain":
        return "diagnostic_search_abstained"
    identical = any(p["identical_coreactants"] for p in audit["requested_depth_programs"])
    if identical != (original_status == "exact_computed_program"):
        raise LibraryAssemblyError("mixed diagnostic and original identical-repeat gate disagree")
    if identical:
        return "original_exact_program"
    if audit["requested_depth_programs"]:
        return "different_coreactants_only_at_requested_depth"
    if audit["shorter_program_example"] is not None:
        return "only_shorter_policy_valid_program"
    if any("exact_transform_rejected_by_role_policy" in k for k in audit["dead_end_counts"]):
        return "role_policy_blocks_first_step"
    return "no_verified_first_step"
