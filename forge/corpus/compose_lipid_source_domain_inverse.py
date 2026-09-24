"""Adjudicate complete saved inverse searches against their pinned terminal domain.

No paths are pruned during search, and no supplied component or target selects the
domain. All candidates and individual rejection checks remain in the result.
"""

from __future__ import annotations

import copy

from forge.assembly.families import constitutional_molecule
from forge.assembly.staged_program import _constraints


def qualify_inverse(program, inverse):
    result = copy.deepcopy(inverse)
    if "unfiltered_candidate_components" in result:
        raise ValueError("Inverse domain has already been adjudicated")
    raw = result["candidate_components"]
    qualified, audit = [], []
    for candidate in raw:
        if set(candidate) != set(program.roles):
            raise ValueError("Inverse candidate lost a complete source role")
        checks = {
            role: _constraints(
                smiles,
                program.specification["terminal_constraints"][role],
                program.bounds.maximum_outcomes,
            )
            for role, smiles in candidate.items()
        }
        accepted = all(v["pass"] for v in checks.values())
        audit.append({"components": candidate, "terminal_constraints": checks, "pass": accepted})
        if accepted:
            qualified.append(candidate)
    result.update(
        unfiltered_candidate_components=raw,
        candidate_components=qualified,
        candidate_domain_audit=audit,
        inverse_scope="all_complete_tuples_within_pinned_source_terminal_constraints",
        target_used_for_domain_selection=False,
    )
    return result


def qualify_saved_replay(program, replay, components):
    """Reuse an authenticated complete search, changing only domain adjudication."""
    result = copy.deepcopy(replay)
    if "inverse" not in result:
        return result
    if program.quantities != result["declared_quantities"]:
        raise ValueError("Saved search source multiplicity differs")
    result["inverse"] = qualify_inverse(program, result["inverse"])
    canonical = dict(sorted((r, constitutional_molecule(s)[0]) for r, s in components.items()))
    result["checks"]["unique_complete_inverse"] = result["inverse"]["candidate_components"] == [
        canonical
    ]
    result["computed_consistency_pass"] = all(v is True for v in result["checks"].values())
    result["disposition"] = (
        "exact_computed_reconstruction"
        if result["computed_consistency_pass"]
        else "unresolved_computed_reconstruction"
    )
    return result
