"""Generated-product assessment through complete qualified source contracts.

Source component predicates augment the unfiltered inverse and forward search.
They cannot remove competing outcomes to manufacture a unique reconstruction.
"""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from forge.assembly.families import LibraryAssemblyError, constitutional_molecule
from forge.assembly.repeated_components import replay_repeated_components
from forge.assembly.repeated_inverse import infer_repeated_components


def source_component_checks(
    executor: dict[str, Any], components: Mapping[str, str]
) -> dict[str, bool]:
    """Reuse preparation's extra component domains, including fixed AEMA identity."""
    checks = {}
    if "fixed_AEMA" in executor:
        from forge.corpus.compose_lipid_aema_replay import component_domain
        from forge.corpus.compose_lipid_zhou_replay import component_domain as zhou_domain

        checks["fixed_AEMA_identity"] = (
            constitutional_molecule(components["AEMA"])[0] == executor["fixed_AEMA"]
        )
        if "allowlists" in executor:
            checks["source_component_domain"] = zhou_domain(components, executor)
        else:
            checks["source_component_domain"] = component_domain(
                components["amine_core"], components["thiol_periphery"], executor
            )
        if "ester_thiol_domain" in executor:
            from forge.corpus.compose_lipid_ester_thiol import component_domain as ester_domain

            checks["ester_thiol_domain"] = ester_domain(
                components["thiol_periphery"], executor["ester_thiol_domain"]
            )
    elif executor.get("domain_kind") == "ester_thiol":
        from forge.corpus.compose_lipid_ester_thiol import component_domain as ester_only_domain

        checks["ester_thiol_domain"] = all(
            ester_only_domain(components[role], executor["domain"])
            for role in executor["domain_roles"]
        )
    elif "domain" in executor and executor["kind"] != "fixed":
        raise ValueError("A source component domain has no qualified assessment binding")
    return checks


def evaluate_executor(executor: dict[str, Any], product: str, *, events: int) -> dict[str, Any]:
    """Assess every inferred tuple with the original source replay and extra guards."""
    if executor.get("full_source_contract") is not True:
        raise ValueError("Generated evaluation requires an explicit complete source contract")
    kind = executor["kind"]
    target = constitutional_molecule(product)[0]
    if kind == "repeated":
        adapter, program, bounds = (executor[k] for k in ("adapter", "program", "bounds"))
        inverse = infer_repeated_components(
            adapter,
            target,
            accumulator_role=program["accumulator_role"],
            events=events,
            bounds=bounds,
        )

        def replay(parts: dict[str, str]) -> dict[str, Any]:
            return replay_repeated_components(
                adapter,
                parts,
                target,
                accumulator_role=program["accumulator_role"],
                events=events,
                byproducts_per_event=program["net_byproducts_per_event"],
                bounds=bounds,
            )

    elif kind in {"grouped", "staged", "program"}:
        program = executor["program"]
        inverse = program.infer(target)

        def replay(parts: dict[str, str]) -> dict[str, Any]:
            return cast(dict[str, Any], program.replay(parts, target))

    elif kind == "fixed":
        candidates = executor["adapter"].decompose(
            target, maximum_outcomes=executor["bounds"].maximum_outcomes
        )
        inverse = {
            "complete_search": True,
            "candidate_components": [dict(c.components) for c in candidates],
        }

        def replay(parts: dict[str, str]) -> dict[str, Any]:
            return cast(dict[str, Any], executor["run"](parts, target))

    else:
        raise ValueError(f"Unsupported complete source executor: {kind}")
    accepted, assessments = [], []
    if inverse["complete_search"]:
        for parts in inverse["candidate_components"]:
            mapping = executor["mapping"]
            if set(parts) != set(mapping):
                raise ValueError("Inverse components differ from the qualified source roles")
            identities: dict[str, set[str]] = {}
            for role, smiles in parts.items():
                identities.setdefault(mapping[role], set()).add(smiles)
            source_identity_exact = all(len(v) == 1 for v in identities.values())
            checked = replay(parts)
            extra = source_component_checks(executor, parts)
            passed = (
                checked["computed_consistency_pass"] is True
                and source_identity_exact
                and all(v is True for v in extra.values())
            )
            assessments.append(
                dict(
                    components=parts,
                    source_identity_exact=source_identity_exact,
                    source_component_checks=extra,
                    replay=checked,
                    **{"pass": passed},
                )
            )
            if passed:
                accepted.append(parts)
    return dict(
        exact_registry_program_roundtrip=bool(accepted) and inverse["complete_search"],
        inverse=inverse,
        candidates=assessments,
        accepted_components=accepted,
    )


def assess_product(
    executors: Sequence[dict[str, Any]], product: str, *, events: int
) -> dict[str, Any]:
    """Preserve bounded abstentions and return all considered source contracts."""
    if not executors:
        return dict(status="abstained", exact=False, reason="no_matching_qualified_executor")
    checks = []
    for executor in executors:
        try:
            checks.append(evaluate_executor(executor, product, events=events))
        except LibraryAssemblyError as error:
            checks.append(dict(error=str(error), exact_registry_program_roundtrip=False))
    return dict(
        status="evaluated",
        exact=any(c["exact_registry_program_roundtrip"] for c in checks),
        checks=checks,
    )
