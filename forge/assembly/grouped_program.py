"""Bounded repeated events grouped by explicitly documented source stages.

All event paths remain visible. Uniqueness is required at each source-stage boundary,
not at a fictitious isolated mono-adduct. Complete inverse tuples are independently
searched with one declared repeated identity per side role in each stage.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.assembly.families import (
    LibraryAssemblyError,
    RegistryAssemblyAdapter,
    constitutional_molecule,
)
from forge.assembly.repeated_components import RepeatBounds, element_inventory
from forge.assembly.repeated_inverse import infer_repeated_components
from forge.assembly.staged_program import _constraints, _validate_constraints
from forge.core.hashing import sha256_file


@dataclass(frozen=True)
class RegistryGroupedProgram:
    specification: dict[str, Any]
    adapters: tuple[RegistryAssemblyAdapter, ...]
    bounds: RepeatBounds

    @classmethod
    def from_registry(
        cls,
        path: Path,
        *,
        program_id: str,
        expected_sha256: str,
        bounds: RepeatBounds = RepeatBounds(),
    ) -> RegistryGroupedProgram:
        if str(sha256_file(path)) != expected_sha256:
            raise LibraryAssemblyError("grouped registry hash mismatch")
        registry = json.loads(path.read_text())
        programs = registry.get("grouped_programs")
        if not isinstance(programs, list) or any(not isinstance(p, dict) for p in programs):
            raise LibraryAssemblyError("grouped registry program list is invalid")
        matching = [p for p in programs if p.get("program_id") == program_id]
        if len(matching) != 1:
            raise LibraryAssemblyError("grouped program must resolve exactly once")
        spec = matching[0]
        if (
            set(spec)
            != {
                "program_id",
                "initial_role",
                "stages",
                "terminal_constraints",
                "product_constraints",
                "architecture_subfamily",
                "scope",
            }
            or not isinstance(spec["initial_role"], str)
            or not spec["initial_role"]
        ):
            raise LibraryAssemblyError("grouped program contract changed")
        stages = spec["stages"]
        if not isinstance(stages, list) or not stages or len(stages) > bounds.maximum_events:
            raise LibraryAssemblyError("invalid grouped stage count")
        roles, adapters, events = {spec["initial_role"]}, [], 0
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict) or set(stage) != {
                "reaction_id",
                "accumulator_role",
                "added_roles",
                "events",
                "net_byproducts_per_event",
                "source_step",
            }:
                raise LibraryAssemblyError("grouped stage contract changed")
            if any(
                not isinstance(stage[k], str) or not stage[k]
                for k in ("reaction_id", "accumulator_role", "source_step")
            ):
                raise LibraryAssemblyError("invalid grouped stage identifiers")
            added = stage["added_roles"]
            if (
                not isinstance(added, list)
                or not added
                or any(not isinstance(role, str) or not role for role in added)
                or len(set(added)) != len(added)
                or set(added) & roles
                or stage["accumulator_role"] in added
                or (index == 0 and stage["accumulator_role"] != spec["initial_role"])
            ):
                raise LibraryAssemblyError("grouped terminal roles or stage order changed")
            if type(stage["events"]) is not int or stage["events"] < 1:
                raise LibraryAssemblyError("grouped events must be positive integers")
            events += stage["events"]
            if events > bounds.maximum_events:
                raise LibraryAssemblyError("total grouped events exceed the program bound")
            byproducts = stage["net_byproducts_per_event"]
            if (
                not isinstance(byproducts, dict)
                or not byproducts
                or any(
                    not isinstance(k, str) or type(v) is not int or v < 0
                    for k, v in byproducts.items()
                )
            ):
                raise LibraryAssemblyError("explicit nonnegative event inventory is required")
            adapter = RegistryAssemblyAdapter.from_registry(
                path,
                reaction_id=stage["reaction_id"],
                expected_sha256=expected_sha256,
            )
            if set(adapter.roles) != {stage["accumulator_role"], *added}:
                raise LibraryAssemblyError("grouped stage roles differ from the reaction")
            roles.update(added)
            adapters.append(adapter)
        if (
            not isinstance(spec["terminal_constraints"], dict)
            or set(spec["terminal_constraints"]) != roles
        ):
            raise LibraryAssemblyError("grouped constraints must cover every terminal")
        for c in [*spec["terminal_constraints"].values(), spec["product_constraints"]]:
            _validate_constraints(c)
        return cls(spec, tuple(adapters), bounds)

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(sorted(self.specification["terminal_constraints"]))

    @property
    def quantities(self) -> dict[str, int]:
        quantities = {self.specification["initial_role"]: 1}
        for stage in self.specification["stages"]:
            quantities.update({role: stage["events"] for role in stage["added_roles"]})
        return dict(sorted(quantities.items()))

    def infer(self, product: str) -> dict[str, Any]:
        states: set[tuple[str, tuple[tuple[str, str], ...]]] = {
            (constitutional_molecule(product)[0], ())
        }
        layers, searches, reasons = [1], [], []
        total_states, transitions = 1, 0
        for index in reversed(range(len(self.adapters))):
            stage, adapter = self.specification["stages"][index], self.adapters[index]
            following = set()
            for state, assigned in sorted(states):
                result = infer_repeated_components(
                    adapter,
                    state,
                    accumulator_role=stage["accumulator_role"],
                    events=stage["events"],
                    bounds=self.bounds,
                )
                searches.append({"stage": index + 1, "product": state, "search": result})
                total_states += sum(result["states_by_depth"][1:])
                transitions += result["transitions"]
                if not result["complete_search"]:
                    reasons.extend(result["bound_reasons"])
                    break
                if (
                    total_states > self.bounds.maximum_states
                    or transitions > self.bounds.maximum_transitions
                ):
                    reasons.append("inverse_global_state_or_transition_bound")
                    break
                for candidate in result["candidate_components"]:
                    extended = tuple(
                        sorted(
                            (*assigned, *((role, candidate[role]) for role in stage["added_roles"]))
                        )
                    )
                    following.add((candidate[stage["accumulator_role"]], extended))
            states = following
            layers.append(len(states))
            if reasons:
                break
        candidates = (
            []
            if reasons
            else [
                dict(sorted(((self.specification["initial_role"], state), *assigned)))
                for state, assigned in sorted(states)
            ]
        )
        return {
            "complete_search": not reasons,
            "candidate_components": candidates,
            "states_by_stage": layers,
            "total_event_states": total_states,
            "transitions": transitions,
            "stage_searches": searches,
            "bound_reasons": reasons,
            "inverse_scope": "all_complete_tuples_with_one_repeated_identity_per_declared_stage_role",
        }

    def replay(self, components: dict[str, str], product: str) -> dict[str, Any]:
        if set(components) != set(self.roles):
            raise LibraryAssemblyError("grouped replay requires complete terminal roles")
        canonical = {role: constitutional_molecule(s)[0] for role, s in components.items()}
        target = constitutional_molecule(product)[0]
        stage_layers = [[canonical[self.specification["initial_role"]]]]
        event_searches, edges, balances, reasons = [], [], [], []
        total_states = 1
        for index, (stage, adapter) in enumerate(
            zip(self.specification["stages"], self.adapters, strict=True)
        ):
            side = {role: canonical[role] for role in stage["added_roles"]}
            event_layers = [stage_layers[-1]]
            for event in range(stage["events"]):
                following = set()
                for state in event_layers[-1]:
                    outcome = adapter.forward_products(
                        {stage["accumulator_role"]: state, **side},
                        maximum_outcomes=self.bounds.maximum_outcomes,
                    )
                    if outcome.saturated:
                        reasons.append("forward_outcome_bound")
                        break
                    for child in outcome.products:
                        following.add(child)
                        edges.append([index + 1, event + 1, state, child])
                        left = element_inventory(state)
                        for s in side.values():
                            left.update(element_inventory(s))
                        right = element_inventory(child)
                        right.update(stage["net_byproducts_per_event"])
                        balances.append(left == right)
                        if (
                            total_states + len(following) > self.bounds.maximum_states
                            or len(edges) > self.bounds.maximum_transitions
                        ):
                            reasons.append("forward_global_state_or_transition_bound")
                            break
                    if reasons:
                        break
                total_states += len(following)
                event_layers.append(sorted(following))
                if reasons:
                    break
            event_searches.append(
                {"stage": index + 1, "declared_events": stage["events"], "layers": event_layers}
            )
            stage_layers.append(event_layers[-1])
            if reasons:
                break
        inverse = self.infer(target)
        terminal_checks = {
            role: _constraints(canonical[role], spec, self.bounds.maximum_outcomes)
            for role, spec in self.specification["terminal_constraints"].items()
        }
        product_checks = _constraints(
            target, self.specification["product_constraints"], self.bounds.maximum_outcomes
        )
        left, right = Counter(), element_inventory(target)
        for role, quantity in self.quantities.items():
            for key, value in element_inventory(canonical[role]).items():
                left[key] += quantity * value
        for stage in self.specification["stages"]:
            for key, value in stage["net_byproducts_per_event"].items():
                right[key] += stage["events"] * value
        checks = {
            "complete_search": not reasons and inverse["complete_search"],
            "every_declared_event_replayed": len(event_searches) == len(self.adapters)
            and all(
                len(s["layers"]) == s["declared_events"] + 1 and all(s["layers"])
                for s in event_searches
            ),
            "unique_each_completed_source_stage": all(len(layer) == 1 for layer in stage_layers),
            "unique_forward_exact": stage_layers[-1] == [target],
            "unique_complete_inverse": inverse["candidate_components"]
            == [dict(sorted(canonical.items()))],
            "all_event_inventories_balance": bool(balances) and all(balances),
            "full_element_hydrogen_charge_balance": left == right,
            "terminal_constraints": all(v["pass"] for v in terminal_checks.values()),
            "product_constraints": product_checks["pass"],
        }
        return {
            "checks": checks,
            "computed_consistency_pass": all(checks.values()),
            "forward_layers": stage_layers,
            "event_searches": event_searches,
            "forward_edges": edges,
            "event_balance_checks": balances,
            "inverse": inverse,
            "bound_reasons": reasons,
            "terminal_constraints": terminal_checks,
            "product_constraints": product_checks,
            "balance": {
                "reactants": dict[str, Any](sorted(left.items())),
                "product_and_net_byproducts": dict[str, Any](sorted(right.items())),
            },
            "declared_quantities": self.quantities,
            "experimental_selectivity_qualified": False,
            "within_stage_event_order_qualified": False,
        }
