"""Exact ordered, multi-component stages from a pinned source registry.

Every stage retains every constitutional outcome. Complete inverse tuples and
all intermediate branches remain observable; bounds or ambiguity cannot pass.
This executor establishes computed consistency, never experimental selectivity.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

from forge.assembly.families import (
    LibraryAssemblyError,
    RegistryAssemblyAdapter,
    constitutional_molecule,
)
from forge.assembly.repeated_components import RepeatBounds, element_inventory
from forge.core.hashing import sha256_file


def _validate_constraints(specification: dict) -> None:
    supported = {
        "allowed_atomic_numbers",
        "exact_element_counts",
        "allow_aromatic_atoms",
        "formal_charge",
        "required_queries",
    }
    if not isinstance(specification, dict) or set(specification) - supported:
        raise LibraryAssemblyError("staged component constraint schema changed")
    if "allowed_atomic_numbers" in specification and (
        not isinstance(specification["allowed_atomic_numbers"], list)
        or not specification["allowed_atomic_numbers"]
        or any(
            type(n) is not int or not 1 <= n <= 118 for n in specification["allowed_atomic_numbers"]
        )
    ):
        raise LibraryAssemblyError("invalid allowed atomic numbers")
    if "formal_charge" in specification and type(specification["formal_charge"]) is not int:
        raise LibraryAssemblyError("invalid formal charge constraint")
    if (
        "allow_aromatic_atoms" in specification
        and type(specification["allow_aromatic_atoms"]) is not bool
    ):
        raise LibraryAssemblyError("invalid aromaticity constraint")
    exact = specification.get("exact_element_counts", {})
    if not isinstance(exact, dict) or any(
        not isinstance(k, str) or type(v) is not int or v < 0 for k, v in exact.items()
    ):
        raise LibraryAssemblyError("invalid exact element counts")
    queries = specification.get("required_queries", [])
    if not isinstance(queries, list):
        raise LibraryAssemblyError("invalid required query list")
    seen = set()
    for q in queries:
        if (
            not isinstance(q, dict)
            or set(q) != {"name", "smarts", "minimum_matches"}
            or not isinstance(q["name"], str)
            or not q["name"]
            or q["name"] in seen
            or not isinstance(q["smarts"], str)
            or Chem.MolFromSmarts(q["smarts"]) is None
            or type(q["minimum_matches"]) is not int
            or q["minimum_matches"] < 1
        ):
            raise LibraryAssemblyError("invalid required query constraint")
        seen.add(q["name"])


def _constraints(smiles: str, specification: dict, maximum_matches: int) -> dict:
    _, molecule = constitutional_molecule(smiles)
    checks = {}
    if "allowed_atomic_numbers" in specification:
        checks["elements"] = all(
            a.GetAtomicNum() in specification["allowed_atomic_numbers"] for a in molecule.GetAtoms()
        )
    if "formal_charge" in specification:
        checks["formal_charge"] = Chem.GetFormalCharge(molecule) == specification["formal_charge"]
    if "allow_aromatic_atoms" in specification:
        checks["aromaticity"] = specification["allow_aromatic_atoms"] or not any(
            a.GetIsAromatic() for a in molecule.GetAtoms()
        )
    inventory = element_inventory(smiles)
    for element, count in specification.get("exact_element_counts", {}).items():
        checks["element_count_" + element] = inventory[element] == count
    query_counts = {}
    for q in specification.get("required_queries", []):
        count = len(
            molecule.GetSubstructMatches(
                Chem.MolFromSmarts(q["smarts"]), maxMatches=maximum_matches
            )
        )
        query_counts[q["name"]] = count
        checks["query_" + q["name"]] = q["minimum_matches"] <= count < maximum_matches
    return {"checks": checks, "query_counts": query_counts, "pass": all(checks.values())}


@dataclass(frozen=True)
class RegistryStagedProgram:
    specification: dict
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
    ) -> RegistryStagedProgram:
        if str(sha256_file(path)) != expected_sha256:
            raise LibraryAssemblyError("staged registry hash mismatch")
        registry = json.loads(path.read_text())
        if not isinstance(registry, dict) or not isinstance(registry.get("programs"), list):
            raise LibraryAssemblyError("staged registry has no program list")
        if any(not isinstance(p, dict) for p in registry["programs"]):
            raise LibraryAssemblyError("staged registry program is not a record")
        programs = [p for p in registry.get("programs", []) if p.get("program_id") == program_id]
        if len(programs) != 1:
            raise LibraryAssemblyError("staged program must resolve exactly once")
        spec = programs[0]
        if set(spec) != {
            "program_id",
            "initial_role",
            "stages",
            "terminal_constraints",
            "product_constraints",
            "architecture_subfamily",
            "scope",
            "equal_component_groups",
        }:
            raise LibraryAssemblyError("staged program contract changed")
        stages = spec["stages"]
        if not isinstance(stages, list) or not 1 <= len(stages) <= bounds.maximum_events:
            raise LibraryAssemblyError("staged stage count exceeds declared bound")
        if not isinstance(spec["initial_role"], str) or not spec["initial_role"]:
            raise LibraryAssemblyError("staged initial role is missing")
        roles, adapters = {spec["initial_role"]}, []
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict) or set(stage) != {
                "reaction_id",
                "accumulator_role",
                "added_roles",
                "net_byproducts",
                "source_step",
            }:
                raise LibraryAssemblyError("staged stage contract changed")
            if any(
                not isinstance(stage[name], str) or not stage[name]
                for name in ("reaction_id", "accumulator_role", "source_step")
            ):
                raise LibraryAssemblyError("staged stage identifiers are invalid")
            added = stage["added_roles"]
            if (
                not isinstance(added, list)
                or not added
                or any(not isinstance(role, str) or not role for role in added)
                or len(set(added)) != len(added)
            ):
                raise LibraryAssemblyError("staged added_roles must be nonempty and distinct")
            if (
                set(added) & roles
                or stage["accumulator_role"] in added
                or (index == 0 and stage["accumulator_role"] != spec["initial_role"])
            ):
                raise LibraryAssemblyError("staged stage order or distinct terminal roles changed")
            byproducts = stage["net_byproducts"]
            if (
                not isinstance(byproducts, dict)
                or not byproducts
                or any(
                    not isinstance(k, str) or type(v) is not int or v < 0
                    for k, v in byproducts.items()
                )
            ):
                raise LibraryAssemblyError(
                    "staged stage requires an explicit nonnegative net inventory"
                )
            adapter = RegistryAssemblyAdapter.from_registry(
                path, reaction_id=stage["reaction_id"], expected_sha256=expected_sha256
            )
            if set(adapter.roles) != {stage["accumulator_role"], *added} or len(
                adapter.roles
            ) != 1 + len(added):
                raise LibraryAssemblyError("staged stage roles differ from registry")
            roles.update(added)
            adapters.append(adapter)
        if (
            not isinstance(spec["terminal_constraints"], dict)
            or set(spec["terminal_constraints"]) != roles
        ):
            raise LibraryAssemblyError("staged terminal constraints do not cover every role")
        for constraint in [*spec["terminal_constraints"].values(), spec["product_constraints"]]:
            _validate_constraints(constraint)
        groups = spec["equal_component_groups"]
        if not isinstance(groups, list) or any(
            not isinstance(group, list)
            or len(group) < 2
            or any(not isinstance(role, str) or role not in roles for role in group)
            or len(set(group)) != len(group)
            for group in groups
        ):
            raise LibraryAssemblyError("staged equality groups must reference distinct terminals")
        if len({r for group in groups for r in group}) != sum(map(len, groups)):
            raise LibraryAssemblyError("staged equality groups cannot overlap")
        return cls(spec, tuple(adapters), bounds)

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(sorted(self.specification["terminal_constraints"]))

    def infer(self, product: str) -> dict:
        target = constitutional_molecule(product)[0]
        states = {(target, ())}
        layers, transitions, reasons = [1], 0, []
        for stage, adapter in reversed(
            tuple(zip(self.specification["stages"], self.adapters, strict=True))
        ):
            following = set()
            for state, assigned in sorted(states):
                try:
                    inverse = adapter.decompose(
                        state, maximum_outcomes=self.bounds.maximum_outcomes
                    )
                except LibraryAssemblyError as exc:
                    if "saturated" not in str(exc):
                        raise
                    reasons.append("inverse_outcome_bound")
                    break
                for candidate in inverse:
                    components = dict(candidate.components)
                    extended = tuple(
                        sorted(
                            (
                                *assigned,
                                *((role, components[role]) for role in stage["added_roles"]),
                            )
                        )
                    )
                    following.add((components[stage["accumulator_role"]], extended))
                    transitions += 1
                    if (
                        sum(layers) + len(following) > self.bounds.maximum_states
                        or transitions > self.bounds.maximum_transitions
                    ):
                        reasons.append("inverse_state_or_transition_bound")
                        break
                if reasons:
                    break
            layers.append(len(following))
            states = following
            if reasons:
                break
        candidates = (
            []
            if reasons
            else [
                dict(sorted(((self.specification["initial_role"], head), *assigned)))
                for head, assigned in sorted(states)
            ]
        )
        return {
            "complete_search": not reasons,
            "candidate_components": candidates,
            "states_by_depth": layers,
            "transitions": transitions,
            "bound_reasons": reasons,
        }

    def replay(self, components: dict[str, str], product: str) -> dict:
        if set(components) != set(self.roles):
            raise LibraryAssemblyError("staged replay requires all terminal roles")
        canonical = {r: constitutional_molecule(s)[0] for r, s in components.items()}
        target = constitutional_molecule(product)[0]
        layers = [[canonical[self.specification["initial_role"]]]]
        edges, balance_checks, reasons = [], [], []
        terminal_checks = {
            r: _constraints(canonical[r], spec, self.bounds.maximum_outcomes)
            for r, spec in self.specification["terminal_constraints"].items()
        }
        for depth, (stage, adapter) in enumerate(
            zip(self.specification["stages"], self.adapters, strict=True)
        ):
            following = set()
            for state in layers[-1]:
                sides = {role: canonical[role] for role in stage["added_roles"]}
                outcome = adapter.forward_products(
                    {stage["accumulator_role"]: state, **sides},
                    maximum_outcomes=self.bounds.maximum_outcomes,
                )
                if outcome.saturated:
                    reasons.append("forward_outcome_bound")
                    break
                for child in outcome.products:
                    following.add(child)
                    edges.append([depth + 1, state, child])
                    left = element_inventory(state)
                    for side in sides.values():
                        left.update(element_inventory(side))
                    right = element_inventory(child)
                    right.update(stage["net_byproducts"])
                    balance_checks.append(left == right)
                    if (
                        sum(map(len, layers)) + len(following) > self.bounds.maximum_states
                        or len(edges) > self.bounds.maximum_transitions
                    ):
                        reasons.append("forward_state_or_transition_bound")
                        break
                if reasons:
                    break
            layers.append(sorted(following))
            if reasons:
                break
        inverse = self.infer(target)
        left = Counter()
        for smiles in canonical.values():
            left.update(element_inventory(smiles))
        right = element_inventory(target)
        for stage in self.specification["stages"]:
            right.update(stage["net_byproducts"])
        product_checks = _constraints(
            target, self.specification["product_constraints"], self.bounds.maximum_outcomes
        )
        checks = {
            "declared_component_equalities": all(
                len({canonical[role] for role in group}) == 1
                for group in self.specification["equal_component_groups"]
            ),
            "complete_search": not reasons and inverse["complete_search"],
            "every_stage_replayed": len(layers) == len(self.adapters) + 1 and all(layers),
            "unique_each_forward_stage": all(len(layer) == 1 for layer in layers),
            "unique_forward_exact": layers[-1] == [target],
            "unique_complete_inverse": inverse["candidate_components"]
            == [dict(sorted(canonical.items()))],
            "all_stage_inventories_balance": bool(balance_checks) and all(balance_checks),
            "full_element_hydrogen_charge_balance": left == right,
            "terminal_constraints": all(c["pass"] for c in terminal_checks.values()),
            "product_constraints": product_checks["pass"],
        }
        return {
            "checks": checks,
            "computed_consistency_pass": all(checks.values()),
            "forward_layers": layers,
            "forward_edges": edges,
            "stage_balance_checks": balance_checks,
            "inverse": inverse,
            "terminal_constraints": terminal_checks,
            "product_constraints": product_checks,
            "bound_reasons": reasons,
            "balance": {
                "reactants": dict(sorted(left.items())),
                "product_and_net_byproducts": dict(sorted(right.items())),
            },
            "experimental_selectivity_qualified": False,
        }
