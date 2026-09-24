"""Input-only original-task binding and complete B5 source-profile qualification.

No product graph is consulted here. Original global identities and quantities survive;
the registry makes task reagent positions, generic roles and net source stages explicit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

from forge.assembly.component_scope import RegistryComponentScopes
from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import constitutional_molecule
from forge.core.hashing import sha256_file


def _stereo_identity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ComposeLipidError("Invalid raw source stereochemistry")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _body_axis_matches(assessment, rule):
    signatures = assessment["fragment_signatures"]
    if not assessment["pass"] or len(signatures) != 1:
        return False
    for smiles in signatures[0]["bodies"]:
        mol = Chem.MolFromSmiles(smiles)
        branches = sum(a.GetAtomicNum() == 6 and a.GetDegree() > 2 for a in mol.GetAtoms())
        doubles = sum(b.GetBondType() == Chem.BondType.DOUBLE for b in mol.GetBonds())
        if branches != rule["branches"] or doubles != rule["double_bonds"]:
            return False
    return True


@dataclass(frozen=True)
class B5SourceProfiles:
    specification: dict
    scopes: RegistryComponentScopes

    @classmethod
    def from_registry(cls, path: Path, *, expected_sha256: str):
        if sha256_file(path) != expected_sha256:
            raise ComposeLipidError("B5 profile registry checksum differs")
        registry = json.loads(path.read_text())
        scopes = RegistryComponentScopes(registry["component_scopes"])
        contract = registry["original_task_contract"]
        programs = {p["program_id"]: p for p in registry["programs"]}
        selectors = set()
        for name, profile in contract["profiles"].items():
            selector = tuple(sorted(profile["selector"].items()))
            if selector in selectors:
                raise ComposeLipidError("Ambiguous original-task profile selector")
            selectors.add(selector)
            program = programs[profile["program_id"]]
            roles = set(profile["component_scopes"])
            actual = {program["initial_role"]}
            for stage in program["stages"]:
                actual.update(stage["added_roles"])
            if (
                actual != roles
                or set(profile["registry_to_source_roles"]) != roles
                or len(set(profile["registry_to_source_roles"].values())) != len(roles)
                or set(r for r, _ in profile["reagent_order"]) != roles
                or len(profile["reagent_order"]) != len(roles)
            ):
                raise ComposeLipidError(f"Incomplete original-task role bijection: {name}")
            if any(s not in scopes.specifications for s in profile["component_scopes"].values()):
                raise ComposeLipidError(f"Undefined source component scope: {name}")
            if (
                set(profile["tail_roles"]) != set(profile["tail_rules"])
                or profile["head_role"] not in roles
                or profile["reported_head_scope"] not in scopes.specifications
            ):
                raise ComposeLipidError(f"Incomplete source axis binding: {name}")
            for rules in profile["tail_rules"].values():
                for rule in rules.values():
                    if rule.get("kind") == "reported" and set(rule) == {"kind", "smiles"}:
                        for smiles in rule["smiles"]:
                            constitutional_molecule(smiles)
                    elif rule.get("kind") == "body" and set(rule) == {
                        "kind",
                        "branches",
                        "double_bonds",
                    }:
                        if any(
                            type(rule[k]) is not int or rule[k] < 0
                            for k in ("branches", "double_bonds")
                        ):
                            raise ComposeLipidError("Invalid original-task hydrophobe class")
                    else:
                        raise ComposeLipidError("Unknown original-task tail rule")
        return cls(contract, scopes)

    def assess(
        self,
        prepared: dict,
        construction: dict,
        metadata: dict,
        task: dict,
        structures: dict[str, str],
    ) -> dict:
        if prepared.get("eligible_for_program_preparation") is not True:
            raise ComposeLipidError("Protected or unassigned record reached B5 component binding")
        spec = self.specification
        matching = [
            (name, p)
            for name, p in spec["profiles"].items()
            if all(task.get(k) == v for k, v in p["selector"].items())
        ]
        output = {
            "profile_qualified": False,
            "training_admitted": False,
            "experimental_execution_admitted": False,
            "checks": {},
            "components": {},
        }
        if len(matching) != 1:
            output["disposition"] = "unqualified_source_profile"
            return output
        name, profile = matching[0]
        output.update(profile_id=name, program_id=profile["program_id"])
        checks = output["checks"]
        checks["source_task_identity"] = task.get("family") == prepared["family"] == spec[
            "family"
        ] and task.get("task_id") == construction.get("task_id")
        checks["source_task_contract"] = (
            task.get("schema") == spec["schema"] and task.get("training_admissible") is False
        )
        checks["source_metadata_agrees"] = all(
            metadata.get(k) == task.get(k) and k in task for k in spec["metadata_fields"]
        )
        checks["source_events_agree"] = task.get("events") == profile["events"]
        checks["source_event_export_agrees"] = construction.get("reaction_steps_sites", {}).get(
            "task"
        ) == {
            "series": task.get("series"),
            "design_lane": task.get("design_lane"),
            "events": task.get("events"),
        }
        source_roles = profile["registry_to_source_roles"]
        instances = prepared["component_instances"]
        for role, identity, quantity in instances:
            if identity not in structures or type(quantity) is not int or quantity < 1:
                raise ComposeLipidError("Unresolved global component identity or invalid quantity")
        checks["complete_source_roles_quantities"] = (
            len(instances) == len(source_roles)
            and set(r for r, _, _ in instances) == set(source_roles.values())
            and all(q == 1 for _, _, q in instances)
        )
        if not checks["complete_source_roles_quantities"]:
            output["disposition"] = "source_component_roles_or_quantities_disagree"
            return output
        ids = {role: identity for role, identity, _ in instances}
        components = {
            role: constitutional_molecule(structures[ids[alias]])[0]
            for role, alias in source_roles.items()
        }
        output["components"] = components
        output["source_component_instances"] = instances
        reagents = task.get("reagents", [])
        checks["complete_original_component_structure_agreement"] = len(reagents) == len(
            profile["reagent_order"]
        )
        raw_core = None
        if checks["complete_original_component_structure_agreement"]:
            for reagent, (role, generic_role) in zip(
                reagents, profile["reagent_order"], strict=True
            ):
                agrees = (
                    reagent.get("role") == generic_role
                    and constitutional_molecule(reagent.get("smiles"))[0] == components[role]
                )
                checks["complete_original_component_structure_agreement"] &= agrees
                if role == spec["core_role"]:
                    raw_core = reagent.get("smiles")
        checks["raw_source_core_stereochemistry"] = raw_core is not None and _stereo_identity(
            raw_core
        ) == _stereo_identity(spec["source_core_smiles"])
        alias = spec["source_head_alias"]
        checks["source_head_alias_binding"] = construction.get(
            "declared_precursor_identifiers"
        ) == [
            {
                **alias,
                "value": task.get("head_id"),
                "scoped_alias": spec["family"]
                + "|"
                + alias["field"]
                + "|"
                + str(task.get("head_id")),
            }
        ]
        assessed = {
            role: self.scopes.assess(scope, components[role])
            for role, scope in profile["component_scopes"].items()
        }
        checks["complete_component_scope"] = all(
            a["pass"] and a["complete_search"] for a in assessed.values()
        )
        output["component_scope"] = assessed
        head_axis = task.get("head_axis")
        if head_axis == "reported_head_" + profile["head_kind"]:
            head = self.scopes.assess(
                profile["reported_head_scope"], components[profile["head_role"]]
            )
            checks["head_axis_identity"] = head["pass"]
            output["reported_head_scope"] = head
        else:
            checks["head_axis_identity"] = head_axis == "same_head_scaffold_spacer"
        tail_axis = task.get("tail_axis")
        axes = tail_axis.split("__") if isinstance(tail_axis, str) else []
        tail_checks, reported_count = {}, 0
        checks["tail_axis_role_coverage"] = len(axes) == len(profile["tail_roles"])
        if checks["tail_axis_role_coverage"]:
            for role, axis in zip(profile["tail_roles"], axes, strict=True):
                rule = profile["tail_rules"][role].get(axis)
                if rule is None:
                    tail_checks[role] = False
                elif rule["kind"] == "reported":
                    reported_count += 1
                    tail_checks[role] = components[role] in {
                        constitutional_molecule(s)[0] for s in rule["smiles"]
                    }
                else:
                    tail_checks[role] = _body_axis_matches(assessed[role], rule)
        checks["tail_axis_identity"] = checks["tail_axis_role_coverage"] and all(
            tail_checks.values()
        )
        checks["declared_fixed_tail_count"] = reported_count >= profile["minimum_reported_tails"]
        output["tail_axis_checks"] = tail_checks
        group = profile["equal_body_roles"]
        checks["matched_body_identity"] = not group or (
            all(assessed[r]["pass"] and len(assessed[r]["fragment_signatures"]) == 1 for r in group)
            and len({tuple(assessed[r]["fragment_signatures"][0]["bodies"]) for r in group}) == 1
        )
        group = profile["equal_component_roles"]
        checks["matched_full_component_identity"] = (
            not group or len({components[r] for r in group}) == 1
        )
        output["profile_qualified"] = all(checks.values())
        output["disposition"] = (
            "original_recipe_and_complete_scope_qualified"
            if output["profile_qualified"]
            else "source_recipe_or_complete_scope_disagreement"
        )
        return output
