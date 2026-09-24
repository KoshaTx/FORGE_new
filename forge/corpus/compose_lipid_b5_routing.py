"""Select an existing B5 source program using a complete input precursor pair.

Original task metadata is never rewritten. This narrow overlay repairs a generation
lane/program conflation; every precursor and mechanical gate remains mandatory.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import constitutional_molecule
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_b5_profiles import B5SourceProfiles


@dataclass(frozen=True)
class B5SourcePairProfiles:
    base: B5SourceProfiles
    alternatives: tuple

    @property
    def specification(self):
        return self.base.specification

    @property
    def scopes(self):
        return self.base.scopes

    @classmethod
    def from_registry(cls, repo: Path, path: Path, *, expected_sha256: str):
        if sha256_file(path) != expected_sha256:
            raise ComposeLipidError("B5 source-pair routing checksum differs")
        overlay = json.loads(path.read_text())
        if (
            overlay.get("schema_version") != "forge.b5_source_pair_routing.v2"
            or overlay.get("training_admitted") is not False
        ):
            raise ComposeLipidError("B5 source-pair routing contract differs")
        parent = resolve_pin(overlay["parent_registry"], repo, label="B5 parent registry")
        resolve_pin(overlay["derivation"], repo, label="B5 routing derivation")
        for key, value in overlay["source_assets"].items():
            resolve_pin(value, repo, label=key)
        base = B5SourceProfiles.from_registry(
            parent, expected_sha256=overlay["parent_registry"]["sha256"]
        )
        alternatives, selectors, identifiers = [], set(), set()
        for rule in overlay["routes"]:
            original = base.specification["profiles"][rule["original_profile"]]
            source = base.specification["profiles"][rule["source_profile"]]
            unchanged = set(source) - {
                "selector",
                "profile",
                "program_id",
                "tail_rules",
                "minimum_reported_tails",
                "equal_component_roles",
            }
            if (
                any(source[k] != original[k] for k in unchanged)
                or source["selector"]["series"] != original["selector"]["series"]
                or set(rule["required_components"]) != set(source["equal_component_roles"])
                or len(rule["required_components"]) != 2
                or rule["minimum_reported_tails"] != 2
                or rule["task_fields"] != {"tail_axis": "reported_tail_acid__reported_tail_acid"}
            ):
                raise ComposeLipidError("B5 routing changes more than the exact source tail pair")
            required = {
                role: constitutional_molecule(smiles)[0]
                for role, smiles in rule["required_components"].items()
            }
            if len(set(required.values())) != 1 or any(
                [smiles]
                != [
                    constitutional_molecule(s)[0]
                    for s in source["tail_rules"][role]["reported_tail_acid"]["smiles"]
                ]
                for role, smiles in required.items()
            ):
                raise ComposeLipidError("B5 routing pair is not the exact existing source pair")
            selector = (rule["original_profile"], tuple(sorted(required.items())))
            if selector in selectors or rule["route_id"] in identifiers:
                raise ComposeLipidError("Ambiguous B5 source-pair routing")
            selectors.add(selector)
            identifiers.add(rule["route_id"])
            profile = copy.deepcopy(source)
            # This is a new registry profile, not an edit to the original task.
            profile["selector"] = original["selector"]
            profile["minimum_reported_tails"] = rule["minimum_reported_tails"]
            contract = {**base.specification, "profiles": {rule["route_id"]: profile}}
            alternatives.append((rule, required, B5SourceProfiles(contract, base.scopes)))
        if not alternatives:
            raise ComposeLipidError("B5 source-pair routing has no source profiles")
        return cls(base, tuple(alternatives))

    def assess(self, prepared, construction, metadata, task, structures):
        baseline = self.base.assess(prepared, construction, metadata, task, structures)
        if baseline["profile_qualified"] or {
            name for name, passed in baseline["checks"].items() if not passed
        } != {"tail_axis_identity"}:
            return baseline
        matches = [
            (rule, alternate)
            for rule, required, alternate in self.alternatives
            if baseline.get("profile_id") == rule["original_profile"]
            and all(task.get(k) == v for k, v in rule["task_fields"].items())
            and all(baseline["components"].get(k) == v for k, v in required.items())
        ]
        if not matches:
            return baseline
        if len(matches) != 1:
            raise ComposeLipidError("Ambiguous B5 input-only source-pair selection")
        rule, alternate = matches[0]
        result = alternate.assess(prepared, construction, metadata, task, structures)
        result["source_pair_routing"] = {
            "route_id": rule["route_id"],
            "original_profile": rule["original_profile"],
            "source_profile": rule["source_profile"],
            "baseline_binding": baseline,
            "selection_basis": "complete_input_pair_and_original_task_fields",
            "target_used_for_selection": False,
        }
        return result
