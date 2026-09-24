"""Reproduce older partition descriptors without reinterpreting chemical components."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import resolve_pin, sha256_file
from forge.corpus.compose_lipid_partition_signatures import SourceSplitSignatures


@dataclass(frozen=True)
class HistoricalMorphology:
    policy: dict
    corrected: SourceSplitSignatures

    @classmethod
    def from_registry(cls, repo: Path, path: Path, *, expected_sha256: str):
        if sha256_file(path) != expected_sha256:
            raise ComposeLipidError("Historical morphology policy checksum differs")
        policy = json.loads(path.read_text())
        if (
            policy.get("schema_version") != "forge.compose_lipid_historical_morphology_policy.v1"
            or policy.get("training_admitted") is not False
        ):
            raise ComposeLipidError("Historical morphology policy scope differs")
        for key in ("derivation", "source_tree"):
            resolve_pin(policy[key], repo, label=key)
        for name, asset in policy["source_assets"].items():
            resolve_pin(asset["file"], repo, label=name)
            resolve_pin(asset["response"], repo, label=name + " response")
        corrected_path = resolve_pin(policy["corrected_policy"], repo, label="corrected policy")
        corrected = SourceSplitSignatures.from_registry(
            repo, corrected_path, expected_sha256=policy["corrected_policy"]["sha256"]
        )
        return cls(policy, corrected)

    def describe(
        self,
        *,
        family: str,
        metadata: dict,
        source_anchor: bool,
        instances: list,
        corrected_description: dict,
    ) -> dict:
        if type(source_anchor) is not bool or not isinstance(metadata, dict):
            raise ComposeLipidError("Invalid historical morphology context")
        roles = Counter()
        if source_anchor:
            # This assertion is qualified against all original anchor signatures before use.
            for role, _, quantity in instances:
                if (
                    not isinstance(role, str)
                    or not role
                    or type(quantity) is not int
                    or quantity < 1
                ):
                    raise ComposeLipidError("Invalid historical source role inventory")
                roles[role] += quantity
        else:
            if family not in self.policy["virtual_role_descriptors"]:
                raise ComposeLipidError("Missing historical family morphology contract")
            for spec in self.policy["virtual_role_descriptors"][family]:
                rule = spec["quantity"]
                quantity = (
                    rule.get("constant") if "constant" in rule else metadata.get(rule["metadata"])
                )
                if type(quantity) is not int or quantity < 1:
                    raise ComposeLipidError("Missing positive historical morphology multiplicity")
                roles[spec["role"]] += quantity
        if not roles:
            raise ComposeLipidError("Empty historical morphology role inventory")
        labels = self.policy["signature_labels"]
        core = self.corrected.digest([labels["core"], family])
        regional = self.corrected.digest(
            [
                labels["regional"],
                family,
                corrected_description["morphology_context"],
                corrected_description["regional_profile"],
                roles,
            ]
        )
        return {
            "core_scaffold_signature": core,
            "regional_topology_signature": regional,
            "structural_group_signature": self.corrected.digest(
                [labels["structural"], family, core, regional]
            ),
        }
