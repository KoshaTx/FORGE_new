"""Reproduce source split keys and project immutable groups before data admission.

This is a partition audit. Descriptor clipping reproduces the source grouping and
never truncates a molecular graph. Reaction decomposition and model training are absent.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import resolve_pin, sha256_file


@dataclass(frozen=True)
class SourceSplitSignatures:
    policy: dict

    @classmethod
    def from_registry(cls, repo: Path, path: Path, *, expected_sha256: str):
        if sha256_file(path) != expected_sha256:
            raise ComposeLipidError("Source split policy checksum differs")
        policy = json.loads(path.read_text())
        if (
            policy.get("schema_version") != "forge.compose_lipid_frozen_split_policy.v1"
            or policy.get("training_admitted") is not False
        ):
            raise ComposeLipidError("Source split policy contract differs")
        for key in ("source_recovery", "derivation", "source_config"):
            resolve_pin(policy[key], repo, label=key)
        source = json.loads(
            resolve_pin(policy["source_recovery"], repo, label="recovered source").read_text()
        )
        for value in source["assets"]:
            resolve_pin(
                {k: value[k] for k in ("path", "sha256")}, repo, label=value["upstream_path"]
            )
        if policy["digest"] != {
            "algorithm": "sha256",
            "sort_keys": True,
            "ensure_ascii": True,
            "separators": [",", ":"],
        }:
            raise ComposeLipidError("Unsupported source split digest")
        return cls(policy)

    def digest(self, value) -> str:
        raw = json.dumps(
            value,
            sort_keys=self.policy["digest"]["sort_keys"],
            ensure_ascii=self.policy["digest"]["ensure_ascii"],
            separators=tuple(self.policy["digest"]["separators"]),
            allow_nan=False,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def profile(self, smiles: str) -> dict:
        if not isinstance(smiles, str) or not smiles:
            raise ComposeLipidError("Source split target must be a nonempty molecular graph")
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ComposeLipidError("Source split target does not parse")
        elements = self.policy["elements"]
        atoms = tuple(molecule.GetAtoms())
        carbons = [a for a in atoms if a.GetAtomicNum() == elements["carbon"]]
        counts = Counter(a.GetAtomicNum() for a in atoms)
        features = {name: counts[number] for name, number in elements.items()}
        features.update(
            heavy_atoms=molecule.GetNumHeavyAtoms(),
            formal_charge=Chem.GetFormalCharge(molecule),
            rings=rdMolDescriptors.CalcNumRings(molecule),
            carbon_branch_points=sum(
                sum(n.GetAtomicNum() == elements["carbon"] for n in a.GetNeighbors())
                >= self.policy["carbon_branch_minimum_carbon_neighbors"]
                for a in carbons
            ),
            cc_double_bonds=sum(
                b.GetBondType() == Chem.BondType.DOUBLE
                and b.GetBeginAtom().GetAtomicNum() == elements["carbon"]
                and b.GetEndAtom().GetAtomicNum() == elements["carbon"]
                for b in molecule.GetBonds()
            ),
        )
        result = {}
        for field, rule in self.policy["profile_fields"].items():
            value = features[rule["feature"]]
            if "divisor" in rule:
                value //= rule["divisor"]
            if "maximum" in rule:
                value = min(rule["maximum"], value)
            result[field] = value
        return result

    def describe(
        self, *, family: str, smiles: str, metadata: dict, source_anchor: bool, instances: list
    ) -> dict:
        if (
            not isinstance(family, str)
            or not family
            or type(source_anchor) is not bool
            or not isinstance(metadata, dict)
        ):
            raise ComposeLipidError("Invalid source split descriptor context")
        expanded, seen = [], set()
        for role, identity, quantity in instances:
            if (
                not isinstance(role, str)
                or not role
                or not isinstance(identity, str)
                or not identity
                or type(quantity) is not int
                or quantity < 1
                or (role, identity) in seen
            ):
                raise ComposeLipidError("Invalid source split component inventory")
            seen.add((role, identity))
            expanded.extend((role, identity) for _ in range(quantity))
        if not expanded:
            raise ComposeLipidError("Missing complete split components")
        context = (
            dict(self.policy["source_anchor_context"])
            if source_anchor
            else {
                k: v
                for k, v in sorted(metadata.items())
                if k in self.policy["metadata_fields"] or k.endswith(self.policy["metadata_suffix"])
            }
        )
        labels = self.policy["signature_labels"]
        profile = self.profile(smiles)
        core = self.digest([labels["core"], family])
        regional = self.digest(
            [labels["regional"], family, context, profile, Counter(r for r, _ in expanded)]
        )
        return {
            "core_scaffold_signature": core,
            "regional_morphology_signature": regional,
            "morphology_group_signature": self.digest(
                [labels["morphology"], family, core, regional]
            ),
            "combination_signature": self.digest([labels["combination"], family, sorted(expanded)]),
            "regional_profile": profile,
            "morphology_context": context,
        }


@dataclass(frozen=True)
class FrozenGroupProjection:
    formal_families: frozenset
    reference_families: frozenset
    components: frozenset
    studies: frozenset
    test_morphologies: frozenset
    test_combinations: frozenset
    calibration_morphologies: frozenset
    calibration_combinations: frozenset

    @classmethod
    def from_selected_groups(cls, selected: dict):
        formal = frozenset(selected["formal_evaluation_families"])
        reference = frozenset(selected["reference_only_families"])
        if not formal or formal & reference:
            raise ComposeLipidError("Overlapping or empty source split family sets")

        def pairs(name):
            values = selected[name]
            result = frozenset(tuple(v) for v in values)
            if any(len(v) != 2 or v[0] not in formal for v in result) or len(result) != len(values):
                raise ComposeLipidError("Invalid frozen split groups: " + name)
            return result

        result = cls(
            formal,
            reference,
            frozenset(selected["selected_components"]),
            frozenset(selected["selected_source_studies"]),
            pairs("selected_morphology_groups"),
            pairs("selected_combination_groups"),
            pairs("selected_calibration_morphology_groups"),
            pairs("selected_calibration_combination_groups"),
        )
        if (
            result.test_morphologies & result.calibration_morphologies
            or result.test_combinations & result.calibration_combinations
        ):
            raise ComposeLipidError("Frozen test and calibration groups overlap")
        return result

    def project(
        self,
        *,
        family: str,
        component_ids: list[str],
        source_pmids: list,
        morphology: str,
        combination: str,
    ) -> dict:
        if (
            not isinstance(component_ids, list)
            or not component_ids
            or any(not isinstance(c, str) or not c for c in component_ids)
            or not isinstance(source_pmids, list)
            or not isinstance(morphology, str)
            or not morphology
            or not isinstance(combination, str)
            or not combination
        ):
            raise ComposeLipidError("Incomplete frozen-group projection inputs")
        if family in self.reference_families:
            return {
                "split": "reference",
                "test_panels": [],
                "calibration_groups": [],
                "training_admitted": False,
            }
        if family not in self.formal_families:
            raise ComposeLipidError("Unknown source split family")
        studies = set()
        for value in source_pmids:
            pmid = str(value).removeprefix("pmid:")
            if (
                type(value) not in (str, int)
                or not pmid.isascii()
                or not pmid.isdigit()
                or int(pmid) <= 0
            ):
                raise ComposeLipidError("Invalid source study identity in frozen-group projection")
            studies.add("pmid:" + pmid)
        tests = {
            "source_study_transfer_global": bool(studies & self.studies),
            "unseen_component_structure": bool(set(component_ids) & self.components),
            "unseen_regional_morphology": (family, morphology) in self.test_morphologies,
            "unseen_exact_component_combination": (family, combination) in self.test_combinations,
        }
        panels = sorted(k for k, hit in tests.items() if hit)
        calibrations = sorted(
            k
            for k, hit in {
                "regional_morphology": (family, morphology) in self.calibration_morphologies,
                "exact_component_combination": (family, combination)
                in self.calibration_combinations,
            }.items()
            if hit
        )
        return {
            "split": "test" if panels else "calibration" if calibrations else "train",
            "test_panels": panels,
            "calibration_groups": calibrations,
            "training_admitted": False,
        }
