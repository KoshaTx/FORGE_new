"""Complete precursor grammars loaded from a registry, independent of product matching.

Internal cuts certify scope only: they never create precursor identities or change quantities.
Root-labelled fragments preserve the attachment position when comparing hydrophobic bodies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem

from forge.assembly.families import LibraryAssemblyError, constitutional_molecule
from forge.core.hashing import sha256_file


def _positive(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise LibraryAssemblyError(f"{name} must be a positive integer")
    return value


def _canonical(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _template_matches(
    canonical: str, atoms: int, specification: dict[str, Any]
) -> list[dict[str, Any]]:
    """Every candidate is a complete graph, never a substructure acceptance test."""
    found = []
    for index, template in enumerate(specification["templates"]):
        counts = specification.get("repeat_counts")
        if counts is None:
            counts = range(specification["minimum_repeats"], atoms + 1)
        for count in counts:
            candidate = template.replace("{spacer}", specification["repeat_smiles"] * count)
            mol = Chem.MolFromSmiles(candidate)
            if mol is None:
                raise LibraryAssemblyError("Invalid complete-scope template expansion")
            Chem.RemoveStereochemistry(mol)
            if _canonical(mol) == canonical:
                found.append({"template_index": index, "repeat_count": count})
    return found


def _validate_template(spec: dict[str, Any]) -> None:
    expected = {"templates", "repeat_smiles", "minimum_repeats"}
    if set(spec) not in (expected, expected | {"repeat_counts"}):
        raise LibraryAssemblyError("Complete template scope fields changed")
    _positive(spec["minimum_repeats"], "minimum_repeats")
    if not isinstance(spec["repeat_smiles"], str) or not spec["repeat_smiles"]:
        raise LibraryAssemblyError("Missing scope repeat unit")
    if not isinstance(spec["templates"], list) or not spec["templates"]:
        raise LibraryAssemblyError("Missing complete scope templates")
    for template in spec["templates"]:
        if not isinstance(template, str) or template.count("{spacer}") != 1:
            raise LibraryAssemblyError("Scope template needs exactly one spacer placeholder")
    if "repeat_counts" in spec:
        if not isinstance(spec["repeat_counts"], list) or not spec["repeat_counts"]:
            raise LibraryAssemblyError("Missing fixed repeat counts")
        for n in spec["repeat_counts"]:
            if _positive(n, "repeat_counts") < spec["minimum_repeats"]:
                raise LibraryAssemblyError("Fixed repeat count below minimum")
    # Eagerly validate syntax; component size is not a corpus-size filter.
    _template_matches("", spec["minimum_repeats"], spec)


def _body_ok(mol: Chem.Mol, spec: dict[str, Any]) -> bool:
    atoms = list(mol.GetAtoms())
    roots = [a for a in atoms if a.GetAtomicNum() == 0]
    if len(roots) != 1 or roots[0].GetDegree() != 1:
        return False
    if any(a.GetAtomicNum() not in spec["allowed_atomic_numbers"] for a in atoms):
        return False
    if any(a.GetFormalCharge() or a.GetIsAromatic() or a.IsInRing() for a in atoms):
        return False
    bonds = list(mol.GetBonds())
    if any(str(b.GetBondType()) not in spec["allowed_bond_types"] for b in bonds):
        return False
    if sum(b.GetBondType() == Chem.BondType.DOUBLE for b in bonds) > spec["maximum_double_bonds"]:
        return False
    carbon_root = roots[0].GetNeighbors()[0]
    if carbon_root.GetAtomicNum() != 6:
        return False
    branches = [a for a in atoms if a.GetAtomicNum() == 6 and a.GetDegree() > 2]
    if len(branches) > spec["maximum_branch_carbons"]:
        return False
    for atom in branches:
        distance = (
            0
            if atom.GetIdx() == carbon_root.GetIdx()
            else len(Chem.GetShortestPath(mol, carbon_root.GetIdx(), atom.GetIdx())) - 1
        )
        positions = spec["branch_root_distances"]
        if (positions is not None and distance not in positions) or atom.GetDegree() > 3:
            return False
    return True


@dataclass(frozen=True)
class RegistryComponentScopes:
    specifications: dict[str, Any]
    maximum_matches: int = 1024

    def __post_init__(self) -> None:
        _positive(self.maximum_matches, "maximum_matches")
        if not isinstance(self.specifications, dict) or not self.specifications:
            raise LibraryAssemblyError("Missing complete-component scopes")
        for name, spec in self.specifications.items():
            kind = spec.get("kind")
            if kind == "exact":
                if set(spec) != {"kind", "smiles"}:
                    raise LibraryAssemblyError(f"Unexpected exact scope fields: {name}")
                constitutional_molecule(spec["smiles"])
            elif kind == "template":
                _validate_template({k: v for k, v in spec.items() if k != "kind"})
            elif kind == "cut":
                if set(spec) != {
                    "kind",
                    "query",
                    "cut_bonds",
                    "central_anchor_map",
                    "central",
                    "body",
                    "equal_bodies",
                }:
                    raise LibraryAssemblyError(f"Unexpected cut scope fields: {name}")
                query = Chem.MolFromSmarts(spec["query"])
                if query is None:
                    raise LibraryAssemblyError(f"Invalid scope query: {name}")
                maps = [a.GetAtomMapNum() for a in query.GetAtoms() if a.GetAtomMapNum()]
                if len(set(maps)) != len(maps) or spec["central_anchor_map"] not in maps:
                    raise LibraryAssemblyError(f"Nonunique/missing scope atom maps: {name}")
                cuts = spec["cut_bonds"]
                if not isinstance(cuts, list) or not cuts or type(spec["equal_bodies"]) is not bool:
                    raise LibraryAssemblyError(f"Invalid cut scope policy: {name}")
                by_map = {a.GetAtomMapNum(): a.GetIdx() for a in query.GetAtoms()}
                seen = set()
                for pair in cuts:
                    if (
                        not isinstance(pair, list)
                        or len(pair) != 2
                        or pair[0] == pair[1]
                        or any(type(n) is not int or n not in maps for n in pair)
                        or tuple(sorted(pair)) in seen
                    ):
                        raise LibraryAssemblyError(f"Invalid scope cut mapping: {name}")
                    bond = query.GetBondBetweenAtoms(*(by_map[n] for n in pair))
                    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                        raise LibraryAssemblyError(
                            f"Scope cut must be an explicit single bond: {name}"
                        )
                    seen.add(tuple(sorted(pair)))
                central = spec["central"]
                if set(central) == {"smiles"}:
                    if Chem.MolFromSmiles(central["smiles"]) is None:
                        raise LibraryAssemblyError("Invalid central fragment")
                else:
                    _validate_template(central)
                body = spec["body"]
                if set(body) != {
                    "allowed_atomic_numbers",
                    "allowed_bond_types",
                    "maximum_double_bonds",
                    "maximum_branch_carbons",
                    "branch_root_distances",
                }:
                    raise LibraryAssemblyError("Unexpected body scope fields")
                for k in ("maximum_double_bonds", "maximum_branch_carbons"):
                    if type(body[k]) is not int or body[k] < 0:
                        raise LibraryAssemblyError(f"Invalid body bound: {k}")
                for k in ("allowed_atomic_numbers", "branch_root_distances"):
                    if k == "branch_root_distances" and body[k] is None:
                        continue
                    if not isinstance(body[k], list) or any(
                        type(n) is not int or n < 0 for n in body[k]
                    ):
                        raise LibraryAssemblyError(f"Invalid body vocabulary: {k}")
            else:
                raise LibraryAssemblyError(f"Unknown complete-component scope: {name}")

    @classmethod
    def from_registry(
        cls, path: Path, *, expected_sha256: str, maximum_matches: int = 1024
    ) -> RegistryComponentScopes:
        if sha256_file(path) != expected_sha256:
            raise LibraryAssemblyError("Complete-component registry checksum differs")
        return cls(json.loads(path.read_text())["component_scopes"], maximum_matches)

    def assess(self, scope_id: str, smiles: str) -> dict[str, Any]:
        if scope_id not in self.specifications:
            raise LibraryAssemblyError(f"Unknown complete-component scope: {scope_id}")
        canonical, mol = constitutional_molecule(smiles)
        if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            raise LibraryAssemblyError("Supplied precursor contains a wildcard atom")
        spec = self.specifications[scope_id]
        result = {
            "scope_id": scope_id,
            "constitution": canonical,
            "pass": False,
            "complete_search": True,
            "template_matches": [],
            "fragment_signatures": [],
        }
        if spec["kind"] == "exact":
            result["pass"] = canonical == constitutional_molecule(spec["smiles"])[0]
        elif spec["kind"] == "template":
            result["template_matches"] = _template_matches(
                canonical, mol.GetNumAtoms(), {k: v for k, v in spec.items() if k != "kind"}
            )
            result["pass"] = bool(result["template_matches"])
        else:
            query = Chem.MolFromSmarts(spec["query"])
            matches = mol.GetSubstructMatches(
                query, uniquify=False, maxMatches=self.maximum_matches + 1
            )
            if len(matches) > self.maximum_matches:
                result["complete_search"] = False
                return result
            maps = {a.GetAtomMapNum(): a.GetIdx() for a in query.GetAtoms() if a.GetAtomMapNum()}
            for atom in mol.GetAtoms():
                atom.SetIntProp("_scope_original_index", atom.GetIdx())
            signatures = set()
            all_valid = bool(matches)
            for match in matches:
                cut_indices = [
                    mol.GetBondBetweenAtoms(match[maps[a]], match[maps[b]]).GetIdx()
                    for a, b in spec["cut_bonds"]
                ]
                cut = Chem.FragmentOnBonds(mol, cut_indices, addDummies=True)
                for atom in cut.GetAtoms():
                    if atom.GetAtomicNum() == 0:
                        atom.SetIsotope(0)
                        atom.SetAtomMapNum(1)
                fragments = Chem.GetMolFrags(cut, asMols=True, sanitizeFrags=True)
                central: list[Chem.Mol] = []
                bodies: list[Chem.Mol] = []
                anchor = match[maps[spec["central_anchor_map"]]]
                for fragment in fragments:
                    indices = {
                        a.GetIntProp("_scope_original_index")
                        for a in fragment.GetAtoms()
                        if a.HasProp("_scope_original_index")
                    }
                    (central if anchor in indices else bodies).append(fragment)
                valid = len(central) == 1 and len(bodies) == len(cut_indices)
                if valid:
                    central_smiles = _canonical(central[0])
                    central_spec = spec["central"]
                    central_ok = (
                        central_smiles == _canonical(Chem.MolFromSmiles(central_spec["smiles"]))
                        if "smiles" in central_spec
                        else bool(
                            _template_matches(central_smiles, mol.GetNumAtoms(), central_spec)
                        )
                    )
                    body_smiles = tuple(sorted(_canonical(b) for b in bodies))
                    valid = (
                        central_ok
                        and all(_body_ok(b, spec["body"]) for b in bodies)
                        and (not spec["equal_bodies"] or len(set(body_smiles)) == 1)
                    )
                    if valid:
                        signatures.add((central_smiles, body_smiles))
                all_valid = all_valid and valid
            result["fragment_signatures"] = [
                {"central": c, "bodies": list(b)} for c, b in sorted(signatures)
            ]
            result["pass"] = all_valid and len(signatures) == 1
        return result
