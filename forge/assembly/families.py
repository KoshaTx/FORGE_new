"""Fixed-arity assembly across qualified, hash-pinned reaction libraries.

This interface checks registry policy and exact constitutional reconstruction. It does not
qualify source-executed synthesis, repeated programs, precursor routes, or procurement.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, rdBase

from forge.assembly.api import ForwardAssemblyCheck, ForwardAssemblyProducts
from forge.assembly.program import repair_template_hydrogens
from forge.assembly.registry import CompiledRegistryReaction, load_compiled_registry_reaction
from forge.chemistry.reactive_sites import (
    RAW_SUBSTRUCTURE_MATCHES,
    SUPPORTED_MULTIPLICITY_SEMANTICS,
    audit_reactive_site_multiplicity,
)
from forge.core.hashing import sha256_file


class LibraryAssemblyError(ValueError):
    """A library or requested assembly violates its declared contract."""


def _bound(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LibraryAssemblyError("maximum_outcomes must be a positive integer")


def constitutional_molecule(smiles: str) -> tuple[str, Chem.Mol]:
    """Normalize stereo-free, unmapped identity; reject disconnected or isotope-labelled graphs."""

    if not isinstance(smiles, str) or not smiles:
        raise LibraryAssemblyError("SMILES must be a nonempty string")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise LibraryAssemblyError("SMILES must describe a valid connected molecular graph")
    if any(atom.GetIsotope() for atom in molecule.GetAtoms()):
        raise LibraryAssemblyError("isotope-labelled input is outside constitutional support")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(molecule)
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    return canonical, molecule


@dataclass(frozen=True)
class LibraryRoleAssessment:
    role: str
    handle_count: int
    multiplicity_semantics: str
    forbidden_match: bool
    qualified: bool


@dataclass(frozen=True)
class LibraryDecomposition:
    reaction_id: str
    product_smiles: str
    components: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RegistryAssemblyAdapter:
    """One fixed-arity reaction with the registry's unmodified role admission policy."""

    reaction: CompiledRegistryReaction
    registry_path: Path
    registry_sha256: str
    multiplicity_semantics: tuple[str, ...]

    @classmethod
    def from_registry(
        cls, registry_path: Path, *, reaction_id: str, expected_sha256: str
    ) -> RegistryAssemblyAdapter:
        path = registry_path.resolve()
        observed = str(sha256_file(path))
        if observed != expected_sha256:
            raise LibraryAssemblyError(f"registry hash mismatch: {path}")
        with rdBase.BlockLogs():
            compiled = load_compiled_registry_reaction(path, reaction_id=reaction_id)
        raw = next(
            row
            for row in json.loads(path.read_text())["reactions"]
            if row["reaction_id"] == reaction_id
        )
        roles = raw["reactant_roles"]
        names = tuple(role["name"] for role in roles)
        if len(set(names)) != len(names):
            raise LibraryAssemblyError(f"{reaction_id}: repeated role names need a program adapter")
        if any(
            type(role.get("count", 1)) is not int or role.get("count", 1) != 1 for role in roles
        ):
            raise LibraryAssemblyError(f"{reaction_id}: role counts require a program adapter")
        if compiled.forward.GetNumProductTemplates() != 1:
            raise LibraryAssemblyError(f"{reaction_id}: a single product template is required")
        semantics = tuple(
            role.get("site_multiplicity_semantics", RAW_SUBSTRUCTURE_MATCHES) for role in roles
        )
        if any(value not in SUPPORTED_MULTIPLICITY_SEMANTICS for value in semantics):
            raise LibraryAssemblyError(f"{reaction_id}: unsupported role multiplicity semantics")
        return cls(compiled, path, observed, semantics)

    @property
    def reaction_id(self) -> str:
        return self.reaction.definition.reaction_id

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(role.name for role in self.reaction.definition.reactant_roles)

    def _reactants(self, components: Mapping[str, str]) -> tuple[Chem.Mol, ...]:
        if set(components) != set(self.roles):
            raise LibraryAssemblyError(f"{self.reaction_id}: expected roles {self.roles}")
        return tuple(constitutional_molecule(components[role])[1] for role in self.roles)

    def _assess(self, reactants: Sequence[Chem.Mol]) -> tuple[LibraryRoleAssessment, ...]:
        result = []
        for molecule, role, handle, forbidden, semantics in zip(
            reactants,
            self.reaction.definition.reactant_roles,
            self.reaction.handles,
            self.reaction.forbidden,
            self.multiplicity_semantics,
            strict=True,
        ):
            count = (
                len(molecule.GetSubstructMatches(handle))
                if semantics == RAW_SUBSTRUCTURE_MATCHES
                else audit_reactive_site_multiplicity(molecule, handle).count(semantics)
            )
            blocked = any(molecule.HasSubstructMatch(query) for query in forbidden)
            result.append(
                LibraryRoleAssessment(
                    role.name,
                    count,
                    semantics,
                    blocked,
                    count in role.allowed_site_multiplicity and not blocked,
                )
            )
        return tuple(result)

    def assess_roles(self, components: Mapping[str, str]) -> tuple[LibraryRoleAssessment, ...]:
        return self._assess(self._reactants(components))

    def forward_products(
        self, components: Mapping[str, str], *, maximum_outcomes: int = 64
    ) -> ForwardAssemblyProducts:
        _bound(maximum_outcomes)
        reactants = self._reactants(components)
        outcomes: tuple[tuple[Chem.Mol, ...], ...] = ()
        if all(item.qualified for item in self._assess(reactants)):
            with rdBase.BlockLogs():
                outcomes = self.reaction.forward.RunReactants(
                    reactants, maxProducts=maximum_outcomes
                )
        products = set()
        for outcome in outcomes:
            if len(outcome) != 1:
                continue
            product = Chem.Mol(outcome[0])
            try:
                with rdBase.BlockLogs():
                    Chem.SanitizeMol(product)
                canonical, _ = constitutional_molecule(Chem.MolToSmiles(product))
            except (ValueError, RuntimeError):
                continue
            products.add(canonical)
        return ForwardAssemblyProducts(
            assembly_id=self.reaction_id,
            reaction_id=self.reaction_id,
            roles=self.roles,
            products=tuple(sorted(products)),
            saturated=len(outcomes) >= maximum_outcomes,
            enumerated_outcomes_by_step=(len(outcomes),),
        )

    def check_forward(
        self, components: Mapping[str, str], product_smiles: str, *, maximum_outcomes: int = 64
    ) -> ForwardAssemblyCheck:
        target, _ = constitutional_molecule(product_smiles)
        products = self.forward_products(components, maximum_outcomes=maximum_outcomes)
        return ForwardAssemblyCheck(
            reaction_id=self.reaction_id,
            roles=self.roles,
            exact=target in products.products,
            saturated=products.saturated,
            enumerated_outcomes=products.enumerated_outcomes_by_step[0],
        )

    def decompose(
        self, product_smiles: str, *, maximum_outcomes: int = 128
    ) -> tuple[LibraryDecomposition, ...]:
        """Return handle-qualified exact inverse/forward traces; never admit a truncated search."""

        _bound(maximum_outcomes)
        target, product = constitutional_molecule(product_smiles)
        with rdBase.BlockLogs():
            outcomes = self.reaction.reverse.RunReactants((product,), maxProducts=maximum_outcomes)
        if len(outcomes) >= maximum_outcomes:
            raise LibraryAssemblyError(f"{self.reaction_id}: reverse enumeration saturated")
        candidates = set()
        for outcome in outcomes:
            if len(outcome) != len(self.roles):
                continue
            repaired = tuple(repair_template_hydrogens(fragment) for fragment in outcome)
            if any(item is None for item in repaired):
                continue
            components = tuple(
                (role, item[0])
                for role, item in zip(self.roles, repaired, strict=True)
                if item is not None
            )
            candidates.add(components)
        result = []
        for components in sorted(candidates):
            check = self.check_forward(dict(components), target, maximum_outcomes=maximum_outcomes)
            if check.saturated:
                raise LibraryAssemblyError(f"{self.reaction_id}: forward enumeration saturated")
            if check.exact:
                result.append(LibraryDecomposition(self.reaction_id, target, components))
        return tuple(result)


def load_assembly_libraries(
    registries: Sequence[tuple[Path, str]], *, expected_families: Sequence[str]
) -> dict[str, RegistryAssemblyAdapter]:
    """Load an explicit family set without importing a corpus or a training workflow."""

    if not expected_families or len(set(expected_families)) != len(expected_families):
        raise LibraryAssemblyError("expected_families must be nonempty and unique")
    result = {}
    for path, expected_hash in registries:
        if str(sha256_file(path)) != expected_hash:
            raise LibraryAssemblyError(f"registry hash mismatch: {path}")
        for raw in json.loads(path.read_text())["reactions"]:
            family = raw["reaction_id"]
            if family in result:
                raise LibraryAssemblyError(f"duplicate registry family: {family}")
            result[family] = RegistryAssemblyAdapter.from_registry(
                path, reaction_id=family, expected_sha256=expected_hash
            )
    if set(result) != set(expected_families):
        raise LibraryAssemblyError("declared families differ from the registry union")
    return dict(sorted(result.items()))
