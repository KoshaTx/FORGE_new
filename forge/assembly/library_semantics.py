"""Exact atom-origin semantics for registry-defined library programs.

This module replays an already qualified computed program with temporary isotope labels.  The
labels are removed before publication and never enter model state.  A semantic record is returned
only when every exact outcome and every graph automorphism agrees on precursor role and reaction
core position for every product atom.

Ugi is intentionally handled by its existing qualified annotator because its amide oxygen is an
assembly-introduced atom under the settled AGILE semantic contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rdkit import Chem, rdBase

from forge.assembly.families import (
    LibraryAssemblyError,
    RegistryAssemblyAdapter,
    constitutional_molecule,
)

_ORIGIN_ISOTOPE_START = 1001
_MAX_CANONICAL_MATCHES = 4096


@dataclass(frozen=True)
class LibraryAtomSemantics:
    reaction_id: str
    canonical_product_smiles: str
    atom_origins: tuple[str, ...]
    core_positions: tuple[str, ...]
    step_count: int

    def __post_init__(self) -> None:
        if (
            not self.reaction_id
            or not self.canonical_product_smiles
            or not self.atom_origins
            or len(self.atom_origins) != len(self.core_positions)
            or self.step_count < 1
        ):
            raise LibraryAssemblyError("library atom semantics are incomplete")


def _tagged(smiles: str, isotope: int) -> Chem.Mol:
    _, molecule = constitutional_molecule(smiles)
    if any(atom.GetIsotope() for atom in molecule.GetAtoms()):
        raise LibraryAssemblyError("semantic replay does not accept isotope-labelled inputs")
    for atom in molecule.GetAtoms():
        atom.SetIsotope(isotope)
    return molecule


def _record_core_positions(molecule: Chem.Mol) -> None:
    for atom in molecule.GetAtoms():
        if atom.HasProp("old_mapno"):
            atom.SetProp("_forge_core_position", f"map_{atom.GetIntProp('old_mapno')}")


def _canonical_semantics(
    molecule: Chem.Mol,
    labels_by_isotope: Mapping[int, str],
    core_position_aliases: Mapping[str, str],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    origins: list[str] = []
    positions: list[str] = []
    cleared = Chem.Mol(molecule)
    for source, target in zip(molecule.GetAtoms(), cleared.GetAtoms(), strict=True):
        isotope = source.GetIsotope()
        if isotope not in labels_by_isotope:
            raise LibraryAssemblyError(
                "generic semantic replay produced an atom without a precursor-role origin"
            )
        origins.append(labels_by_isotope[isotope])
        position = (
            source.GetProp("_forge_core_position") if source.HasProp("_forge_core_position") else ""
        )
        if position and core_position_aliases:
            try:
                position = core_position_aliases[position]
            except KeyError as error:
                raise LibraryAssemblyError(
                    f"semantic core position lacks a declared alias: {position}"
                ) from error
        positions.append(position)
        target.SetIsotope(0)
        target.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(cleared)
    smiles = Chem.MolToSmiles(cleared, canonical=True, isomericSmiles=False)
    canonical = Chem.MolFromSmiles(smiles)
    if canonical is None:
        raise LibraryAssemblyError("canonical semantic product did not parse")
    matches = cleared.GetSubstructMatches(
        canonical, uniquify=False, maxMatches=_MAX_CANONICAL_MATCHES
    )
    if not matches or len(matches) >= _MAX_CANONICAL_MATCHES:
        raise LibraryAssemblyError("canonical semantic mapping is absent or saturated")
    assignments = {
        (
            tuple(origins[index] for index in match),
            tuple(positions[index] for index in match),
        )
        for match in matches
    }
    if len(assignments) != 1:
        raise LibraryAssemblyError(
            "precursor-role or core-position states are ambiguous under graph symmetry"
        )
    atom_origins, core_positions = next(iter(assignments))
    return smiles, atom_origins, core_positions


def trace_library_atom_semantics(
    adapter: RegistryAssemblyAdapter,
    components: Mapping[str, str],
    intermediate_products: Sequence[str],
    *,
    accumulator_role: str | None,
    maximum_outcomes: int = 256,
    core_position_aliases: Mapping[str, str] | None = None,
) -> LibraryAtomSemantics:
    """Replay one unique computed program and require exact, unambiguous atom semantics."""

    if set(components) != set(adapter.roles):
        raise LibraryAssemblyError("semantic program components differ from registry roles")
    if not intermediate_products:
        raise LibraryAssemblyError("semantic replay requires at least one product")
    if accumulator_role is None:
        if len(intermediate_products) != 1:
            raise LibraryAssemblyError("fixed-arity semantic replay requires exactly one step")
    elif len(adapter.roles) != 2 or accumulator_role not in adapter.roles:
        raise LibraryAssemblyError("repeated semantic replay has an invalid accumulator role")
    if (
        isinstance(maximum_outcomes, bool)
        or not isinstance(maximum_outcomes, int)
        or maximum_outcomes < 1
    ):
        raise LibraryAssemblyError("maximum_outcomes must be a positive integer")
    aliases = dict(core_position_aliases or {})
    if any(not key or not value for key, value in aliases.items()):
        raise LibraryAssemblyError("semantic core-position aliases must be nonempty strings")

    labels_by_isotope = {
        _ORIGIN_ISOTOPE_START + index: role for index, role in enumerate(adapter.roles)
    }
    canonical_components = {
        role: constitutional_molecule(smiles)[0] for role, smiles in components.items()
    }
    current = (
        _tagged(
            canonical_components[accumulator_role],
            _ORIGIN_ISOTOPE_START + adapter.roles.index(accumulator_role),
        )
        if accumulator_role is not None
        else None
    )
    final: tuple[str, tuple[str, ...], tuple[str, ...]] | None = None
    for expected in intermediate_products:
        reactants = []
        for index, role in enumerate(adapter.roles):
            if role == accumulator_role:
                assert current is not None
                reactants.append(current)
            else:
                reactants.append(_tagged(canonical_components[role], _ORIGIN_ISOTOPE_START + index))
        if not all(value.qualified for value in adapter._assess(tuple(reactants))):
            raise LibraryAssemblyError("semantic replay step violates registry role policy")
        with rdBase.BlockLogs():
            outcomes = adapter.reaction.forward.RunReactants(
                tuple(reactants), maxProducts=maximum_outcomes
            )
        if len(outcomes) >= maximum_outcomes:
            raise LibraryAssemblyError("semantic replay outcome enumeration saturated")
        target = constitutional_molecule(expected)[0]
        candidates: list[tuple[Chem.Mol, str, tuple[str, ...], tuple[str, ...]]] = []
        for outcome in outcomes:
            if len(outcome) != 1:
                continue
            product = Chem.Mol(outcome[0])
            try:
                with rdBase.BlockLogs():
                    Chem.SanitizeMol(product)
                _record_core_positions(product)
                smiles, origins, positions = _canonical_semantics(
                    product, labels_by_isotope, aliases
                )
            except (LibraryAssemblyError, ValueError, RuntimeError):
                continue
            if smiles == target:
                candidates.append((product, smiles, origins, positions))
        if not candidates:
            raise LibraryAssemblyError("semantic replay did not reconstruct the exact step")
        assignments = {(value[2], value[3]) for value in candidates}
        if len(assignments) != 1:
            raise LibraryAssemblyError(
                "exact outcomes disagree on precursor-role or core-position states"
            )
        candidates.sort(key=lambda value: Chem.MolToSmiles(value[0], canonical=False))
        current, smiles, origins, positions = candidates[0]
        final = smiles, origins, positions
    assert final is not None
    return LibraryAtomSemantics(
        reaction_id=adapter.reaction_id,
        canonical_product_smiles=final[0],
        atom_origins=final[1],
        core_positions=final[2],
        step_count=len(intermediate_products),
    )


__all__ = ["LibraryAtomSemantics", "trace_library_atom_semantics"]
