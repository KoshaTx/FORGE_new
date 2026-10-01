"""Recover precursor occurrences by exhaustive labelled replay of exact computed programs.

Connected same-role components are not precursor occurrences: later steps may join two
occurrences. Labels here track individual precursor atoms across steps and are removed before
graph comparison. Only a unique occurrence/correspondence relation is admitted, modulo names.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from rdkit import Chem, rdBase

from forge.assembly.families import (
    LibraryAssemblyError,
    RegistryAssemblyAdapter,
    constitutional_molecule,
)
from forge.assembly.library_programs import LibraryProgramLimits


class OccurrenceTraceError(LibraryAssemblyError):
    pass


def normalize_occurrences(occurrences: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Remove occurrence and common-atom naming while preserving their equality relation."""
    if not occurrences or len({len(o) for o in occurrences}) != 1 or not occurrences[0]:
        raise OccurrenceTraceError("incomplete occurrence correspondence")
    flat = [i for o in occurrences for i in o]
    if any(type(i) is not int or i < 0 for i in flat) or len(set(flat)) != len(flat):
        raise OccurrenceTraceError("overlapping or invalid occurrence atoms")
    # Sort whole occurrences by their atom sets, then common columns by their node vectors.
    ordered = sorted(map(tuple, occurrences), key=lambda o: tuple(sorted(o)))
    columns = sorted(zip(*ordered, strict=True))
    return tuple(zip(*columns, strict=True))


def trace_precursor_occurrences(
    adapter: RegistryAssemblyAdapter,
    components: Mapping[str, str],
    intermediate_products: Sequence[str],
    *,
    accumulator_role: str,
    limits: LibraryProgramLimits,
    maximum_matches: int = 4096,
) -> tuple[tuple[int, ...], ...]:
    """Return canonical product atom indices aligned across repeated co-reactant occurrences.

    Every exact labelled state is propagated, rather than selecting one convenient outcome.
    Every final graph automorphism must induce the same normalized relation. Lost precursor
    atoms may be omitted only if the same precursor coordinates survive in every occurrence.
    Saturation, unidentified product atoms and conflicting relations all fail closed.
    """
    if (
        len(adapter.roles) != 2
        or accumulator_role not in adapter.roles
        or set(components) != set(adapter.roles)
        or not 2 <= len(intermediate_products) <= limits.maximum_steps
        or type(maximum_matches) is not int
        or maximum_matches < 2
    ):
        raise OccurrenceTraceError("invalid repeated occurrence-trace contract")
    co_role = next(role for role in adapter.roles if role != accumulator_role)
    precursor = {role: constitutional_molecule(s)[1] for role, s in components.items()}
    if any(a.GetIsotope() for mol in precursor.values() for a in mol.GetAtoms()):
        raise OccurrenceTraceError("occurrence tracing requires isotope-free inputs")
    metadata = {}
    next_label = 1001

    def tagged(role: str, occurrence: int) -> Chem.Mol:
        nonlocal next_label
        mol = Chem.Mol(precursor[role])
        for atom in mol.GetAtoms():
            if next_label >= 65535:
                raise OccurrenceTraceError("occurrence isotope label bound exceeded")
            metadata[next_label] = (role, occurrence, atom.GetIdx())
            atom.SetIsotope(next_label)
            next_label += 1
        return mol

    def clear(mol: Chem.Mol) -> Chem.Mol:
        cleared = Chem.Mol(mol)
        for atom in cleared.GetAtoms():
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
        Chem.RemoveStereochemistry(cleared)
        return cleared

    states = [tagged(accumulator_role, 0)]
    expansions = 0
    for step, expected in enumerate(intermediate_products, 1):
        reagent = tagged(co_role, step)
        target = constitutional_molecule(expected)[0]
        following = {}
        for current in states:
            expansions += 1
            if expansions > limits.maximum_expansions:
                raise OccurrenceTraceError("occurrence expansion limit")
            inputs = {accumulator_role: current, co_role: reagent}
            reactants = tuple(inputs[role] for role in adapter.roles)
            if not all(value.qualified for value in adapter._assess(reactants)):
                raise OccurrenceTraceError("occurrence step violates registry role policy")
            with rdBase.BlockLogs():
                outcomes = adapter.reaction.forward.RunReactants(
                    reactants, maxProducts=limits.maximum_outcomes
                )
            if len(outcomes) >= limits.maximum_outcomes:
                raise OccurrenceTraceError("occurrence outcome limit")
            for outcome in outcomes:
                if len(outcome) != 1:
                    continue
                mol = Chem.Mol(outcome[0])
                try:
                    with rdBase.BlockLogs():
                        Chem.SanitizeMol(mol)
                    smiles = Chem.MolToSmiles(clear(mol), canonical=True, isomericSmiles=False)
                except (ValueError, RuntimeError):
                    continue
                if smiles != target:
                    continue
                labels = [a.GetIsotope() for a in mol.GetAtoms()]
                if len(set(labels)) != len(labels) or any(i not in metadata for i in labels):
                    raise OccurrenceTraceError("lost or duplicated precursor atom identity")
                # Isotope-labelled graph identity preserves every surviving precursor atom.
                key = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                following[key] = mol
                if len(following) > limits.maximum_states:
                    raise OccurrenceTraceError("occurrence state limit")
        if not following:
            raise OccurrenceTraceError("occurrence replay did not reconstruct exact step")
        states = [following[key] for key in sorted(following)]
    canonical = constitutional_molecule(intermediate_products[-1])[1]
    relations = set()
    for mol in states:
        matches = clear(mol).GetSubstructMatches(
            canonical, uniquify=False, maxMatches=maximum_matches
        )
        if not matches or len(matches) >= maximum_matches:
            raise OccurrenceTraceError("occurrence canonical mapping absent or saturated")
        for match in matches:
            occurrences: defaultdict[int, dict[int, int]] = defaultdict(dict)
            for index, old in enumerate(match):
                role, occurrence, atom = metadata[mol.GetAtomWithIdx(old).GetIsotope()]
                if role == co_role:
                    occurrences[occurrence][atom] = index
            if set(occurrences) != set(range(1, len(intermediate_products) + 1)):
                raise OccurrenceTraceError("precursor occurrence missing from product")
            retained = [tuple(sorted(v)) for _, v in sorted(occurrences.items())]
            if len(set(retained)) != 1 or not retained[0]:
                raise OccurrenceTraceError("occurrences retain different precursor atoms")
            relation = normalize_occurrences(
                [tuple(occurrences[i][j] for j in retained[0]) for i in sorted(occurrences)]
            )
            relations.add(relation)
            if len(relations) != 1:
                raise OccurrenceTraceError("ambiguous occurrence correspondence")
    return next(iter(relations))
