"""Trace all bounded forward paths without merging different atom-origin states.

Temporary isotope labels encode precursor role and registry core position together.
They are removed before role assessment and before constitutional identity comparison.
No target product, inverse decomposition, or selected reaction lineage enters this API.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from rdkit import Chem, rdBase

from forge.assembly.families import (
    LibraryAssemblyError,
    RegistryAssemblyAdapter,
    constitutional_molecule,
)
from forge.assembly.repeated_components import RepeatBounds


@dataclass(frozen=True)
class ProgramAtomOrigins:
    canonical_product_smiles: str
    atom_roles: tuple[str, ...]
    core_positions: tuple[str, ...]


@dataclass(frozen=True)
class SemanticReplay:
    disposition: str
    complete_search: bool
    annotations: ProgramAtomOrigins | None
    constitutional_products: tuple[str, ...]
    states_by_layer: tuple[int, ...]
    raw_outcomes: int


def canonical_coordinates(
    molecule: Chem.Mol, coordinates: Sequence[tuple[str, str]]
) -> ProgramAtomOrigins | None:
    """Require labels to be invariant under every constitutional graph symmetry.

    RDKit's untied canonical ranks partition atoms into symmetry classes. Labels must
    be constant on each class before any one isomorphism is used to serialize them.
    This avoids truncated automorphism enumeration for repeated identical tails.
    """
    if molecule.GetNumAtoms() != len(coordinates) or not coordinates:
        raise LibraryAssemblyError("Atom-coordinate count does not match the complete product")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    ranks = Chem.CanonicalRankAtoms(
        molecule,
        breakTies=False,
        includeChirality=False,
        includeIsotopes=False,
        includeAtomMaps=False,
    )
    labels: dict[int, tuple[str, str]] = {}
    for rank, coordinate in zip(ranks, coordinates, strict=True):
        if rank in labels and labels[rank] != coordinate:
            return None
        labels[rank] = coordinate
    reference = Chem.MolFromSmiles(canonical)
    match = molecule.GetSubstructMatch(reference)
    if len(match) != len(coordinates):
        raise LibraryAssemblyError("Canonical semantic product did not map onto its source")
    return ProgramAtomOrigins(
        canonical, tuple(coordinates[i][0] for i in match), tuple(coordinates[i][1] for i in match)
    )


def _clear_labels(molecule: Chem.Mol) -> Chem.Mol:
    cleared = Chem.Mol(molecule)
    for atom in cleared.GetAtoms():
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(cleared)
    return cleared


def trace_repeated_program(
    adapter: RegistryAssemblyAdapter,
    components: Mapping[str, str],
    *,
    accumulator_role: str,
    events: int,
    source_roles: Mapping[str, str],
    bounds: RepeatBounds,
) -> SemanticReplay:
    """Propagate all role/core assignments for one qualified fixed-component program.

    This supplies semantics only. Upstream exact-evidence admission must separately
    require full inventory, inverse, source-control and protection checks. An atom
    introduced by the assembly currently abstains instead of acquiring a false origin.
    """
    if (
        set(components) != set(adapter.roles)
        or set(source_roles) != set(adapter.roles)
        or len(set(source_roles.values())) != len(source_roles)
        or any(not isinstance(value, str) or not value for value in source_roles.values())
        or accumulator_role not in components
        or type(events) is not int
        or not 1 <= events <= bounds.maximum_events
    ):
        raise LibraryAssemblyError("Complete source roles and a bounded event count are required")
    if any(
        atom.GetIsotope()
        for template in (
            *adapter.reaction.forward.GetReactants(),
            *adapter.reaction.forward.GetProducts(),
        )
        for atom in template.GetAtoms()
    ):
        raise LibraryAssemblyError(
            "Temporary origin labels require an isotope-free registry transform"
        )
    core_positions = sorted(
        {
            f"map_{atom.GetAtomMapNum()}"
            for template in adapter.reaction.forward.GetProducts()
            for atom in template.GetAtoms()
            if atom.GetAtomMapNum()
        }
    )
    if not core_positions:
        raise LibraryAssemblyError("Reaction registry supplies no mapped product-core positions")
    pairs = [
        (role, core)
        for role in sorted(source_roles.values())
        for core in ("exterior", *core_positions)
    ]
    encode = {pair: index + 1 for index, pair in enumerate(pairs)}
    decode = {value: key for key, value in encode.items()}
    molecules = {}
    for role in adapter.roles:
        _, molecule = constitutional_molecule(components[role])
        for atom in molecule.GetAtoms():
            atom.SetIsotope(encode[(source_roles[role], "exterior")])
        molecules[role] = molecule
    initial = molecules[accumulator_role]
    current = {Chem.MolToSmiles(initial, isomericSmiles=True): initial}
    layers, transitions = [1], 0

    def result(
        disposition: str,
        *,
        complete: bool = False,
        annotations: ProgramAtomOrigins | None = None,
        products: Collection[str] = (),
    ) -> SemanticReplay:
        return SemanticReplay(
            disposition, complete, annotations, tuple(products), tuple(layers), transitions
        )

    for _ in range(events):
        following: dict[str, Chem.Mol] = {}
        for _, state in sorted(current.items()):
            reactants = tuple(
                state if role == accumulator_role else molecules[role] for role in adapter.roles
            )
            clean = tuple(_clear_labels(mol) for mol in reactants)
            # Use the very same compiled role/handle policy as RegistryAssemblyAdapter.
            if not all(value.qualified for value in adapter._assess(clean)):
                continue
            with rdBase.BlockLogs():
                outcomes = adapter.reaction.forward.RunReactants(
                    reactants, maxProducts=bounds.maximum_outcomes
                )
            transitions += len(outcomes)
            if len(outcomes) >= bounds.maximum_outcomes:
                return result("forward_outcome_bound")
            if transitions > bounds.maximum_transitions:
                return result("semantic_transition_bound")
            for outcome in outcomes:
                if len(outcome) != 1:
                    continue
                product = Chem.Mol(outcome[0])
                try:
                    with rdBase.BlockLogs():
                        Chem.SanitizeMol(product)
                    if len(Chem.GetMolFrags(product)) != 1:
                        continue
                except (ValueError, RuntimeError):
                    continue
                for atom in product.GetAtoms():
                    if atom.GetIsotope() not in decode:
                        return result("assembly_introduced_or_unresolved_atom_origin")
                    role, core = decode[atom.GetIsotope()]
                    if atom.HasProp("old_mapno"):
                        core = f"map_{atom.GetIntProp('old_mapno')}"
                    if (role, core) not in encode:
                        raise LibraryAssemblyError("Reaction produced an undeclared core position")
                    atom.SetIsotope(encode[(role, core)])
                    atom.SetAtomMapNum(0)
                Chem.RemoveStereochemistry(product)
                key = Chem.MolToSmiles(product, isomericSmiles=True, canonical=True)
                following.setdefault(key, product)
                if sum(layers) + len(following) > bounds.maximum_states:
                    return result("semantic_state_bound")
        layers.append(len(following))
        current = following
        if not current:
            return result("no_complete_forward_product", complete=True)
    products, annotations, symmetry_ambiguous = set(), set(), False
    for state in current.values():
        clean = _clear_labels(state)
        products.add(Chem.MolToSmiles(clean, isomericSmiles=False, canonical=True))
        coordinates = tuple(decode[atom.GetIsotope()] for atom in state.GetAtoms())
        assignment = canonical_coordinates(clean, coordinates)
        if assignment is None:
            symmetry_ambiguous = True
        else:
            annotations.add(assignment)
    ordered_products = tuple(sorted(products))
    if len(ordered_products) != 1:
        return result("ambiguous_forward_products", complete=True, products=ordered_products)
    if symmetry_ambiguous or len(annotations) != 1:
        return result("ambiguous_atom_coordinates", complete=True, products=ordered_products)
    return result(
        "unique_forward_atom_coordinates",
        complete=True,
        products=ordered_products,
        annotations=next(iter(annotations)),
    )
