"""Single-event all-path coordinates with one explicitly introduced registry atom."""

from __future__ import annotations

from collections.abc import Mapping

from rdkit import Chem, rdBase

from forge.assembly.families import (
    LibraryAssemblyError,
    RegistryAssemblyAdapter,
    constitutional_molecule,
)
from forge.assembly.program_atom_origins import (
    ProgramAtomOrigins,
    SemanticReplay,
    _clear_labels,
    canonical_coordinates,
)
from forge.assembly.repeated_components import RepeatBounds

ASSEMBLY_INTRODUCED = "assembly_introduced"
INTRODUCED_POSITION = "template_introduced_0"


def trace_single_introduction_program(
    adapter: RegistryAssemblyAdapter,
    components: Mapping[str, str],
    *,
    source_roles: Mapping[str, str],
    bounds: RepeatBounds,
) -> SemanticReplay:
    """Trace every outcome without assigning the introduced atom to a precursor.

    Scope is one connected product template containing exactly one unmapped atom,
    whose neighbors are mapped. Element, charge, bonds and reaction bookkeeping must
    independently identify that atom in every outcome. Unknown origins abstain.
    """
    if (
        set(components) != set(adapter.roles)
        or set(source_roles) != set(adapter.roles)
        or len(set(source_roles.values())) != len(source_roles)
        or any(not isinstance(r, str) or not r for r in source_roles.values())
        or ASSEMBLY_INTRODUCED in source_roles.values()
    ):
        raise LibraryAssemblyError("Complete distinct precursor roles are required")
    reaction = adapter.reaction.forward
    templates = tuple(reaction.GetProducts())
    if len(templates) != 1:
        raise LibraryAssemblyError("Single introduction requires one product template")
    for template in (*reaction.GetReactants(), *templates):
        if any(a.GetIsotope() or "Isotope" in a.DescribeQuery() for a in template.GetAtoms()):
            raise LibraryAssemblyError("Temporary origins require isotope-free registry queries")
    introduced = [a for a in templates[0].GetAtoms() if not a.GetAtomMapNum()]
    if len(introduced) != 1 or introduced[0].GetAtomicNum() < 1:
        raise LibraryAssemblyError("Exactly one element-defined introduced atom is required")
    new_atom = introduced[0]
    neighbor_bonds = tuple(
        sorted(
            (bond.GetOtherAtom(new_atom).GetAtomMapNum(), bond.GetBondTypeAsDouble())
            for bond in new_atom.GetBonds()
        )
    )
    if not neighbor_bonds or any(number < 1 for number, _ in neighbor_bonds):
        raise LibraryAssemblyError("Introduced atom neighbors must be registry-mapped")
    signature = (new_atom.GetAtomicNum(), new_atom.GetFormalCharge(), neighbor_bonds)
    maps = {a.GetAtomMapNum() for a in templates[0].GetAtoms() if a.GetAtomMapNum()}
    positions = ("exterior", *(f"map_{number}" for number in sorted(maps)))
    pairs = [(role, core) for role in sorted(source_roles.values()) for core in positions]
    pairs.append((ASSEMBLY_INTRODUCED, INTRODUCED_POSITION))
    colors = {pair: index + 1 for index, pair in enumerate(pairs)}
    labels = {index + 1: source_roles[role] for index, role in enumerate(adapter.roles)}
    reactants = tuple(constitutional_molecule(components[role])[1] for role in adapter.roles)
    if not all(value.qualified for value in adapter._assess(reactants)):
        return SemanticReplay("no_complete_forward_product", True, None, (), (1, 0), 0)
    for index, molecule in enumerate(reactants, start=1):
        for atom in molecule.GetAtoms():
            atom.SetIsotope(index)
    with rdBase.BlockLogs():
        outcomes = reaction.RunReactants(reactants, maxProducts=bounds.maximum_outcomes)
    states: set[str] = set()
    products: set[str] = set()
    assignments: set[ProgramAtomOrigins] = set()
    ambiguous = False

    def result(
        disposition: str, *, complete: bool = False, annotation: ProgramAtomOrigins | None = None
    ) -> SemanticReplay:
        return SemanticReplay(
            disposition,
            complete,
            annotation,
            tuple(sorted(products)),
            (1, len(states)),
            len(outcomes),
        )

    if len(outcomes) >= bounds.maximum_outcomes:
        return result("forward_outcome_bound")
    if len(outcomes) > bounds.maximum_transitions:
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
        coordinates, introduced_count = [], 0
        for atom in product.GetAtoms():
            if atom.GetIsotope() in labels:
                role = labels[atom.GetIsotope()]
                core = "exterior"
                if atom.HasProp("old_mapno"):
                    number = atom.GetIntProp("old_mapno")
                    if number not in maps:
                        raise LibraryAssemblyError("Reaction produced an undeclared mapped core")
                    core = f"map_{number}"
                coordinates.append((role, core))
            else:
                if atom.GetIsotope() or any(
                    atom.HasProp(name) for name in ("old_mapno", "react_atom_idx", "react_idx")
                ):
                    return result("unresolved_introduced_atom_origin")
                neighbors = tuple(
                    sorted(
                        (
                            (
                                bond.GetOtherAtom(atom).GetIntProp("old_mapno")
                                if bond.GetOtherAtom(atom).HasProp("old_mapno")
                                else 0
                            ),
                            bond.GetBondTypeAsDouble(),
                        )
                        for bond in atom.GetBonds()
                    )
                )
                if (atom.GetAtomicNum(), atom.GetFormalCharge(), neighbors) != signature:
                    return result("unresolved_introduced_atom_origin")
                introduced_count += 1
                coordinates.append((ASSEMBLY_INTRODUCED, INTRODUCED_POSITION))
        if introduced_count != 1:
            return result("unresolved_introduced_atom_origin")
        clean = _clear_labels(product)
        products.add(Chem.MolToSmiles(clean, canonical=True, isomericSmiles=False))
        assignment = canonical_coordinates(clean, coordinates)
        if assignment is None:
            ambiguous = True
        else:
            assignments.add(assignment)
        for atom, coordinate in zip(product.GetAtoms(), coordinates, strict=True):
            atom.SetIsotope(colors[coordinate])
            atom.SetAtomMapNum(0)
        Chem.RemoveStereochemistry(product)
        states.add(Chem.MolToSmiles(product, canonical=True, isomericSmiles=True))
        if len(states) + 1 > bounds.maximum_states:
            return result("semantic_state_bound")
    if not products:
        return result("no_complete_forward_product", complete=True)
    if len(products) != 1:
        return result("ambiguous_forward_products", complete=True)
    if ambiguous or len(assignments) != 1:
        return result("ambiguous_atom_coordinates", complete=True)
    return result(
        "unique_forward_atom_coordinates", complete=True, annotation=next(iter(assignments))
    )
