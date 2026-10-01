"""All-path origins and stage-local reaction participation for repeated event groups."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping

from rdkit import Chem, rdBase

from forge.assembly.families import LibraryAssemblyError, constitutional_molecule
from forge.assembly.grouped_program import RegistryGroupedProgram
from forge.assembly.program_atom_origins import (
    ProgramAtomOrigins,
    SemanticReplay,
    _clear_labels,
    canonical_coordinates,
)
from forge.assembly.staged_program import _constraints


def trace_grouped_program(
    program: RegistryGroupedProgram,
    components: Mapping[str, str],
    *,
    source_roles: Mapping[str, str],
) -> SemanticReplay:
    """Count each atom's participation in source stages without inventing event order.

    Repeated identical components can exchange within-stage event order. Stage/map
    visit counts retain repeated participation while staying invariant to that exchange.
    Every colored intermediate remains visible until the complete bounded search ends.
    """
    if set(components) != set(program.roles) or set(source_roles) != set(program.roles):
        raise LibraryAssemblyError("Grouped origins require every qualified terminal role")
    if any(not isinstance(role, str) or not role for role in source_roles.values()):
        raise LibraryAssemblyError("Source origin roles must be nonempty strings")
    spec, bounds = program.specification, program.bounds
    if len(set(source_roles.values())) != len(source_roles):
        raise LibraryAssemblyError("Grouped source terminal roles must remain distinct")
    stages = spec["stages"]
    if (
        len(stages) != len(program.adapters)
        or not stages
        or any(type(stage["events"]) is not int or stage["events"] < 1 for stage in stages)
        or sum(stage["events"] for stage in stages) > bounds.maximum_events
    ):
        raise LibraryAssemblyError("Grouped event support differs from the qualified program")
    for adapter in program.adapters:
        for template in (
            *adapter.reaction.forward.GetReactants(),
            *adapter.reaction.forward.GetProducts(),
        ):
            if any(
                atom.GetIsotope() or "Isotope" in atom.DescribeQuery()
                for atom in template.GetAtoms()
            ):
                raise LibraryAssemblyError(
                    "Temporary origin labels require isotope-free registry queries"
                )
    canonical = {role: constitutional_molecule(smiles)[0] for role, smiles in components.items()}
    terminal_pass = all(
        _constraints(smiles, spec["terminal_constraints"][role], bounds.maximum_outcomes)["pass"]
        for role, smiles in canonical.items()
    )
    encode: dict[tuple[str, tuple[tuple[str, int], ...]], int] = {}
    decode: dict[int, tuple[str, tuple[tuple[str, int], ...]]] = {}

    def tag(role: str, core: Mapping[str, int]) -> int:
        pair = (role, tuple(sorted(core.items())))
        if pair not in encode:
            index = len(encode) + 1
            if index > 65535:
                raise LibraryAssemblyError("Semantic isotope label capacity exceeded")
            encode[pair], decode[index] = index, pair
        return encode[pair]

    molecules = {}
    for role, smiles in sorted(canonical.items()):
        molecule = Chem.MolFromSmiles(smiles)
        for atom in molecule.GetAtoms():
            atom.SetIsotope(tag(source_roles[role], {}))
        molecules[role] = molecule
    initial = molecules[spec["initial_role"]]
    current = {Chem.MolToSmiles(initial, isomericSmiles=True): initial}
    layers, transitions = [1], 0
    ambiguous_stage = False

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

    for depth, (stage, adapter) in enumerate(zip(stages, program.adapters, strict=True), start=1):
        core_maps = {
            atom.GetAtomMapNum()
            for template in adapter.reaction.forward.GetProducts()
            for atom in template.GetAtoms()
            if atom.GetAtomMapNum()
        }
        if not core_maps:
            raise LibraryAssemblyError("Grouped registry has no mapped product-core positions")
        added = stage["added_roles"]
        if set(adapter.roles) != {stage["accumulator_role"], *added}:
            raise LibraryAssemblyError("Grouped adapter roles differ from the source stage")
        for _ in range(stage["events"]):
            following: dict[str, Chem.Mol] = {}
            for _, state in sorted(current.items()):
                reactants = tuple(
                    Chem.Mol(state if role == stage["accumulator_role"] else molecules[role])
                    for role in adapter.roles
                )
                for molecule in reactants:
                    # Untouched atoms can otherwise retain old_mapno from the preceding reaction.
                    for atom in molecule.GetAtoms():
                        for name in ("old_mapno", "react_atom_idx", "react_idx"):
                            if atom.HasProp(name):
                                atom.ClearProp(name)
                if not all(
                    assessment.qualified
                    for assessment in adapter._assess(
                        tuple(_clear_labels(mol) for mol in reactants)
                    )
                ):
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
                        positions = Counter(dict(core))
                        if atom.HasProp("old_mapno"):
                            number = atom.GetIntProp("old_mapno")
                            if number not in core_maps:
                                raise LibraryAssemblyError(
                                    "Reaction produced an undeclared core position"
                                )
                            positions[f"step_{depth}:map_{number}"] += 1
                        atom.SetIsotope(tag(role, positions))
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
        completed = {
            Chem.MolToSmiles(_clear_labels(state), canonical=True, isomericSmiles=False)
            for state in current.values()
        }
        ambiguous_stage |= len(completed) != 1
    products, annotations, ambiguous = set(), set(), False
    for state in current.values():
        coordinates = [
            (
                decode[atom.GetIsotope()][0],
                "|".join(
                    f"{position}:visits_{count}" for position, count in decode[atom.GetIsotope()][1]
                )
                or "exterior",
            )
            for atom in state.GetAtoms()
        ]
        clean = _clear_labels(state)
        canonical_product = Chem.MolToSmiles(clean, canonical=True, isomericSmiles=False)
        products.add(canonical_product)
        value = canonical_coordinates(clean, coordinates)
        if value is None:
            ambiguous = True
        else:
            annotations.add(value)
    ordered_products = tuple(sorted(products))
    if len(ordered_products) != 1:
        return result("ambiguous_forward_products", complete=True, products=ordered_products)
    if ambiguous_stage:
        return result("ambiguous_completed_source_stage", complete=True, products=ordered_products)
    if ambiguous or len(annotations) != 1:
        return result(
            "ambiguous_forward_atom_coordinates", complete=True, products=ordered_products
        )
    if (
        not terminal_pass
        or not _constraints(
            ordered_products[0], spec["product_constraints"], bounds.maximum_outcomes
        )["pass"]
    ):
        return result(
            "outside_qualified_program_constraints", complete=True, products=ordered_products
        )
    return result(
        "unique_forward_atom_coordinates",
        complete=True,
        annotations=next(iter(annotations)),
        products=ordered_products,
    )
