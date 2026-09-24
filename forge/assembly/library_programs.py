"""Bounded recovery of minimal computed programs from saved component identities.

Raw transform execution diagnoses the corpus; only paths passing the unchanged registry role
policy at EVERY step are eligible for transform-consistency supervision. No path is evidence of
experimental execution. Distinct ordered constitutional paths are counted (capped at two), not
silently replaced by a unique-source-order claim.
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


@dataclass(frozen=True)
class LibraryProgramLimits:
    maximum_steps: int
    maximum_outcomes: int = 256
    maximum_states: int = 4096
    maximum_expansions: int = 16384

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise LibraryAssemblyError("program limits must be positive integers")


@dataclass(frozen=True)
class RecoveredLibraryProgram:
    product_smiles: str
    minimum_steps: int | None
    intermediate_products: tuple[str, ...]
    policy_intermediate_products: tuple[str, ...]
    minimum_path_count_capped_at_two: int
    policy_path_count_capped_at_two: int
    search_complete_through_minimum_depth: bool
    unresolved_reason: str

    @property
    def exact(self) -> bool:
        return self.minimum_steps is not None

    @property
    def qualified_unique_program(self) -> bool:
        return (
            self.search_complete_through_minimum_depth and self.policy_path_count_capped_at_two == 1
        )


@dataclass(frozen=True)
class LibraryProgramSearch:
    reaction_id: str
    accumulator_role: str | None
    targets: tuple[RecoveredLibraryProgram, ...]
    completed_depth: int
    expansions: int
    peak_states: int
    termination: str


@dataclass(frozen=True)
class _State:
    path: tuple[str, ...]
    count: int
    policy_path: tuple[str, ...]
    policy_count: int


def _layer(
    adapter: RegistryAssemblyAdapter,
    components: Mapping[str, str],
    maximum_outcomes: int,
) -> tuple[tuple[str, ...], bool, bool]:
    reactants = adapter._reactants(components)
    policy_passes = all(value.qualified for value in adapter._assess(reactants))
    with rdBase.BlockLogs():
        raw = adapter.reaction.forward.RunReactants(reactants, maxProducts=maximum_outcomes)
    products = set()
    for outcome in raw:
        if len(outcome) != 1:
            continue
        molecule = Chem.Mol(outcome[0])
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(molecule)
            canonical, _ = constitutional_molecule(Chem.MolToSmiles(molecule))
        except (ValueError, RuntimeError):
            continue
        products.add(canonical)
    return tuple(sorted(products)), len(raw) >= maximum_outcomes, policy_passes


def _strictly_appends(adapter: RegistryAssemblyAdapter, accumulator_role: str | None) -> bool:
    """Prove growth for a single retained accumulator atom with a new heavy-atom attachment.

    With only one accumulator atom in the template no existing accumulator bond can be deleted.
    Its mapped atom and all context survive. More complicated templates get no size shortcut.
    """
    if accumulator_role is None:
        return False
    reaction = adapter.reaction.forward
    template = reaction.GetReactantTemplate(adapter.roles.index(accumulator_role))
    if template.GetNumAtoms() != 1:
        return False
    atom = template.GetAtomWithIdx(0)
    mapped = atom.GetAtomMapNum()
    product = reaction.GetProductTemplate(0)
    retained = [a for a in product.GetAtoms() if a.GetAtomMapNum() == mapped]
    return (
        mapped > 0
        and len(retained) == 1
        and atom.GetAtomicNum() > 1
        and retained[0].GetAtomicNum() == atom.GetAtomicNum()
        and any(a.GetAtomicNum() > 1 and a.GetAtomMapNum() != mapped for a in product.GetAtoms())
    )


def recover_library_programs(
    adapter: RegistryAssemblyAdapter,
    components: Mapping[str, str],
    targets: Sequence[str],
    *,
    accumulator_role: str | None,
    limits: LibraryProgramLimits,
    append_only_size_bound: bool = True,
) -> LibraryProgramSearch:
    """Exhaust breadth-first layers until all saved targets have minimal witnesses or a bound hits.

    For repeated programs the designated accumulator can occupy either registry input position.
    Other component identities stay fixed. A state cap invalidates the incomplete layer, even if
    it happened to contain a target. Earlier COMPLETE layers remain valid. The source target is
    used for exact membership and a proven append-only size bound, never to select reactive sites.
    """

    if set(components) != set(adapter.roles):
        raise LibraryAssemblyError("program component roles differ from registry")
    if type(append_only_size_bound) is not bool:
        raise LibraryAssemblyError("append_only_size_bound must be boolean")
    if accumulator_role is None:
        if limits.maximum_steps != 1:
            raise LibraryAssemblyError("a fixed-arity program must have exactly one step")
    elif len(adapter.roles) != 2 or accumulator_role not in adapter.roles:
        raise LibraryAssemblyError("repeated programs require two roles and a declared accumulator")
    canonical_components = {
        role: constitutional_molecule(value)[0] for role, value in components.items()
    }
    target_set = {constitutional_molecule(value)[0] for value in targets}
    if not target_set:
        raise LibraryAssemblyError("program search requires saved targets")
    start = canonical_components[accumulator_role] if accumulator_role is not None else ""
    current = {start: _State((), 1, (), 1)}
    recovered: dict[str, RecoveredLibraryProgram] = {}
    expansions, peak, completed_depth = 0, 1, 0
    termination = "maximum_steps"
    growing = append_only_size_bound and _strictly_appends(adapter, accumulator_role)
    largest_target = max(constitutional_molecule(p)[1].GetNumHeavyAtoms() for p in target_set)
    for depth in range(1, limits.maximum_steps + 1):
        if growing and all(
            constitutional_molecule(p)[1].GetNumHeavyAtoms() >= largest_target for p in current
        ):
            termination = "append_only_frontier_above_targets"
            break
        following: dict[str, _State] = {}
        failure = ""
        for accumulator, state in sorted(current.items()):
            if expansions >= limits.maximum_expansions:
                failure = "expansion_limit"
                break
            inputs = dict(canonical_components)
            if accumulator_role is not None:
                inputs[accumulator_role] = accumulator
            products, saturated, policy = _layer(adapter, inputs, limits.maximum_outcomes)
            expansions += 1
            if saturated:
                failure = "outcome_limit"
                break
            for product in products:
                path = (*state.path, product)
                policy_count = state.policy_count if policy else 0
                policy_path = (*state.policy_path, product) if policy_count else ()
                old = following.get(product)
                if old is None:
                    following[product] = _State(path, state.count, policy_path, policy_count)
                else:
                    allowed = [p for p in (old.policy_path, policy_path) if p]
                    following[product] = _State(
                        min(old.path, path),
                        min(2, old.count + state.count),
                        min(allowed) if allowed else (),
                        min(2, old.policy_count + policy_count),
                    )
            peak = max(peak, len(following))
            if len(following) > limits.maximum_states:
                failure = "state_limit"
                break
        if failure:
            termination = failure
            break
        completed_depth = depth
        for product in sorted(target_set.difference(recovered).intersection(following)):
            state = following[product]
            recovered[product] = RecoveredLibraryProgram(
                product,
                depth,
                state.path,
                state.policy_path,
                state.count,
                state.policy_count,
                True,
                "",
            )
        if len(recovered) == len(target_set):
            termination = "all_targets_recovered"
            break
        if not following:
            termination = "exhausted"
            break
        current = following
    for product in sorted(target_set.difference(recovered)):
        recovered[product] = RecoveredLibraryProgram(
            product, None, (), (), 0, 0, False, termination
        )
    return LibraryProgramSearch(
        adapter.reaction_id,
        accumulator_role,
        tuple(recovered[p] for p in sorted(recovered)),
        completed_depth,
        expansions,
        peak,
        termination,
    )


def replay_library_program(
    adapter: RegistryAssemblyAdapter,
    components: Mapping[str, str],
    intermediate_products: Sequence[str],
    *,
    accumulator_role: str | None,
    maximum_outcomes: int = 256,
) -> bool:
    """Authenticate every supplied intermediate through the STRICT adapter, including saturation."""

    if not intermediate_products or (accumulator_role is None and len(intermediate_products) != 1):
        raise LibraryAssemblyError("invalid fixed/repeated program depth")
    if accumulator_role is not None and (
        len(adapter.roles) != 2 or accumulator_role not in adapter.roles
    ):
        raise LibraryAssemblyError("invalid accumulator role")
    current = dict(components)
    for target in intermediate_products:
        check = adapter.check_forward(current, target, maximum_outcomes=maximum_outcomes)
        if not check.exact or check.saturated:
            return False
        if accumulator_role is not None:
            current[accumulator_role] = target
    return True
