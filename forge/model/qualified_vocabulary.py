"""Explicit, versioned valence support for new corpora; frozen defaults stay unchanged."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload

from rdkit import Chem

from forge.model.defog_feasibility import AtomState, FeasibilityError
from forge.model.sparse_topology_feasibility import _maximum_valence_units


@dataclass(frozen=True)
class QualifiedAtomVocabulary(Sequence[AtomState]):
    """An immutable vocabulary with narrowly validated, opt-in neutral halogen support.

    Unsupported states still fail. Existing atom states retain exactly their historical
    capacities, including explicit-hydrogen accounting. Merely adding an atom to an ordinary
    vocabulary never changes decoder policy.
    """

    states: tuple[AtomState, ...]
    neutral_monovalent_extensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.states, tuple)
            or not self.states
            or len(set(self.states)) != len(self.states)
        ):
            raise ValueError("qualified vocabulary requires a nonempty unique tuple of states")
        if not isinstance(self.neutral_monovalent_extensions, tuple):
            raise ValueError("valence extensions must be immutable")
        for symbol in self.neutral_monovalent_extensions:
            # The artifact names the scope. RDKit validates it; no unbounded generic fallback.
            if list(Chem.GetPeriodicTable().GetValenceList(symbol)) != [1]:
                raise ValueError(f"extension is not exclusively monovalent in RDKit: {symbol}")
            if symbol not in {state.symbol for state in self.states}:
                raise ValueError(f"extension is absent from the vocabulary: {symbol}")
        self.capacities()

    def __len__(self) -> int:
        return len(self.states)

    @overload
    def __getitem__(self, index: int) -> AtomState: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[AtomState, ...]: ...

    def __getitem__(self, index: int | slice) -> AtomState | tuple[AtomState, ...]:
        return self.states[index]

    def __iter__(self) -> Iterator[AtomState]:
        return iter(self.states)

    def capacities(self) -> tuple[int, ...]:
        values = []
        for state in self.states:
            try:
                total = _maximum_valence_units(state)
            except FeasibilityError:
                if (
                    state.symbol not in self.neutral_monovalent_extensions
                    or state.formal_charge != 0
                    or state.aromatic
                    or state.explicit_hydrogens != 0
                ):
                    raise
                total = 2
            capacity = total - 2 * state.explicit_hydrogens
            if capacity < 0:
                raise ValueError(f"explicit hydrogens exceed capacity: {state}")
            values.append(capacity)
        return tuple(values)
