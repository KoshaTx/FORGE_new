"""Public contracts for exact L1 assembly checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ForwardAssemblyCheck:
    """Outcome of replaying a registry-defined L1 transform.

    ``exact`` establishes forward consistency only.  It is deliberately not named success,
    synthesis probability, or route certification because L2/L3 may still be open.
    """

    reaction_id: str
    roles: tuple[str, ...]
    exact: bool
    saturated: bool
    enumerated_outcomes: int

    def __post_init__(self) -> None:
        if not self.reaction_id or not self.roles:
            raise ValueError("assembly checks require a reaction id and role order")
        if self.enumerated_outcomes < 0:
            raise ValueError("enumerated outcome count must be non-negative")


class AssemblyAdapter(Protocol):
    """The stable seam used by generation and routing for one L1 transform."""

    @property
    def reaction_id(self) -> str: ...

    @property
    def roles(self) -> tuple[str, ...]: ...

    def check_forward(
        self,
        components: Mapping[str, str],
        product_smiles: str,
        *,
        maximum_outcomes: int = 64,
    ) -> ForwardAssemblyCheck: ...


__all__ = ["AssemblyAdapter", "ForwardAssemblyCheck"]
