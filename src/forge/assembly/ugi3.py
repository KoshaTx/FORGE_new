"""Registry-backed AGILE-type Ugi 3CR assembly adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.assembly.api import ForwardAssemblyCheck
from forge.core.hashing import sha256_file


class Ugi3AssemblyError(RuntimeError):
    """The qualified registry cannot provide the declared Ugi assembly contract."""


@dataclass(frozen=True)
class Ugi3AssemblyAdapter:
    """Exact forward validation through the hash-pinned qualified registry."""

    _compiled: Any
    registry_path: Path
    registry_sha256: str

    @classmethod
    def from_registry(
        cls,
        registry_path: Path,
        *,
        expected_sha256: str | None = None,
    ) -> Ugi3AssemblyAdapter:
        from forge.data.r1_prime_audit import QUALIFIED_STATUS
        from forge.product.ugi_held_component_gate import load_ugi_reaction_contract

        resolved = registry_path.resolve()
        observed = str(sha256_file(resolved))
        if expected_sha256 is not None and observed != expected_sha256:
            raise Ugi3AssemblyError(
                f"qualified reaction registry changed: expected {expected_sha256}, found {observed}"
            )
        compiled = load_ugi_reaction_contract(resolved)
        if compiled.definition.status != QUALIFIED_STATUS:
            raise Ugi3AssemblyError(
                f"reaction {compiled.definition.reaction_id!r} is not qualified"
            )
        return cls(_compiled=compiled, registry_path=resolved, registry_sha256=observed)

    @property
    def reaction_id(self) -> str:
        return str(self._compiled.definition.reaction_id)

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(str(role.name) for role in self._compiled.definition.reactant_roles)

    def check_forward(
        self,
        components: Mapping[str, str],
        product_smiles: str,
        *,
        maximum_outcomes: int = 64,
    ) -> ForwardAssemblyCheck:
        from forge.product.ugi_held_component_gate import (
            exact_forward_reconstructs_ugi_product,
        )

        if isinstance(maximum_outcomes, bool) or maximum_outcomes < 1:
            raise Ugi3AssemblyError("maximum_outcomes must be a positive integer")
        missing = set(self.roles) - set(components)
        unknown = set(components) - set(self.roles)
        if missing or unknown:
            raise Ugi3AssemblyError(
                f"component roles differ from registry; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        exact, saturated, outcomes = exact_forward_reconstructs_ugi_product(
            self._compiled,
            components,
            product_smiles,
            maximum_outcomes=maximum_outcomes,
        )
        return ForwardAssemblyCheck(
            reaction_id=self.reaction_id,
            roles=self.roles,
            exact=exact,
            saturated=saturated,
            enumerated_outcomes=outcomes,
        )


__all__ = ["Ugi3AssemblyAdapter", "Ugi3AssemblyError"]
