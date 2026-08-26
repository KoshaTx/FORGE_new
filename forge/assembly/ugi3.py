"""Registry-backed AGILE-type Ugi 3CR assembly adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.assembly.api import ForwardAssemblyCheck, ForwardAssemblyProducts
from forge.assembly.program import repair_template_hydrogens
from forge.core.hashing import sha256_file


class Ugi3AssemblyError(RuntimeError):
    """The qualified registry cannot provide the declared Ugi assembly contract."""


@dataclass(frozen=True)
class Ugi3DecompositionTrace:
    """One exact open Ugi decomposition whose forward replay returns the product."""

    product_smiles: str
    components: tuple[tuple[str, str], ...]

    def as_mapping(self) -> dict[str, str]:
        return dict(self.components)


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
        from forge.corpus.r1_prime_audit import QUALIFIED_STATUS
        from forge.corpus.ugi_held_component_gate import load_ugi_reaction_contract

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
        products = self.forward_products(components, maximum_outcomes=maximum_outcomes)
        with rdBase.BlockLogs():
            product = Chem.MolFromSmiles(product_smiles)
        if product is None or len(Chem.GetMolFrags(product)) != 1:
            raise Ugi3AssemblyError("target product must be a valid connected molecular graph")
        canonical = Chem.MolToSmiles(product, canonical=True, isomericSmiles=False)
        return ForwardAssemblyCheck(
            reaction_id=self.reaction_id,
            roles=self.roles,
            exact=canonical in products.products,
            saturated=products.saturated,
            enumerated_outcomes=products.enumerated_outcomes_by_step[0],
        )

    def forward_products(
        self,
        components: Mapping[str, str],
        *,
        maximum_outcomes: int = 64,
    ) -> ForwardAssemblyProducts:
        """Assemble a precursor tuple without selecting a component or product outcome."""

        if isinstance(maximum_outcomes, bool) or maximum_outcomes < 1:
            raise Ugi3AssemblyError("maximum_outcomes must be a positive integer")
        missing = set(self.roles) - set(components)
        unknown = set(components) - set(self.roles)
        if missing or unknown:
            raise Ugi3AssemblyError(
                f"component roles differ from registry; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        reactants = []
        for role in self.roles:
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(components[role])
            if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
                raise Ugi3AssemblyError(f"{role} is not a valid connected component")
            reactants.append(molecule)
        with rdBase.BlockLogs():
            outcomes = self._compiled.forward.RunReactants(
                tuple(reactants), maxProducts=maximum_outcomes
            )
        products: set[str] = set()
        for outcome in outcomes:
            if len(outcome) != 1:
                continue
            product = Chem.Mol(outcome[0])
            try:
                with rdBase.BlockLogs():
                    Chem.SanitizeMol(product)
            except (ValueError, RuntimeError):
                continue
            products.add(Chem.MolToSmiles(product, canonical=True, isomericSmiles=False))
        return ForwardAssemblyProducts(
            assembly_id=self.reaction_id,
            reaction_id=self.reaction_id,
            roles=self.roles,
            products=tuple(sorted(products)),
            saturated=len(outcomes) >= maximum_outcomes,
            enumerated_outcomes_by_step=(len(outcomes),),
        )

    def decompose(
        self,
        product_smiles: str,
        *,
        maximum_outcomes: int = 128,
    ) -> tuple[Ugi3DecompositionTrace, ...]:
        """Enumerate handle-qualified reverse traces and retain exact forward round trips."""

        from forge.corpus.ugi_held_component_gate import reaction_handle_qualification

        if isinstance(maximum_outcomes, bool) or maximum_outcomes < 1:
            raise Ugi3AssemblyError("maximum_outcomes must be a positive integer")
        with rdBase.BlockLogs():
            product = Chem.MolFromSmiles(product_smiles)
        if product is None or len(Chem.GetMolFrags(product)) != 1:
            return ()
        canonical = Chem.MolToSmiles(product, canonical=True, isomericSmiles=False)
        with rdBase.BlockLogs():
            outcomes = self._compiled.reverse.RunReactants((product,), maxProducts=maximum_outcomes)
        if len(outcomes) >= maximum_outcomes:
            raise Ugi3AssemblyError(
                f"Ugi reverse decomposition reached maximum_outcomes={maximum_outcomes}"
            )
        traces: set[tuple[tuple[str, str], ...]] = set()
        policies = self._compiled.definition.reactant_roles
        for outcome in outcomes:
            if len(outcome) != len(self.roles):
                continue
            components: list[tuple[str, str]] = []
            qualified = True
            for index, (role, fragment) in enumerate(zip(self.roles, outcome, strict=True)):
                repaired = repair_template_hydrogens(fragment)
                if repaired is None:
                    qualified = False
                    break
                smiles, molecule = repaired
                handle_check = reaction_handle_qualification(
                    molecule,
                    query=self._compiled.handles[index],
                    forbidden=self._compiled.forbidden[index],
                    allowed_site_multiplicity=policies[index].allowed_site_multiplicity,
                )
                if not handle_check["passes_registry_handle_policy"]:
                    qualified = False
                    break
                components.append((role, smiles))
            if not qualified:
                continue
            trace = tuple(components)
            forward_check = self.check_forward(
                dict(trace), canonical, maximum_outcomes=maximum_outcomes
            )
            if forward_check.exact and not forward_check.saturated:
                traces.add(trace)
        return tuple(
            Ugi3DecompositionTrace(product_smiles=canonical, components=trace)
            for trace in sorted(traces)
        )


__all__ = ["Ugi3AssemblyAdapter", "Ugi3AssemblyError", "Ugi3DecompositionTrace"]
